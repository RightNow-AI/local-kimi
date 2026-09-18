"""SwiGLU dense and sparse feed-forward paths for Kimi-Linear."""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import nn

from .quantized import LinearFactory, W4A16Linear, make_linear
from .router import KLinearRouter

ExpertLinear = torch.Tensor | W4A16Linear
ExpertProvider = Callable[
    [int, int, torch.device, torch.dtype],
    tuple[ExpertLinear, ExpertLinear, ExpertLinear],
]


class ExpertMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        *,
        tensor_prefix: str = "expert",
        linear_factory: LinearFactory | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.w1 = make_linear(
            f"{tensor_prefix}.w1.weight",
            hidden_size,
            intermediate_size,
            linear_factory=linear_factory,
            device=device,
            dtype=dtype,
        )
        self.w2 = make_linear(
            f"{tensor_prefix}.w2.weight",
            intermediate_size,
            hidden_size,
            linear_factory=linear_factory,
            device=device,
            dtype=dtype,
        )
        self.w3 = make_linear(
            f"{tensor_prefix}.w3.weight",
            hidden_size,
            intermediate_size,
            linear_factory=linear_factory,
            device=device,
            dtype=dtype,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(hidden_states)) * self.w3(hidden_states))


class DenseMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        *,
        tensor_prefix: str = "mlp",
        linear_factory: LinearFactory | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.gate_proj = make_linear(
            f"{tensor_prefix}.gate_proj.weight",
            hidden_size,
            intermediate_size,
            linear_factory=linear_factory,
            device=device,
            dtype=dtype,
        )
        self.up_proj = make_linear(
            f"{tensor_prefix}.up_proj.weight",
            hidden_size,
            intermediate_size,
            linear_factory=linear_factory,
            device=device,
            dtype=dtype,
        )
        self.down_proj = make_linear(
            f"{tensor_prefix}.down_proj.weight",
            intermediate_size,
            hidden_size,
            linear_factory=linear_factory,
            device=device,
            dtype=dtype,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.down_proj(
            F.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states)
        )


class KLinearMoE(nn.Module):
    def __init__(
        self,
        layer_idx: int,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
        top_k: int,
        *,
        num_shared_experts: int = 0,
        use_grouped_topk: bool = True,
        num_expert_group: int = 1,
        topk_group: int = 1,
        renormalize: bool = True,
        routed_scaling_factor: float = 1.0,
        router_activation: str = "sigmoid",
        expert_provider: ExpertProvider | None = None,
        tensor_prefix: str = "block_sparse_moe",
        linear_factory: LinearFactory | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_experts = num_experts
        self.top_k = top_k
        self.expert_provider = expert_provider
        self.gate = KLinearRouter(
            hidden_size,
            num_experts,
            top_k,
            use_grouped_topk=use_grouped_topk,
            num_expert_group=num_expert_group,
            topk_group=topk_group,
            renormalize=renormalize,
            routed_scaling_factor=routed_scaling_factor,
            activation=router_activation,
            device=device,
            dtype=dtype,
        )
        if expert_provider is None:
            self.experts = nn.ModuleList(
                [
                    ExpertMLP(
                        hidden_size,
                        intermediate_size,
                        tensor_prefix=f"{tensor_prefix}.experts.{expert_id}",
                        linear_factory=linear_factory,
                        device=device,
                        dtype=dtype,
                    )
                    for expert_id in range(num_experts)
                ]
            )
        else:
            self.experts = nn.ModuleList()

        shared_intermediate = intermediate_size * num_shared_experts
        self.shared_experts = (
            DenseMLP(
                hidden_size,
                shared_intermediate,
                tensor_prefix=f"{tensor_prefix}.shared_experts",
                linear_factory=linear_factory,
                device=device,
                dtype=dtype,
            )
            if shared_intermediate
            else None
        )

    def _run_expert(self, expert_id: int, tokens: torch.Tensor) -> torch.Tensor:
        if self.expert_provider is None:
            return self.experts[expert_id](tokens)
        w1, w2, w3 = self.expert_provider(
            self.layer_idx, expert_id, tokens.device, tokens.dtype
        )
        if isinstance(w1, W4A16Linear):
            if not isinstance(w2, W4A16Linear) or not isinstance(w3, W4A16Linear):
                raise TypeError("expert provider returned mixed linear weight types")
            return w2(F.silu(w1(tokens)) * w3(tokens))
        if isinstance(w2, W4A16Linear) or isinstance(w3, W4A16Linear):
            raise TypeError("expert provider returned mixed linear weight types")
        return F.linear(F.silu(F.linear(tokens, w1)) * F.linear(tokens, w3), w2)

    @torch.no_grad()
    def _route_experts(
        self,
        hidden_states: torch.Tensor,
        expert_indices: torch.Tensor,
        expert_weights: torch.Tensor,
    ) -> torch.Tensor:
        expert_outputs = torch.empty(
            (*expert_indices.shape, hidden_states.shape[-1]),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        for expert_id in range(self.num_experts):
            token_indices, slots = torch.where(expert_indices == expert_id)
            if token_indices.numel() == 0:
                continue
            outputs = self._run_expert(expert_id, hidden_states[token_indices])
            expert_outputs[token_indices, slots] = outputs
        return (
            expert_outputs.float()
            .mul(expert_weights.unsqueeze(-1))
            .sum(dim=1)
            .to(hidden_states.dtype)
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        return_router: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.training:
            raise NotImplementedError("training mode is not supported by Kimi-Linear MoE")
        identity = hidden_states
        original_shape = hidden_states.shape
        expert_indices, expert_weights = self.gate(hidden_states)
        flat_states = hidden_states.reshape(-1, original_shape[-1])
        routed = self._route_experts(flat_states, expert_indices, expert_weights)
        routed = routed.view(original_shape)
        if self.shared_experts is not None:
            routed = routed + self.shared_experts(identity)
        if return_router:
            return routed, expert_indices, expert_weights
        return routed

