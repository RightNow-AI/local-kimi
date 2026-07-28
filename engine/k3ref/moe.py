"""Slow, direct implementation of K3's latent mixture of experts."""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import nn

from .norm import RMSNorm, situ
from .router import K3Router

ExpertProvider = Callable[
    [int, torch.device, torch.dtype],
    tuple[torch.Tensor, torch.Tensor, torch.Tensor],
]


class K3ExpertMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        *,
        situ_beta: float = 4.0,
        situ_linear_beta: float | None = 25.0,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        factory = {"device": device, "dtype": dtype}
        self.w1 = nn.Linear(hidden_size, intermediate_size, bias=False, **factory)
        self.w2 = nn.Linear(intermediate_size, hidden_size, bias=False, **factory)
        self.w3 = nn.Linear(hidden_size, intermediate_size, bias=False, **factory)
        self.situ_beta = situ_beta
        self.situ_linear_beta = situ_linear_beta

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        activated = situ(
            self.w1(hidden_states),
            self.w3(hidden_states),
            self.situ_beta,
            self.situ_linear_beta,
        )
        return self.w2(activated)


class K3SharedMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        *,
        situ_beta: float = 4.0,
        situ_linear_beta: float | None = 25.0,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        factory = {"device": device, "dtype": dtype}
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False, **factory)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False, **factory)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False, **factory)
        self.situ_beta = situ_beta
        self.situ_linear_beta = situ_linear_beta

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        activated = situ(
            self.gate_proj(hidden_states),
            self.up_proj(hidden_states),
            self.situ_beta,
            self.situ_linear_beta,
        )
        return self.down_proj(activated)


class LatentMoE(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        latent_size: int,
        expert_intermediate_size: int,
        num_experts: int,
        top_k: int,
        *,
        num_shared_experts: int = 0,
        num_expert_group: int = 1,
        topk_group: int = 1,
        renormalize: bool = True,
        routed_scaling_factor: float = 1.0,
        rms_norm_eps: float = 1e-5,
        situ_beta: float = 4.0,
        situ_linear_beta: float | None = 25.0,
        expert_provider: ExpertProvider | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        factory = {"device": device, "dtype": dtype}
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.expert_intermediate_size = expert_intermediate_size
        self.num_experts = num_experts
        self.top_k = top_k
        self.situ_beta = situ_beta
        self.situ_linear_beta = situ_linear_beta
        self.expert_provider = expert_provider

        self.gate = K3Router(
            hidden_size,
            num_experts,
            top_k,
            num_expert_group=num_expert_group,
            topk_group=topk_group,
            renormalize=renormalize,
            routed_scaling_factor=routed_scaling_factor,
            device=device,
            dtype=dtype,
        )
        self.routed_expert_down_proj = nn.Linear(
            hidden_size, latent_size, bias=False, **factory
        )
        self.routed_expert_norm = RMSNorm(
            latent_size, eps=rms_norm_eps, device=device, dtype=dtype
        )
        self.routed_expert_up_proj = nn.Linear(
            latent_size, hidden_size, bias=False, **factory
        )

        if expert_provider is None:
            self.experts = nn.ModuleList(
                [
                    K3ExpertMLP(
                        latent_size,
                        expert_intermediate_size,
                        situ_beta=situ_beta,
                        situ_linear_beta=situ_linear_beta,
                        device=device,
                        dtype=dtype,
                    )
                    for _ in range(num_experts)
                ]
            )
        else:
            # Real checkpoints are decoded only for experts selected by the router.
            self.experts = nn.ModuleList()

        shared_intermediate = expert_intermediate_size * num_shared_experts
        self.shared_experts = (
            K3SharedMLP(
                hidden_size,
                shared_intermediate,
                situ_beta=situ_beta,
                situ_linear_beta=situ_linear_beta,
                device=device,
                dtype=dtype,
            )
            if shared_intermediate
            else None
        )

    def _run_expert(self, expert_id: int, tokens: torch.Tensor) -> torch.Tensor:
        if self.expert_provider is None:
            return self.experts[expert_id](tokens)
        w1, w2, w3 = self.expert_provider(expert_id, tokens.device, tokens.dtype)
        activated = situ(
            F.linear(tokens, w1),
            F.linear(tokens, w3),
            self.situ_beta,
            self.situ_linear_beta,
        )
        return F.linear(activated, w2)

    @torch.no_grad()
    def _route_experts(
        self,
        latent_states: torch.Tensor,
        expert_indices: torch.Tensor,
        expert_weights: torch.Tensor,
    ) -> torch.Tensor:
        expert_outputs = torch.empty(
            (*expert_indices.shape, latent_states.shape[-1]),
            dtype=latent_states.dtype,
            device=latent_states.device,
        )
        for expert_id in range(self.num_experts):
            token_indices, slots = torch.where(expert_indices == expert_id)
            if token_indices.numel() == 0:
                continue
            expert_output = self._run_expert(expert_id, latent_states[token_indices])
            expert_outputs[token_indices, slots] = expert_output
        return (
            expert_outputs.float()
            .mul(expert_weights.unsqueeze(-1))
            .sum(dim=1)
            .to(latent_states.dtype)
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        return_router: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.training:
            raise NotImplementedError("training mode is not supported by the K3 MoE reference")
        identity = hidden_states
        original_shape = hidden_states.shape
        expert_indices, expert_weights = self.gate(hidden_states)
        flat_states = hidden_states.view(-1, original_shape[-1])

        # These names look reversed geometrically, but match Moonshot's module names.
        latent_states = self.routed_expert_down_proj(flat_states)
        routed = self._route_experts(latent_states, expert_indices, expert_weights)
        routed = self.routed_expert_norm(routed)
        routed = self.routed_expert_up_proj(routed).view(original_shape)

        if self.shared_experts is not None:
            routed = routed + self.shared_experts(identity)
        if return_router:
            return routed, expert_indices, expert_weights
        return routed
