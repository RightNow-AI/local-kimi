"""Inference-time sigmoid router used by Kimi-Linear."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class KLinearRouter(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        top_k: int,
        *,
        use_grouped_topk: bool = True,
        num_expert_group: int = 1,
        topk_group: int = 1,
        renormalize: bool = True,
        routed_scaling_factor: float = 1.0,
        activation: str = "sigmoid",
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if num_experts % num_expert_group:
            raise ValueError("num_experts must be divisible by num_expert_group")
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.top_k = top_k
        self.use_grouped_topk = use_grouped_topk
        self.num_expert_group = num_expert_group
        self.topk_group = topk_group
        self.renormalize = renormalize
        self.routed_scaling_factor = routed_scaling_factor
        self.activation = activation
        self.weight = nn.Parameter(
            torch.empty(num_experts, hidden_size, device=device, dtype=dtype)
        )
        self.e_score_correction_bias = nn.Parameter(
            torch.empty(num_experts, device=device, dtype=dtype)
        )
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        nn.init.zeros_(self.e_score_correction_bias)

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.training:
            raise NotImplementedError("training mode is not supported by the Kimi-Linear router")
        flat_states = hidden_states.reshape(-1, hidden_states.shape[-1])
        logits = F.linear(flat_states.float(), self.weight.float(), None)
        if self.activation == "sigmoid":
            scores = logits.sigmoid()
        elif self.activation == "softmax":
            scores = logits.softmax(dim=1)
        else:
            raise NotImplementedError(f"unsupported router activation: {self.activation}")

        scores_for_choice = scores + self.e_score_correction_bias.float().unsqueeze(0)
        if self.use_grouped_topk:
            experts_per_group = self.num_experts // self.num_expert_group
            grouped = scores_for_choice.view(
                flat_states.shape[0], self.num_expert_group, experts_per_group
            )
            group_width = min(2, experts_per_group)
            group_scores = grouped.topk(group_width, dim=-1)[0].sum(dim=-1)
            group_indices = group_scores.topk(
                self.topk_group, dim=-1, sorted=False
            )[1]
            group_mask = torch.zeros_like(group_scores)
            group_mask.scatter_(1, group_indices, 1)
            score_mask = (
                group_mask.unsqueeze(-1)
                .expand(-1, self.num_expert_group, experts_per_group)
                .reshape(flat_states.shape[0], self.num_experts)
            )
            choice_scores = scores_for_choice.masked_fill(~score_mask.bool(), 0.0)
        else:
            choice_scores = scores_for_choice

        indices = choice_scores.topk(self.top_k, dim=-1, sorted=False)[1]
        weights = scores.gather(1, indices)
        if self.top_k > 1 and self.renormalize:
            weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
        return indices, weights * self.routed_scaling_factor

