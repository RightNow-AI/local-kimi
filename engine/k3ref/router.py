"""Exact inference-time Kimi K3 noaux_tc router."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class K3Router(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        top_k: int,
        *,
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
        self.num_expert_group = num_expert_group
        self.topk_group = topk_group
        self.renormalize = renormalize
        self.routed_scaling_factor = routed_scaling_factor
        self.activation = activation
        self.weight = nn.Parameter(
            torch.empty(num_experts, hidden_size, device=device, dtype=dtype)
        )
        self.e_score_correction_bias = nn.Parameter(
            torch.empty(num_experts, device=device, dtype=torch.float32)
        )
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        nn.init.zeros_(self.e_score_correction_bias)

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        assert not self.training
        hidden_size = hidden_states.shape[-1]
        flat_states = hidden_states.view(-1, hidden_size)
        logits = F.linear(flat_states.float(), self.weight.float(), None)
        if self.activation == "sigmoid":
            scores = logits.sigmoid()
        elif self.activation == "softmax":
            scores = logits.softmax(dim=1)
        else:
            raise NotImplementedError(f"unsupported router activation: {self.activation}")

        scores_for_choice = scores + self.e_score_correction_bias.float().unsqueeze(0)
        if self.num_expert_group > 1 and self.num_expert_group > self.topk_group:
            grouped = scores_for_choice.view(
                flat_states.shape[0], self.num_expert_group, -1
            )
            group_scores = grouped.topk(2, dim=-1)[0].sum(dim=-1)
            group_indices = torch.topk(
                group_scores, k=self.topk_group, dim=-1, sorted=False
            )[1]
            group_mask = torch.zeros_like(group_scores)
            group_mask.scatter_(1, group_indices, 1)
            score_mask = (
                group_mask.unsqueeze(-1)
                .expand(
                    flat_states.shape[0],
                    self.num_expert_group,
                    self.num_experts // self.num_expert_group,
                )
                .reshape(flat_states.shape[0], -1)
            )
            choice_scores = scores_for_choice.masked_fill(
                ~score_mask.bool(), float("-inf")
            )
        else:
            choice_scores = scores_for_choice

        indices = torch.topk(
            choice_scores, k=self.top_k, dim=-1, sorted=False
        )[1]
        weights = scores.gather(1, indices)
        if self.top_k > 1 and self.renormalize:
            weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
        return indices, weights * self.routed_scaling_factor
