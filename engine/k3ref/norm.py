"""Small normalization and activation primitives used by K3."""

from __future__ import annotations

import torch
from torch import nn


def rms_norm(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    dtype = hidden_states.dtype
    values = hidden_states.float()
    values = values * torch.rsqrt(values.square().mean(dim=-1, keepdim=True) + eps)
    return weight * values.to(dtype)


def situ(
    gate: torch.Tensor,
    up: torch.Tensor,
    beta: float,
    linear_beta: float | None,
) -> torch.Tensor:
    dtype = gate.dtype
    gate_float = gate.float()
    up_float = up.float()
    activated = beta * torch.tanh(gate_float / beta) * torch.sigmoid(gate_float)
    if linear_beta is not None:
        up_float = linear_beta * torch.tanh(up_float / linear_beta)
    return (activated * up_float).to(dtype)


def apply_attention_residual(
    prefix_sum: torch.Tensor,
    block_residual: torch.Tensor,
    projection_weight: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Moonshot's learned softmax residual mixer."""
    values = torch.cat((block_residual, prefix_sum.unsqueeze(1)), dim=1)
    values_float = values.float()
    variance = values_float.square().mean(dim=-1, keepdim=True)
    normalized = values_float * torch.rsqrt(variance + eps)
    score_weight = norm_weight.float() * projection_weight.squeeze(0).float()
    scores = (normalized * score_weight).sum(dim=-1)
    probabilities = scores.softmax(dim=-1).unsqueeze(1)
    mixed = torch.matmul(probabilities, values_float).squeeze(1)
    return mixed.to(values.dtype)


class RMSNorm(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.variance_epsilon = eps
        self.weight = nn.Parameter(torch.ones(hidden_size, device=device, dtype=dtype))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return rms_norm(hidden_states, self.weight, self.variance_epsilon)


class RMSGatedNorm(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-5,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size, device=device, dtype=dtype))

    def forward(self, values: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        dtype = values.dtype
        normalized = values.float() * torch.rsqrt(
            values.float().square().mean(dim=-1, keepdim=True) + self.eps
        )
        output = normalized * self.weight.float() * torch.sigmoid(gate.float())
        return output.to(dtype)
