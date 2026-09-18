"""Normalization primitives used by the Kimi-Linear engine."""

from __future__ import annotations

import torch
from torch import nn


def rms_norm(hidden_states: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    input_dtype = hidden_states.dtype
    values = hidden_states.float()
    values = values * torch.rsqrt(values.square().mean(dim=-1, keepdim=True) + eps)
    return weight * values.to(input_dtype)


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
        input_dtype = values.dtype
        values_float = values.float()
        normalized = values_float * torch.rsqrt(
            values_float.square().mean(dim=-1, keepdim=True) + self.eps
        )
        output = normalized * self.weight.float() * torch.sigmoid(gate.float())
        return output.to(input_dtype)

