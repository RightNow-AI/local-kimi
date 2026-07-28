"""Obvious, slow PyTorch oracles for the packed K3 expert path."""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

import torch
import torch.nn.functional as F

MXFP4_GROUP_SIZE = 32
MXFP4_EXPONENT_BIAS = 127
_E2M1_POSITIVE = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
_E2M1_CODEBOOK = _E2M1_POSITIVE + tuple(-value for value in _E2M1_POSITIVE)


class PackedExpertWeightsLike(Protocol):
    w1_packed: torch.Tensor
    w1_scale: torch.Tensor
    w2_packed: torch.Tensor
    w2_scale: torch.Tensor
    w3_packed: torch.Tensor
    w3_scale: torch.Tensor


ExpertFunction = Callable[[int, torch.Tensor], torch.Tensor]


def dequantize_mxfp4_reference(
    packed: torch.Tensor,
    scale: torch.Tensor,
    *,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Decode low-nibble-first E2M1 values with one E8M0 scale per 32 values."""
    if packed.dtype != torch.uint8 or scale.dtype != torch.uint8:
        raise TypeError("packed and scale tensors must both use torch.uint8")
    if packed.ndim != 2 or scale.ndim != 2:
        raise ValueError("packed and scale tensors must both be matrices")
    if packed.device != scale.device:
        raise ValueError("packed and scale tensors must be on the same device")
    if not torch.empty((), dtype=dtype).is_floating_point():
        raise TypeError("MXFP4 output dtype must be floating point")

    rows, packed_columns = packed.shape
    value_columns = packed_columns * 2
    if (
        scale.shape[0] != rows
        or value_columns != scale.shape[1] * MXFP4_GROUP_SIZE
    ):
        raise ValueError(
            "MXFP4 scale shape must provide one exponent for every 32 decoded values"
        )

    nibbles = torch.empty(
        (rows, value_columns), dtype=torch.long, device=packed.device
    )
    nibbles[:, 0::2] = (packed & 0x0F).long()
    nibbles[:, 1::2] = (packed >> 4).long()
    codebook = torch.tensor(_E2M1_CODEBOOK, dtype=torch.float32, device=packed.device)
    values = codebook[nibbles]
    exponents = torch.exp2(scale.to(torch.int16).float() - MXFP4_EXPONENT_BIAS)
    values = values * exponents.repeat_interleave(MXFP4_GROUP_SIZE, dim=1)
    return values.to(dtype)


def mxfp4_linear_reference(
    activations: torch.Tensor,
    weight_packed: torch.Tensor,
    weight_scale: torch.Tensor,
) -> torch.Tensor:
    """Materialize the decoded weight, then run the linear operation."""
    if activations.ndim != 2:
        raise ValueError("activations must have shape [tokens, reduction]")
    weight = dequantize_mxfp4_reference(
        weight_packed, weight_scale, dtype=activations.dtype
    )
    if activations.shape[1] != weight.shape[1]:
        raise ValueError("activation reduction dimension does not match the weight")
    return F.linear(activations, weight)


def situ_reference(
    gate: torch.Tensor,
    up: torch.Tensor,
    *,
    beta: float = 4.0,
    linear_beta: float | None = 25.0,
) -> torch.Tensor:
    """Moonshot's SiTU activation, evaluated in FP32 like the model source."""
    if gate.shape != up.shape:
        raise ValueError("gate and up tensors must have the same shape")
    gate_float = gate.float()
    up_float = up.float()
    activated_gate = beta * torch.tanh(gate_float / beta) * torch.sigmoid(gate_float)
    if linear_beta is not None:
        up_float = linear_beta * torch.tanh(up_float / linear_beta)
    return (activated_gate * up_float).to(gate.dtype)


def mxfp4_expert_reference(
    hidden_states: torch.Tensor,
    weights: PackedExpertWeightsLike,
    *,
    situ_beta: float = 4.0,
    situ_linear_beta: float | None = 25.0,
) -> torch.Tensor:
    """Unfused packed expert MLP used only as a numerical and timing baseline."""
    gate = mxfp4_linear_reference(
        hidden_states, weights.w1_packed, weights.w1_scale
    )
    up = mxfp4_linear_reference(hidden_states, weights.w3_packed, weights.w3_scale)
    activated = situ_reference(
        gate, up, beta=situ_beta, linear_beta=situ_linear_beta
    )
    return mxfp4_linear_reference(activated, weights.w2_packed, weights.w2_scale)


def naive_token_major_moe(
    hidden_states: torch.Tensor,
    expert_indices: torch.Tensor,
    routing_weights: torch.Tensor,
    expert_fn: ExpertFunction,
    *,
    output_size: int | None = None,
) -> torch.Tensor:
    """Slow token-major routing oracle that preserves every route slot."""
    if hidden_states.ndim != 2:
        raise ValueError("hidden_states must have shape [tokens, hidden]")
    if expert_indices.ndim != 2:
        raise ValueError("expert_indices must have shape [tokens, routes]")
    if expert_indices.shape != routing_weights.shape:
        raise ValueError("expert_indices and routing_weights must have the same shape")
    if expert_indices.shape[0] != hidden_states.shape[0]:
        raise ValueError("routing tables must have one row per token")
    if expert_indices.device != hidden_states.device:
        raise ValueError("routing tables and hidden_states must share a device")
    if routing_weights.device != hidden_states.device:
        raise ValueError("routing tables and hidden_states must share a device")
    if torch.any(expert_indices < -1):
        raise ValueError("expert indices may use only -1 as the empty-route sentinel")

    tokens, routes = expert_indices.shape
    output_size = hidden_states.shape[1] if output_size is None else output_size
    slot_outputs = hidden_states.new_zeros((tokens, routes, output_size))
    for token_index in range(tokens):
        token = hidden_states[token_index : token_index + 1]
        for route_slot in range(routes):
            expert_id = int(expert_indices[token_index, route_slot].item())
            if expert_id < 0:
                continue
            expert_output = expert_fn(expert_id, token)
            if expert_output.shape != (1, output_size):
                raise ValueError("expert_fn returned an unexpected output shape")
            slot_outputs[token_index, route_slot] = expert_output[0]

    valid_weights = torch.where(
        expert_indices >= 0, routing_weights, torch.zeros_like(routing_weights)
    )
    return (
        slot_outputs.float()
        .mul(valid_weights.float().unsqueeze(-1))
        .sum(dim=1)
        .to(hidden_states.dtype)
    )
