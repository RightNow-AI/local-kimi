"""Expert-major scheduling for the K3 routed-expert path."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch

from .mxfp4_gemm import mxfp4_linear

ExpertFunction = Callable[[int, torch.Tensor], torch.Tensor]


@dataclass(frozen=True)
class PackedExpertWeights:
    w1_packed: torch.Tensor
    w1_scale: torch.Tensor
    w2_packed: torch.Tensor
    w2_scale: torch.Tensor
    w3_packed: torch.Tensor
    w3_scale: torch.Tensor


PackedExpertProvider = Callable[[int], PackedExpertWeights]


@dataclass(frozen=True)
class ExpertAssignment:
    expert_id: int
    token_indices: torch.Tensor
    route_slots: torch.Tensor

    @property
    def occurrence_count(self) -> int:
        return self.token_indices.numel()


@dataclass(frozen=True)
class ExpertSchedule:
    assignments: tuple[ExpertAssignment, ...]

    @property
    def union_size(self) -> int:
        return len(self.assignments)

    @property
    def expert_token_counts(self) -> tuple[tuple[int, int], ...]:
        return tuple(
            (assignment.expert_id, assignment.occurrence_count)
            for assignment in self.assignments
        )


@dataclass(frozen=True)
class GroupedMoEResult:
    values: torch.Tensor
    union_size: int
    expert_token_counts: tuple[tuple[int, int], ...]


def _validate_expert_indices(expert_indices: torch.Tensor) -> None:
    if expert_indices.ndim != 2:
        raise ValueError("expert_indices must have shape [tokens, routes]")
    if expert_indices.dtype not in (
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    ):
        raise TypeError("expert_indices must use an integer dtype")
    if torch.any(expert_indices < -1):
        raise ValueError("expert indices may use only -1 as the empty-route sentinel")


def build_expert_schedule(expert_indices: torch.Tensor) -> ExpertSchedule:
    """Sort the routed union and retain every token-slot occurrence."""
    _validate_expert_indices(expert_indices)
    valid = expert_indices[expert_indices >= 0]
    if valid.numel() == 0:
        return ExpertSchedule(assignments=())

    expert_ids = torch.unique(valid, sorted=True).tolist()
    assignments = []
    for expert_id in expert_ids:
        token_indices, route_slots = torch.where(expert_indices == expert_id)
        assignments.append(
            ExpertAssignment(
                expert_id=int(expert_id),
                token_indices=token_indices,
                route_slots=route_slots,
            )
        )
    return ExpertSchedule(assignments=tuple(assignments))


def expert_union_size(expert_indices: torch.Tensor) -> int:
    """Return the number of distinct nonempty routed experts in a batch."""
    return build_expert_schedule(expert_indices).union_size


def expert_major_moe(
    hidden_states: torch.Tensor,
    expert_indices: torch.Tensor,
    routing_weights: torch.Tensor,
    expert_fn: ExpertFunction,
    *,
    output_size: int | None = None,
) -> GroupedMoEResult:
    """Apply each routed expert once to all token-slot occurrences in its batch."""
    if hidden_states.ndim != 2:
        raise ValueError("hidden_states must have shape [tokens, hidden]")
    _validate_expert_indices(expert_indices)
    if expert_indices.shape != routing_weights.shape:
        raise ValueError("expert_indices and routing_weights must have the same shape")
    if expert_indices.shape[0] != hidden_states.shape[0]:
        raise ValueError("routing tables must have one row per token")
    if not (
        expert_indices.device == routing_weights.device == hidden_states.device
    ):
        raise ValueError("routing tables and hidden_states must share a device")

    output_size = hidden_states.shape[1] if output_size is None else output_size
    if output_size <= 0:
        raise ValueError("output_size must be positive")
    tokens, routes = expert_indices.shape
    schedule = build_expert_schedule(expert_indices)
    slot_outputs = hidden_states.new_zeros((tokens, routes, output_size))

    for assignment in schedule.assignments:
        expert_inputs = hidden_states.index_select(0, assignment.token_indices)
        expert_outputs = expert_fn(assignment.expert_id, expert_inputs)
        expected_shape = (assignment.occurrence_count, output_size)
        if expert_outputs.shape != expected_shape:
            raise ValueError(
                f"expert {assignment.expert_id} returned {tuple(expert_outputs.shape)}, "
                f"expected {expected_shape}"
            )
        if expert_outputs.device != hidden_states.device:
            raise ValueError("expert outputs must stay on the hidden-state device")
        if expert_outputs.dtype != hidden_states.dtype:
            raise TypeError("expert outputs must preserve the hidden-state dtype")
        slot_outputs[assignment.token_indices, assignment.route_slots] = expert_outputs

    # Slot preservation prevents duplicate routes from overwriting one another.
    valid_weights = torch.where(
        expert_indices >= 0, routing_weights, torch.zeros_like(routing_weights)
    )
    values = (
        slot_outputs.float()
        .mul(valid_weights.float().unsqueeze(-1))
        .sum(dim=1)
        .to(hidden_states.dtype)
    )
    return GroupedMoEResult(
        values=values,
        union_size=schedule.union_size,
        expert_token_counts=schedule.expert_token_counts,
    )


def situ_and_mul(
    gate: torch.Tensor,
    up: torch.Tensor,
    *,
    beta: float = 4.0,
    linear_beta: float | None = 25.0,
) -> torch.Tensor:
    """Apply K3's SiTU activation in FP32 and return the input dtype."""
    if gate.shape != up.shape:
        raise ValueError("gate and up tensors must have the same shape")
    gate_float = gate.float()
    up_float = up.float()
    activated_gate = beta * torch.tanh(gate_float / beta) * torch.sigmoid(gate_float)
    if linear_beta is not None:
        up_float = linear_beta * torch.tanh(up_float / linear_beta)
    return (activated_gate * up_float).to(gate.dtype)


def mxfp4_expert_mlp(
    hidden_states: torch.Tensor,
    weights: PackedExpertWeights,
    *,
    situ_beta: float = 4.0,
    situ_linear_beta: float | None = 25.0,
) -> torch.Tensor:
    """Run one K3 routed expert with packed weights consumed by fused GEMMs."""
    gate = mxfp4_linear(hidden_states, weights.w1_packed, weights.w1_scale)
    up = mxfp4_linear(hidden_states, weights.w3_packed, weights.w3_scale)
    activated = situ_and_mul(
        gate, up, beta=situ_beta, linear_beta=situ_linear_beta
    )
    return mxfp4_linear(activated, weights.w2_packed, weights.w2_scale)


def expert_major_mxfp4(
    hidden_states: torch.Tensor,
    expert_indices: torch.Tensor,
    routing_weights: torch.Tensor,
    expert_provider: PackedExpertProvider,
    *,
    situ_beta: float = 4.0,
    situ_linear_beta: float | None = 25.0,
) -> GroupedMoEResult:
    """Load each expert once, then run all tokens routed to it as one batch."""

    def run_expert(expert_id: int, expert_inputs: torch.Tensor) -> torch.Tensor:
        weights = expert_provider(expert_id)
        return mxfp4_expert_mlp(
            expert_inputs,
            weights,
            situ_beta=situ_beta,
            situ_linear_beta=situ_linear_beta,
        )

    return expert_major_moe(
        hidden_states,
        expert_indices,
        routing_weights,
        run_expert,
        output_size=hidden_states.shape[1],
    )
