"""Union, routed-byte, throughput, and fairness measurements."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Iterable, Sequence

from engine.batching.union_model import (
    FUSED_DEQUANT,
    DequantMode,
    ExpertUnionModel,
    HardwareConfig,
    RoutingPrior,
    ThroughputPrediction,
)

from .compose import PendingToken


@dataclass(frozen=True)
class UnionMeasurement:
    per_layer_union: tuple[int, ...]
    total_union: int
    mean_union_per_layer: float
    max_union_per_layer: int


@dataclass(frozen=True)
class UnionReduction:
    composed: UnionMeasurement
    random: UnionMeasurement
    experts_saved_across_layers: int
    reduction_fraction: float
    bytes_saved_per_batch: int
    bytes_saved_per_token: float


@dataclass(frozen=True)
class FairnessMetrics:
    served_tokens: int
    outstanding_tokens: int
    average_deferred_rounds: float
    worst_case_deferred_rounds: int
    max_served_deferred_rounds: int
    max_outstanding_deferred_rounds: int
    average_wait_seconds: float
    worst_case_wait_seconds: float


@dataclass(frozen=True)
class ThroughputFeedback:
    random: ThroughputPrediction
    composed: ThroughputPrediction
    aggregate_speedup_fraction: float
    seconds_saved_per_batch: float


def measure_union(batch: Sequence[PendingToken]) -> UnionMeasurement:
    """Measure distinct experts independently at each layer."""

    tokens = tuple(batch)
    if not tokens:
        raise ValueError("cannot measure the union of an empty batch")
    layer_count = len(tokens[0].layer_masks)
    if any(len(token.layer_masks) != layer_count for token in tokens):
        raise ValueError("all tokens in a batch must describe the same number of layers")

    layer_masks = [0] * layer_count
    for token in tokens:
        for layer, mask in enumerate(token.layer_masks):
            layer_masks[layer] |= mask
    per_layer = tuple(mask.bit_count() for mask in layer_masks)
    total = sum(per_layer)
    return UnionMeasurement(
        per_layer_union=per_layer,
        total_union=total,
        mean_union_per_layer=total / layer_count,
        max_union_per_layer=max(per_layer),
    )


def measure_union_reduction(
    composed_batch: Sequence[PendingToken],
    random_batch: Sequence[PendingToken],
    *,
    expert_bytes: int = 17_547_264,
) -> UnionReduction:
    """Compare equal-sized batches and turn saved expert-layer pairs into bytes."""

    if len(composed_batch) != len(random_batch):
        raise ValueError("composed and random batches must have equal sizes")
    if not composed_batch:
        raise ValueError("cannot compare empty batches")
    if isinstance(expert_bytes, bool) or not isinstance(expert_bytes, int):
        raise TypeError("expert_bytes must be an integer")
    if expert_bytes <= 0:
        raise ValueError("expert_bytes must be positive")

    composed = measure_union(composed_batch)
    random = measure_union(random_batch)
    if len(composed.per_layer_union) != len(random.per_layer_union):
        raise ValueError("composed and random batches must have equal layer counts")
    saved = random.total_union - composed.total_union
    reduction = saved / random.total_union if random.total_union else 0.0
    return UnionReduction(
        composed=composed,
        random=random,
        experts_saved_across_layers=saved,
        reduction_fraction=reduction,
        bytes_saved_per_batch=saved * expert_bytes,
        bytes_saved_per_token=saved * expert_bytes / len(composed_batch),
    )


def deferred_rounds(arrival_round: int, selected_round: int) -> int:
    """Count completed selection rounds in which a token was not selected."""

    if arrival_round < 0 or selected_round < 0:
        raise ValueError("round numbers must be non-negative")
    if selected_round < arrival_round:
        raise ValueError("selected_round cannot precede arrival_round")
    return selected_round - arrival_round


def _non_negative_ints(values: Iterable[int], name: str) -> tuple[int, ...]:
    normalized = tuple(values)
    if any(isinstance(value, bool) or not isinstance(value, int) for value in normalized):
        raise TypeError(f"{name} must contain integers")
    if any(value < 0 for value in normalized):
        raise ValueError(f"{name} must be non-negative")
    return normalized


def _non_negative_floats(values: Iterable[float], name: str) -> tuple[float, ...]:
    normalized = tuple(float(value) for value in values)
    if any(value < 0.0 or not math.isfinite(value) for value in normalized):
        raise ValueError(f"{name} must be finite and non-negative")
    return normalized


def measure_fairness(
    served_deferred_rounds: Iterable[int],
    *,
    outstanding_deferred_rounds: Iterable[int] = (),
    served_wait_seconds: Iterable[float] = (),
    outstanding_wait_seconds: Iterable[float] = (),
) -> FairnessMetrics:
    """Report completed waits and current waits of tokens still outstanding."""

    served_rounds = _non_negative_ints(served_deferred_rounds, "served_deferred_rounds")
    outstanding_rounds = _non_negative_ints(
        outstanding_deferred_rounds, "outstanding_deferred_rounds"
    )
    served_seconds = _non_negative_floats(served_wait_seconds, "served_wait_seconds")
    outstanding_seconds = _non_negative_floats(
        outstanding_wait_seconds, "outstanding_wait_seconds"
    )
    if served_seconds and len(served_seconds) != len(served_rounds):
        raise ValueError("served wait seconds and round counts must have equal lengths")
    if outstanding_seconds and len(outstanding_seconds) != len(outstanding_rounds):
        raise ValueError("outstanding wait seconds and round counts must have equal lengths")
    if not served_seconds:
        served_seconds = (0.0,) * len(served_rounds)
    if not outstanding_seconds:
        outstanding_seconds = (0.0,) * len(outstanding_rounds)

    all_rounds = served_rounds + outstanding_rounds
    all_seconds = served_seconds + outstanding_seconds
    return FairnessMetrics(
        served_tokens=len(served_rounds),
        outstanding_tokens=len(outstanding_rounds),
        average_deferred_rounds=(
            math.fsum(all_rounds) / len(all_rounds) if all_rounds else 0.0
        ),
        worst_case_deferred_rounds=max(all_rounds, default=0),
        max_served_deferred_rounds=max(served_rounds, default=0),
        max_outstanding_deferred_rounds=max(outstanding_rounds, default=0),
        average_wait_seconds=(
            math.fsum(all_seconds) / len(all_seconds) if all_seconds else 0.0
        ),
        worst_case_wait_seconds=max(all_seconds, default=0.0),
    )


def predict_observed_union(
    hardware: HardwareConfig,
    model: ExpertUnionModel,
    batch_size: int,
    observed_mean_union_per_layer: float,
    *,
    prior: RoutingPrior | None = None,
    compute_scale: float = 1.0,
    dequant_mode: DequantMode = FUSED_DEQUANT,
) -> ThroughputPrediction:
    """Feed an observed union into the existing calibrated throughput model.

    ``HardwareConfig.predict`` supplies every non-union term. This function
    replaces only the union-derived routed bytes and dequant work, so the
    scheduling lane does not reimplement the analytic union mathematics.
    """

    if observed_mean_union_per_layer <= 0.0 or not math.isfinite(
        observed_mean_union_per_layer
    ):
        raise ValueError("observed_mean_union_per_layer must be finite and positive")
    if observed_mean_union_per_layer > model.total_experts:
        raise ValueError("observed union cannot exceed total_experts")

    baseline = hardware.predict(
        model,
        batch_size,
        prior,
        compute_scale=compute_scale,
        dequant_mode=dequant_mode,
    )
    batch_bytes = (
        observed_mean_union_per_layer * model.expert_bytes * model.moe_layers
    )
    union_seconds = batch_bytes / 1e9 / hardware.effective_bandwidth_gb_s
    weight_seconds = union_seconds + baseline.dense_seconds_per_batch
    dequant_seconds = (
        observed_mean_union_per_layer
        * model.moe_layers
        * model.expert_tensors
        * dequant_mode.seconds_per_tensor
    )
    total_seconds = (
        weight_seconds
        + dequant_seconds
        + baseline.token_compute_seconds_per_batch
    )
    aggregate = batch_size / total_seconds
    return replace(
        baseline,
        expected_union=observed_mean_union_per_layer,
        batch_routed_traffic_gb=batch_bytes / 1e9,
        routed_traffic_gb_per_token=batch_bytes / batch_size / 1e9,
        aggregate_tokens_per_second=aggregate,
        per_agent_tokens_per_second=aggregate / batch_size,
        union_seconds_per_batch=union_seconds,
        weight_seconds_per_batch=weight_seconds,
        dequant_seconds_per_batch=dequant_seconds,
        dequant_seconds_per_token=dequant_seconds / batch_size,
        total_seconds_per_batch=total_seconds,
    )


def compare_throughput(
    hardware: HardwareConfig,
    model: ExpertUnionModel,
    batch_size: int,
    random_mean_union_per_layer: float,
    composed_mean_union_per_layer: float,
    *,
    prior: RoutingPrior | None = None,
    compute_scale: float = 1.0,
    dequant_mode: DequantMode = FUSED_DEQUANT,
) -> ThroughputFeedback:
    random_prediction = predict_observed_union(
        hardware,
        model,
        batch_size,
        random_mean_union_per_layer,
        prior=prior,
        compute_scale=compute_scale,
        dequant_mode=dequant_mode,
    )
    composed_prediction = predict_observed_union(
        hardware,
        model,
        batch_size,
        composed_mean_union_per_layer,
        prior=prior,
        compute_scale=compute_scale,
        dequant_mode=dequant_mode,
    )
    return ThroughputFeedback(
        random=random_prediction,
        composed=composed_prediction,
        aggregate_speedup_fraction=(
            composed_prediction.aggregate_tokens_per_second
            / random_prediction.aggregate_tokens_per_second
            - 1.0
        ),
        seconds_saved_per_batch=(
            random_prediction.total_seconds_per_batch
            - composed_prediction.total_seconds_per_batch
        ),
    )
