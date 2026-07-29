"""Weighted routing metrics that preserve rank and probability mass."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from .records import LayerRoutingTrace, RoutingRun


def probability_mass_overlap(
    reference_expert_ids: Sequence[int],
    reference_weights: Sequence[float],
    candidate_expert_ids: Sequence[int],
    candidate_weights: Sequence[float],
) -> float:
    """Return sum(min(p_ref, p_candidate)) over experts selected by both sides."""

    ref_ids = np.asarray(reference_expert_ids, dtype=np.int64)
    ref_weights = np.asarray(reference_weights, dtype=np.float64)
    cand_ids = np.asarray(candidate_expert_ids, dtype=np.int64)
    cand_weights = np.asarray(candidate_weights, dtype=np.float64)
    if ref_ids.ndim != 1 or cand_ids.ndim != 1:
        raise ValueError("token expert IDs must be one-dimensional")
    if ref_ids.shape != ref_weights.shape or cand_ids.shape != cand_weights.shape:
        raise ValueError("token expert ID and weight shapes differ")
    _validate_token_selection(ref_ids, ref_weights)
    _validate_token_selection(cand_ids, cand_weights)
    ref_probabilities = ref_weights / ref_weights.sum()
    cand_probabilities = cand_weights / cand_weights.sum()
    candidate_by_id = {
        int(expert_id): float(weight)
        for expert_id, weight in zip(cand_ids, cand_probabilities, strict=True)
    }
    return float(
        sum(
            min(float(weight), candidate_by_id[int(expert_id)])
            for expert_id, weight in zip(ref_ids, ref_probabilities, strict=True)
            if int(expert_id) in candidate_by_id
        )
    )


def top1_expert_agreement(
    reference_expert_ids: Sequence[Sequence[int]],
    reference_weights: Sequence[Sequence[float]],
    candidate_expert_ids: Sequence[Sequence[int]],
    candidate_weights: Sequence[Sequence[float]],
) -> float:
    """Return the fraction of tokens with the same highest-weighted expert."""

    arrays = _validated_matrices(
        reference_expert_ids,
        reference_weights,
        candidate_expert_ids,
        candidate_weights,
    )
    ref_ids, ref_weights, cand_ids, cand_weights = arrays
    reference_top1 = _top1_ids(ref_ids, ref_weights)
    candidate_top1 = _top1_ids(cand_ids, cand_weights)
    return float(np.mean(reference_top1 == candidate_top1))


def rank_position_histogram(
    reference_expert_ids: Sequence[Sequence[int]],
    reference_weights: Sequence[Sequence[float]],
    candidate_expert_ids: Sequence[Sequence[int]],
    candidate_weights: Sequence[Sequence[float]],
    *,
    expected_top_k: int = 8,
) -> dict[int, int]:
    """Count one-sided experts at their weight-ranked positions on each side."""

    arrays = _validated_matrices(
        reference_expert_ids,
        reference_weights,
        candidate_expert_ids,
        candidate_weights,
    )
    ref_ids, ref_weights, cand_ids, cand_weights = arrays
    if ref_ids.shape[1] != expected_top_k:
        raise ValueError(
            f"expected top-k width {expected_top_k}, got {ref_ids.shape[1]}"
        )
    histogram = _disagreement_histogram(ref_ids, ref_weights, cand_ids, cand_weights)
    return {rank: int(histogram[rank - 1]) for rank in range(1, expected_top_k + 1)}


def compare_routing_runs(reference: RoutingRun, candidate: RoutingRun) -> dict[str, Any]:
    """Compare matched runs and refuse any prompt or tokenization mismatch."""

    if reference.prompt_set_sha256 != candidate.prompt_set_sha256:
        raise ValueError("cannot compare runs from different prompt sets")
    if reference.router_config != candidate.router_config:
        raise ValueError("router configurations differ between checkpoints")
    if len(reference.prompts) != len(candidate.prompts):
        raise ValueError("the two runs contain different prompt counts")

    overall = _Accumulator(top_k=_configured_top_k(reference))
    per_layer: dict[int, _Accumulator] = {}
    for ref_prompt, cand_prompt in zip(reference.prompts, candidate.prompts, strict=True):
        if ref_prompt.prompt_id != cand_prompt.prompt_id:
            raise ValueError("prompt order or IDs differ between runs")
        if ref_prompt.token_ids != cand_prompt.token_ids:
            raise ValueError(f"tokenized prompt differs for {ref_prompt.prompt_id}")
        if len(ref_prompt.layers) != len(cand_prompt.layers):
            raise ValueError(f"routed layer count differs for {ref_prompt.prompt_id}")
        for ref_layer, cand_layer in zip(
            ref_prompt.layers, cand_prompt.layers, strict=True
        ):
            if ref_layer.layer_index != cand_layer.layer_index:
                raise ValueError(f"routed layer IDs differ for {ref_prompt.prompt_id}")
            layer_accumulator = per_layer.setdefault(
                ref_layer.layer_index,
                _Accumulator(top_k=ref_layer.top_k),
            )
            components = _compare_layer(ref_layer, cand_layer)
            layer_accumulator.update(components)
            overall.update(components)

    overall_summary = overall.summary()
    per_layer_summary = [
        {"layer_index": layer_index, **per_layer[layer_index].summary()}
        for layer_index in sorted(per_layer)
    ]
    histogram = overall_summary["disagreement_rank_histogram"]
    disagreement_count = sum(histogram.values())
    low_rank_count = histogram.get(7, 0) + histogram.get(8, 0)
    low_rank_share = (
        float(low_rank_count / disagreement_count) if disagreement_count else None
    )
    if disagreement_count == 0:
        hypothesis = "no disagreements were observed"
        concentrated = None
    elif low_rank_share is not None and low_rank_share > 0.5:
        hypothesis = "disagreements are concentrated at ranks 7 and 8"
        concentrated = True
    else:
        hypothesis = "disagreements are not concentrated at ranks 7 and 8"
        concentrated = False

    return {
        "schema_version": "runinfra.kimi_linear.router_analysis.v1",
        "reference_checkpoint": reference.checkpoint,
        "candidate_checkpoint": candidate.checkpoint,
        "prompt_set_sha256": reference.prompt_set_sha256,
        "prompt_count": len(reference.prompts),
        "prompt_token_count": reference.prompt_token_count,
        "router_config": dict(reference.router_config),
        "capture_weight_point": (
            "KLinearRouter output after selected-weight renormalization and "
            "routed_scaling_factor, exactly as passed to KLinearMoE._route_experts"
        ),
        "metric_definitions": {
            "top1_expert_agreement": (
                "fraction of token-layer observations with the same "
                "highest-weighted expert"
            ),
            "router_probability_mass_overlap": (
                "per token and layer, normalize each selected weight vector to sum "
                "to one, then sum min(reference_weight, candidate_weight) for "
                "shared expert IDs"
            ),
            "disagreement_rank_histogram": (
                "one count for every expert present on only one side, assigned to "
                "its descending-weight rank on the side that selected it"
            ),
            "mean_disagreeing_expert_weight": (
                "mean unnormalized post-scaling MoE weight across one-sided expert "
                "occurrences from both checkpoints"
            ),
        },
        "overall": overall_summary,
        "per_layer": per_layer_summary,
        "rank_7_8_hypothesis": {
            "assessment": hypothesis,
            "concentrated_at_ranks_7_8": concentrated,
            "decision_rule": "more than half of disagreement occurrences are ranks 7 or 8",
            "rank_7_8_count": low_rank_count,
            "all_disagreement_count": disagreement_count,
            "rank_7_8_share": low_rank_share,
        },
    }


def _validate_token_selection(ids: np.ndarray, weights: np.ndarray) -> None:
    if ids.size == 0:
        raise ValueError("a token selection cannot be empty")
    if len(set(int(value) for value in ids)) != ids.size:
        raise ValueError("a token selection contains duplicate expert IDs")
    if not np.all(np.isfinite(weights)) or np.any(weights < 0):
        raise ValueError("router weights must be finite and non-negative")
    if weights.sum() <= 0:
        raise ValueError("router weights must carry positive mass")


def _validated_matrices(
    reference_expert_ids: Sequence[Sequence[int]],
    reference_weights: Sequence[Sequence[float]],
    candidate_expert_ids: Sequence[Sequence[int]],
    candidate_weights: Sequence[Sequence[float]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ref_ids = np.asarray(reference_expert_ids, dtype=np.int64)
    ref_weights = np.asarray(reference_weights, dtype=np.float64)
    cand_ids = np.asarray(candidate_expert_ids, dtype=np.int64)
    cand_weights = np.asarray(candidate_weights, dtype=np.float64)
    if any(array.ndim != 2 for array in (ref_ids, ref_weights, cand_ids, cand_weights)):
        raise ValueError("router matrices must have shape [tokens, top_k]")
    if not (
        ref_ids.shape
        == ref_weights.shape
        == cand_ids.shape
        == cand_weights.shape
    ):
        raise ValueError("router matrices have different shapes")
    if ref_ids.shape[0] == 0 or ref_ids.shape[1] == 0:
        raise ValueError("router matrices cannot be empty")
    for ids, weights in ((ref_ids, ref_weights), (cand_ids, cand_weights)):
        if not np.all(np.isfinite(weights)) or np.any(weights < 0):
            raise ValueError("router weights must be finite and non-negative")
        if np.any(weights.sum(axis=1) <= 0):
            raise ValueError("every token must carry positive routing mass")
        sorted_ids = np.sort(ids, axis=1)
        if np.any(sorted_ids[:, 1:] == sorted_ids[:, :-1]):
            raise ValueError("a token selection contains duplicate expert IDs")
    return ref_ids, ref_weights, cand_ids, cand_weights


def _top1_ids(ids: np.ndarray, weights: np.ndarray) -> np.ndarray:
    maximum = weights.max(axis=1, keepdims=True)
    largest_id = np.iinfo(np.int64).max
    return np.where(weights == maximum, ids, largest_id).min(axis=1)


def _rank_order(ids: np.ndarray, weights: np.ndarray) -> np.ndarray:
    id_order = np.argsort(ids, axis=1, kind="stable")
    weights_by_id = np.take_along_axis(weights, id_order, axis=1)
    weight_order = np.argsort(-weights_by_id, axis=1, kind="stable")
    return np.take_along_axis(id_order, weight_order, axis=1)


def _disagreement_histogram(
    ref_ids: np.ndarray,
    ref_weights: np.ndarray,
    cand_ids: np.ndarray,
    cand_weights: np.ndarray,
) -> np.ndarray:
    top_k = ref_ids.shape[1]
    ref_ranked = np.take_along_axis(ref_ids, _rank_order(ref_ids, ref_weights), axis=1)
    cand_ranked = np.take_along_axis(cand_ids, _rank_order(cand_ids, cand_weights), axis=1)
    histogram = np.zeros(top_k, dtype=np.int64)
    for rank in range(top_k):
        ref_present = np.any(ref_ranked[:, rank, None] == cand_ids, axis=1)
        cand_present = np.any(cand_ranked[:, rank, None] == ref_ids, axis=1)
        histogram[rank] = np.count_nonzero(~ref_present) + np.count_nonzero(~cand_present)
    return histogram


def _compare_layer(
    reference: LayerRoutingTrace,
    candidate: LayerRoutingTrace,
) -> "_LayerComponents":
    ref_ids, ref_weights, cand_ids, cand_weights = _validated_matrices(
        reference.expert_ids,
        reference.expert_weights,
        candidate.expert_ids,
        candidate.expert_weights,
    )
    ref_probabilities = ref_weights / ref_weights.sum(axis=1, keepdims=True)
    cand_probabilities = cand_weights / cand_weights.sum(axis=1, keepdims=True)
    matches = ref_ids[:, :, None] == cand_ids[:, None, :]
    overlaps = (
        np.minimum(ref_probabilities[:, :, None], cand_probabilities[:, None, :])
        * matches
    ).sum(axis=(1, 2))
    reference_top1 = _top1_ids(ref_ids, ref_weights)
    candidate_top1 = _top1_ids(cand_ids, cand_weights)
    ref_agrees = matches.any(axis=2)
    cand_agrees = matches.any(axis=1)
    ref_disagreeing = ref_weights[~ref_agrees]
    cand_disagreeing = cand_weights[~cand_agrees]
    return _LayerComponents(
        token_count=int(ref_ids.shape[0]),
        top1_match_count=int(np.count_nonzero(reference_top1 == candidate_top1)),
        overlaps=overlaps,
        disagreement_histogram=_disagreement_histogram(
            ref_ids, ref_weights, cand_ids, cand_weights
        ),
        disagreeing_weight_sum=float(ref_disagreeing.sum() + cand_disagreeing.sum()),
        disagreeing_weight_count=int(ref_disagreeing.size + cand_disagreeing.size),
    )


def _configured_top_k(run: RoutingRun) -> int:
    configured = run.router_config.get("num_experts_per_token")
    if not isinstance(configured, int) or isinstance(configured, bool) or configured <= 0:
        raise ValueError("run router config has an invalid num_experts_per_token")
    captured = run.prompts[0].layers[0].top_k
    if configured != captured:
        raise ValueError("captured top-k width differs from the router config")
    return configured


@dataclass(frozen=True)
class _LayerComponents:
    token_count: int
    top1_match_count: int
    overlaps: np.ndarray
    disagreement_histogram: np.ndarray
    disagreeing_weight_sum: float
    disagreeing_weight_count: int


@dataclass
class _Accumulator:
    top_k: int
    token_count: int = 0
    top1_match_count: int = 0
    overlap_chunks: list[np.ndarray] = field(default_factory=list)
    disagreement_histogram: np.ndarray = field(init=False)
    disagreeing_weight_sum: float = 0.0
    disagreeing_weight_count: int = 0

    def __post_init__(self) -> None:
        self.disagreement_histogram = np.zeros(self.top_k, dtype=np.int64)

    def update(self, components: _LayerComponents) -> None:
        if components.disagreement_histogram.shape != self.disagreement_histogram.shape:
            raise ValueError("top-k width changed while accumulating routing metrics")
        self.token_count += components.token_count
        self.top1_match_count += components.top1_match_count
        self.overlap_chunks.append(components.overlaps)
        self.disagreement_histogram += components.disagreement_histogram
        self.disagreeing_weight_sum += components.disagreeing_weight_sum
        self.disagreeing_weight_count += components.disagreeing_weight_count

    def summary(self) -> dict[str, Any]:
        if not self.token_count or not self.overlap_chunks:
            raise ValueError("cannot summarize an empty routing comparison")
        overlaps = np.concatenate(self.overlap_chunks)
        histogram = {
            rank: int(self.disagreement_histogram[rank - 1])
            for rank in range(1, self.top_k + 1)
        }
        total_disagreements = sum(histogram.values())
        return {
            "routing_observation_count": self.token_count,
            "top1_expert_agreement": self.top1_match_count / self.token_count,
            "router_probability_mass_overlap": {
                "mean": float(np.mean(overlaps)),
                "median": float(np.median(overlaps)),
                "p10": float(np.percentile(overlaps, 10)),
            },
            "disagreement_rank_histogram": histogram,
            "disagreement_rank_fraction": {
                rank: (count / total_disagreements if total_disagreements else None)
                for rank, count in histogram.items()
            },
            "disagreeing_expert_weight": {
                "mean": (
                    self.disagreeing_weight_sum / self.disagreeing_weight_count
                    if self.disagreeing_weight_count
                    else None
                ),
                "occurrence_count": self.disagreeing_weight_count,
                "weight_basis": (
                    "post-renormalization, post-routed-scaling-factor weight used by MoE"
                ),
            },
        }
