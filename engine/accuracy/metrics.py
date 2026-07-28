"""Accuracy metrics shared by the Modal job and offline evidence verifier.

The distribution and router implementations are the existing, independently
tested functions in ``engine.bench.metrics``. This module adds only the token
sequence comparisons and log-probability reductions needed by this experiment.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from engine.bench.metrics import (
    routing_agreement as router_set_agreement,
)
from engine.bench.metrics import (
    token_kl_divergence,
    top1_agreement,
)

# The three names above are re-exports, not dead imports: analyze.py and the
# tests import them from here so callers have one metrics surface. __all__
# states that intent, which is also what stops a lint autofix from deleting
# them and breaking those callers.
__all__ = [
    "first_divergence_index",
    "greedy_identity_rate",
    "kl_per_position_from_logprobs",
    "perplexity_from_gold_logprobs",
    "router_set_agreement",
    "token_kl_divergence",
    "top1_agreement",
]


def first_divergence_index(
    reference_token_ids: Sequence[int],
    candidate_token_ids: Sequence[int],
) -> int | None:
    """Return the zero-based first differing token index, or ``None``.

    A strict-prefix length difference diverges at the first missing token.
    """
    common = min(len(reference_token_ids), len(candidate_token_ids))
    for index in range(common):
        if reference_token_ids[index] != candidate_token_ids[index]:
            return index
    if len(reference_token_ids) != len(candidate_token_ids):
        return common
    return None


def greedy_identity_rate(
    reference_token_sequences: Sequence[Sequence[int]],
    candidate_token_sequences: Sequence[Sequence[int]],
) -> float:
    """Return exact generated-token identity rate across paired prompts."""
    if len(reference_token_sequences) != len(candidate_token_sequences):
        raise ValueError("reference and candidate must contain the same prompt count")
    if not reference_token_sequences:
        raise ValueError("at least one prompt is required")
    identical = sum(
        list(reference) == list(candidate)
        for reference, candidate in zip(
            reference_token_sequences,
            candidate_token_sequences,
            strict=True,
        )
    )
    return identical / len(reference_token_sequences)


def perplexity_from_gold_logprobs(gold_logprobs: Sequence[float]) -> float:
    """Return exp(mean negative log likelihood) from teacher-forced log p."""
    if not gold_logprobs:
        raise ValueError("at least one teacher-forced position is required")
    values = [float(value) for value in gold_logprobs]
    if not all(math.isfinite(value) and value <= 0.0 for value in values):
        raise ValueError("gold log probabilities must be finite and non-positive")
    return math.exp(-math.fsum(values) / len(values))


def kl_per_position_from_logprobs(
    reference_logprobs,
    candidate_logprobs,
):
    """Return KL(reference || candidate) per position from normalized log p.

    This accepts NumPy arrays inside the Modal and offline analysis paths. The
    full vocabulary remains present, so this is exact rather than a top-k KL.
    """
    import numpy as np

    reference = np.asarray(reference_logprobs, dtype=np.float64)
    candidate = np.asarray(candidate_logprobs, dtype=np.float64)
    if reference.shape != candidate.shape:
        raise ValueError("reference and candidate log probabilities must match")
    if reference.ndim != 2 or reference.shape[0] == 0 or reference.shape[1] == 0:
        raise ValueError("log probabilities must have shape [positions, vocabulary]")
    if not np.isfinite(reference).all() or not np.isfinite(candidate).all():
        raise ValueError("full-vocabulary log probabilities must all be finite")

    reference_norm = np.logaddexp.reduce(reference, axis=-1, keepdims=True)
    candidate_norm = np.logaddexp.reduce(candidate, axis=-1, keepdims=True)
    reference_normalized = reference - reference_norm
    candidate_normalized = candidate - candidate_norm
    values = np.sum(
        np.exp(reference_normalized)
        * (reference_normalized - candidate_normalized),
        axis=-1,
        dtype=np.float64,
    )
    return np.maximum(values, 0.0)
