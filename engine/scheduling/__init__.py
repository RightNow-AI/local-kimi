"""Routing-aware batch composition and simulation for sparse MoE decoding."""

from .compose import (
    CompositionDecision,
    PendingToken,
    compare_composers,
    compose_batch,
)
from .metrics import (
    FairnessMetrics,
    ThroughputFeedback,
    UnionMeasurement,
    UnionReduction,
    compare_throughput,
    deferred_rounds,
    measure_fairness,
    measure_union,
    measure_union_reduction,
    predict_observed_union,
)

__all__ = [
    "CompositionDecision",
    "FairnessMetrics",
    "PendingToken",
    "ThroughputFeedback",
    "UnionMeasurement",
    "UnionReduction",
    "compare_composers",
    "compare_throughput",
    "compose_batch",
    "deferred_rounds",
    "measure_fairness",
    "measure_union",
    "measure_union_reduction",
    "predict_observed_union",
]
