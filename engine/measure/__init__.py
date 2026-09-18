"""Fail-closed, same-invocation serving measurement for Kimi-Linear."""

from .compare import ComparisonRefused, compare_record, load_record, render_comparison_table
from .record import (
    MIN_REPETITIONS,
    REQUIRED_CONCURRENCY_LEVELS,
    SCHEMA_VERSION,
    percentile_nearest_rank,
    summarize_series,
    validate_measurement_record,
)

__all__ = [
    "ComparisonRefused",
    "MIN_REPETITIONS",
    "REQUIRED_CONCURRENCY_LEVELS",
    "SCHEMA_VERSION",
    "compare_record",
    "load_record",
    "percentile_nearest_rank",
    "render_comparison_table",
    "summarize_series",
    "validate_measurement_record",
]
