import pytest

from engine.measure.record import percentile_nearest_rank, summarize_series


def test_p95_uses_nearest_rank_on_known_synthetic_series() -> None:
    synthetic_samples = list(range(1, 21))

    assert percentile_nearest_rank(synthetic_samples, 0.95) == 19.0
    summary = summarize_series(synthetic_samples, unit="synthetic-ms")
    assert summary == {
        "median": 10.5,
        "p95": 19.0,
        "sample_count": 20,
        "unit": "synthetic-ms",
        "percentile_method": "nearest-rank",
    }


def test_percentile_rejects_empty_or_nonfinite_synthetic_samples() -> None:
    with pytest.raises(ValueError, match="at least one"):
        percentile_nearest_rank([], 0.95)
    with pytest.raises(ValueError, match="finite"):
        percentile_nearest_rank([1.0, float("inf")], 0.95)
