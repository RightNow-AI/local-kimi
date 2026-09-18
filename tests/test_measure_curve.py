from __future__ import annotations

import re

import pytest

from engine.measure.compare import (
    ComparisonRefused,
    aggregate_speedup,
    compare_record,
    render_comparison_table,
)

CONCURRENCIES = (8, 32, 64)


def _summary(aggregate_throughput: float) -> dict:
    return {
        "time_to_first_token_ms": {
            "median": 10.0,
            "p95": 12.0,
            "unit": "ms",
        },
        "inter_token_latency_ms": {
            "median": 2.0,
            "p95": 3.0,
            "unit": "ms",
        },
        "end_to_end_latency_ms": {
            "median": 50.0,
            "p95": 60.0,
            "unit": "ms",
        },
        "output_tokens_per_second_per_stream": {
            "median": aggregate_throughput / 8.0,
            "p95": aggregate_throughput / 7.0,
            "unit": "tokens/s",
        },
        "aggregate_output_tokens_per_second": {
            "median": aggregate_throughput,
            "p95": aggregate_throughput * 1.1,
            "unit": "tokens/s",
        },
    }


def _side(
    throughputs: tuple[float, ...],
    *,
    weight_bytes: int,
    gpu_count: int,
) -> dict:
    assert len(throughputs) == len(CONCURRENCIES)
    return {
        "status": "ok",
        "prompt_set_id": "sha256:synthetic-curve",
        "gpu": {
            "count": gpu_count,
            "names": [f"Synthetic GPU {index}" for index in range(gpu_count)],
            "device_uuids": [
                f"GPU-synthetic-curve-{index}" for index in range(gpu_count)
            ],
            "driver_version": "synthetic-driver",
            "cuda_driver_version": "synthetic-cuda-driver",
            "cuda_runtime_version": "synthetic-cuda-runtime",
        },
        "model": {
            "id": "synthetic/model",
            "resolved_revision": "a" * 40,
            "weights_resident_bytes": weight_bytes,
        },
        "runtime": {"tensor_parallel_size": 1, "max_model_len": 1024},
        "memory": {
            "steady_state_gpu_memory_bytes": weight_bytes + 100,
            "peak_gpu_memory_bytes": weight_bytes + 200,
        },
        "measurements": [
            {
                "concurrency": concurrency,
                "summary": _summary(throughput),
                "raw_batches": [
                    {
                        "repetition": 0,
                        "requests": [
                            {
                                "request_id": f"synthetic-c{concurrency}",
                                "prompt_id": "synthetic-prompt",
                                "seed": 17,
                                "prompt_tokens": 16,
                            }
                        ],
                    }
                ],
            }
            for concurrency, throughput in zip(CONCURRENCIES, throughputs)
        ],
    }


def _record(
    baseline_throughputs: tuple[float, ...],
    candidate_throughputs: tuple[float, ...],
    *,
    gpu_count: int = 1,
    gpu_hourly_cost_usd: float | None = None,
) -> dict:
    record = {
        "status": "complete",
        "sides": {
            "baseline": _side(
                baseline_throughputs,
                weight_bytes=98_000,
                gpu_count=gpu_count,
            ),
            "candidate": _side(
                candidate_throughputs,
                weight_bytes=28_000,
                gpu_count=gpu_count,
            ),
        },
    }
    if gpu_hourly_cost_usd is not None:
        record["environment"] = {
            "gpu_hourly_cost_usd": gpu_hourly_cost_usd,
        }
    return record


def test_aggregate_across_concurrencies_speedup_is_refused() -> None:
    record = _record((100.0, 200.0, 300.0), (120.0, 180.0, 270.0))

    with pytest.raises(ComparisonRefused, match="aggregate-across-concurrencies"):
        aggregate_speedup(record)

    with pytest.raises(ComparisonRefused, match="aggregate-across-concurrencies"):
        compare_record(record, aggregate_across_concurrencies=True)


def test_crossover_is_computed_for_curve_that_inverts() -> None:
    record = _record((100.0, 200.0, 300.0), (120.0, 180.0, 270.0))

    crossover = compare_record(record)["crossover"]

    assert crossover["status"] == "inverts_within_swept_range"
    assert crossover["inversion"] is True
    assert crossover["inversion_count"] == 1
    assert crossover["trend"] == "decreasing"
    assert len(crossover["crossings"]) == 1
    crossing = crossover["crossings"][0]
    assert crossing["kind"] == "interpolated_between_measured_points"
    assert crossing["lower_concurrency"] == 8
    assert crossing["upper_concurrency"] == 32
    assert crossing["lower_ratio"] == pytest.approx(1.2)
    assert crossing["upper_ratio"] == pytest.approx(0.9)
    assert crossing["estimated_concurrency"] == pytest.approx(24.0)
    assert crossing["estimation_method"] == (
        "linear interpolation of measured median ratios"
    )
    assert "inverts" in crossover["statement"]


def test_curve_that_never_crosses_reports_that_explicitly() -> None:
    record = _record((100.0, 200.0, 300.0), (120.0, 220.0, 330.0))

    crossover = compare_record(record)["crossover"]

    assert crossover["status"] == "does_not_cross_within_swept_range"
    assert crossover["crossings"] == []
    assert crossover["trend"] == "decreasing"
    assert "No throughput crossover occurs" in crossover["statement"]
    assert "candidate remains faster than baseline" in crossover["statement"]


def test_non_monotonic_curve_reports_every_inversion() -> None:
    record = _record((100.0, 200.0, 300.0), (120.0, 160.0, 330.0))

    crossover = compare_record(record)["crossover"]

    assert crossover["inversion"] is True
    assert crossover["inversion_count"] == 2
    assert crossover["trend"] == "mixed"
    assert len(crossover["crossings"]) == 2
    assert "inverts 2 times" in crossover["statement"]


def test_measured_parity_between_win_and_loss_is_reported_as_inversion() -> None:
    record = _record((100.0, 200.0, 300.0), (120.0, 200.0, 240.0))

    crossover = compare_record(record)["crossover"]

    assert crossover["status"] == "inverts_within_swept_range"
    assert crossover["inversion_count"] == 1
    assert crossover["crossings"] == [
        {
            "kind": "measured_parity",
            "concurrency": 32,
            "ratio": 1.0,
        }
    ]
    assert crossover["inversion_events"][0] == {
        "from": "candidate_faster",
        "to": "candidate_slower",
        "kind": "measured_parity_transition",
        "parity_concurrencies": [32],
    }
    assert "measured parity c32" in crossover["statement"]


def test_near_parity_does_not_create_a_duplicate_interpolated_crossing() -> None:
    record = _record(
        (100.0, 200.0, 300.0),
        (100.00000000005, 180.0, 240.0),
    )

    crossover = compare_record(record)["crossover"]

    assert crossover["status"] == "reaches_parity_without_inversion"
    assert crossover["inversion"] is False
    assert len(crossover["crossings"]) == 1
    assert crossover["crossings"][0]["kind"] == "measured_parity"
    assert crossover["crossings"][0]["concurrency"] == 8


def test_rendered_ratios_always_name_concurrency_or_independent_scope() -> None:
    record = _record((100.0, 200.0, 300.0), (120.0, 180.0, 270.0))

    comparison = compare_record(record)
    rendered = render_comparison_table(record)
    ratio_matches = [
        (line, match)
        for line in rendered.splitlines()
        for match in re.finditer(r"\b\d+(?:\.\d+)?x\b", line)
    ]

    assert ratio_matches
    for line, match in ratio_matches:
        context = line[match.end() :].split("|", 1)[0]
        assert re.match(r" at c\d+", context) or context.startswith(
            ", concurrency-independent"
        )
    assert rendered.startswith("## Throughput curve\n")
    assert rendered.index("### Curve headline") > rendered.index("| c64 |")
    assert rendered.index("## Throughput curve") < rendered.index(
        "## Footprint, independent from throughput"
    )
    assert comparison["score_policy"]["combined_score_allowed"] is False


def test_cost_per_million_uses_explicit_hourly_rate_and_same_concurrency() -> None:
    record = _record(
        (100.0, 200.0, 300.0),
        (120.0, 180.0, 270.0),
        gpu_count=2,
    )

    comparison = compare_record(record, gpu_hourly_cost_usd=3.6)
    first_cost = comparison["throughput_curve"][0][
        "cost_per_million_output_tokens_usd"
    ]

    assert first_cost["baseline"] == pytest.approx(20.0)
    assert first_cost["candidate"] == pytest.approx(50.0 / 3.0)
    assert first_cost["gpu_count"] == 2
    assert first_cost["total_gpu_hourly_cost_usd"] == pytest.approx(7.2)
    assert first_cost["gpu_hourly_cost_source"] == "caller argument"


def test_cost_per_million_can_use_rate_recorded_in_environment() -> None:
    record = _record(
        (100.0, 200.0, 300.0),
        (120.0, 180.0, 270.0),
        gpu_hourly_cost_usd=3.6,
    )

    comparison = compare_record(record)
    first_cost = comparison["throughput_curve"][0][
        "cost_per_million_output_tokens_usd"
    ]

    assert comparison["cost_basis"]["source"] == (
        "record.environment.gpu_hourly_cost_usd"
    )
    assert first_cost["baseline"] == pytest.approx(10.0)
    assert first_cost["candidate"] == pytest.approx(25.0 / 3.0)


def test_cost_is_omitted_when_no_hourly_rate_is_available() -> None:
    comparison = compare_record(
        _record((100.0, 200.0, 300.0), (120.0, 180.0, 270.0))
    )

    assert comparison["cost_basis"] is None
    assert all(
        "cost_per_million_output_tokens_usd" not in point
        for point in comparison["throughput_curve"]
    )


@pytest.mark.parametrize("hourly_rate", [0.0, -1.0, float("inf"), float("nan")])
def test_invalid_hourly_rate_is_refused(hourly_rate: float) -> None:
    record = _record((100.0, 200.0, 300.0), (120.0, 180.0, 270.0))

    with pytest.raises(ComparisonRefused, match="positive finite"):
        compare_record(record, gpu_hourly_cost_usd=hourly_rate)
