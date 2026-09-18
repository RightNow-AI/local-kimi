from copy import deepcopy

import pytest

from engine.measure.compare import ComparisonRefused, compare_record


def _synthetic_summary(multiplier: float) -> dict:
    units = {
        "time_to_first_token_ms": "ms",
        "inter_token_latency_ms": "ms",
        "end_to_end_latency_ms": "ms",
        "output_tokens_per_second_per_stream": "tokens/s",
        "aggregate_output_tokens_per_second": "tokens/s",
    }
    return {
        metric: {
            "median": multiplier * (index + 1),
            "p95": multiplier * (index + 2),
            "sample_count": 4,
            "unit": unit,
            "percentile_method": "nearest-rank",
        }
        for index, (metric, unit) in enumerate(units.items())
    }


def _synthetic_side(multiplier: float) -> dict:
    return {
        "status": "ok",
        "prompt_set_id": "sha256:synthetic-prompt-set",
        "gpu": {
            "count": 1,
            "names": ["Synthetic GPU"],
            "device_uuids": ["GPU-synthetic-1"],
            "driver_version": "synthetic-driver",
            "cuda_driver_version": "synthetic-cuda-driver",
            "cuda_runtime_version": "synthetic-cuda-runtime",
        },
        "model": {
            "id": "synthetic/model",
            "resolved_revision": "a" * 40,
            "weights_resident_bytes": int(1000 * multiplier),
        },
        "runtime": {"tensor_parallel_size": 1, "max_model_len": 1024},
        "memory": {
            "steady_state_gpu_memory_bytes": 2000 * multiplier,
            "peak_gpu_memory_bytes": 3000 * multiplier,
        },
        "measurements": [
            {
                "concurrency": concurrency,
                "summary": _synthetic_summary(multiplier),
                "raw_batches": [
                    {
                        "repetition": 0,
                        "requests": [
                            {
                                "request_id": f"synthetic-c{concurrency}-s{slot}",
                                "prompt_id": f"synthetic-prompt-{slot % 2}",
                                "seed": 100 + slot,
                                "prompt_tokens": 10 + slot % 2,
                            }
                            for slot in range(concurrency)
                        ],
                    }
                ],
            }
            for concurrency in (1, 4, 16, 64)
        ],
    }


def _synthetic_record() -> dict:
    return {
        "status": "complete",
        "sides": {
            "baseline": _synthetic_side(2.0),
            "candidate": _synthetic_side(1.0),
        },
    }


def test_comparator_refuses_mismatched_gpu() -> None:
    synthetic_record = _synthetic_record()
    synthetic_record["sides"]["candidate"]["gpu"]["device_uuids"] = [
        "GPU-synthetic-2"
    ]

    with pytest.raises(ComparisonRefused, match="GPU mismatch"):
        compare_record(synthetic_record)


def test_comparator_refuses_mismatched_concurrency() -> None:
    synthetic_record = _synthetic_record()
    synthetic_record["sides"]["candidate"]["measurements"].pop()

    with pytest.raises(ComparisonRefused, match="concurrency mismatch"):
        compare_record(synthetic_record)


def test_comparator_refuses_when_one_side_failed_to_start() -> None:
    synthetic_record = _synthetic_record()
    synthetic_record["status"] = "failed"
    synthetic_record["sides"]["candidate"] = {
        "status": "failed",
        "stage": "start server",
        "error": {
            "type": "SyntheticStartError",
            "message": "synthetic candidate did not start",
        },
    }

    with pytest.raises(ComparisonRefused, match="not complete"):
        compare_record(synthetic_record)


def test_comparator_refuses_mismatched_prompt_set() -> None:
    synthetic_record = _synthetic_record()
    synthetic_record["sides"]["candidate"]["prompt_set_id"] = (
        "sha256:different-synthetic-prompt-set"
    )

    with pytest.raises(ComparisonRefused, match="prompt set mismatch"):
        compare_record(synthetic_record)


def test_matched_synthetic_record_produces_ratios() -> None:
    synthetic_record = deepcopy(_synthetic_record())

    comparison = compare_record(synthetic_record)

    assert comparison["serving"]
    assert comparison["footprint"][0]["baseline_over_candidate"] == 2.0
