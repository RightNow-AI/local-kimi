"""Load and compare complete serving records without third-party packages."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .record import SUMMARY_METRICS, validate_measurement_record


class ComparisonRefused(ValueError):
    """Raised when a speedup ratio would compare unmatched measurements."""


def load_record(path: str | Path) -> dict[str, Any]:
    """Load and validate a measurement JSON file."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return validate_measurement_record(payload)


def _side(record: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    sides = record.get("sides")
    if not isinstance(sides, Mapping) or not isinstance(sides.get(name), Mapping):
        raise ComparisonRefused(f"record is missing the {name} side")
    return sides[name]


def _gpu_identity(side: Mapping[str, Any]) -> tuple[Any, ...]:
    gpu = side.get("gpu")
    if not isinstance(gpu, Mapping):
        raise ComparisonRefused("a side is missing GPU identity")
    return (
        gpu.get("count"),
        tuple(gpu.get("names") or ()),
        tuple(gpu.get("device_uuids") or ()),
        gpu.get("driver_version"),
        gpu.get("cuda_driver_version"),
        gpu.get("cuda_runtime_version"),
    )


def _concurrency_map(side: Mapping[str, Any]) -> dict[int, Mapping[str, Any]]:
    points = side.get("measurements")
    if not isinstance(points, Sequence) or isinstance(points, (str, bytes)):
        raise ComparisonRefused("a side is missing concurrency measurements")
    mapped: dict[int, Mapping[str, Any]] = {}
    for point in points:
        if not isinstance(point, Mapping) or not isinstance(point.get("concurrency"), int):
            raise ComparisonRefused("a concurrency measurement is malformed")
        concurrency = point["concurrency"]
        if concurrency in mapped:
            raise ComparisonRefused("a side contains duplicate concurrency levels")
        mapped[concurrency] = point
    return mapped


def _runtime_key(side: Mapping[str, Any], field: str) -> Any:
    runtime = side.get("runtime")
    if not isinstance(runtime, Mapping):
        raise ComparisonRefused("a side is missing runtime metadata")
    return runtime.get(field)


def _prompt_schedule(side: Mapping[str, Any]) -> tuple[tuple[Any, ...], ...]:
    schedule = []
    for concurrency, point in sorted(_concurrency_map(side).items()):
        batches = point.get("raw_batches")
        if not isinstance(batches, Sequence) or isinstance(batches, (str, bytes)):
            raise ComparisonRefused("a side is missing raw prompt schedule samples")
        for batch in batches:
            if not isinstance(batch, Mapping):
                raise ComparisonRefused("a raw batch is malformed")
            repetition = batch.get("repetition")
            requests = batch.get("requests")
            if not isinstance(requests, Sequence) or isinstance(requests, (str, bytes)):
                raise ComparisonRefused("a raw batch is missing requests")
            for request in requests:
                if not isinstance(request, Mapping):
                    raise ComparisonRefused("a raw request is malformed")
                schedule.append(
                    (
                        concurrency,
                        repetition,
                        request.get("request_id"),
                        request.get("prompt_id"),
                        request.get("seed"),
                        request.get("prompt_tokens"),
                    )
                )
    return tuple(schedule)


def _model_key(side: Mapping[str, Any], field: str) -> Any:
    model = side.get("model")
    if not isinstance(model, Mapping):
        raise ComparisonRefused("a side is missing model metadata")
    return model.get(field)


def require_comparable(record: Mapping[str, Any]) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    """Refuse ratios unless the two sides share the measurement law."""

    if record.get("status") != "complete":
        raise ComparisonRefused("record is not complete; no comparison is allowed")
    baseline = _side(record, "baseline")
    candidate = _side(record, "candidate")
    if baseline.get("status") != "ok" or candidate.get("status") != "ok":
        raise ComparisonRefused("both sides must start and finish before comparison")
    if _gpu_identity(baseline) != _gpu_identity(candidate):
        raise ComparisonRefused("GPU mismatch: speedup requires the same physical GPU")
    baseline_levels = set(_concurrency_map(baseline))
    candidate_levels = set(_concurrency_map(candidate))
    if baseline_levels != candidate_levels:
        raise ComparisonRefused(
            "concurrency mismatch: speedup requires identical concurrency levels"
        )
    if baseline.get("prompt_set_id") != candidate.get("prompt_set_id"):
        raise ComparisonRefused(
            "prompt set mismatch: speedup requires the identical prompt schedule"
        )
    if _prompt_schedule(baseline) != _prompt_schedule(candidate):
        raise ComparisonRefused(
            "prompt schedule mismatch: speedup requires identical requests per batch"
        )
    for field in ("id", "resolved_revision"):
        if _model_key(baseline, field) != _model_key(candidate, field):
            raise ComparisonRefused(f"model {field} mismatch")
    for field in ("tensor_parallel_size", "max_model_len"):
        if _runtime_key(baseline, field) != _runtime_key(candidate, field):
            raise ComparisonRefused(f"runtime {field} mismatch")
    return baseline, candidate


def _metric_speedup(metric: str, baseline: float, candidate: float) -> float:
    if baseline <= 0.0 or candidate <= 0.0:
        raise ComparisonRefused(f"metric {metric} contains a zero and has no finite ratio")
    if metric in {
        "time_to_first_token_ms",
        "inter_token_latency_ms",
        "end_to_end_latency_ms",
    }:
        return baseline / candidate
    return candidate / baseline


def compare_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Compute matched median and p95 ratios for a complete record."""

    baseline, candidate = require_comparable(record)
    baseline_points = _concurrency_map(baseline)
    candidate_points = _concurrency_map(candidate)
    rows: list[dict[str, Any]] = []
    for concurrency in sorted(baseline_points):
        baseline_summary = baseline_points[concurrency]["summary"]
        candidate_summary = candidate_points[concurrency]["summary"]
        for metric in SUMMARY_METRICS:
            base_metric = baseline_summary[metric]
            cand_metric = candidate_summary[metric]
            if base_metric["unit"] != cand_metric["unit"]:
                raise ComparisonRefused(f"metric {metric} unit mismatch")
            rows.append(
                {
                    "concurrency": concurrency,
                    "metric": metric,
                    "unit": base_metric["unit"],
                    "baseline_median": float(base_metric["median"]),
                    "baseline_p95": float(base_metric["p95"]),
                    "candidate_median": float(cand_metric["median"]),
                    "candidate_p95": float(cand_metric["p95"]),
                    "median_speedup": _metric_speedup(
                        metric,
                        float(base_metric["median"]),
                        float(cand_metric["median"]),
                    ),
                    "p95_speedup": _metric_speedup(
                        metric,
                        float(base_metric["p95"]),
                        float(cand_metric["p95"]),
                    ),
                }
            )

    baseline_memory = baseline["memory"]
    candidate_memory = candidate["memory"]
    baseline_weights = baseline["model"]["weights_resident_bytes"]
    candidate_weights = candidate["model"]["weights_resident_bytes"]
    footprint = []
    for metric, baseline_value, candidate_value in (
        (
            "weights_resident_bytes",
            float(baseline_weights),
            float(candidate_weights),
        ),
        (
            "steady_state_gpu_memory_bytes",
            float(baseline_memory["steady_state_gpu_memory_bytes"]),
            float(candidate_memory["steady_state_gpu_memory_bytes"]),
        ),
        (
            "peak_gpu_memory_bytes",
            float(baseline_memory["peak_gpu_memory_bytes"]),
            float(candidate_memory["peak_gpu_memory_bytes"]),
        ),
    ):
        if candidate_value <= 0.0:
            raise ComparisonRefused(f"footprint metric {metric} has no finite ratio")
        footprint.append(
            {
                "metric": metric,
                "unit": "bytes",
                "baseline": baseline_value,
                "candidate": candidate_value,
                "baseline_over_candidate": baseline_value / candidate_value,
            }
        )
    return {"serving": rows, "footprint": footprint}


def _number(value: float) -> str:
    return f"{value:.6g}"


def render_comparison_table(record: Mapping[str, Any]) -> str:
    """Render matched serving and footprint results as Markdown tables."""

    comparison = compare_record(record)
    lines = [
        "| concurrency | metric | baseline median | baseline p95 | candidate median | candidate p95 | median speedup | p95 speedup |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in comparison["serving"]:
        lines.append(
            "| {concurrency} | {metric} ({unit}) | {baseline_median} | {baseline_p95} | "
            "{candidate_median} | {candidate_p95} | {median_speedup}x | {p95_speedup}x |".format(
                concurrency=row["concurrency"],
                metric=row["metric"],
                unit=row["unit"],
                baseline_median=_number(row["baseline_median"]),
                baseline_p95=_number(row["baseline_p95"]),
                candidate_median=_number(row["candidate_median"]),
                candidate_p95=_number(row["candidate_p95"]),
                median_speedup=_number(row["median_speedup"]),
                p95_speedup=_number(row["p95_speedup"]),
            )
        )
    lines.extend(
        [
            "",
            "| footprint metric | baseline bytes | candidate bytes | baseline / candidate |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in comparison["footprint"]:
        lines.append(
            "| {metric} | {baseline} | {candidate} | {ratio}x |".format(
                metric=row["metric"],
                baseline=int(row["baseline"]),
                candidate=int(row["candidate"]),
                ratio=_number(row["baseline_over_candidate"]),
            )
        )
    return "\n".join(lines)


def main(path: str) -> None:
    print(render_comparison_table(load_record(path)))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("record")
    arguments = parser.parse_args()
    main(arguments.record)
