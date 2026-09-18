"""Load and compare complete serving records without third-party packages."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from numbers import Real
from pathlib import Path
from typing import Any, NoReturn

from .record import SUMMARY_METRICS, validate_measurement_record

THROUGHPUT_METRIC = "aggregate_output_tokens_per_second"


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


def _gpu_hourly_cost(
    record: Mapping[str, Any],
    supplied: float | None,
    *,
    gpu_count: int,
) -> dict[str, Any] | None:
    value: Any = supplied
    source = "caller argument"
    if value is None:
        environment = record.get("environment")
        if isinstance(environment, Mapping):
            value = environment.get("gpu_hourly_cost_usd")
            source = "record.environment.gpu_hourly_cost_usd"
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ComparisonRefused("GPU hourly cost must be a positive finite number")
    converted = float(value)
    if not math.isfinite(converted) or converted <= 0.0:
        raise ComparisonRefused("GPU hourly cost must be a positive finite number")
    if isinstance(gpu_count, bool) or not isinstance(gpu_count, int) or gpu_count < 1:
        raise ComparisonRefused("GPU count must be a positive integer for token cost")
    return {
        "usd_per_gpu_hour": converted,
        "gpu_count": gpu_count,
        "total_usd_per_hour": converted * gpu_count,
        "source": source,
    }


def _cost_per_million_output_tokens(
    throughput_tokens_per_second: float,
    gpu_hourly_cost_usd: float,
) -> float:
    if throughput_tokens_per_second <= 0.0:
        raise ComparisonRefused("throughput must be positive to derive token cost")
    return gpu_hourly_cost_usd * 1_000_000.0 / (
        throughput_tokens_per_second * 3600.0
    )


def _trend(ratios: Sequence[float]) -> tuple[str, str]:
    deltas = [right - left for left, right in zip(ratios, ratios[1:])]
    tolerance = 1e-12
    positive = any(delta > tolerance for delta in deltas)
    negative = any(delta < -tolerance for delta in deltas)
    if positive and negative:
        shape = "mixed"
    elif positive:
        shape = "increasing"
    elif negative:
        shape = "decreasing"
    else:
        shape = "flat"
    net = ratios[-1] - ratios[0]
    if net > tolerance:
        net_direction = "increasing"
    elif net < -tolerance:
        net_direction = "decreasing"
    else:
        net_direction = "flat"
    return shape, net_direction


def _crossing_analysis(curve: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not curve:
        raise ComparisonRefused("throughput curve is empty")
    ratios = [float(point["candidate_over_baseline_median"]) for point in curve]
    trend, net_direction = _trend(ratios)
    signs = [
        0
        if math.isclose(ratio, 1.0, rel_tol=1e-12, abs_tol=1e-12)
        else (1 if ratio > 1.0 else -1)
        for ratio in ratios
    ]
    crossings: list[dict[str, Any]] = []
    measured_parity: list[int] = []
    for point, sign in zip(curve, signs):
        ratio = float(point["candidate_over_baseline_median"])
        concurrency = int(point["concurrency"])
        if sign == 0:
            measured_parity.append(concurrency)
            crossings.append(
                {
                    "kind": "measured_parity",
                    "concurrency": concurrency,
                    "ratio": ratio,
                }
            )
    for index, (left, right) in enumerate(zip(curve, curve[1:])):
        if signs[index] * signs[index + 1] != -1:
            continue
        left_ratio = float(left["candidate_over_baseline_median"])
        right_ratio = float(right["candidate_over_baseline_median"])
        left_concurrency = int(left["concurrency"])
        right_concurrency = int(right["concurrency"])
        estimated = left_concurrency + (1.0 - left_ratio) * (
            right_concurrency - left_concurrency
        ) / (right_ratio - left_ratio)
        crossings.append(
            {
                "kind": "interpolated_between_measured_points",
                "lower_concurrency": left_concurrency,
                "upper_concurrency": right_concurrency,
                "lower_ratio": left_ratio,
                "upper_ratio": right_ratio,
                "estimated_concurrency": estimated,
                "estimation_method": "linear interpolation of measured median ratios",
            }
        )

    inversion_events: list[dict[str, Any]] = []
    previous_non_parity_index: int | None = None
    for index, sign in enumerate(signs):
        if sign == 0:
            continue
        if (
            previous_non_parity_index is not None
            and sign != signs[previous_non_parity_index]
        ):
            parity_concurrencies = [
                int(curve[parity_index]["concurrency"])
                for parity_index in range(previous_non_parity_index + 1, index)
                if signs[parity_index] == 0
            ]
            event: dict[str, Any] = {
                "from": (
                    "candidate_faster"
                    if signs[previous_non_parity_index] > 0
                    else "candidate_slower"
                ),
                "to": "candidate_faster" if sign > 0 else "candidate_slower",
            }
            if parity_concurrencies:
                event.update(
                    {
                        "kind": "measured_parity_transition",
                        "parity_concurrencies": parity_concurrencies,
                    }
                )
            else:
                lower = int(curve[previous_non_parity_index]["concurrency"])
                upper = int(curve[index]["concurrency"])
                crossing = next(
                    item
                    for item in crossings
                    if item.get("lower_concurrency") == lower
                    and item.get("upper_concurrency") == upper
                )
                event.update(
                    {
                        "kind": "interpolated_transition",
                        "lower_concurrency": lower,
                        "upper_concurrency": upper,
                        "lower_ratio": crossing["lower_ratio"],
                        "upper_ratio": crossing["upper_ratio"],
                        "estimated_concurrency": crossing["estimated_concurrency"],
                    }
                )
            inversion_events.append(event)
        previous_non_parity_index = index

    first_concurrency = int(curve[0]["concurrency"])
    last_concurrency = int(curve[-1]["concurrency"])
    trend_text = {
        "increasing": "increases as concurrency rises",
        "decreasing": "decreases as concurrency rises",
        "flat": "is flat across the swept range",
        "mixed": "is non-monotonic across the swept range",
    }[trend]
    inversion = bool(inversion_events)

    if inversion_events:
        first_event = inversion_events[0]
        prefix = (
            "Throughput advantage inverts from candidate win to candidate loss"
            if first_event["from"] == "candidate_faster"
            else "Throughput advantage inverts from candidate loss to candidate win"
        )
        if first_event["kind"] == "interpolated_transition":
            statement = (
                f"{prefix} between {first_event['lower_ratio']:.6g}x at "
                f"c{first_event['lower_concurrency']} and "
                f"{first_event['upper_ratio']:.6g}x at "
                f"c{first_event['upper_concurrency']}. Linear interpolation "
                f"estimates parity at c{first_event['estimated_concurrency']:.6g}. "
                f"The candidate/baseline median throughput ratio {trend_text}."
            )
        else:
            parity_concurrencies = first_event["parity_concurrencies"]
            if len(parity_concurrencies) == 1:
                parity_location = f"c{parity_concurrencies[0]}"
            else:
                parity_location = (
                    f"c{parity_concurrencies[0]} to c{parity_concurrencies[-1]}"
                )
            statement = (
                f"{prefix} at measured parity {parity_location}. The "
                f"candidate/baseline median throughput ratio {trend_text}."
            )
        if len(inversion_events) > 1:
            statement += f" The measured curve inverts {len(inversion_events)} times."
        status = "inverts_within_swept_range"
    elif crossings:
        parity_text = ", ".join(f"c{value}" for value in measured_parity)
        statement = (
            f"Median aggregate throughput reaches measured parity at {parity_text} "
            f"without inverting. The candidate/baseline median throughput ratio "
            f"{trend_text}."
        )
        status = "reaches_parity_without_inversion"
    else:
        if all(ratio > 1.0 for ratio in ratios):
            relation = "candidate remains faster than baseline"
        elif all(ratio < 1.0 for ratio in ratios):
            relation = "candidate remains slower than baseline"
        else:
            relation = "candidate remains at parity with baseline"
        statement = (
            f"No throughput crossover occurs within the swept range c{first_concurrency} "
            f"to c{last_concurrency}; {relation}. The candidate/baseline median "
            f"throughput ratio {trend_text}."
        )
        status = "does_not_cross_within_swept_range"

    return {
        "metric": THROUGHPUT_METRIC,
        "ratio": "candidate / baseline",
        "statistic": "median",
        "status": status,
        "inversion": inversion,
        "inversion_count": len(inversion_events),
        "inversion_events": inversion_events,
        "trend": trend,
        "net_direction": net_direction,
        "swept_concurrency_range": [first_concurrency, last_concurrency],
        "crossings": crossings,
        "statement": statement,
    }


def aggregate_speedup(record: Mapping[str, Any]) -> NoReturn:
    """Refuse a run-wide scalar because concurrency changes the comparison."""

    raise ComparisonRefused(
        "aggregate-across-concurrencies speedup is forbidden; every ratio "
        "must name one measured concurrency"
    )


def compare_record(
    record: Mapping[str, Any],
    *,
    aggregate_across_concurrencies: bool = False,
    gpu_hourly_cost_usd: float | None = None,
) -> dict[str, Any]:
    """Compute only matched, concurrency-scoped ratios for a complete record."""

    if aggregate_across_concurrencies:
        aggregate_speedup(record)

    baseline, candidate = require_comparable(record)
    baseline_points = _concurrency_map(baseline)
    candidate_points = _concurrency_map(candidate)
    baseline_gpu = baseline["gpu"]
    cost = _gpu_hourly_cost(
        record,
        gpu_hourly_cost_usd,
        gpu_count=baseline_gpu["count"],
    )
    throughput_curve: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for concurrency in sorted(baseline_points):
        baseline_summary = baseline_points[concurrency]["summary"]
        candidate_summary = candidate_points[concurrency]["summary"]
        baseline_throughput = baseline_summary[THROUGHPUT_METRIC]
        candidate_throughput = candidate_summary[THROUGHPUT_METRIC]
        if baseline_throughput["unit"] != candidate_throughput["unit"]:
            raise ComparisonRefused(f"metric {THROUGHPUT_METRIC} unit mismatch")
        baseline_median_throughput = float(baseline_throughput["median"])
        candidate_median_throughput = float(candidate_throughput["median"])
        baseline_p95_throughput = float(baseline_throughput["p95"])
        candidate_p95_throughput = float(candidate_throughput["p95"])
        curve_point: dict[str, Any] = {
            "concurrency": concurrency,
            "unit": baseline_throughput["unit"],
            "baseline_median": baseline_median_throughput,
            "baseline_p95": baseline_p95_throughput,
            "candidate_median": candidate_median_throughput,
            "candidate_p95": candidate_p95_throughput,
            "candidate_over_baseline_median": _metric_speedup(
                THROUGHPUT_METRIC,
                baseline_median_throughput,
                candidate_median_throughput,
            ),
            "candidate_over_baseline_p95": _metric_speedup(
                THROUGHPUT_METRIC,
                baseline_p95_throughput,
                candidate_p95_throughput,
            ),
        }
        if cost is not None:
            curve_point["cost_per_million_output_tokens_usd"] = {
                "baseline": _cost_per_million_output_tokens(
                    baseline_median_throughput,
                    cost["total_usd_per_hour"],
                ),
                "candidate": _cost_per_million_output_tokens(
                    candidate_median_throughput,
                    cost["total_usd_per_hour"],
                ),
                "gpu_hourly_cost_usd": cost["usd_per_gpu_hour"],
                "gpu_count": cost["gpu_count"],
                "total_gpu_hourly_cost_usd": cost["total_usd_per_hour"],
                "gpu_hourly_cost_source": cost["source"],
                "throughput_statistic": "median",
            }
        throughput_curve.append(curve_point)
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
    return {
        "throughput_curve": throughput_curve,
        "crossover": _crossing_analysis(throughput_curve),
        "cost_basis": cost,
        "serving": rows,
        "footprint": footprint,
        "score_policy": {
            "combined_score_allowed": False,
            "reason": (
                "Throughput and footprint are separate measured axes and are not "
                "combined into one score"
            ),
        },
    }


def _number(value: float) -> str:
    return f"{value:.6g}"


def _money(value: float) -> str:
    return f"${value:.6g}"


def render_comparison_table(
    record: Mapping[str, Any],
    *,
    gpu_hourly_cost_usd: float | None = None,
) -> str:
    """Render the throughput curve first, then separate footprint and detail."""

    comparison = compare_record(
        record,
        gpu_hourly_cost_usd=gpu_hourly_cost_usd,
    )
    cost_available = comparison["cost_basis"] is not None
    lines = ["## Throughput curve"]
    if cost_available:
        lines.extend(
            [
                "| concurrency | baseline median tok/s | candidate median tok/s | "
                "candidate / baseline median | baseline p95 tok/s | candidate p95 tok/s | "
                "candidate / baseline p95 | baseline $/1M output tokens | "
                "candidate $/1M output tokens |",
                "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
    else:
        lines.extend(
            [
                "| concurrency | baseline median tok/s | candidate median tok/s | "
                "candidate / baseline median | baseline p95 tok/s | candidate p95 tok/s | "
                "candidate / baseline p95 |",
                "|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
    for point in comparison["throughput_curve"]:
        concurrency = int(point["concurrency"])
        values = {
            "concurrency": concurrency,
            "baseline_median": _number(point["baseline_median"]),
            "candidate_median": _number(point["candidate_median"]),
            "median_ratio": _number(point["candidate_over_baseline_median"]),
            "baseline_p95": _number(point["baseline_p95"]),
            "candidate_p95": _number(point["candidate_p95"]),
            "p95_ratio": _number(point["candidate_over_baseline_p95"]),
        }
        if cost_available:
            token_cost = point["cost_per_million_output_tokens_usd"]
            values["baseline_cost"] = _money(token_cost["baseline"])
            values["candidate_cost"] = _money(token_cost["candidate"])
            lines.append(
                "| c{concurrency} | {baseline_median} | {candidate_median} | "
                "{median_ratio}x at c{concurrency} | {baseline_p95} | "
                "{candidate_p95} | {p95_ratio}x at c{concurrency} | "
                "{baseline_cost} | {candidate_cost} |".format(**values)
            )
        else:
            lines.append(
                "| c{concurrency} | {baseline_median} | {candidate_median} | "
                "{median_ratio}x at c{concurrency} | {baseline_p95} | "
                "{candidate_p95} | {p95_ratio}x at c{concurrency} |".format(
                    **values
                )
            )

    lines.extend(
        [
            "",
            "### Curve headline",
            f"**{comparison['crossover']['statement']}**",
        ]
    )
    if cost_available:
        cost_basis = comparison["cost_basis"]
        lines.append(
            "Cost uses {rate} per GPU-hour from {source} across {gpu_count} GPU(s), "
            "for {total_rate}/hour total. Each value is derived from the median "
            "aggregate throughput at that same concurrency.".format(
                rate=_money(cost_basis["usd_per_gpu_hour"]),
                source=cost_basis["source"],
                gpu_count=cost_basis["gpu_count"],
                total_rate=_money(cost_basis["total_usd_per_hour"]),
            )
        )
    else:
        lines.append(
            "Cost per million output tokens is not derived because the record and "
            "caller supplied no positive GPU hourly price."
        )

    lines.extend(
        [
            "",
            "## Footprint, independent from throughput",
            comparison["score_policy"]["reason"] + ".",
            "",
            "| footprint metric | baseline bytes | candidate bytes | baseline / candidate |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in comparison["footprint"]:
        lines.append(
            "| {metric} | {baseline} | {candidate} | "
            "{ratio}x, concurrency-independent footprint |".format(
                metric=row["metric"],
                baseline=int(row["baseline"]),
                candidate=int(row["candidate"]),
                ratio=_number(row["baseline_over_candidate"]),
            )
        )

    lines.extend(
        [
            "",
            "## Matched serving detail by concurrency",
            "| concurrency | metric | ratio direction | baseline median | baseline p95 | "
            "candidate median | candidate p95 | median ratio | p95 ratio |",
            "|---:|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in comparison["serving"]:
        concurrency = int(row["concurrency"])
        ratio_direction = (
            "baseline / candidate"
            if row["metric"]
            in {
                "time_to_first_token_ms",
                "inter_token_latency_ms",
                "end_to_end_latency_ms",
            }
            else "candidate / baseline"
        )
        lines.append(
            "| c{concurrency} | {metric} ({unit}) | {ratio_direction} | "
            "{baseline_median} | {baseline_p95} | {candidate_median} | "
            "{candidate_p95} | {median_speedup}x at c{concurrency} | "
            "{p95_speedup}x at c{concurrency} |".format(
                concurrency=concurrency,
                metric=row["metric"],
                unit=row["unit"],
                ratio_direction=ratio_direction,
                baseline_median=_number(row["baseline_median"]),
                baseline_p95=_number(row["baseline_p95"]),
                candidate_median=_number(row["candidate_median"]),
                candidate_p95=_number(row["candidate_p95"]),
                median_speedup=_number(row["median_speedup"]),
                p95_speedup=_number(row["p95_speedup"]),
            )
        )
    return "\n".join(lines)


def main(path: str, *, gpu_hourly_cost_usd: float | None = None) -> None:
    print(
        render_comparison_table(
            load_record(path),
            gpu_hourly_cost_usd=gpu_hourly_cost_usd,
        )
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("record")
    parser.add_argument("--gpu-hourly-cost-usd", type=float)
    arguments = parser.parse_args()
    main(
        arguments.record,
        gpu_hourly_cost_usd=arguments.gpu_hourly_cost_usd,
    )
