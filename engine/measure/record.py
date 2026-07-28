"""Pure-Python validation and statistics for serving measurement records."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from collections.abc import Mapping, Sequence
from numbers import Real
from typing import Any

from engine.bench.artifacts import IMMUTABLE_REVISION

SCHEMA_VERSION = 1
REQUIRED_CONCURRENCY_LEVELS = (1, 4, 16, 64)
MIN_REPETITIONS = 20
SUMMARY_METRICS = (
    "time_to_first_token_ms",
    "inter_token_latency_ms",
    "end_to_end_latency_ms",
    "output_tokens_per_second_per_stream",
    "aggregate_output_tokens_per_second",
)
SUMMARY_UNITS = {
    "time_to_first_token_ms": "ms",
    "inter_token_latency_ms": "ms",
    "end_to_end_latency_ms": "ms",
    "output_tokens_per_second_per_stream": "tokens/s",
    "aggregate_output_tokens_per_second": "tokens/s",
}


class MeasurementRecordError(ValueError):
    """Raised when a record cannot support an auditable serving claim."""


def _finite_values(values: Sequence[Real]) -> list[float]:
    if isinstance(values, (str, bytes)) or not values:
        raise ValueError("a measured series must contain at least one sample")
    converted = [float(value) for value in values]
    if not all(math.isfinite(value) and value >= 0.0 for value in converted):
        raise ValueError("measured samples must be finite and nonnegative")
    return converted


def percentile_nearest_rank(values: Sequence[Real], percentile: float) -> float:
    """Return the nearest-rank percentile used by the measurement record.

    The rank is ``ceil(percentile * sample_count)`` with one-based indexing.
    This keeps the stored p95 equal to an observed sample and makes independent
    recomputation possible without depending on a statistics library default.
    """

    samples = sorted(_finite_values(values))
    if not math.isfinite(percentile) or not 0.0 < percentile <= 1.0:
        raise ValueError("percentile must be finite and in the interval (0, 1]")
    rank = max(1, math.ceil(percentile * len(samples)))
    return samples[rank - 1]


def summarize_series(values: Sequence[Real], *, unit: str) -> dict[str, Any]:
    """Return median and p95 without discarding the raw series elsewhere."""

    samples = _finite_values(values)
    if not unit.strip():
        raise ValueError("summary unit must not be empty")
    return {
        "median": float(statistics.median(samples)),
        "p95": percentile_nearest_rank(samples, 0.95),
        "sample_count": len(samples),
        "unit": unit,
        "percentile_method": "nearest-rank",
    }


def validate_concurrency_levels(values: Any) -> tuple[int, ...]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise MeasurementRecordError("concurrency levels must be a sequence")
    levels: list[int] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise MeasurementRecordError("concurrency levels must be positive integers")
        levels.append(value)
    if len(levels) != len(set(levels)):
        raise MeasurementRecordError("concurrency levels must not contain duplicates")
    missing = sorted(set(REQUIRED_CONCURRENCY_LEVELS) - set(levels))
    if missing:
        raise MeasurementRecordError(
            f"concurrency sweep is missing required levels: {missing}"
        )
    return tuple(levels)


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise MeasurementRecordError(f"{field} must be a mapping")
    return value


def _nonempty_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise MeasurementRecordError(f"{field} must be nonempty text")
    return value


def _nonnegative_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise MeasurementRecordError(f"{field} must be numeric")
    converted = float(value)
    if not math.isfinite(converted) or converted < 0.0:
        raise MeasurementRecordError(f"{field} must be finite and nonnegative")
    return converted


def _validate_prompt_set(prompt_set: Mapping[str, Any]) -> list[dict[str, Any]]:
    prompt_set_id = _nonempty_text(prompt_set.get("id"), "prompt_set.id")
    model_id = _nonempty_text(prompt_set.get("model_id"), "prompt_set.model_id")
    revision = _nonempty_text(
        prompt_set.get("resolved_revision"), "prompt_set.resolved_revision"
    )
    if IMMUTABLE_REVISION.fullmatch(revision) is None:
        raise MeasurementRecordError("prompt set revision must be immutable")
    max_prompt_tokens = prompt_set.get("max_prompt_tokens")
    max_output_tokens = prompt_set.get("max_output_tokens")
    seed = prompt_set.get("seed")
    if (
        isinstance(max_prompt_tokens, bool)
        or not isinstance(max_prompt_tokens, int)
        or max_prompt_tokens < 1
    ):
        raise MeasurementRecordError("prompt_set.max_prompt_tokens must be positive")
    if (
        isinstance(max_output_tokens, bool)
        or not isinstance(max_output_tokens, int)
        or max_output_tokens < 2
    ):
        raise MeasurementRecordError("prompt_set.max_output_tokens must be at least two")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise MeasurementRecordError("prompt_set.seed must be a nonnegative integer")
    request_parameters = _mapping(
        prompt_set.get("request_parameters"), "prompt_set.request_parameters"
    )
    expected_request_parameters = {
        "temperature": 0.0,
        "min_tokens": max_output_tokens,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if dict(request_parameters) != expected_request_parameters:
        raise MeasurementRecordError("prompt set request parameters violate the protocol")
    prompts = prompt_set.get("prompts")
    if (
        not isinstance(prompts, Sequence)
        or isinstance(prompts, (str, bytes))
        or not prompts
    ):
        raise MeasurementRecordError("prompt_set.prompts must be a nonempty sequence")
    prompt_records = []
    prompt_token_counts = []
    seen_ids = set()
    for index, prompt_value in enumerate(prompts):
        prompt = _mapping(prompt_value, f"prompt_set.prompts[{index}]")
        prompt_id = _nonempty_text(prompt.get("id"), f"prompt_set.prompts[{index}].id")
        if prompt_id in seen_ids:
            raise MeasurementRecordError("prompt IDs must be unique")
        seen_ids.add(prompt_id)
        text = _nonempty_text(prompt.get("text"), f"prompt_set.prompts[{index}].text")
        if prompt.get("text_sha256") != hashlib.sha256(text.encode("utf-8")).hexdigest():
            raise MeasurementRecordError("prompt text digest does not match its text")
        rendered_digest = prompt.get("rendered_prompt_sha256")
        if not isinstance(rendered_digest, str) or len(rendered_digest) != 64:
            raise MeasurementRecordError("rendered prompt digest must be a SHA256 string")
        token_ids = prompt.get("token_ids")
        if (
            not isinstance(token_ids, Sequence)
            or isinstance(token_ids, (str, bytes))
            or not token_ids
            or not all(
                isinstance(token, int) and not isinstance(token, bool) and token >= 0
                for token in token_ids
            )
        ):
            raise MeasurementRecordError("prompt token IDs must be nonnegative integers")
        prompt_tokens = prompt.get("prompt_tokens")
        if prompt_tokens != len(token_ids):
            raise MeasurementRecordError("prompt token count does not match token IDs")
        if prompt_tokens > max_prompt_tokens:
            raise MeasurementRecordError("prompt token count exceeds max_prompt_tokens")
        prompt_record = {
            "id": prompt_id,
            "text": text,
            "text_sha256": prompt["text_sha256"],
            "rendered_prompt_sha256": rendered_digest,
            "token_ids": list(token_ids),
            "prompt_tokens": prompt_tokens,
        }
        prompt_records.append(prompt_record)
        prompt_token_counts.append({"id": prompt_id, "prompt_tokens": prompt_tokens})
    identity_payload = {
        "model_id": model_id,
        "resolved_revision": revision,
        "max_prompt_tokens": prompt_set["max_prompt_tokens"],
        "max_output_tokens": prompt_set["max_output_tokens"],
        "seed": prompt_set["seed"],
        "prompts": prompt_records,
        "request_parameters": dict(request_parameters),
    }
    encoded = json.dumps(
        identity_payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    expected_id = "sha256:" + hashlib.sha256(encoded).hexdigest()
    if prompt_set_id != expected_id:
        raise MeasurementRecordError("prompt set identity does not match its contents")
    return prompt_token_counts


def _validate_gpu(gpu: Any, field: str) -> None:
    value = _mapping(gpu, field)
    count = value.get("count")
    names = value.get("names")
    uuids = value.get("device_uuids")
    if count != 1:
        raise MeasurementRecordError(f"{field}.count must equal one")
    for sequence, name in ((names, "names"), (uuids, "device_uuids")):
        if (
            not isinstance(sequence, Sequence)
            or isinstance(sequence, (str, bytes))
            or len(sequence) != count
            or not all(isinstance(item, str) and item for item in sequence)
        ):
            raise MeasurementRecordError(f"{field}.{name} must identify every GPU")
    _nonempty_text(value.get("driver_version"), f"{field}.driver_version")
    _nonempty_text(value.get("cuda_driver_version"), f"{field}.cuda_driver_version")
    _nonempty_text(value.get("cuda_runtime_version"), f"{field}.cuda_runtime_version")


def _close(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=1e-9, abs_tol=1e-6)


def _validate_summary(
    summary: Any,
    field: str,
    *,
    expected_series: Mapping[str, Sequence[Real]],
) -> None:
    value = _mapping(summary, field)
    for metric in SUMMARY_METRICS:
        metric_summary = _mapping(value.get(metric), f"{field}.{metric}")
        median = _nonnegative_number(
            metric_summary.get("median"), f"{field}.{metric}.median"
        )
        p95 = _nonnegative_number(metric_summary.get("p95"), f"{field}.{metric}.p95")
        count = metric_summary.get("sample_count")
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise MeasurementRecordError(
                f"{field}.{metric}.sample_count must be a positive integer"
            )
        if metric_summary.get("percentile_method") != "nearest-rank":
            raise MeasurementRecordError(
                f"{field}.{metric} must disclose nearest-rank percentiles"
            )
        if metric_summary.get("unit") != SUMMARY_UNITS[metric]:
            raise MeasurementRecordError(f"{field}.{metric} has the wrong unit")
        recomputed = summarize_series(
            expected_series[metric],
            unit=str(metric_summary.get("unit") or ""),
        )
        if count != recomputed["sample_count"]:
            raise MeasurementRecordError(f"{field}.{metric} sample count is incorrect")
        if not _close(median, recomputed["median"]) or not _close(p95, recomputed["p95"]):
            raise MeasurementRecordError(
                f"{field}.{metric} median or p95 does not match raw samples"
            )


def _validate_request_sample(
    request: Any,
    field: str,
    *,
    expected_output_tokens: int,
) -> dict[str, Any]:
    value = _mapping(request, field)
    request_id = _nonempty_text(value.get("request_id"), f"{field}.request_id")
    prompt_id = _nonempty_text(value.get("prompt_id"), f"{field}.prompt_id")
    seed = value.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise MeasurementRecordError(f"{field}.seed must be a nonnegative integer")
    prompt_tokens = value.get("prompt_tokens")
    output_tokens = value.get("output_tokens")
    if (
        isinstance(prompt_tokens, bool)
        or not isinstance(prompt_tokens, int)
        or prompt_tokens < 1
    ):
        raise MeasurementRecordError(f"{field}.prompt_tokens must be positive")
    if (
        isinstance(output_tokens, bool)
        or not isinstance(output_tokens, int)
        or output_tokens < 2
    ):
        raise MeasurementRecordError(f"{field}.output_tokens must be at least two")
    if output_tokens != expected_output_tokens:
        raise MeasurementRecordError(f"{field}.output_tokens violates the prompt protocol")
    if value.get("content_chunk_count") != output_tokens:
        raise MeasurementRecordError(
            f"{field} must retain one observed content chunk per output token"
        )
    offsets = {
        name: _nonnegative_number(value.get(name), f"{field}.{name}")
        for name in (
            "request_started_offset_ms",
            "first_token_offset_ms",
            "last_token_offset_ms",
            "completed_offset_ms",
        )
    }
    if not (
        offsets["request_started_offset_ms"]
        <= offsets["first_token_offset_ms"]
        < offsets["last_token_offset_ms"]
        <= offsets["completed_offset_ms"]
    ):
        raise MeasurementRecordError(f"{field} token timestamps are not ordered")
    token_span_ms = offsets["last_token_offset_ms"] - offsets["first_token_offset_ms"]
    recomputed = {
        "time_to_first_token_ms": (
            offsets["first_token_offset_ms"] - offsets["request_started_offset_ms"]
        ),
        "inter_token_latency_ms": token_span_ms / (output_tokens - 1),
        "end_to_end_latency_ms": (
            offsets["completed_offset_ms"] - offsets["request_started_offset_ms"]
        ),
        "output_tokens_per_second_per_stream": (output_tokens - 1) / (token_span_ms / 1000.0),
    }
    for metric, expected in recomputed.items():
        observed = _nonnegative_number(value.get(metric), f"{field}.{metric}")
        if not _close(observed, expected):
            raise MeasurementRecordError(f"{field}.{metric} does not match raw timestamps")
    return {
        **recomputed,
        "request_id": request_id,
        "prompt_id": prompt_id,
        "seed": seed,
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
    }


def _validate_measurement_points(
    side: Mapping[str, Any],
    expected: tuple[int, ...],
    prompt_token_counts: list[dict[str, Any]],
    expected_output_tokens: int,
    prompt_seed: int,
) -> None:
    expected_prompt_tokens = {
        prompt["id"]: prompt["prompt_tokens"] for prompt in prompt_token_counts
    }
    points = side.get("measurements")
    if not isinstance(points, Sequence) or isinstance(points, (str, bytes)):
        raise MeasurementRecordError("successful side measurements must be a sequence")
    observed: list[int] = []
    for index, point_value in enumerate(points):
        point = _mapping(point_value, f"measurements[{index}]")
        concurrency = point.get("concurrency")
        if isinstance(concurrency, bool) or not isinstance(concurrency, int):
            raise MeasurementRecordError("measurement concurrency must be an integer")
        observed.append(concurrency)
        batches = point.get("raw_batches")
        if (
            not isinstance(batches, Sequence)
            or isinstance(batches, (str, bytes))
            or not batches
        ):
            raise MeasurementRecordError("every concurrency must retain raw batch samples")
        repetitions = point.get("repetitions")
        if (
            isinstance(repetitions, bool)
            or not isinstance(repetitions, int)
            or repetitions < MIN_REPETITIONS
            or repetitions != len(batches)
        ):
            raise MeasurementRecordError(
                f"each concurrency must retain at least {MIN_REPETITIONS} declared repetitions"
            )
        series: dict[str, list[float]] = {metric: [] for metric in SUMMARY_METRICS}
        for batch_index, batch_value in enumerate(batches):
            batch = _mapping(batch_value, f"raw_batches[{batch_index}]")
            if batch.get("repetition") != batch_index:
                raise MeasurementRecordError("raw batch repetition indices must be ordered")
            if batch.get("concurrency") != concurrency:
                raise MeasurementRecordError("raw batch concurrency differs from its point")
            requests = batch.get("requests")
            if (
                not isinstance(requests, Sequence)
                or isinstance(requests, (str, bytes))
                or len(requests) != concurrency
            ):
                raise MeasurementRecordError(
                    "each raw batch must contain exactly concurrency request samples"
                )
            wall_time_ms = _nonnegative_number(
                batch.get("wall_time_ms"), "raw batch wall_time_ms"
            )
            if wall_time_ms <= 0.0:
                raise MeasurementRecordError("raw batch wall_time_ms must be positive")
            request_values = [
                _validate_request_sample(
                    request,
                    f"measurements[{index}].raw_batches[{batch_index}].requests[{request_index}]",
                    expected_output_tokens=expected_output_tokens,
                )
                for request_index, request in enumerate(requests)
            ]
            for request_index, request in enumerate(request_values):
                if expected_prompt_tokens.get(request["prompt_id"]) != request["prompt_tokens"]:
                    raise MeasurementRecordError(
                        "raw request prompt identity or token count is not in the prompt set"
                    )
                expected_prompt = prompt_token_counts[
                    (batch_index * concurrency + request_index) % len(prompt_token_counts)
                ]
                if (
                    request["prompt_id"] != expected_prompt["id"]
                    or request["prompt_tokens"] != expected_prompt["prompt_tokens"]
                ):
                    raise MeasurementRecordError("raw request prompt schedule is incorrect")
                expected_seed = prompt_seed + batch_index * concurrency + request_index
                if request["seed"] != expected_seed:
                    raise MeasurementRecordError("raw request seed schedule is incorrect")
            total_output_tokens = sum(
                int(request["output_tokens"]) for request in request_values
            )
            if batch.get("total_output_tokens") != total_output_tokens:
                raise MeasurementRecordError("raw batch total_output_tokens is incorrect")
            aggregate = _nonnegative_number(
                batch.get("aggregate_output_tokens_per_second"),
                "raw batch aggregate_output_tokens_per_second",
            )
            expected_aggregate = total_output_tokens / (wall_time_ms / 1000.0)
            if not _close(aggregate, expected_aggregate):
                raise MeasurementRecordError("raw batch aggregate throughput is incorrect")
            for request in request_values:
                for metric in SUMMARY_METRICS[:-1]:
                    series[metric].append(float(request[metric]))
            series["aggregate_output_tokens_per_second"].append(aggregate)
        _validate_summary(
            point.get("summary"),
            f"measurements[{index}].summary",
            expected_series=series,
        )
    if tuple(observed) != expected:
        raise MeasurementRecordError(
            f"side concurrency levels {observed} do not match record levels {list(expected)}"
        )


def _validate_successful_side(
    side: Mapping[str, Any],
    *,
    expected_concurrencies: tuple[int, ...],
    prompt_set_id: str,
    prompt_token_counts: list[dict[str, Any]],
    prompt_model_id: str,
    prompt_revision: str,
    expected_output_tokens: int,
    prompt_seed: int,
) -> None:
    if side.get("status") != "ok":
        raise MeasurementRecordError("successful record contains a non-ok side")
    if side.get("prompt_set_id") != prompt_set_id:
        raise MeasurementRecordError("side prompt set identity differs from the record")
    if side.get("prompt_token_counts") != prompt_token_counts:
        raise MeasurementRecordError("side prompt token counts differ from the record")
    _validate_gpu(side.get("gpu"), "side.gpu")
    model = _mapping(side.get("model"), "side.model")
    model_id = _nonempty_text(model.get("id"), "side.model.id")
    revision = model.get("resolved_revision")
    digest = model.get("weights_digest_sha256")
    if not (
        isinstance(revision, str)
        and IMMUTABLE_REVISION.fullmatch(revision) is not None
    ) and not (isinstance(digest, str) and len(digest) == 64):
        raise MeasurementRecordError(
            "side model must include an immutable revision or weights digest"
        )
    if model_id != prompt_model_id or revision != prompt_revision:
        raise MeasurementRecordError("side model identity differs from the prompt set")
    weights_resident_bytes = model.get("weights_resident_bytes")
    if (
        isinstance(weights_resident_bytes, bool)
        or not isinstance(weights_resident_bytes, int)
        or weights_resident_bytes < 1
    ):
        raise MeasurementRecordError("weights_resident_bytes must be a positive integer")
    runtime = _mapping(side.get("runtime"), "side.runtime")
    for field in ("name", "version", "quantization_format"):
        _nonempty_text(runtime.get(field), f"side.runtime.{field}")
    if runtime.get("tensor_parallel_size") != 1:
        raise MeasurementRecordError("tensor_parallel_size must equal one")
    max_model_len = runtime.get("max_model_len")
    if isinstance(max_model_len, bool) or not isinstance(max_model_len, int) or max_model_len < 1:
        raise MeasurementRecordError("max_model_len must be a positive integer")
    command = runtime.get("server_command")
    if (
        not isinstance(command, Sequence)
        or isinstance(command, (str, bytes))
        or not command
        or not all(isinstance(item, str) and item for item in command)
    ):
        raise MeasurementRecordError("runtime.server_command must retain exact arguments")
    disclosures = runtime.get("disclosures")
    if disclosures is not None and (
        not isinstance(disclosures, Sequence)
        or isinstance(disclosures, (str, bytes))
        or not all(isinstance(item, str) and item.strip() for item in disclosures)
    ):
        raise MeasurementRecordError("runtime.disclosures must be nonempty text entries")
    memory = _mapping(side.get("memory"), "side.memory")
    memory_values = {}
    for field in (
        "idle_before_launch_bytes",
        "steady_state_gpu_memory_bytes",
        "peak_gpu_memory_bytes",
        "peak_device_used_bytes",
    ):
        memory_values[field] = _nonnegative_number(
            memory.get(field), f"side.memory.{field}"
        )
    if memory_values["steady_state_gpu_memory_bytes"] < weights_resident_bytes:
        raise MeasurementRecordError(
            "steady-state GPU memory is smaller than the claimed resident weight bytes"
        )
    if memory_values["peak_gpu_memory_bytes"] < memory_values["steady_state_gpu_memory_bytes"]:
        raise MeasurementRecordError("peak GPU memory is smaller than steady-state memory")
    if not _close(
        memory_values["peak_device_used_bytes"],
        memory_values["idle_before_launch_bytes"] + memory_values["peak_gpu_memory_bytes"],
    ):
        raise MeasurementRecordError("peak GPU memory does not match device minus idle bytes")
    warmup = _mapping(side.get("warmup"), "side.warmup")
    requested = warmup.get("requested")
    dropped = warmup.get("dropped")
    if (
        isinstance(requested, bool)
        or not isinstance(requested, int)
        or requested < 1
        or dropped != requested
    ):
        raise MeasurementRecordError("all requested warmup requests must be dropped")
    _validate_measurement_points(
        side,
        expected_concurrencies,
        prompt_token_counts,
        expected_output_tokens,
        prompt_seed,
    )


def validate_measurement_record(record: Any) -> dict[str, Any]:
    """Validate the fail-closed top-level record and return a shallow copy."""

    value = _mapping(record, "record")
    if value.get("schema_version") != SCHEMA_VERSION:
        raise MeasurementRecordError("measurement record schema version is unsupported")
    status = value.get("status")
    if status not in {"complete", "failed"}:
        raise MeasurementRecordError("record status must be complete or failed")
    concurrencies = validate_concurrency_levels(value.get("concurrency_levels"))
    prompt_set = _mapping(value.get("prompt_set"), "prompt_set")
    prompt_set_id = _nonempty_text(prompt_set.get("id"), "prompt_set.id")
    prompt_token_counts = _validate_prompt_set(prompt_set)
    prompt_model_id = str(prompt_set["model_id"])
    prompt_revision = str(prompt_set["resolved_revision"])
    expected_output_tokens = int(prompt_set["max_output_tokens"])
    prompt_seed = int(prompt_set["seed"])
    environment = _mapping(value.get("environment"), "environment")
    environment_gpu = _mapping(environment.get("gpu"), "environment.gpu")
    _validate_gpu(environment_gpu, "environment.gpu")
    sides = _mapping(value.get("sides"), "sides")
    if set(sides) != {"baseline", "candidate"}:
        raise MeasurementRecordError("record must contain baseline and candidate sides")

    side_statuses = {
        name: _mapping(side, f"sides.{name}").get("status")
        for name, side in sides.items()
    }
    if status == "complete":
        if any(side_status != "ok" for side_status in side_statuses.values()):
            raise MeasurementRecordError("complete record requires two successful sides")
        comparison = value.get("comparison")
        if not isinstance(comparison, Mapping):
            raise MeasurementRecordError("complete record must disclose comparison eligibility")
        if (
            comparison.get("eligible") is not True
            or comparison.get("gpu_device_uuids") != environment_gpu.get("device_uuids")
            or comparison.get("prompt_set_id") != prompt_set_id
            or comparison.get("concurrency_levels") != list(concurrencies)
        ):
            raise MeasurementRecordError(
                "comparison eligibility does not match the invocation contract"
            )
        for side in sides.values():
            _validate_successful_side(
                _mapping(side, "side"),
                expected_concurrencies=concurrencies,
                prompt_set_id=prompt_set_id,
                prompt_token_counts=prompt_token_counts,
                prompt_model_id=prompt_model_id,
                prompt_revision=prompt_revision,
                expected_output_tokens=expected_output_tokens,
                prompt_seed=prompt_seed,
            )
            if _mapping(side, "side").get("gpu") != environment_gpu:
                raise MeasurementRecordError(
                    "successful sides must carry the invocation GPU identity"
                )
    else:
        if value.get("comparison") is not None:
            raise MeasurementRecordError("failed record must not emit a comparison")
        if all(side_status == "ok" for side_status in side_statuses.values()):
            raise MeasurementRecordError("failed record must identify at least one failed side")
        for name, side_value in sides.items():
            side = _mapping(side_value, f"sides.{name}")
            if side.get("status") == "ok":
                _validate_successful_side(
                    side,
                    expected_concurrencies=concurrencies,
                    prompt_set_id=prompt_set_id,
                    prompt_token_counts=prompt_token_counts,
                    prompt_model_id=prompt_model_id,
                    prompt_revision=prompt_revision,
                    expected_output_tokens=expected_output_tokens,
                    prompt_seed=prompt_seed,
                )
                if side.get("gpu") != environment_gpu:
                    raise MeasurementRecordError(
                        "successful side must carry the invocation GPU identity"
                    )
            elif side.get("status") == "failed":
                error = _mapping(side.get("error"), f"sides.{name}.error")
                _nonempty_text(error.get("type"), f"sides.{name}.error.type")
                _nonempty_text(error.get("message"), f"sides.{name}.error.message")
            else:
                raise MeasurementRecordError(f"sides.{name}.status must be ok or failed")
    return dict(value)
