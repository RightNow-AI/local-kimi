"""Token-stream workload runner with raw, independently recomputable samples."""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Mapping, Sequence
from typing import Any

from .record import MIN_REPETITIONS, summarize_series


class RequestFailure(RuntimeError):
    """Raised when a server cannot satisfy the exact measurement protocol."""


def _prompt_for_slot(
    prompts: Sequence[Mapping[str, Any]],
    *,
    concurrency: int,
    repetition: int,
    slot: int,
) -> Mapping[str, Any]:
    index = repetition * concurrency + slot
    return prompts[index % len(prompts)]


async def _measure_request(
    client: Any,
    *,
    url: str,
    served_model_name: str,
    prompt: Mapping[str, Any],
    max_output_tokens: int,
    seed: int,
    request_id: str,
    gate: asyncio.Event,
    batch_clock: Mapping[str, float],
) -> dict[str, Any]:
    await gate.wait()
    batch_origin = batch_clock["origin"]
    started = time.perf_counter()
    first_token_at: float | None = None
    last_token_at: float | None = None
    completed_at: float | None = None
    usage: Mapping[str, Any] | None = None
    response_id: str | None = None
    finish_reason: str | None = None
    saw_done = False
    content_chunks = 0
    payload = {
        "model": served_model_name,
        "prompt": prompt["token_ids"],
        "max_tokens": max_output_tokens,
        "min_tokens": max_output_tokens,
        "ignore_eos": True,
        "temperature": 0.0,
        "seed": seed,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    async with client.stream("POST", url, json=payload) as response:
        if response.status_code != 200:
            body = (await response.aread()).decode("utf-8", errors="replace")
            raise RequestFailure(
                f"{request_id} returned HTTP {response.status_code}: {body[-2000:]}"
            )
        async for line in response.aiter_lines():
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                saw_done = True
                completed_at = time.perf_counter()
                break
            if not data:
                continue
            try:
                event = json.loads(data)
            except json.JSONDecodeError as exc:
                raise RequestFailure(f"{request_id} returned invalid SSE JSON") from exc
            if not isinstance(event, Mapping):
                raise RequestFailure(f"{request_id} returned a non-object SSE event")
            if isinstance(event.get("id"), str):
                response_id = event["id"]
            event_usage = event.get("usage")
            if isinstance(event_usage, Mapping):
                usage = event_usage
            choices = event.get("choices") or []
            if not isinstance(choices, list):
                raise RequestFailure(f"{request_id} returned malformed choices")
            for choice in choices:
                if not isinstance(choice, Mapping):
                    continue
                if isinstance(choice.get("finish_reason"), str):
                    finish_reason = choice["finish_reason"]
                text = choice.get("text")
                if isinstance(text, str) and text:
                    observed = time.perf_counter()
                    if first_token_at is None:
                        first_token_at = observed
                    last_token_at = observed
                    content_chunks += 1
    if completed_at is None:
        completed_at = time.perf_counter()
    if not saw_done:
        raise RequestFailure(f"{request_id} stream ended without [DONE]")
    if first_token_at is None or last_token_at is None:
        raise RequestFailure(f"{request_id} emitted no nonempty token content")
    if not isinstance(usage, Mapping):
        raise RequestFailure(f"{request_id} omitted final token usage")
    prompt_tokens = usage.get("prompt_tokens")
    output_tokens = usage.get("completion_tokens")
    if prompt_tokens != prompt["prompt_tokens"]:
        raise RequestFailure(
            f"{request_id} reported {prompt_tokens} prompt tokens, expected "
            f"{prompt['prompt_tokens']}"
        )
    if isinstance(output_tokens, bool) or not isinstance(output_tokens, int):
        raise RequestFailure(f"{request_id} omitted an integer completion token count")
    if output_tokens < 2:
        raise RequestFailure(
            f"{request_id} produced fewer than two tokens, so inter-token latency is undefined"
        )
    if output_tokens != max_output_tokens:
        raise RequestFailure(
            f"{request_id} produced {output_tokens} tokens, expected {max_output_tokens}"
        )

    ttft_seconds = first_token_at - started
    token_span_seconds = last_token_at - first_token_at
    e2e_seconds = completed_at - started
    token_aligned = content_chunks == output_tokens
    interval_count = (output_tokens - 1) if token_aligned else (content_chunks - 1)
    itl_exclusion_reason: str | None = None
    if interval_count < 1:
        itl_exclusion_reason = "fewer than two nonempty content chunks"
    elif token_span_seconds <= 0.0:
        itl_exclusion_reason = "nonpositive observed content-chunk span"

    if itl_exclusion_reason is None:
        inter_token_seconds: float | None = token_span_seconds / interval_count
        per_stream_tokens_per_second: float | None = (
            (output_tokens - 1) / token_span_seconds
        )
    else:
        inter_token_seconds = None
        per_stream_tokens_per_second = None

    if token_aligned:
        itl_basis = "token-aligned content-chunk intervals"
        itl_arithmetic = (
            "(last_token_offset_ms - first_token_offset_ms) / (output_tokens - 1)"
        )
    elif itl_exclusion_reason is None:
        itl_basis = "observed content-chunk intervals"
        itl_arithmetic = (
            "(last_token_offset_ms - first_token_offset_ms) / "
            "(content_chunk_count - 1)"
        )
    else:
        itl_basis = "excluded"
        itl_arithmetic = f"excluded: {itl_exclusion_reason}"

    return {
        "request_id": request_id,
        "prompt_id": prompt["id"],
        "seed": seed,
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "request_started_offset_ms": (started - batch_origin) * 1000.0,
        "first_token_offset_ms": (first_token_at - batch_origin) * 1000.0,
        "last_token_offset_ms": (last_token_at - batch_origin) * 1000.0,
        "completed_offset_ms": (completed_at - batch_origin) * 1000.0,
        "time_to_first_token_ms": ttft_seconds * 1000.0,
        "inter_token_latency_ms": (
            inter_token_seconds * 1000.0
            if inter_token_seconds is not None
            else None
        ),
        "inter_token_latency_approximate": not token_aligned,
        "inter_token_latency_excluded": itl_exclusion_reason is not None,
        "inter_token_latency_exclusion_reason": itl_exclusion_reason,
        "inter_token_latency_basis": itl_basis,
        "inter_token_interval_count": max(interval_count, 0),
        "end_to_end_latency_ms": e2e_seconds * 1000.0,
        "output_tokens_per_second_per_stream": per_stream_tokens_per_second,
        "output_tokens_per_second_per_stream_approximate": not token_aligned,
        "content_chunk_count": content_chunks,
        "chunk_token_count_discrepancy": content_chunks - output_tokens,
        "response_id": response_id,
        "finish_reason": finish_reason,
        "arithmetic": {
            "time_to_first_token_ms": "first_token_offset_ms - request_started_offset_ms",
            "inter_token_latency_ms": itl_arithmetic,
            "end_to_end_latency_ms": "completed_offset_ms - request_started_offset_ms",
            "output_tokens_per_second_per_stream": (
                "(output_tokens - 1) / ((last_token_offset_ms - first_token_offset_ms) / 1000)"
                if per_stream_tokens_per_second is not None
                else f"excluded: {itl_exclusion_reason}"
            ),
        },
    }


async def _measure_batch(
    client: Any,
    *,
    url: str,
    served_model_name: str,
    prompts: Sequence[Mapping[str, Any]],
    concurrency: int,
    repetition: int,
    max_output_tokens: int,
    seed: int,
    request_prefix: str,
) -> dict[str, Any]:
    gate = asyncio.Event()
    batch_clock: dict[str, float] = {}
    tasks = []
    for slot in range(concurrency):
        prompt = _prompt_for_slot(
            prompts,
            concurrency=concurrency,
            repetition=repetition,
            slot=slot,
        )
        request_id = f"{request_prefix}-c{concurrency}-r{repetition}-s{slot}"
        tasks.append(
            asyncio.create_task(
                _measure_request(
                    client,
                    url=url,
                    served_model_name=served_model_name,
                    prompt=prompt,
                    max_output_tokens=max_output_tokens,
                    seed=seed + repetition * concurrency + slot,
                    request_id=request_id,
                    gate=gate,
                    batch_clock=batch_clock,
                )
            )
        )
    batch_origin = time.perf_counter()
    batch_clock["origin"] = batch_origin
    gate.set()
    requests = await asyncio.gather(*tasks)
    completed = time.perf_counter()
    wall_seconds = completed - batch_origin
    total_output_tokens = sum(sample["output_tokens"] for sample in requests)
    return {
        "repetition": repetition,
        "concurrency": concurrency,
        "wall_time_ms": wall_seconds * 1000.0,
        "total_output_tokens": total_output_tokens,
        "aggregate_output_tokens_per_second": total_output_tokens / wall_seconds,
        "requests": requests,
        "arithmetic": {
            "aggregate_output_tokens_per_second": (
                "sum(request.output_tokens) / (wall_time_ms / 1000)"
            )
        },
    }


def _summarize_batches(batches: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    requests = [request for batch in batches for request in batch["requests"]]
    inter_token_samples = [
        request["inter_token_latency_ms"]
        for request in requests
        if request["inter_token_latency_ms"] is not None
    ]
    per_stream_samples = [
        request["output_tokens_per_second_per_stream"]
        for request in requests
        if request["output_tokens_per_second_per_stream"] is not None
    ]
    inter_token_summary = _summarize_optional_series(inter_token_samples, unit="ms")
    inter_token_summary.update(
        {
            "total_request_count": len(requests),
            "excluded_sample_count": sum(
                bool(request["inter_token_latency_excluded"])
                for request in requests
            ),
            "approximate_sample_count": sum(
                bool(request["inter_token_latency_approximate"])
                and not bool(request["inter_token_latency_excluded"])
                for request in requests
            ),
        }
    )
    per_stream_summary = _summarize_optional_series(
        per_stream_samples,
        unit="tokens/s",
    )
    per_stream_summary["excluded_sample_count"] = (
        len(requests) - len(per_stream_samples)
    )
    return {
        "time_to_first_token_ms": summarize_series(
            [request["time_to_first_token_ms"] for request in requests],
            unit="ms",
        ),
        "inter_token_latency_ms": inter_token_summary,
        "end_to_end_latency_ms": summarize_series(
            [request["end_to_end_latency_ms"] for request in requests],
            unit="ms",
        ),
        "output_tokens_per_second_per_stream": per_stream_summary,
        "aggregate_output_tokens_per_second": summarize_series(
            [batch["aggregate_output_tokens_per_second"] for batch in batches],
            unit="tokens/s",
        ),
    }


def _summarize_optional_series(
    values: Sequence[float],
    *,
    unit: str,
) -> dict[str, Any]:
    if values:
        return summarize_series(values, unit=unit)
    return {
        "median": None,
        "p95": None,
        "sample_count": 0,
        "unit": unit,
        "percentile_method": "nearest-rank",
    }


async def _run_workload_async(
    *,
    base_url: str,
    served_model_name: str,
    prompt_set: Mapping[str, Any],
    concurrency_levels: Sequence[int],
    repetitions: int,
    warmup_requests: int,
    request_timeout_seconds: float,
) -> dict[str, Any]:
    import httpx

    if repetitions < MIN_REPETITIONS:
        raise ValueError(
            f"repetitions must be at least {MIN_REPETITIONS} for an observed p95 rank"
        )
    if warmup_requests < 1:
        raise ValueError("warmup_requests must be positive")
    prompts = prompt_set["prompts"]
    max_output_tokens = prompt_set["max_output_tokens"]
    seed = prompt_set["seed"]
    limits = httpx.Limits(
        max_connections=max(max(concurrency_levels), warmup_requests) + 8,
        max_keepalive_connections=max(max(concurrency_levels), warmup_requests) + 8,
    )
    timeout = httpx.Timeout(request_timeout_seconds)
    async with httpx.AsyncClient(limits=limits, timeout=timeout) as client:
        warmup_batch = await _measure_batch(
            client,
            url=f"{base_url.rstrip('/')}/v1/completions",
            served_model_name=served_model_name,
            prompts=prompts,
            concurrency=warmup_requests,
            repetition=0,
            max_output_tokens=max_output_tokens,
            seed=seed,
            request_prefix="warmup",
        )
        measurements = []
        for concurrency in concurrency_levels:
            batches = []
            for repetition in range(repetitions):
                batches.append(
                    await _measure_batch(
                        client,
                        url=f"{base_url.rstrip('/')}/v1/completions",
                        served_model_name=served_model_name,
                        prompts=prompts,
                        concurrency=concurrency,
                        repetition=repetition,
                        max_output_tokens=max_output_tokens,
                        seed=seed,
                        request_prefix="measured",
                    )
                )
            measurements.append(
                {
                    "concurrency": concurrency,
                    "repetitions": repetitions,
                    "summary": _summarize_batches(batches),
                    "raw_batches": batches,
                }
            )
    return {
        "warmup": {
            "requested": warmup_requests,
            "dropped": len(warmup_batch["requests"]),
            "mode": "one discarded concurrent batch",
        },
        "measurements": measurements,
    }


def run_workload(**kwargs: Any) -> dict[str, Any]:
    """Run the exact async workload from synchronous process orchestration."""

    return asyncio.run(_run_workload_async(**kwargs))
