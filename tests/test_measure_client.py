import asyncio
import json

import pytest

import engine.measure.client as measure_client


class _StreamResponse:
    status_code = 200

    def __init__(self, events: list[dict]) -> None:
        self._lines = [
            f"data: {json.dumps(event, separators=(',', ':'))}"
            for event in events
        ]
        self._lines.append("data: [DONE]")

    async def aread(self) -> bytes:
        return b""

    async def aiter_lines(self):
        for line in self._lines:
            yield line


class _StreamContext:
    def __init__(self, response: _StreamResponse) -> None:
        self.response = response

    async def __aenter__(self) -> _StreamResponse:
        return self.response

    async def __aexit__(self, exc_type, exc, traceback) -> None:
        return None


class _FakeClient:
    def __init__(self, events: list[dict]) -> None:
        self.response = _StreamResponse(events)
        self.requests: list[tuple[str, str, dict]] = []

    def stream(self, method: str, url: str, *, json: dict) -> _StreamContext:
        self.requests.append((method, url, json))
        return _StreamContext(self.response)


def _content_event(text: str) -> dict:
    return {
        "id": "cmpl-measured",
        "choices": [{"index": 0, "text": text, "finish_reason": None}],
        "usage": None,
    }


def _usage_event(*, prompt_tokens: int, completion_tokens: int) -> dict:
    return {
        "id": "cmpl-measured",
        "choices": [],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


async def _measure(
    monkeypatch,
    *,
    content_chunks: list[str],
    output_tokens: int,
    clock_values: list[float],
) -> dict:
    events = [_content_event(text) for text in content_chunks]
    events.append(_usage_event(prompt_tokens=3, completion_tokens=output_tokens))
    client = _FakeClient(events)
    clock = iter(clock_values)
    monkeypatch.setattr(measure_client.time, "perf_counter", lambda: next(clock))
    gate = asyncio.Event()
    gate.set()

    sample = await measure_client._measure_request(
        client,
        url="http://127.0.0.1:8000/v1/completions",
        served_model_name="kimi-linear",
        prompt={"id": "prompt-1", "token_ids": [1, 2, 3], "prompt_tokens": 3},
        max_output_tokens=output_tokens,
        seed=17,
        request_id="measured-c4-r5-s3",
        gate=gate,
        batch_clock={"origin": 90.0},
    )

    assert client.requests[0][2]["stream_options"] == {"include_usage": True}
    return sample


@pytest.mark.asyncio
async def test_chunk_token_mismatch_records_both_counts_and_marks_itl_approximate(
    monkeypatch,
):
    sample = await _measure(
        monkeypatch,
        content_chunks=["a", "b", "cd"],
        output_tokens=4,
        clock_values=[100.0, 101.0, 102.0, 103.0, 104.0],
    )

    assert sample["output_tokens"] == 4
    assert sample["content_chunk_count"] == 3
    assert sample["chunk_token_count_discrepancy"] == -1
    assert sample["inter_token_latency_approximate"] is True
    assert sample["inter_token_latency_excluded"] is False
    assert sample["inter_token_latency_basis"] == "observed content-chunk intervals"
    assert sample["inter_token_interval_count"] == 2
    assert sample["inter_token_latency_ms"] == pytest.approx(1000.0)
    assert sample["output_tokens_per_second_per_stream"] == pytest.approx(1.5)
    assert sample["arithmetic"]["inter_token_latency_ms"] == (
        "(last_token_offset_ms - first_token_offset_ms) / "
        "(content_chunk_count - 1)"
    )


@pytest.mark.asyncio
async def test_unusable_chunk_mapping_is_excluded_from_itl_summary_without_raising(
    monkeypatch,
):
    sample = await _measure(
        monkeypatch,
        content_chunks=["all tokens coalesced"],
        output_tokens=4,
        clock_values=[100.0, 101.0, 104.0],
    )

    assert sample["inter_token_latency_ms"] is None
    assert sample["inter_token_latency_excluded"] is True
    assert sample["inter_token_latency_exclusion_reason"] == (
        "fewer than two nonempty content chunks"
    )

    summary = measure_client._summarize_batches(
        [
            {
                "aggregate_output_tokens_per_second": 1.0,
                "requests": [sample],
            }
        ]
    )

    assert summary["inter_token_latency_ms"] == {
        "median": None,
        "p95": None,
        "sample_count": 0,
        "unit": "ms",
        "percentile_method": "nearest-rank",
        "total_request_count": 1,
        "excluded_sample_count": 1,
        "approximate_sample_count": 0,
    }
