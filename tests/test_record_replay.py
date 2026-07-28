"""Recording and replay — the machinery the conformance suite is built on.

``tests/test_conformance.py`` asserts that the committed cassettes still match.
This file asserts that the *thing doing the matching* is trustworthy: that a
cassette survives a disk round trip, that credentials never reach disk, that SSE
parsing copes with what real clients emit, and — most importantly — that the
normalisation which hides volatile ids does not also hide a reasoning
regression. A conformance suite that normalises too hard passes forever.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from k3.reasoning import encode_signature
from k3.record import (
    Cassette,
    Recorder,
    RecordingEngine,
    ReplayEngine,
    load_cassettes,
    parse_sse,
)
from k3.replay import diff_json, normalize, normalize_upstream, replay_cassette
from k3.server import ServerConfig, create_app
from k3.upstream import MockUpstream, UpstreamConfig

UNICODE = "héllo wörld — 日本語 ✅ Ünicode ⟨λ⟩"


def _cassette(**overrides: Any) -> Cassette:
    base: dict[str, Any] = dict(
        preset="claude-code",
        path="/v1/messages",
        detected_via="user-agent ~ /claude-cli//",
        request_headers={"user-agent": "claude-cli/1.0.60", "anthropic-version": "2023-06-01"},
        request_body={"model": "k3", "messages": [{"role": "user", "content": UNICODE}]},
        upstream_request={"model": "k3", "messages": [{"role": "user", "content": UNICODE}]},
        upstream_chunks=[{"choices": [{"index": 0, "delta": {"content": UNICODE}}]}],
        upstream_response={"id": "chatcmpl-x", "choices": []},
        client_status=200,
        client_body={"type": "message", "text": UNICODE},
        client_sse=["event: ping\ndata: {}\n\n"],
        streamed=True,
        recorded_at=1_700_000_000.5,
        name="round-trip-fixture",
        source="synthetic",
    )
    base.update(overrides)
    return Cassette(**base)


def _stream_payload(stream: bool = True) -> dict[str, Any]:
    return {
        "model": "k3",
        "messages": [{"role": "user", "content": "list the files in this repo"}],
        "stream": stream,
    }


async def _drain(engine: Any, payload: dict[str, Any]) -> list[dict[str, Any]]:
    return [chunk async for chunk in engine.chat_stream(payload)]


# --------------------------------------------------------------------------
# 1. cassette round trip
# --------------------------------------------------------------------------


def test_cassette_survives_a_disk_round_trip_both_ways(tmp_path: Path) -> None:
    original = _cassette()
    expected = original.to_dict()

    plain = original.save(tmp_path)
    packed = original.save(tmp_path, compress=True)

    assert plain.name == "round-trip-fixture.json"
    assert packed.name == "round-trip-fixture.json.gz"

    for target in (plain, packed):
        loaded = Cassette.load(target)
        assert loaded.to_dict() == expected, f"{target.name} lost a field"
        # Called out explicitly because both are easy to drop in to_dict/from_dict
        # and neither has a loud failure mode.
        assert loaded.source == "synthetic"
        assert loaded.streamed is True
        assert loaded.recorded_at == 1_700_000_000.5
        assert loaded.request_body["messages"][0]["content"] == UNICODE
        assert loaded.client_body["text"] == UNICODE
        assert loaded.client_sse == original.client_sse
        assert loaded.upstream_chunks == original.upstream_chunks
        assert loaded.upstream_response == original.upstream_response

    # gzip is on by default for committed fixtures precisely because real client
    # traffic is huge and repetitive; prove that actually buys something.
    big = _cassette(
        name="big-fixture",
        request_body={"messages": [{"role": "user", "content": "réquest body line\n" * 4000}]},
    )
    big_plain = big.save(tmp_path)
    big_packed = big.save(tmp_path, compress=True)
    assert big_plain.stat().st_size > 50_000
    assert big_packed.stat().st_size * 10 < big_plain.stat().st_size

    # The compressed file really is gzip, not a plain file with a .gz name.
    with gzip.open(big_packed, "rt", encoding="utf-8") as fh:
        assert json.load(fh)["name"] == "big-fixture"


# --------------------------------------------------------------------------
# 2. redaction
# --------------------------------------------------------------------------


def test_recorder_redacts_credentials_and_lowercases_names(tmp_path: Path) -> None:
    recorder = Recorder(tmp_path)
    cassette = recorder.start(
        path="/v1/messages",
        headers={
            "Authorization": "Bearer sk-super-secret",
            "X-Api-Key": "sk-ant-also-secret",
            "Cookie": "session=abc123",
            "Proxy-Authorization": "Basic dXNlcjpwYXNz",
            "User-Agent": "claude-cli/1.0.60",
            "Anthropic-Version": "2023-06-01",
        },
        body={"model": "k3"},
        preset="claude-code",
        detected_via="test",
    )
    assert cassette is not None
    headers = cassette.request_headers

    for name in ("authorization", "x-api-key", "cookie", "proxy-authorization"):
        assert headers[name] == "<redacted>", f"{name} was written to disk"

    assert headers["user-agent"] == "claude-cli/1.0.60"
    assert headers["anthropic-version"] == "2023-06-01"

    assert list(headers) == [k.lower() for k in headers], "header names must be lowercased"
    blob = json.dumps(cassette.to_dict())
    for secret in ("sk-super-secret", "sk-ant-also-secret", "session=abc123", "dXNlcjpwYXNz"):
        assert secret not in blob


def test_recorder_is_a_no_op_when_disabled() -> None:
    recorder = Recorder(None)
    assert recorder.enabled is False
    assert recorder.start(path="/v1/messages", headers={}, body={}, preset="openai", detected_via="") is None


def test_recorder_keeps_only_the_most_recent_written_paths(tmp_path: Path) -> None:
    recorder = Recorder(tmp_path)
    targets = []

    for index in range(105):
        target = recorder.finish(_cassette(name=f"recording-{index:03d}"))
        assert target is not None
        targets.append(target)

    assert len(recorder.written) == 100
    assert recorder.written == targets[-100:]


# --------------------------------------------------------------------------
# 3. loading a directory
# --------------------------------------------------------------------------


def test_load_cassettes_reads_both_extensions_in_a_stable_order(tmp_path: Path) -> None:
    _cassette(name="zulu", source="recorded").save(tmp_path)
    _cassette(name="alpha", source="recorded").save(tmp_path, compress=True)
    _cassette(name="mike", source="synthetic").save(tmp_path)
    _cassette(name="bravo", source="synthetic").save(tmp_path, compress=True)
    (tmp_path / "notes.txt").write_text("not a cassette", encoding="utf-8")

    loaded = load_cassettes(tmp_path)
    # alpha/bravo are .json.gz, mike/zulu are .json: both extensions, one order.
    assert [c.name for c in loaded] == ["alpha", "bravo", "mike", "zulu"]
    assert [c.name for c in load_cassettes(tmp_path)] == [c.name for c in loaded]
    assert {c.source for c in loaded} == {"recorded", "synthetic"}
    assert load_cassettes(tmp_path / "does-not-exist") == []


# --------------------------------------------------------------------------
# 4. SSE parsing
# --------------------------------------------------------------------------


def test_parse_sse_handles_everything_real_clients_emit() -> None:
    raw = (
        "event: message_start\n"
        'data: {"type": "message_start", "text": "h\\u00e9llo"}\n'
        "\n"
        ": a bare comment keepalive on its own\n"
        "\n"
        'data: {"bare": true}\n'
        "\n"
        ": a comment sharing a block with data\n"
        'data: {"after_comment": true}\n'
        "\n"
        'data: {"multi_line": [1,\n'
        "data: 2, 3]}\n"
        "\n"
        "data: {this is not json\n"
        "\n"
        "data: [DONE]\n"
        "\n"
    )
    records = parse_sse(raw)

    assert records == [
        {"event": "message_start", "data": {"type": "message_start", "text": "héllo"}},
        {"event": None, "data": {"bare": True}},
        {"event": None, "data": {"after_comment": True}},
        {"event": None, "data": {"multi_line": [1, 2, 3]}},
        {"event": None, "data": "{this is not json"},
        {"event": None, "data": "[DONE]"},
    ]

    # bytes in, same records out
    assert parse_sse(raw.encode("utf-8")) == records
    # nothing at all is not an error
    assert parse_sse("") == []
    assert parse_sse("\n\n\n") == []
    assert parse_sse(": only comments\n\n") == []


# --------------------------------------------------------------------------
# 5. RecordingEngine is transparent
# --------------------------------------------------------------------------


async def test_recording_engine_does_not_change_what_the_pipeline_sees() -> None:
    payload = _stream_payload()

    reference = await _drain(MockUpstream(UpstreamConfig(model="k3")), dict(payload))

    cassette = _cassette(upstream_chunks=[], upstream_response=None, upstream_request=None)
    recorder = RecordingEngine(MockUpstream(UpstreamConfig(model="k3")), cassette)
    observed = await _drain(recorder, dict(payload))

    assert observed == reference
    assert cassette.upstream_chunks == reference
    assert cassette.upstream_request == payload
    assert cassette.upstream_response is None

    # Health/models/close are pass-throughs too, or `--record` would change
    # what `/health` reports.
    assert await recorder.health() == (True, "mock")
    assert await recorder.models() == {"object": "list", "data": [{"id": "k3", "object": "model"}]}
    await recorder.aclose()


async def test_recording_engine_records_non_streaming_calls() -> None:
    cassette = _cassette(upstream_chunks=[], upstream_response=None, upstream_request=None)
    inner = MockUpstream(UpstreamConfig(model="k3"))
    recorder = RecordingEngine(inner, cassette)

    payload = _stream_payload(stream=False)
    result = await recorder.chat(dict(payload))

    assert cassette.upstream_request == payload
    assert cassette.upstream_response == result
    assert cassette.upstream_chunks == []


async def test_recording_engine_without_a_cassette_still_works() -> None:
    engine = RecordingEngine(MockUpstream(UpstreamConfig(model="k3")), None)
    assert await _drain(engine, _stream_payload()) != []


# --------------------------------------------------------------------------
# 6. ReplayEngine
# --------------------------------------------------------------------------


async def test_replay_engine_streams_recorded_chunks_verbatim() -> None:
    chunks = [
        {"id": "chatcmpl-1", "choices": [{"index": 0, "delta": {"reasoning_content": "thinking"}}]},
        {"id": "chatcmpl-1", "choices": [{"index": 0, "delta": {"content": "answer"}}]},
        {"id": "chatcmpl-1", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
    ]
    engine = ReplayEngine(_cassette(upstream_chunks=chunks, upstream_response=None))

    payload = _stream_payload()
    assert await _drain(engine, payload) == chunks
    assert engine.seen_payload == payload


async def test_replay_engine_synthesises_chunks_from_a_recorded_response() -> None:
    response = {
        "id": "chatcmpl-json",
        "model": "k3",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "the answer",
                    "reasoning_content": "the thinking",
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7},
    }
    cassette = _cassette(upstream_chunks=[], upstream_response=response)
    engine = ReplayEngine(cassette)

    payload = _stream_payload()
    chunks = await _drain(engine, payload)
    assert engine.seen_payload == payload

    from k3.pipeline import response_to_chunks

    assert chunks == response_to_chunks(response)

    reasoning = "".join(
        c["delta"].get("reasoning_content", "")
        for chunk in chunks
        for c in chunk.get("choices") or []
        if "delta" in c
    )
    content = "".join(
        c["delta"].get("content", "")
        for chunk in chunks
        for c in chunk.get("choices") or []
        if "delta" in c
    )
    assert reasoning == "the thinking"
    assert content == "the answer"
    assert chunks[-1]["choices"][0]["finish_reason"] == "stop"

    # Non-streaming replay of the same cassette hands back the response itself.
    assert await engine.chat(payload) == response
    assert engine.seen_payload == payload


async def test_replay_engine_folds_chunks_into_a_response_when_asked_for_one() -> None:
    chunks = [
        {"id": "chatcmpl-folded", "model": "k3-folded", "choices": [{"index": 0, "delta": {"role": "assistant"}}]},
        {"choices": [{"index": 0, "delta": {"reasoning_content": "step one. "}}]},
        {"choices": [{"index": 0, "delta": {"reasoning_content": "step two."}}]},
        {"choices": [{"index": 0, "delta": {"content": "final "}}]},
        {"choices": [{"index": 0, "delta": {"content": "answer"}}]},
        {"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]},
        {"choices": [], "usage": {"prompt_tokens": 11, "completion_tokens": 5, "total_tokens": 16}},
    ]
    engine = ReplayEngine(_cassette(upstream_chunks=chunks, upstream_response=None))

    payload = _stream_payload(stream=False)
    folded = await engine.chat(payload)

    assert engine.seen_payload == payload
    assert folded["object"] == "chat.completion"
    assert folded["id"] == "chatcmpl-folded"
    assert folded["model"] == "k3-folded"
    message = folded["choices"][0]["message"]
    assert message["role"] == "assistant"
    assert message["content"] == "final answer"
    assert message["reasoning_content"] == "step one. step two."
    assert folded["choices"][0]["finish_reason"] == "tool_calls"
    assert folded["usage"] == {"prompt_tokens": 11, "completion_tokens": 5, "total_tokens": 16}


async def test_replay_engine_reports_itself_healthy_and_closes_cleanly() -> None:
    engine = ReplayEngine(_cassette())
    assert await engine.health() == (True, "replay")
    assert await engine.models() == {"object": "list", "data": []}
    assert await engine.aclose() is None


# --------------------------------------------------------------------------
# 7. normalize renumbers volatile identity
# --------------------------------------------------------------------------


def _volatile(suffix: str, stamp: int) -> dict[str, Any]:
    return {
        "id": f"msg_{suffix}",
        "message_id": f"chatcmpl-{suffix}",
        "item_id": f"resp_{suffix}",
        "response_id": f"rs_{suffix}",
        "tool_use_id": f"toolu_{suffix}",
        "call_id": f"call_{suffix}",
        "tool_call_id": f"fc_{suffix}",
        "created": stamp,
        "created_at": stamp,
        "content": "the part that actually matters",
        "nested": [{"id": f"msg_{suffix}_deep", "created": stamp}],
    }


def test_normalize_renumbers_ids_and_collapses_timestamps() -> None:
    a = _volatile("aaaaaaaa1111", 1_700_000_000)
    b = _volatile("bbbbbbbb2222", 1_800_000_000)
    assert a != b

    na, nb = normalize(a), normalize(b)
    assert na == nb
    assert diff_json(na, nb) == []

    assert na["id"] == "<id:1>"
    assert na["message_id"] == "<id:2>"
    assert na["item_id"] == "<id:3>"
    assert na["response_id"] == "<id:4>"
    assert na["tool_use_id"] == "<id:5>"
    assert na["call_id"] == "<id:6>"
    assert na["tool_call_id"] == "<id:7>"
    assert na["nested"][0]["id"] == "<id:8>"
    assert na["created"] == 0
    assert na["created_at"] == 0
    assert na["content"] == "the part that actually matters"


def test_normalize_exposes_a_broken_tool_id_relationship() -> None:
    matching = {
        "content": [
            {"type": "tool_use", "id": "toolu_aaaaaaaa1111"},
            {"type": "tool_result", "tool_use_id": "toolu_aaaaaaaa1111"},
        ]
    }
    broken = {
        "content": [
            {"type": "tool_use", "id": "toolu_aaaaaaaa1111"},
            {"type": "tool_result", "tool_use_id": "toolu_bbbbbbbb2222"},
        ]
    }

    expected = normalize(matching)
    actual = normalize(broken)

    assert expected != actual
    assert diff_json(expected, actual) == [
        "$.content[1].tool_use_id: '<id:1>' != '<id:2>'"
    ]


def test_normalize_erases_random_id_values_but_keeps_relationships() -> None:
    first = {
        "content": [
            {"type": "tool_use", "id": "toolu_aaaaaaaa1111"},
            {"type": "tool_result", "tool_use_id": "toolu_aaaaaaaa1111"},
        ]
    }
    second = {
        "content": [
            {"type": "tool_use", "id": "toolu_bbbbbbbb2222"},
            {"type": "tool_result", "tool_use_id": "toolu_bbbbbbbb2222"},
        ]
    }

    normalized_first = normalize(first)
    normalized_second = normalize(second)

    assert normalized_first == normalized_second
    assert normalized_first["content"][0]["id"] == "<id:1>"
    assert normalized_first["content"][1]["tool_use_id"] == "<id:1>"


def test_normalize_shares_id_mapping_between_fields_and_free_text() -> None:
    normalized = normalize(
        {
            "tool_call_id": "call_abcdef123",
            "content": "completed call_abcdef123 successfully",
        }
    )

    assert normalized["tool_call_id"] == "<id:1>"
    assert normalized["content"] == "completed <id:1> successfully"


def test_normalize_id_numbering_is_deterministic_and_call_local() -> None:
    payload = {
        "items": [
            {"id": "msg_aaaaaa1111"},
            {"call_id": "call_bbbbbb2222"},
            {"tool_use_id": "msg_aaaaaa1111"},
        ]
    }
    expected = {
        "items": [
            {"id": "<id:1>"},
            {"call_id": "<id:2>"},
            {"tool_use_id": "<id:1>"},
        ]
    }

    assert normalize(payload) == expected
    assert normalize(payload) == expected
    assert normalize({"response_id": "resp_cccccc3333"}) == {
        "response_id": "<id:1>"
    }


def test_normalize_leaves_real_content_alone() -> None:
    """Normalisation must not be a blanket wildcard."""
    a = {"id": "msg_one", "text": "the model said A", "count": 1}
    b = {"id": "msg_two", "text": "the model said B", "count": 2}
    assert normalize(a) != normalize(b)
    diffs = diff_json(normalize(a), normalize(b))
    assert any("text" in d for d in diffs)
    assert any("count" in d for d in diffs)
    # An id that is not one of ours is left exactly as it is.
    assert normalize({"id": "sess-not-ours"})["id"] == "sess-not-ours"


# --------------------------------------------------------------------------
# 8. normalize preserves meaning
# --------------------------------------------------------------------------


def test_normalize_digests_signatures_without_hiding_a_reasoning_regression() -> None:
    same_a = encode_signature("the model reasoned carefully about the file layout", "ledger-one")
    same_b = encode_signature("the model reasoned carefully about the file layout", "ledger-two")
    different = encode_signature("the model reasoned about something else entirely", "ledger-one")

    assert same_a != same_b, "fixture is not exercising the volatile ledger id"

    na = normalize({"signature": same_a})["signature"]
    nb = normalize({"signature": same_b})["signature"]
    nc = normalize({"signature": different})["signature"]

    assert na.startswith("<sig:") and na.endswith(">")
    # A different ledger id must not show up as a diff...
    assert na == nb
    # ...but different reasoning absolutely must.
    assert na != nc
    assert diff_json(normalize({"signature": same_a}), normalize({"signature": different})) != []

    # The Responses-API vehicle is normalised the same way.
    assert normalize({"encrypted_content": same_a})["encrypted_content"] == na
    assert normalize({"encrypted_content": different})["encrypted_content"] == nc

    # Signatures we did not mint are passed through rather than swallowed.
    assert normalize({"signature": "ErUBCkYIBBgCKkA..."})["signature"] == "ErUBCkYIBBgCKkA..."


# --------------------------------------------------------------------------
# 9. normalize_upstream
# --------------------------------------------------------------------------


def _upstream(args: str, reasoning: str, call_id: str) -> dict[str, Any]:
    return {
        "model": "k3",
        "messages": [
            {"role": "user", "content": "read the file"},
            {
                "role": "assistant",
                "content": "",
                "reasoning_content": reasoning,
                "tool_calls": [
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {"name": "Read", "arguments": args},
                    }
                ],
            },
        ],
    }


def test_normalize_upstream_ignores_tool_argument_formatting() -> None:
    a = _upstream('{"path": "a.txt", "limit": 5}', "same thinking", "call_aaaaaa")
    b = _upstream('{"limit":5,"path":"a.txt"}', "same thinking", "call_bbbbbb")
    assert a != b
    assert normalize_upstream(a) == normalize_upstream(b)
    assert diff_json(normalize_upstream(a), normalize_upstream(b)) == []

    call = normalize_upstream(a)["messages"][1]["tool_calls"][0]
    assert "id" not in call, "tool-call ids are regenerated per run and must be dropped"
    assert call["function"]["arguments"] == '{"limit":5,"path":"a.txt"}'

    # Arguments that are not JSON at all are left alone rather than exploding.
    ugly = _upstream("not json", "same thinking", "call_cccccc")
    assert normalize_upstream(ugly)["messages"][1]["tool_calls"][0]["function"]["arguments"] == "not json"


def test_normalize_upstream_drop_reasoning_is_opt_in() -> None:
    a = _upstream('{"path":"a.txt"}', "reasoning A", "call_a")
    b = _upstream('{"path":"a.txt"}', "reasoning B", "call_b")

    dropped = normalize_upstream(a, drop_reasoning=True)
    assert "reasoning_content" not in dropped["messages"][1]
    assert dropped == normalize_upstream(b, drop_reasoning=True)

    kept_a = normalize_upstream(a, drop_reasoning=False)
    kept_b = normalize_upstream(b, drop_reasoning=False)
    assert kept_a["messages"][1]["reasoning_content"] == "reasoning A"
    assert kept_a != kept_b
    diffs = diff_json(kept_a, kept_b)
    assert diffs and any("reasoning_content" in d for d in diffs), diffs

    # The default is to keep reasoning: losing it silently is the whole failure
    # mode this suite exists to catch.
    assert normalize_upstream(a)["messages"][1]["reasoning_content"] == "reasoning A"

    # Non-dict payloads pass straight through.
    assert normalize_upstream(None) is None
    assert normalize_upstream("nope") == "nope"


# --------------------------------------------------------------------------
# 10. diff_json
# --------------------------------------------------------------------------


def test_diff_json_is_empty_for_equal_structures() -> None:
    obj = {"a": [1, 2, {"b": "c"}], "d": None, "e": 1.5}
    assert diff_json(obj, json.loads(json.dumps(obj))) == []


def test_diff_json_reports_missing_and_unexpected_keys() -> None:
    expected = {"kept": 1, "only_expected": "gone"}
    actual = {"kept": 1, "only_actual": "new"}
    diffs = diff_json(expected, actual)

    missing = [d for d in diffs if "only_expected" in d]
    unexpected = [d for d in diffs if "only_actual" in d]
    assert len(missing) == 1 and "missing" in missing[0]
    assert len(unexpected) == 1 and "unexpected" in unexpected[0]
    assert all(d.startswith("$.") for d in diffs), diffs


def test_diff_json_reports_length_and_scalar_differences_with_a_path() -> None:
    length = diff_json({"items": [1, 2, 3]}, {"items": [1, 2]})
    assert any("$.items" in d and "length 3 != 2" in d for d in length), length

    scalar = diff_json(
        {"choices": [{"message": {"content": "expected"}}]},
        {"choices": [{"message": {"content": "actual"}}]},
    )
    assert scalar == ["$.choices[0].message.content: 'expected' != 'actual'"]

    typed = diff_json({"n": "1"}, {"n": 1})
    assert typed and "type str != int" in typed[0]

    rooted = diff_json({"a": 1}, {"a": 2}, path="$.upstream")
    assert rooted == ["$.upstream.a: 1 != 2"]


def test_diff_json_respects_the_limit_for_value_differences() -> None:
    expected = {f"k{i}": "expected" for i in range(40)}
    actual = {f"k{i}": "actual" for i in range(40)}

    assert len(diff_json(expected, actual, limit=5)) == 5
    assert len(diff_json(expected, actual, limit=1)) == 1
    assert len(diff_json(expected, actual)) == 25  # documented default
    assert len(diff_json(expected, actual, limit=100)) == 40


def test_diff_json_respects_the_limit_for_key_differences() -> None:
    """``limit`` must cap the result whatever *kind* of difference is found.

    ``diff_json`` is documented as "capped so output stays usable", and the
    guard at the top of its ``walk`` shows the intent. That guard is only
    consulted on entry, so the missing/unexpected-key branch appends one line
    per key with no cap at all: 40 keys come back whatever ``limit`` says.
    """
    expected = {f"k{i}": i for i in range(40)}
    missing = diff_json(expected, {}, limit=5)
    assert len(missing) <= 5, f"limit=5 but got {len(missing)} diffs"

    unexpected = diff_json({}, expected, limit=5)
    assert len(unexpected) <= 5, f"limit=5 but got {len(unexpected)} diffs"

    nested = diff_json({"items": [dict(expected)] * 5}, {"items": [{}] * 5})
    assert len(nested) <= 25, f"default limit is 25 but got {len(nested)} diffs"


# --------------------------------------------------------------------------
# 11. record -> replay, end to end
# --------------------------------------------------------------------------


async def _record_one(tmp_path: Path, body: dict[str, Any]) -> Cassette:
    app = create_app(
        ServerConfig(mock=True, record_dir=str(tmp_path)),
        engine=MockUpstream(UpstreamConfig(model="k3")),
    )
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://k3.test") as client:
        response = await client.post("/v1/chat/completions", json=body)
    assert response.status_code == 200, response.text

    cassettes = load_cassettes(tmp_path)
    assert len(cassettes) == 1, [c.name for c in cassettes]
    return cassettes[0]


@pytest.mark.parametrize("stream", [False, True], ids=["json", "stream"])
async def test_recorded_traffic_replays_identically(tmp_path: Path, stream: bool) -> None:
    body = {
        "model": "k3",
        "messages": [{"role": "user", "content": "what files are in this repo?"}],
        "stream": stream,
    }
    cassette = await _record_one(tmp_path, body)

    assert cassette.preset == "openai"
    assert cassette.streamed is stream
    assert cassette.client_status == 200
    assert cassette.upstream_request is not None
    assert cassette.request_body == body
    if stream:
        assert cassette.upstream_chunks, "a streamed request must record engine chunks"
        assert cassette.client_sse, "a streamed request must record the SSE it sent"
    else:
        assert cassette.client_body, "a JSON request must record the body it sent"

    result = await replay_cassette(cassette)
    assert result.ok is True, "\n  ".join(["replay diverged:", *result.diffs[:15]])
    assert result.actual_preset == "openai"
    assert result.status == 200


async def test_recording_is_off_by_default(tmp_path: Path) -> None:
    app = create_app(ServerConfig(mock=True), engine=MockUpstream(UpstreamConfig(model="k3")))
    assert app.state.recorder.enabled is False

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://k3.test") as client:
        response = await client.post(
            "/v1/chat/completions",
            json={"model": "k3", "messages": [{"role": "user", "content": "hi"}]},
        )
    assert response.status_code == 200
    assert app.state.recorder.written == []
    assert list(tmp_path.iterdir()) == []
