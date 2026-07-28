"""End-to-end tests for the HTTP surface.

These drive :func:`k3.server.create_app` through a real ASGI transport with the
scripted :class:`k3.upstream.MockUpstream` behind it, so every layer runs:
detection, dialect ingress, payload assembly, the pipeline, the tool-call text
parser, the reasoning ledger, and dialect egress.

The centrepiece is :func:`test_round_trip_restores_engine_bytes`. Everything
else is scaffolding around the one property that matters: what Claude Code hands
back to us on turn two has to reach K3 as the exact bytes K3 emitted on turn one
— reasoning included, raw tool-call argument string included, engine-side ids
included. If that breaks, agent loops degrade silently rather than loudly.
"""

from __future__ import annotations

import json
from typing import Any, Optional

import httpx

from k3 import presets as presets_mod
from k3.dialects.anthropic_messages import (
    TOOL_ID_PREFIX,
    from_anthropic_tool_id,
    to_anthropic_tool_id,
)
from k3.reasoning import decode_signature
from k3.record import parse_sse
from k3.server import ServerConfig, create_app
from k3.upstream import MockUpstream, UpstreamConfig, UpstreamError

# --------------------------------------------------------------------------
# harness
# --------------------------------------------------------------------------

CLAUDE_CODE_HEADERS = {
    "anthropic-version": "2023-06-01",
    "user-agent": "claude-cli/1.0.60 (external, cli)",
}

KIMI_CLI_HEADERS = {"user-agent": "kimi-cli/0.4.1"}

TOOL_RESULT_TEXT = "22C sunny"


def build(cfg: Optional[ServerConfig] = None) -> tuple[Any, MockUpstream]:
    """A fresh app plus the mock engine behind it, isolated per test."""
    cfg = cfg or ServerConfig(mock=True, upstream=UpstreamConfig(model="k3"))
    engine = MockUpstream(cfg.upstream)
    return create_app(cfg, engine=engine), engine


async def post(
    app: Any,
    path: str,
    body: Any,
    headers: Optional[dict[str, str]] = None,
) -> httpx.Response:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://k3.test") as client:
        return await client.post(path, json=body, headers=headers or {})


async def post_raw(
    app: Any,
    path: str,
    content: bytes,
    headers: Optional[dict[str, str]] = None,
) -> httpx.Response:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://k3.test") as client:
        return await client.post(path, content=content, headers=headers or {})


async def get(
    app: Any,
    path: str,
    headers: Optional[dict[str, str]] = None,
) -> httpx.Response:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://k3.test") as client:
        return await client.get(path, headers=headers or {})


# --------------------------------------------------------------------------
# request bodies
# --------------------------------------------------------------------------

WEATHER_TOOL = {
    "name": "get_weather",
    "description": "Look up the current weather for a city.",
    "input_schema": {
        "type": "object",
        "properties": {"city": {"type": "string", "description": "City name"}},
        "required": ["city"],
    },
}

USER_QUESTION = "What is the weather in Paris right now?"


def anthropic_body(stream: bool = False) -> dict[str, Any]:
    """A Claude Code request with system, a tool, and extended thinking on."""
    return {
        "model": "claude-sonnet-4-5-20250929",
        "max_tokens": 4096,
        "system": "You are a careful assistant.",
        "messages": [{"role": "user", "content": USER_QUESTION}],
        "tools": [WEATHER_TOOL],
        "thinking": {"type": "enabled", "budget_tokens": 8000},
        "stream": stream,
    }


def openai_body(stream: bool = False) -> dict[str, Any]:
    return {
        "model": "k3",
        "messages": [
            {"role": "system", "content": "You are a careful assistant."},
            {"role": "user", "content": USER_QUESTION},
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": WEATHER_TOOL["name"],
                    "description": WEATHER_TOOL["description"],
                    "parameters": WEATHER_TOOL["input_schema"],
                },
            }
        ],
        "stream": stream,
    }


# --------------------------------------------------------------------------
# block / stream helpers
# --------------------------------------------------------------------------


def blocks_of(payload: dict[str, Any], kind: str) -> list[dict[str, Any]]:
    return [b for b in payload["content"] if b.get("type") == kind]


def only_block(payload: dict[str, Any], kind: str) -> dict[str, Any]:
    found = blocks_of(payload, kind)
    assert len(found) == 1, f"expected exactly one {kind} block, got {len(found)}"
    return found[0]


def event_names(records: list[dict[str, Any]]) -> list[str]:
    """The ``type`` of every Anthropic SSE record, in order."""
    return [r["data"]["type"] for r in records if isinstance(r["data"], dict)]


def delta_types(records: list[dict[str, Any]]) -> list[str]:
    out = []
    for record in records:
        data = record["data"]
        if isinstance(data, dict) and data.get("type") == "content_block_delta":
            out.append(data["delta"]["type"])
    return out


def deltas_of(records: list[dict[str, Any]], delta_type: str) -> list[dict[str, Any]]:
    return [
        r["data"]["delta"]
        for r in records
        if isinstance(r["data"], dict)
        and r["data"].get("type") == "content_block_delta"
        and r["data"]["delta"]["type"] == delta_type
    ]


def rebuild_blocks(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Fold an Anthropic SSE stream back into the content blocks it describes."""
    blocks: dict[int, dict[str, Any]] = {}
    order: list[int] = []
    for record in records:
        data = record["data"]
        if not isinstance(data, dict):
            continue
        kind = data.get("type")
        if kind == "content_block_start":
            index = data["index"]
            blocks[index] = dict(data["content_block"])
            order.append(index)
        elif kind == "content_block_delta":
            block = blocks[data["index"]]
            delta = data["delta"]
            if delta["type"] == "thinking_delta":
                block["thinking"] = block.get("thinking", "") + delta["thinking"]
            elif delta["type"] == "signature_delta":
                block["signature"] = block.get("signature", "") + delta["signature"]
            elif delta["type"] == "text_delta":
                block["text"] = block.get("text", "") + delta["text"]
            elif delta["type"] == "input_json_delta":
                block["_partial_json"] = block.get("_partial_json", "") + delta["partial_json"]
    out = []
    for index in order:
        block = blocks[index]
        if "_partial_json" in block:
            raw = block.pop("_partial_json")
            block["input"] = json.loads(raw) if raw.strip() else {}
        out.append(block)
    return out


def block_indices(records: list[dict[str, Any]], kind: str) -> list[int]:
    return [
        r["data"]["index"]
        for r in records
        if isinstance(r["data"], dict) and r["data"].get("type") == kind
    ]


def ledger_entry_for(app: Any, thinking_block: dict[str, Any]) -> Any:
    """The turn's ledger entry, reached the way the server reaches it."""
    _, ledger_id = decode_signature(thinking_block["signature"])
    assert ledger_id, "thinking signature carried no ledger id"
    entry = app.state.ledger.get(ledger_id)
    assert entry is not None, f"ledger has no entry for {ledger_id}"
    return entry


async def anthropic_turn_one(app: Any) -> dict[str, Any]:
    response = await post(app, "/v1/messages", anthropic_body(), CLAUDE_CODE_HEADERS)
    assert response.status_code == 200, response.text
    return response.json()


# --------------------------------------------------------------------------
# 1-3: Anthropic non-streaming shape
# --------------------------------------------------------------------------


async def test_anthropic_non_streaming_returns_thinking_then_tool_use() -> None:
    app, _ = build()
    body = await anthropic_turn_one(app)

    assert body["type"] == "message"
    assert body["role"] == "assistant"

    kinds = [b["type"] for b in body["content"]]
    assert kinds[0] == "thinking", f"thinking must lead the content, got {kinds}"
    assert "tool_use" in kinds, f"no tool_use block in {kinds}"

    thinking = only_block(body, "thinking")
    assert thinking["thinking"], "thinking block carried no text"
    assert thinking["signature"], "thinking block carried no signature"

    tool_use = only_block(body, "tool_use")
    assert tool_use["name"] == WEATHER_TOOL["name"]
    assert isinstance(tool_use["input"], dict) and tool_use["input"]

    assert body["stop_reason"] == "tool_use"
    assert set(body["usage"]) == {
        "input_tokens",
        "output_tokens",
        "cache_creation_input_tokens",
        "cache_read_input_tokens",
    }


async def test_thinking_signature_decodes_to_the_thinking_text() -> None:
    app, _ = build()
    body = await anthropic_turn_one(app)
    thinking = only_block(body, "thinking")

    recovered, ledger_id = decode_signature(thinking["signature"])
    assert ledger_id, "signature must carry a ledger id"
    assert recovered == thinking["thinking"], "signature does not round-trip the reasoning"


async def test_tool_use_id_is_a_reversible_namespaced_engine_id() -> None:
    app, _ = build()
    body = await anthropic_turn_one(app)
    tool_use = only_block(body, "tool_use")
    client_id = tool_use["id"]

    assert client_id.startswith(TOOL_ID_PREFIX), client_id

    engine_id = from_anthropic_tool_id(client_id)
    assert to_anthropic_tool_id(engine_id) == client_id, "id mapping is not reversible"

    entry = ledger_entry_for(app, only_block(body, "thinking"))
    upstream_id = entry.upstream_message["tool_calls"][0]["id"]
    assert engine_id == upstream_id, "stripped id is not the id the engine produced"


# --------------------------------------------------------------------------
# 4: the round trip
# --------------------------------------------------------------------------


async def test_round_trip_restores_engine_bytes() -> None:
    """Echo turn one back and check K3 sees its own bytes again."""
    app, engine = build()

    first = await anthropic_turn_one(app)
    thinking = only_block(first, "thinking")
    tool_use = only_block(first, "tool_use")
    engine_id = from_anthropic_tool_id(tool_use["id"])

    hits_before = app.state.ledger.stats()["hits"]

    second_body = anthropic_body()
    second_body["messages"] = [
        {"role": "user", "content": USER_QUESTION},
        {"role": "assistant", "content": first["content"]},
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": tool_use["id"],
                    "content": TOOL_RESULT_TEXT,
                }
            ],
        },
    ]
    response = await post(app, "/v1/messages", second_body, CLAUDE_CODE_HEADERS)
    assert response.status_code == 200, response.text

    hits_after = app.state.ledger.stats()["hits"]
    assert hits_after >= hits_before + 1, "the ledger was not consulted on the echo"
    assert app.state.ledger.stats()["hits"] >= 1

    messages = engine.calls[-1]["messages"]
    index = next(i for i, m in enumerate(messages) if m["role"] == "assistant")
    assistant = messages[index]

    entry = ledger_entry_for(app, thinking)
    original = entry.upstream_message
    raw_args = original["tool_calls"][0]["function"]["arguments"]

    assert assistant["reasoning_content"] == thinking["thinking"]
    assert assistant["reasoning_content"] == original["reasoning_content"]

    sent_args = assistant["tool_calls"][0]["function"]["arguments"]
    assert isinstance(sent_args, str)
    assert sent_args == raw_args, "the raw argument string was not restored verbatim"

    assert assistant["tool_calls"][0]["id"] == engine_id
    assert not assistant["tool_calls"][0]["id"].startswith(TOOL_ID_PREFIX)

    assert messages[index + 1] == {
        "role": "tool",
        "tool_call_id": engine_id,
        "content": TOOL_RESULT_TEXT,
    }


# --------------------------------------------------------------------------
# 5-8: Anthropic streaming
# --------------------------------------------------------------------------


async def anthropic_stream_records(app: Any) -> list[dict[str, Any]]:
    response = await post(app, "/v1/messages", anthropic_body(stream=True), CLAUDE_CODE_HEADERS)
    assert response.status_code == 200, response.text
    assert response.headers["content-type"].startswith("text/event-stream")
    return parse_sse(response.text)


async def test_anthropic_stream_event_order() -> None:
    app, _ = build()
    records = await anthropic_stream_records(app)
    names = event_names(records)

    assert names[0] == "message_start"
    cursor = 1
    if names[cursor] == "ping":
        cursor += 1

    # thinking block
    assert names[cursor] == "content_block_start"
    assert records[cursor]["data"]["content_block"]["type"] == "thinking"
    cursor += 1

    thinking_deltas = 0
    while names[cursor] == "content_block_delta" and (
        records[cursor]["data"]["delta"]["type"] == "thinking_delta"
    ):
        thinking_deltas += 1
        cursor += 1
    assert thinking_deltas >= 1, "no thinking_delta events"

    assert names[cursor] == "content_block_delta"
    assert records[cursor]["data"]["delta"]["type"] == "signature_delta"
    cursor += 1
    assert names[cursor] == "content_block_stop"
    cursor += 1

    # tool_use block
    assert names[cursor] == "content_block_start"
    assert records[cursor]["data"]["content_block"]["type"] == "tool_use"
    cursor += 1

    while names[cursor] == "content_block_delta":
        assert records[cursor]["data"]["delta"]["type"] == "input_json_delta"
        cursor += 1

    assert names[cursor] == "content_block_stop"
    cursor += 1

    assert names[cursor:] == ["message_delta", "message_stop"], names[cursor:]
    assert delta_types(records).count("signature_delta") == 1
    assert names.count("message_delta") == 1
    assert names.count("message_stop") == 1


async def test_anthropic_stream_deltas_reassemble() -> None:
    app, _ = build()
    records = await anthropic_stream_records(app)

    reasoning = "".join(d["thinking"] for d in deltas_of(records, "thinking_delta"))
    assert reasoning

    signatures = deltas_of(records, "signature_delta")
    assert len(signatures) == 1
    recovered, ledger_id = decode_signature(signatures[0]["signature"])
    assert ledger_id
    assert recovered == reasoning, "streamed signature does not match the streamed reasoning"

    partial = "".join(d["partial_json"] for d in deltas_of(records, "input_json_delta"))
    assert json.loads(partial), "concatenated input_json_delta is not a JSON object"


async def test_anthropic_stream_block_indices_are_well_formed() -> None:
    app, _ = build()
    records = await anthropic_stream_records(app)

    starts = block_indices(records, "content_block_start")
    stops = block_indices(records, "content_block_stop")

    assert starts == list(range(len(starts))), f"indices not 0..n increasing: {starts}"
    assert starts == stops, f"unmatched start/stop indices: {starts} vs {stops}"

    for record in records:
        data = record["data"]
        if isinstance(data, dict) and data.get("type") == "content_block_delta":
            assert data["index"] in starts


async def test_streaming_and_non_streaming_agree() -> None:
    app, _ = build()

    non_streaming = (await anthropic_turn_one(app))["content"]
    streamed = rebuild_blocks(await anthropic_stream_records(app))

    assert [b["type"] for b in streamed] == [b["type"] for b in non_streaming]

    for streamed_block, direct in zip(streamed, non_streaming):
        if direct["type"] == "thinking":
            assert streamed_block["thinking"] == direct["thinking"]
        elif direct["type"] == "text":
            assert streamed_block["text"] == direct["text"]
        elif direct["type"] == "tool_use":
            assert streamed_block["name"] == direct["name"]
            assert streamed_block["input"] == direct["input"]


# --------------------------------------------------------------------------
# 9-11: OpenAI chat
# --------------------------------------------------------------------------


async def test_openai_non_streaming_tool_call_strips_reasoning() -> None:
    app, _ = build()
    response = await post(app, "/v1/chat/completions", openai_body())
    assert response.status_code == 200, response.text
    body = response.json()

    assert body["object"] == "chat.completion"
    choice = body["choices"][0]
    assert choice["finish_reason"] == "tool_calls"

    message = choice["message"]
    arguments = message["tool_calls"][0]["function"]["arguments"]
    assert isinstance(arguments, str)
    assert isinstance(json.loads(arguments), dict)

    assert "reasoning_content" not in message, "the openai preset must strip reasoning"


async def openai_stream_records(
    app: Any, headers: Optional[dict[str, str]] = None
) -> tuple[httpx.Response, list[dict[str, Any]]]:
    response = await post(app, "/v1/chat/completions", openai_body(stream=True), headers)
    assert response.status_code == 200, response.text
    assert response.headers["content-type"].startswith("text/event-stream")
    return response, parse_sse(response.text)


async def test_openai_stream_frames() -> None:
    app, _ = build()
    response, records = await openai_stream_records(app)

    lines = [line for line in response.text.splitlines() if line.strip()]
    assert lines[-1] == "data: [DONE]"
    assert lines.count("data: [DONE]") == 1

    dones = [r for r in records if r["data"] == "[DONE]"]
    assert len(dones) == 1
    assert records[-1] is dones[0]

    first = records[0]["data"]
    assert first["choices"][0]["delta"]["role"] == "assistant"

    finishes = [
        r["data"]["choices"][0]["finish_reason"]
        for r in records
        if isinstance(r["data"], dict) and r["data"].get("choices")
    ]
    assert any(f for f in finishes), "no chunk carried a finish_reason"

    usage_chunks = [
        r["data"]
        for r in records
        if isinstance(r["data"], dict) and r["data"].get("choices") == []
    ]
    assert len(usage_chunks) == 1
    assert "usage" in usage_chunks[0]


async def test_kimi_cli_user_agent_keeps_reasoning_in_the_stream() -> None:
    generic_app, _ = build()
    generic_response, generic_records = await openai_stream_records(generic_app)

    kimi_app, _ = build()
    kimi_response, kimi_records = await openai_stream_records(kimi_app, KIMI_CLI_HEADERS)

    assert (
        kimi_response.headers["x-k3-preset"] != generic_response.headers["x-k3-preset"]
    ), "the kimi-cli user-agent did not change the detected preset"

    def reasoning_of(records: list[dict[str, Any]]) -> str:
        out = []
        for record in records:
            data = record["data"]
            if not isinstance(data, dict):
                continue
            for choice in data.get("choices") or []:
                text = (choice.get("delta") or {}).get("reasoning_content")
                if text:
                    out.append(text)
        return "".join(out)

    assert reasoning_of(kimi_records), "kimi-cli must receive delta.reasoning_content"
    assert not reasoning_of(generic_records), "the generic preset must strip reasoning"


# --------------------------------------------------------------------------
# 12-14: the other endpoints
# --------------------------------------------------------------------------


async def test_count_tokens_returns_a_positive_estimate() -> None:
    app, _ = build()
    response = await post(
        app, "/v1/messages/count_tokens", anthropic_body(), CLAUDE_CODE_HEADERS
    )
    assert response.status_code == 200, response.text
    body = response.json()

    assert isinstance(body["input_tokens"], int)
    assert not isinstance(body["input_tokens"], bool)
    assert body["input_tokens"] > 0


async def test_models_shape_follows_the_detected_client() -> None:
    app, _ = build()

    anthropic = await get(app, "/v1/models", CLAUDE_CODE_HEADERS)
    assert anthropic.status_code == 200
    anthropic_body_ = anthropic.json()
    assert "has_more" in anthropic_body_
    assert anthropic_body_["data"]
    assert all(item["type"] == "model" for item in anthropic_body_["data"])

    openai = await get(app, "/v1/models")
    assert openai.status_code == 200
    openai_body_ = openai.json()
    assert openai_body_["object"] == "list"
    assert all(item["object"] == "model" for item in openai_body_["data"])


async def test_health_presets_and_stats() -> None:
    app, _ = build()

    health = await get(app, "/health")
    assert health.status_code == 200
    assert health.json()["status"] == "ok"

    presets = await get(app, "/k3/presets")
    assert presets.status_code == 200
    listed = [p["name"] for p in presets.json()["presets"]]
    assert listed == presets_mod.names()

    before = (await get(app, "/k3/stats")).json()["counters"]["requests"]
    await post(app, "/v1/messages", anthropic_body(), CLAUDE_CODE_HEADERS)
    await post(app, "/v1/chat/completions", openai_body())
    after = (await get(app, "/k3/stats")).json()

    assert after["counters"]["requests"] == before + 2
    assert after["by_preset"]["claude-code"] == 1
    assert after["by_preset"]["openai"] == 1


# --------------------------------------------------------------------------
# 15-17: error paths
# --------------------------------------------------------------------------


async def test_malformed_json_body_is_a_400() -> None:
    app, _ = build()
    response = await post_raw(
        app,
        "/v1/messages",
        b"{nope",
        {**CLAUDE_CODE_HEADERS, "content-type": "application/json"},
    )
    assert response.status_code == 400, response.text
    assert "error" in response.json()


async def test_malformed_json_error_is_rendered_in_the_client_dialect() -> None:
    """A client that cannot parse your error message reports it as a hang.

    ``/v1/messages/count_tokens`` already renders this exact failure in the
    Anthropic shape, so the sibling route must too.
    """
    app, _ = build()
    headers = {**CLAUDE_CODE_HEADERS, "content-type": "application/json"}

    count_tokens = await post_raw(app, "/v1/messages/count_tokens", b"{nope", headers)
    assert count_tokens.status_code == 400
    assert count_tokens.json()["type"] == "error"

    messages = await post_raw(app, "/v1/messages", b"{nope", headers)
    assert messages.status_code == 400
    assert messages.json().get("type") == "error", (
        "Claude Code got an OpenAI-shaped error body from /v1/messages: "
        f"{messages.text}"
    )


async def test_non_dict_json_body_is_a_400_not_a_500() -> None:
    """A body that parses but is not an object is still a malformed request."""
    app, _ = build()
    try:
        response = await post(app, "/v1/messages", ["not", "a", "dict"], CLAUDE_CODE_HEADERS)
    except AttributeError as exc:  # pragma: no cover - the bug being pinned
        raise AssertionError(
            "a top-level JSON array crashed the request handler instead of "
            f"producing a 400: {type(exc).__name__}: {exc}"
        ) from exc
    assert response.status_code == 400, response.text
    assert "error" in response.json()


class ExplodingEngine:
    """An engine that is down, in the way the transport reports it."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def chat(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.calls.append(payload)
        raise UpstreamError(502, "engine down", "connection")

    def chat_stream(self, payload: dict[str, Any]) -> Any:
        self.calls.append(payload)
        raise UpstreamError(502, "engine down", "connection")

    async def models(self) -> Optional[dict[str, Any]]:
        return None

    async def health(self) -> tuple[bool, str]:
        return False, "engine down"

    async def aclose(self) -> None:
        return None


async def test_engine_failure_renders_in_the_client_dialect() -> None:
    cfg = ServerConfig(mock=True, upstream=UpstreamConfig(model="k3"))

    anthropic_app = create_app(cfg, engine=ExplodingEngine())
    response = await post(anthropic_app, "/v1/messages", anthropic_body(), CLAUDE_CODE_HEADERS)
    assert response.status_code == 502, response.text
    body = response.json()
    assert body["type"] == "error"
    assert isinstance(body["error"], dict)
    assert body["error"]["type"] == "connection_error"
    assert "engine down" in body["error"]["message"]

    openai_app = create_app(cfg, engine=ExplodingEngine())
    response = await post(openai_app, "/v1/chat/completions", openai_body())
    assert response.status_code == 502, response.text
    body = response.json()
    assert "type" not in body
    assert isinstance(body["error"], dict)
    assert "engine down" in body["error"]["message"]

    # A stream that cannot start must still produce a status code, not a 200
    # with a dead body.
    streaming_app = create_app(cfg, engine=ExplodingEngine())
    response = await post(
        streaming_app, "/v1/messages", anthropic_body(stream=True), CLAUDE_CODE_HEADERS
    )
    assert response.status_code == 502, response.text
    assert response.json()["type"] == "error"


async def test_auth_token_is_enforced() -> None:
    cfg = ServerConfig(mock=True, auth_token="secret", upstream=UpstreamConfig(model="k3"))
    app, _ = build(cfg)

    unauthorized = await post(app, "/v1/messages", anthropic_body(), CLAUDE_CODE_HEADERS)
    assert unauthorized.status_code == 401, unauthorized.text
    assert unauthorized.json()["type"] == "error"

    authorized = await post(
        app,
        "/v1/messages",
        anthropic_body(),
        {**CLAUDE_CODE_HEADERS, "authorization": "Bearer secret"},
    )
    assert authorized.status_code == 200, authorized.text
