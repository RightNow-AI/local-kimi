"""Regressions from the adversarial review.

One test per finding, named after what it protects. These are the failures that
were found by reading the code rather than by using it, which is exactly the
class that comes back if nobody pins them.
"""

from __future__ import annotations

import json
from typing import Any, AsyncIterator

import httpx
import pytest

from k3 import pipeline
from k3 import presets as presets_mod
from k3.dialects import anthropic_messages, openai_chat
from k3.dialects.base import DialectContext
from k3.ir import StreamEnd, StreamEvent, ToolCallArgsDelta, ToolCallEnd, ToolCallStart
from k3.reasoning import ReasoningLedger
from k3.record import parse_sse
from k3.server import ServerConfig, create_app
from k3.upstream import MockUpstream, UpstreamConfig, UpstreamError

CLAUDE_CODE = {"anthropic-version": "2023-06-01", "user-agent": "claude-cli/1.0.60"}
CODEX = {"user-agent": "codex_cli_rs/0.20.0"}


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


class ChunkEngine:
    def __init__(self, chunks: list[dict[str, Any]]) -> None:
        self.chunks = chunks

    async def chat(self, payload: dict[str, Any]) -> dict[str, Any]:  # pragma: no cover
        return {}

    async def chat_stream(self, payload: dict[str, Any]) -> AsyncIterator[dict[str, Any]]:
        for chunk in self.chunks:
            yield chunk


class DyingEngine:
    """Streams one chunk, then the engine drops the connection."""

    async def chat(self, payload: dict[str, Any]) -> dict[str, Any]:
        raise UpstreamError(502, "engine down", "connection")

    async def chat_stream(self, payload: dict[str, Any]) -> AsyncIterator[dict[str, Any]]:
        yield {"id": "c1", "model": "k3", "choices": [{"index": 0, "delta": {"content": "partial "}}]}
        raise UpstreamError(502, "engine down", "connection")

    async def health(self) -> tuple[bool, str]:
        return True, "dying"

    async def aclose(self) -> None:
        return None


def tool_delta(call: Any, finish: str | None = None) -> dict[str, Any]:
    delta = {"tool_calls": [call]} if call else {}
    return {
        "id": "c1",
        "model": "k3",
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }


async def canonical(chunks: list[dict[str, Any]]) -> list[StreamEvent]:
    return [
        ev
        async for ev in pipeline.run(
            ChunkEngine(chunks),
            {"model": "k3"},
            tool_parser="passthrough",
            ledger=ReasoningLedger(),
            stream=True,
        )
    ]


async def render(dialect: Any, events: list[StreamEvent], preset: str) -> list[dict[str, Any]]:
    ctx = DialectContext(preset=presets_mod.get(preset), ledger=ReasoningLedger(), client_model="k3")

    async def gen() -> AsyncIterator[StreamEvent]:
        for ev in events:
            yield ev

    raw = b"".join([chunk async for chunk in dialect.egress_stream(gen(), ctx)])
    return parse_sse(raw)


def build_app(engine: Any = None, **kwargs: Any) -> Any:
    cfg = ServerConfig(upstream=UpstreamConfig(model="k3"), **kwargs)
    return create_app(cfg, engine=engine or MockUpstream(cfg.upstream))


async def post(app: Any, path: str, body: Any, headers: dict[str, str] | None = None) -> httpx.Response:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://k3.test") as client:
        return await client.post(path, json=body, headers=headers or {})


async def get(app: Any, path: str, headers: dict[str, str] | None = None) -> httpx.Response:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://k3.test") as client:
        return await client.get(path, headers=headers or {})


# --------------------------------------------------------------------------
# native tool-call identity
# --------------------------------------------------------------------------


async def test_tool_call_id_announced_matches_the_id_sent_to_the_engine() -> None:
    """The client's tool_result must reference a call that exists upstream.

    The engine may send the name before the id. Announcing a placeholder and
    revising it later leaves the next turn referencing a call K3 never made.
    """
    events = await canonical(
        [
            tool_delta({"index": 0, "function": {"name": "get_weather"}}),
            tool_delta({"index": 0, "id": "call_REAL", "function": {"arguments": '{"city":"SF"}'}}),
            tool_delta(None, "tool_calls"),
        ]
    )
    start = next(e for e in events if isinstance(e, ToolCallStart))
    end = next(e for e in events if isinstance(e, ToolCallEnd))
    upstream = next(e for e in events if isinstance(e, StreamEnd)).response.upstream_message

    assert start.id == end.id
    assert upstream["tool_calls"][0]["id"] == start.id

    records = await render(anthropic_messages, events, "claude-code")
    block = next(
        r["data"]["content_block"]
        for r in records
        if r["data"].get("type") == "content_block_start"
        and r["data"]["content_block"]["type"] == "tool_use"
    )
    assert anthropic_messages.from_anthropic_tool_id(block["id"]) == start.id


@pytest.mark.parametrize(
    "dialect,preset",
    [(anthropic_messages, "claude-code"), (openai_chat, "openai")],
)
async def test_arguments_streamed_before_the_name_still_reach_the_client(dialect, preset) -> None:
    """Dialects rebuild arguments from deltas alone, so none may be skipped."""
    events = await canonical(
        [
            tool_delta({"index": 0, "id": "call_2", "function": {"arguments": '{"ci'}}),
            tool_delta({"index": 0, "function": {"name": "get_weather"}}),
            tool_delta({"index": 0, "function": {"arguments": 'ty":"SF"}'}}),
            tool_delta(None, "tool_calls"),
        ]
    )
    records = await render(dialect, events, preset)

    if dialect is anthropic_messages:
        args = "".join(
            r["data"]["delta"]["partial_json"]
            for r in records
            if r["data"].get("type") == "content_block_delta"
            and r["data"]["delta"]["type"] == "input_json_delta"
        )
    else:
        args = "".join(
            call["function"].get("arguments", "")
            for r in records
            if isinstance(r["data"], dict)
            for choice in r["data"].get("choices", [])
            for call in (choice.get("delta") or {}).get("tool_calls", [])
        )

    assert json.loads(args) == {"city": "SF"}


async def test_tool_name_split_across_fragments_is_announced_whole() -> None:
    """`function.name` reaches the client once; a partial one dispatches wrong."""
    events = await canonical(
        [
            tool_delta({"index": 0, "id": "call_9", "function": {"name": "get_"}}),
            tool_delta({"index": 0, "function": {"name": "weather", "arguments": "{}"}}),
            tool_delta(None, "tool_calls"),
        ]
    )
    start = next(e for e in events if isinstance(e, ToolCallStart))
    end = next(e for e in events if isinstance(e, ToolCallEnd))
    assert start.name == end.name == "get_weather"


async def test_engine_repeating_the_full_name_does_not_duplicate_it() -> None:
    events = await canonical(
        [
            tool_delta({"index": 0, "id": "call_9", "function": {"name": "ls", "arguments": "{"}}),
            tool_delta({"index": 0, "function": {"name": "ls", "arguments": "}"}}),
            tool_delta(None, "tool_calls"),
        ]
    )
    assert next(e for e in events if isinstance(e, ToolCallEnd)).name == "ls"


async def test_argument_deltas_are_not_replayed() -> None:
    """Each argument fragment reaches the client exactly once."""
    events = await canonical(
        [
            tool_delta({"index": 0, "id": "c", "function": {"name": "f", "arguments": '{"a":'}}),
            tool_delta({"index": 0, "function": {"arguments": "1}"}}),
            tool_delta(None, "tool_calls"),
        ]
    )
    streamed = "".join(e.text for e in events if isinstance(e, ToolCallArgsDelta))
    assert streamed == next(e for e in events if isinstance(e, ToolCallEnd)).arguments


# --------------------------------------------------------------------------
# tool-id namespacing
# --------------------------------------------------------------------------


@pytest.mark.parametrize("upstream_id", ["abc", "toolu_k3_abc", "toolu_k3_", "", "call_x"])
def test_tool_id_prefixing_is_reversible(upstream_id: str) -> None:
    """Skipping the prefix for ids that already carry it breaks injectivity."""
    client_id = anthropic_messages.to_anthropic_tool_id(upstream_id)
    assert anthropic_messages.from_anthropic_tool_id(client_id) == upstream_id


# --------------------------------------------------------------------------
# thinking: disabled
# --------------------------------------------------------------------------


async def test_thinking_disabled_suppresses_thinking_blocks() -> None:
    """The Messages API returns no thinking blocks when thinking is off, and
    rejects an assistant turn carrying them on the way back in."""
    app = build_app(mock=True)
    body = {
        "model": "k3",
        "max_tokens": 256,
        "messages": [{"role": "user", "content": "hi"}],
        "thinking": {"type": "disabled"},
    }
    response = await post(app, "/v1/messages", body, CLAUDE_CODE)
    assert response.status_code == 200
    assert not [b for b in response.json()["content"] if b["type"] == "thinking"]


@pytest.mark.parametrize("kind", ["enabled", "adaptive"])
async def test_thinking_enabled_or_adaptive_still_returns_thinking(kind: str) -> None:
    app = build_app(mock=True)
    body = {
        "model": "k3",
        "max_tokens": 256,
        "messages": [{"role": "user", "content": "hi"}],
        "thinking": {"type": kind, "budget_tokens": 8000},
    }
    response = await post(app, "/v1/messages", body, CLAUDE_CODE)
    assert [b for b in response.json()["content"] if b["type"] == "thinking"]


async def test_thinking_disabled_suppresses_thinking_in_the_stream() -> None:
    app = build_app(mock=True)
    body = {
        "model": "k3",
        "max_tokens": 256,
        "stream": True,
        "messages": [{"role": "user", "content": "hi"}],
        "thinking": {"type": "disabled"},
    }
    response = await post(app, "/v1/messages", body, CLAUDE_CODE)
    kinds = [
        r["data"]["content_block"]["type"]
        for r in parse_sse(response.text)
        if r["data"].get("type") == "content_block_start"
    ]
    assert "thinking" not in kinds


# --------------------------------------------------------------------------
# error rendering
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "body",
    [
        {"model": "k3", "tools": 5, "messages": []},
        {"model": "k3", "messages": {"not": "a list"}},
        {"model": "k3", "system": 7, "messages": [{"role": "user", "content": None}]},
        {},
    ],
)
async def test_count_tokens_tolerates_malformed_bodies(body: dict[str, Any]) -> None:
    """Claude Code calls this before every request, so it must not crash."""
    app = build_app(mock=True)
    response = await post(app, "/v1/messages/count_tokens", body, CLAUDE_CODE)
    assert response.status_code == 200
    assert isinstance(response.json()["input_tokens"], int)


async def test_count_tokens_renders_an_unexpected_failure_as_json(monkeypatch) -> None:
    """The backstop: a text/plain 500 is a parse error on the client, which
    reads as a hang rather than as an error."""
    from k3.dialects import anthropic_messages as am

    def boom(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        raise RuntimeError("tokenizer exploded")

    monkeypatch.setattr(am, "count_tokens", boom)
    app = build_app(mock=True)
    response = await post(
        app, "/v1/messages/count_tokens", {"model": "k3", "messages": []}, CLAUDE_CODE
    )
    assert response.status_code == 400
    assert response.headers["content-type"].startswith("application/json")
    assert response.json()["type"] == "error"


async def test_mid_stream_failure_is_a_named_event_for_anthropic() -> None:
    app = build_app(engine=DyingEngine())
    body = {"model": "k3", "max_tokens": 64, "stream": True, "messages": [{"role": "user", "content": "hi"}]}
    records = parse_sse((await post(app, "/v1/messages", body, CLAUDE_CODE)).text)
    assert records[-1]["event"] == "error"
    assert records[-1]["data"]["type"] == "error"


async def test_mid_stream_failure_is_a_named_event_for_responses() -> None:
    """The Responses parser dispatches on the event name; unnamed frames vanish."""
    app = build_app(engine=DyingEngine())
    body = {"model": "k3", "input": "hi", "stream": True}
    records = parse_sse((await post(app, "/v1/responses", body, CODEX)).text)
    assert records[-1]["event"] == "error"
    assert records[-1]["data"]["type"] == "error"


async def test_mid_stream_failure_terminates_the_chat_stream() -> None:
    """Chat clients block until [DONE]; without it the error never surfaces."""
    app = build_app(engine=DyingEngine())
    body = {"model": "k3", "stream": True, "messages": [{"role": "user", "content": "hi"}]}
    response = await post(app, "/v1/chat/completions", body)
    lines = [line for line in response.text.splitlines() if line.strip()]
    assert lines[-1] == "data: [DONE]"
    assert "error" in json.loads(lines[-2][len("data: ") :])


# --------------------------------------------------------------------------
# exposure
# --------------------------------------------------------------------------


@pytest.mark.parametrize("path", ["/v1/models", "/k3/stats", "/k3/presets"])
async def test_api_key_gates_the_read_only_routes(path: str) -> None:
    app = build_app(mock=True, auth_token="secret")
    assert (await get(app, path)).status_code == 401
    assert (await get(app, path, {"authorization": "Bearer secret"})).status_code == 200


async def test_health_stays_reachable_but_withholds_config() -> None:
    """Container health checks keep working; the engine URL does not leak."""
    app = build_app(mock=True, auth_token="secret")
    open_body = (await get(app, "/health")).json()
    assert open_body["status"] == "ok"
    assert "upstream" not in open_body
    assert "upstream" in (await get(app, "/health", {"authorization": "Bearer secret"})).json()


async def test_cors_is_off_unless_an_origin_is_configured() -> None:
    """A wildcard lets any page the user visits drive their local engine."""
    app = build_app(mock=True)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://k3.test") as client:
        response = await client.options(
            "/v1/chat/completions",
            headers={"origin": "https://evil.example", "access-control-request-method": "POST"},
        )
    assert response.headers.get("access-control-allow-origin") is None


async def test_configured_cors_origin_does_not_admit_others() -> None:
    app = build_app(mock=True, cors_origins=["https://trusted.example"])
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://k3.test") as client:
        evil = await client.options(
            "/v1/chat/completions",
            headers={"origin": "https://evil.example", "access-control-request-method": "POST"},
        )
        good = await client.options(
            "/v1/chat/completions",
            headers={"origin": "https://trusted.example", "access-control-request-method": "POST"},
        )
    assert evil.headers.get("access-control-allow-origin") is None
    assert good.headers.get("access-control-allow-origin") == "https://trusted.example"


# --------------------------------------------------------------------------
# streaming usage opt-out
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "options,expected",
    [(None, 1), ({"include_usage": False}, 0), ({"include_usage": True}, 1)],
)
async def test_include_usage_opt_out_is_honoured(options: Any, expected: int) -> None:
    """The extra `choices: []` chunk crashes clients that index choices[0]."""
    app = build_app(mock=True)
    body: dict[str, Any] = {
        "model": "k3",
        "messages": [{"role": "user", "content": "hi"}],
        "stream": True,
    }
    if options is not None:
        body["stream_options"] = options
    response = await post(app, "/v1/chat/completions", body)
    usage_chunks = [
        r
        for r in parse_sse(response.text)
        if isinstance(r["data"], dict) and r["data"].get("choices") == []
    ]
    assert len(usage_chunks) == expected
