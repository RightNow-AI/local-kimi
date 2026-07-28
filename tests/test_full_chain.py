"""The whole product, end to end, with no GPU and no weights.

Every other test in this repository exercises one half. The proxy tests drive
`k3` against a mock upstream, and the server tests drive `engine/serve` against
a stub engine. Nothing has ever run a request through BOTH, which means the one
thing a user actually does has never been tested:

    coding agent -> k3 (dialect translation) -> HTTP -> engine/serve -> engine

The seam between those halves is exactly where this project has already been
bitten. `engine/serve` was written against `k3/server.py` as a reader rather
than as a caller, and two lanes that never saw each other's code produced a
plan resolver and a plan factory that could not be called together.

`httpx.ASGITransport` gives a real HTTP round trip, request line and headers and
status codes included, without binding a socket, so this stays a fast offline
test that still proves the wire contract rather than an in-process shortcut.
"""

from __future__ import annotations

import httpx
import pytest

from engine.serve.api import create_stub_app
from k3.server import ServerConfig, create_app
from k3.upstream import Upstream, UpstreamConfig

CLAUDE_CODE_USER_AGENT = "claude-cli/1.0.60 (external, cli)"


def _chain() -> tuple[object, Upstream]:
    """Wire k3 to engine/serve over a real, socket-free HTTP transport."""
    serve_app = create_stub_app()

    upstream_config = UpstreamConfig(base_url="http://engine.test/v1", model="k3")
    upstream = Upstream(upstream_config)
    # Inject the transport rather than letting Upstream build its own client, so
    # the proxy takes its ordinary HTTP path and lands on our server.
    upstream._client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=serve_app),
        headers={"content-type": "application/json"},
        timeout=httpx.Timeout(30.0),
    )

    proxy_app = create_app(ServerConfig(upstream=upstream_config), engine=upstream)
    return proxy_app, upstream


@pytest.mark.asyncio
async def test_anthropic_request_survives_the_whole_chain():
    """A Claude Code shaped request comes back Anthropic shaped."""
    proxy_app, upstream = _chain()
    try:
        transport = httpx.ASGITransport(app=proxy_app)
        async with httpx.AsyncClient(transport=transport, base_url="http://k3.test") as client:
            response = await client.post(
                "/v1/messages",
                headers={
                    "anthropic-version": "2023-06-01",
                    "user-agent": CLAUDE_CODE_USER_AGENT,
                },
                json={
                    "model": "k3",
                    "max_tokens": 64,
                    "messages": [{"role": "user", "content": "ping"}],
                },
            )
    finally:
        await upstream.aclose()

    assert response.status_code == 200, response.text
    body = response.json()

    # Anthropic shape, not OpenAI shape. Getting this wrong is the single most
    # common way a proxy breaks a client, and it is invisible from either half.
    assert body["type"] == "message"
    assert body["role"] == "assistant"
    assert isinstance(body["content"], list)
    assert body["content"], "an assistant turn must carry at least one block"
    assert {block["type"] for block in body["content"]} <= {"text", "thinking", "tool_use"}
    assert body["stop_reason"] in {"end_turn", "max_tokens", "tool_use", "stop_sequence"}


@pytest.mark.asyncio
async def test_usage_crosses_the_chain_as_real_counts():
    """Token counts must come from the server, not be invented by the proxy.

    engine/serve counts real token ids. If the proxy substituted an estimate,
    or dropped usage and defaulted to zero, billing and context accounting
    downstream would both be wrong while everything still looked healthy.
    """
    proxy_app, upstream = _chain()
    try:
        transport = httpx.ASGITransport(app=proxy_app)
        async with httpx.AsyncClient(transport=transport, base_url="http://k3.test") as client:
            response = await client.post(
                "/v1/messages",
                headers={
                    "anthropic-version": "2023-06-01",
                    "user-agent": CLAUDE_CODE_USER_AGENT,
                },
                json={
                    "model": "k3",
                    "max_tokens": 32,
                    "messages": [{"role": "user", "content": "count these tokens"}],
                },
            )
    finally:
        await upstream.aclose()

    assert response.status_code == 200, response.text
    usage = response.json()["usage"]
    assert usage["input_tokens"] > 0
    assert usage["output_tokens"] > 0


@pytest.mark.asyncio
async def test_streaming_crosses_the_chain_and_terminates():
    """SSE must survive translation from our server's frames to Anthropic's.

    The two dialects do not share an event vocabulary, so this asserts the
    Anthropic lifecycle rather than a passthrough: a stream that starts and
    never stops hangs the client forever, which is worse than an error.
    """
    proxy_app, upstream = _chain()
    events: list[str] = []
    try:
        transport = httpx.ASGITransport(app=proxy_app)
        async with httpx.AsyncClient(transport=transport, base_url="http://k3.test") as client:
            async with client.stream(
                "POST",
                "/v1/messages",
                headers={
                    "anthropic-version": "2023-06-01",
                    "user-agent": CLAUDE_CODE_USER_AGENT,
                },
                json={
                    "model": "k3",
                    "max_tokens": 32,
                    "stream": True,
                    "messages": [{"role": "user", "content": "stream please"}],
                },
            ) as response:
                assert response.status_code == 200
                async for line in response.aiter_lines():
                    if line.startswith("event: "):
                        events.append(line.removeprefix("event: ").strip())
    finally:
        await upstream.aclose()

    assert "message_start" in events, events
    assert "message_stop" in events, events
    assert events.index("message_start") < events.index("message_stop")


@pytest.mark.asyncio
async def test_openai_dialect_reaches_the_same_server():
    """The same server serves a second dialect through the same proxy.

    This is the claim the preset system exists to make. If the OpenAI path were
    quietly translating into a different upstream shape, this would fail while
    the Anthropic tests above still passed.
    """
    proxy_app, upstream = _chain()
    try:
        transport = httpx.ASGITransport(app=proxy_app)
        async with httpx.AsyncClient(transport=transport, base_url="http://k3.test") as client:
            response = await client.post(
                "/v1/chat/completions",
                headers={"user-agent": "OpenAI/Python 1.40.0"},
                json={
                    "model": "k3",
                    "max_tokens": 32,
                    "messages": [{"role": "user", "content": "ping"}],
                },
            )
    finally:
        await upstream.aclose()

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["object"] == "chat.completion"
    assert body["choices"][0]["message"]["role"] == "assistant"
    assert body["usage"]["completion_tokens"] > 0
