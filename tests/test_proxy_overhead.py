"""What the proxy costs, measured, because the headline depends on it.

The recommended setup is a fast local backend, llama.cpp, with k3 in front of it
translating dialects. That pitch quietly assumes the proxy adds no meaningful
latency. Nobody had checked.

This measures the same request twice against the same server: once directly, and
once through k3. The difference is the proxy's cost. Both paths use
httpx.ASGITransport, so there is no socket and no network variance, and what is
left is the translation work itself.

The absolute microseconds here are Python overhead on a stub engine and do not
transfer to a real deployment. The RATIO is what transfers: whether translation
is a rounding error next to inference or a tax on every token. Against a backend
generating at 32 tokens per second, one token takes about 31 milliseconds, so
anything under a millisecond of proxy time per request is invisible.
"""

from __future__ import annotations

import statistics
import time

import httpx
import pytest

from engine.serve.api import create_stub_app
from k3.server import ServerConfig, create_app
from k3.upstream import Upstream, UpstreamConfig

CLAUDE_CODE_USER_AGENT = "claude-cli/1.0.60 (external, cli)"
ITERATIONS = 40
WARMUP = 5

#: A request costs about 31 ms against a 32 token per second backend. The proxy
#: is allowed a small fraction of that. Declared before measuring.
MAX_ACCEPTABLE_OVERHEAD_MS = 5.0


def _upstream_to(app: object) -> Upstream:
    upstream = Upstream(UpstreamConfig(base_url="http://engine.test/v1", model="k3"))
    upstream._client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        headers={"content-type": "application/json"},
        timeout=httpx.Timeout(30.0),
    )
    return upstream


async def _time_calls(client: httpx.AsyncClient, path: str, headers: dict, body: dict):
    for _ in range(WARMUP):
        response = await client.post(path, headers=headers, json=body)
        assert response.status_code == 200, response.text
    samples = []
    for _ in range(ITERATIONS):
        started = time.perf_counter()
        response = await client.post(path, headers=headers, json=body)
        samples.append((time.perf_counter() - started) * 1000.0)
        assert response.status_code == 200, response.text
    return samples


@pytest.mark.asyncio
async def test_proxy_overhead_is_a_rounding_error_next_to_inference():
    """Translating a dialect must cost far less than generating one token."""
    serve_app = create_stub_app()

    direct_transport = httpx.ASGITransport(app=serve_app)
    async with httpx.AsyncClient(
        transport=direct_transport, base_url="http://engine.test"
    ) as direct:
        direct_samples = await _time_calls(
            direct,
            "/v1/chat/completions",
            {"content-type": "application/json"},
            {
                "model": "k3",
                "max_tokens": 16,
                "messages": [{"role": "user", "content": "ping"}],
            },
        )

    upstream = _upstream_to(serve_app)
    proxy_app = create_app(
        ServerConfig(upstream=UpstreamConfig(base_url="http://engine.test/v1")),
        engine=upstream,
    )
    try:
        proxy_transport = httpx.ASGITransport(app=proxy_app)
        async with httpx.AsyncClient(
            transport=proxy_transport, base_url="http://k3.test"
        ) as proxied:
            proxy_samples = await _time_calls(
                proxied,
                "/v1/messages",
                {
                    "anthropic-version": "2023-06-01",
                    "user-agent": CLAUDE_CODE_USER_AGENT,
                },
                {
                    "model": "k3",
                    "max_tokens": 16,
                    "messages": [{"role": "user", "content": "ping"}],
                },
            )
    finally:
        await upstream.aclose()

    direct_median = statistics.median(direct_samples)
    proxy_median = statistics.median(proxy_samples)
    overhead = proxy_median - direct_median

    print(
        f"\ndirect median {direct_median:.3f} ms, "
        f"through k3 {proxy_median:.3f} ms, "
        f"overhead {overhead:.3f} ms"
    )

    assert overhead < MAX_ACCEPTABLE_OVERHEAD_MS, (
        f"k3 added {overhead:.3f} ms per request against a declared budget of "
        f"{MAX_ACCEPTABLE_OVERHEAD_MS} ms. At 32 tokens per second a single "
        "token costs about 31 ms, so the proxy must stay well under that or the "
        "recommended local setup is paying a real tax for dialect translation."
    )


@pytest.mark.asyncio
async def test_translation_happened_rather_than_passthrough():
    """Guard the measurement above: it is only meaningful if work occurred.

    A proxy that forwarded the body unchanged would be fast and useless. This
    asserts the response actually came back in Anthropic shape, so the timing
    above is the cost of real translation.
    """
    serve_app = create_stub_app()
    upstream = _upstream_to(serve_app)
    proxy_app = create_app(
        ServerConfig(upstream=UpstreamConfig(base_url="http://engine.test/v1")),
        engine=upstream,
    )
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
                    "max_tokens": 16,
                    "messages": [{"role": "user", "content": "ping"}],
                },
            )
    finally:
        await upstream.aclose()

    body = response.json()
    assert body["type"] == "message"
    assert body["role"] == "assistant"
    assert isinstance(body["content"], list) and body["content"]
    assert "choices" not in body, "an OpenAI body leaked through untranslated"
