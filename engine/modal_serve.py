"""Serve the k3 proxy as a public Modal endpoint.

This is the product surface: one URL that Claude Code, Codex and any OpenAI
client can point at. The proxy auto-detects which client is calling and speaks
that client's dialect to whatever engine sits behind it.

Two backends, selected by env:
  K3_UPSTREAM unset  -> the built-in mock engine, so the serving path can be
                        exercised end to end without a GPU
  K3_UPSTREAM set    -> a real OpenAI-compatible engine at that base URL

The endpoint is NOT open. A token is required, because an unauthenticated LLM
endpoint on a public URL is somebody else's free compute.

    modal deploy engine/modal_serve.py
    modal run engine/modal_serve.py::smoke
"""

from __future__ import annotations

import os

import modal

app = modal.App("k3-serve")

IMAGE = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "fastapi>=0.115",
        "uvicorn[standard]>=0.30",
        "httpx>=0.27",
        "pydantic>=2.7",
        "typer>=0.12",
        "rich>=13.7",
    )
    .add_local_python_source("k3")
)

#: Shared secret the clients must present. Overridden by a Modal Secret named
#: `k3-token` when one exists, so the default is only for a smoke test.
DEFAULT_TOKEN = "k3-local-dev-token"


def _build_app():
    from k3.server import ServerConfig, create_app
    from k3.upstream import MockUpstream, Upstream, UpstreamConfig

    upstream_url = os.environ.get("K3_UPSTREAM", "").strip()
    token = os.environ.get("K3_TOKEN", DEFAULT_TOKEN)
    model = os.environ.get("K3_MODEL", "k3")

    cfg = ServerConfig(
        mock=not upstream_url,
        auth_token=token,
        upstream=UpstreamConfig(
            base_url=upstream_url or "http://127.0.0.1:8000/v1",
            api_key=os.environ.get("K3_UPSTREAM_KEY") or None,
            model=model,
        ),
    )
    engine = MockUpstream(cfg.upstream) if cfg.mock else Upstream(cfg.upstream)
    return create_app(cfg, engine=engine)


@app.function(
    image=IMAGE,
    min_containers=1,
    timeout=60 * 60,
    # Streaming responses must not be cut off mid-generation by a scale-down.
    scaledown_window=60 * 5,
)
@modal.concurrent(max_inputs=32)
@modal.asgi_app()
def api():
    return _build_app()


@app.function(image=IMAGE, timeout=900)
def smoke(base_url: str = "", token: str = DEFAULT_TOKEN) -> dict:
    """Drive the deployed endpoint the way each real client would.

    Runs inside Modal so the laptop is never in the path. Exercises all three
    dialects plus the auth boundary, and checks the reasoning signature actually
    round-trips, which is the thing this proxy exists to get right.
    """
    import json

    import httpx

    from k3.reasoning import decode_signature

    base = (base_url or os.environ.get("K3_BASE_URL", "")).rstrip("/")
    if not base:
        return {"error": "pass --base-url or set K3_BASE_URL"}

    auth = {"authorization": f"Bearer {token}"}
    results: dict[str, object] = {"base_url": base}
    tool = {
        "name": "get_weather",
        "description": "Look up the weather.",
        "input_schema": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    }

    with httpx.Client(timeout=120.0) as c:
        # 1. unauthenticated request must be refused
        r = c.post(f"{base}/v1/messages", json={"model": "k3", "max_tokens": 16,
                                                "messages": [{"role": "user", "content": "hi"}]})
        results["auth_enforced"] = r.status_code == 401

        # 2. Claude Code dialect, non-streaming, with a tool
        cc = {**auth, "anthropic-version": "2023-06-01",
              "user-agent": "claude-cli/1.0.60 (external, cli)"}
        r = c.post(f"{base}/v1/messages", headers=cc, json={
            "model": "claude-sonnet-4-5-20250929", "max_tokens": 1024,
            "system": "You are terse.",
            "messages": [{"role": "user", "content": "weather in Beijing?"}],
            "tools": [tool],
            "thinking": {"type": "enabled", "budget_tokens": 8000},
        })
        body = r.json() if r.status_code == 200 else {}
        blocks = [b.get("type") for b in body.get("content", [])]
        sig = next((b.get("signature") for b in body.get("content", [])
                    if b.get("type") == "thinking"), None)
        recovered, _ = decode_signature(sig) if sig else (None, None)
        thinking = next((b.get("thinking") for b in body.get("content", [])
                         if b.get("type") == "thinking"), None)
        results["claude_code"] = {
            "status": r.status_code,
            "blocks": blocks,
            "stop_reason": body.get("stop_reason"),
            "signature_round_trips": bool(recovered) and recovered == thinking,
        }

        # 3. Claude Code streaming
        with c.stream("POST", f"{base}/v1/messages", headers=cc, json={
            "model": "k3", "max_tokens": 512, "stream": True,
            "messages": [{"role": "user", "content": "hello"}],
        }) as s:
            events = [ln[7:] for ln in s.iter_lines() if ln.startswith("event: ")]
        results["claude_code_stream"] = {
            "first": events[0] if events else None,
            "last": events[-1] if events else None,
            "count": len(events),
        }

        # 4. Codex / Responses dialect
        r = c.post(f"{base}/v1/responses", headers={**auth, "user-agent": "codex_cli_rs/0.20.0"},
                   json={"model": "k3", "instructions": "be terse", "input": "hello",
                         "max_output_tokens": 256})
        rb = r.json() if r.status_code == 200 else {}
        results["codex"] = {
            "status": r.status_code,
            "object": rb.get("object"),
            "output_types": [i.get("type") for i in rb.get("output", [])],
        }

        # 5. Generic OpenAI dialect, streaming
        with c.stream("POST", f"{base}/v1/chat/completions", headers=auth, json={
            "model": "k3", "stream": True,
            "messages": [{"role": "user", "content": "hello"}],
            "stream_options": {"include_usage": True},
        }) as s:
            lines = [ln for ln in s.iter_lines() if ln.strip()]
        results["openai_stream"] = {
            "chunks": len(lines),
            "terminated_with_done": lines[-1] == "data: [DONE]" if lines else False,
        }

        # 6. model listing shape must follow the detected dialect
        ra = c.get(f"{base}/v1/models", headers=cc)
        ro = c.get(f"{base}/v1/models", headers=auth)
        results["models"] = {
            "anthropic_shape": "has_more" in (ra.json() if ra.status_code == 200 else {}),
            "openai_shape": (ro.json() or {}).get("object") == "list" if ro.status_code == 200 else False,
        }

        r = c.get(f"{base}/health", headers=auth)
        results["health"] = r.json() if r.status_code == 200 else {"status": r.status_code}

    ok = (
        results["auth_enforced"]
        and results["claude_code"]["status"] == 200
        and results["claude_code"]["signature_round_trips"]
        and results["codex"]["status"] == 200
        and results["openai_stream"]["terminated_with_done"]
    )
    results["ALL_GOOD"] = ok
    print(json.dumps(results, indent=2))
    return results


@app.local_entrypoint()
def main(base_url: str = "", token: str = DEFAULT_TOKEN):
    smoke.remote(base_url=base_url, token=token)
