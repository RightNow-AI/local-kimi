"""A real inference backend for k3, so the proxy stops talking to a mock.

Everything in this project has been validated against a scripted stand-in. That
proves the wire formats and nothing about the contract that actually matters:
K3 emits ``reasoning_content`` and expects it back verbatim on the next turn.
A mock cannot fail that contract, so it cannot verify it either.

This serves a small reasoning model through vLLM with the deepseek_r1 reasoning
parser, which emits ``reasoning_content`` in exactly the shape K3 does. Pointing
k3 at it exercises the whole reasoning round trip against real weights, real
sampling, and a real engine, at a size that fits one cheap GPU.

It is not Kimi K3. It is the same CONTRACT as K3, which is the part the proxy is
responsible for.

    modal deploy engine/modal_vllm.py
    modal run engine/modal_vllm.py::smoke
"""

from __future__ import annotations

import json
import os
import subprocess
import time

import modal

app = modal.App("k3-vllm")

MODEL = os.environ.get("K3_VLLM_MODEL", "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
PORT = 8000

IMAGE = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("vllm>=0.8", "huggingface_hub[hf_transfer]>=0.26", "httpx>=0.27")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "VLLM_USE_V1": "1"})
)

HF_CACHE = modal.Volume.from_name("hf-cache", create_if_missing=True)

# The smoke test needs the k3 package but not vLLM, and rebuilding the vLLM
# image to add a source mount costs minutes for nothing.
SMOKE_IMAGE = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("httpx>=0.27", "fastapi>=0.115", "pydantic>=2.7")
    .add_local_python_source("k3")
)


@app.function(
    image=IMAGE,
    gpu="A10G",
    volumes={"/root/.cache/huggingface": HF_CACHE},
    timeout=60 * 60,
    scaledown_window=60 * 10,
)
@modal.concurrent(max_inputs=16)
@modal.web_server(PORT, startup_timeout=60 * 15)
def engine():
    """vLLM's own OpenAI-compatible server, with reasoning enabled.

    --reasoning-parser deepseek_r1 makes the server split the model's <think>
    channel into `reasoning_content`, which is the field k3 round-trips.
    """
    subprocess.Popen(
        [
            "vllm", "serve", MODEL,
            "--host", "0.0.0.0",
            "--port", str(PORT),
            "--reasoning-parser", "deepseek_r1",
            "--enable-auto-tool-choice",
            "--tool-call-parser", "hermes",
            "--max-model-len", "8192",
            "--gpu-memory-utilization", "0.90",
            "--served-model-name", "k3",
        ]
    )


@app.function(image=SMOKE_IMAGE, timeout=60 * 20)
def smoke(engine_url: str = "", k3_url: str = "", token: str = "k3-local-dev-token") -> dict:
    """Prove real reasoning survives the round trip through k3.

    Two halves. First the engine alone, to confirm it emits reasoning_content.
    Then the same conversation through k3 in the Anthropic dialect, to confirm
    the reasoning arrives as a thinking block whose signature restores the exact
    bytes - which is the claim this whole project rests on.
    """
    import httpx

    from k3.reasoning import decode_signature

    out: dict[str, object] = {}
    engine_url = engine_url.rstrip("/")
    k3_url = k3_url.rstrip("/")

    with httpx.Client(timeout=300.0) as c:
        if engine_url:
            for attempt in range(30):
                try:
                    if c.get(f"{engine_url}/v1/models").status_code == 200:
                        break
                except Exception:
                    pass
                time.sleep(10)

            r = c.post(f"{engine_url}/v1/chat/completions", json={
                "model": "k3",
                "messages": [{"role": "user", "content": "What is 17 * 23? Think briefly."}],
                "max_tokens": 400,
            })
            body = r.json() if r.status_code == 200 else {}
            msg = (body.get("choices") or [{}])[0].get("message") or {}
            reasoning = msg.get("reasoning_content") or ""
            out["engine"] = {
                "status": r.status_code,
                "has_reasoning_content": bool(reasoning),
                "reasoning_chars": len(reasoning),
                "content_chars": len(msg.get("content") or ""),
            }

        if k3_url:
            r = c.post(f"{k3_url}/v1/messages", headers={
                "authorization": f"Bearer {token}",
                "anthropic-version": "2023-06-01",
                "user-agent": "claude-cli/1.0.60 (external, cli)",
            }, json={
                "model": "k3", "max_tokens": 600,
                "messages": [{"role": "user", "content": "What is 17 * 23? Think briefly."}],
                "thinking": {"type": "enabled", "budget_tokens": 4000},
            })
            body = r.json() if r.status_code == 200 else {}
            blocks = body.get("content") or []
            thinking = next((b for b in blocks if b.get("type") == "thinking"), None)
            text = next((b for b in blocks if b.get("type") == "text"), None)
            recovered = None
            if thinking:
                recovered, _ = decode_signature(thinking.get("signature"))
            out["through_k3"] = {
                "status": r.status_code,
                "block_types": [b.get("type") for b in blocks],
                "thinking_chars": len(thinking.get("thinking", "")) if thinking else 0,
                "text_preview": (text.get("text") or "")[:120] if text else None,
                "signature_restores_reasoning_exactly": bool(recovered)
                and recovered == (thinking or {}).get("thinking"),
                "usage": body.get("usage"),
            }

    ok = (
        out.get("engine", {}).get("has_reasoning_content", False)
        and out.get("through_k3", {}).get("status") == 200
        and out.get("through_k3", {}).get("signature_restores_reasoning_exactly", False)
    )
    out["REAL_REASONING_ROUND_TRIP"] = ok
    print(json.dumps(out, indent=2))
    return out


@app.local_entrypoint()
def main(engine_url: str = "", k3_url: str = "", token: str = "k3-local-dev-token"):
    smoke.remote(engine_url=engine_url, k3_url=k3_url, token=token)
