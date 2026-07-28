"""Serve Kimi-Linear from the existing Modal weight volume.

Deploy and run the proof smoke with:

    modal deploy engine/modal_serve_klinear.py
    modal run engine/modal_serve_klinear.py
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import modal

app = modal.App("kimi-linear-engine")

MODEL_NAME = "moonshotai/Kimi-Linear-48B-A3B-Instruct"
WEIGHTS_MOUNT = "/weights"
QUANTIZED_MOUNT = "/quantized"
BF16_DIRECTORY = f"{WEIGHTS_MOUNT}/Kimi-Linear-48B-A3B-Instruct"
INT4_DIRECTORY = f"{QUANTIZED_MOUNT}/Kimi-Linear-48B-A3B-Instruct-W4A16"
WEIGHTS = modal.Volume.from_name("kimi-linear-weights", create_if_missing=False)
QUANTIZED = modal.Volume.from_name("kimi-linear-quantized", create_if_missing=False)

#: Which checkpoint to serve. Both volumes are mounted so the choice is a
#: setting rather than a redeploy, and the INT4 path is the one the product
#: argument rests on: 28,803,304,448 measured bytes against 98,245,528,576 for
#: BF16. Defaults to int4 for exactly that reason. Set K3_CHECKPOINT=bf16 to
#: serve the as-shipped weights, which is what the comparison baseline needs.
CHECKPOINT_KIND = os.environ.get("K3_CHECKPOINT", "int4").strip().lower()


def _model_directory() -> str:
    """Resolve the checkpoint directory, refusing an unrecognised setting.

    Defaulting silently to BF16 on a typo would serve a 91.5 GiB checkpoint
    while the operator believed they were serving 26.8 GiB, and the only
    symptom would be a memory figure nobody was looking at.
    """
    if CHECKPOINT_KIND == "int4":
        return INT4_DIRECTORY
    if CHECKPOINT_KIND == "bf16":
        return BF16_DIRECTORY
    raise ValueError(
        f"K3_CHECKPOINT must be 'int4' or 'bf16', got {CHECKPOINT_KIND!r}"
    )

# The BF16 checkpoint is measured at 91.51 GiB. An 80 GB GPU cannot hold that
# checkpoint, so this service deliberately requests one H200. The devel image
# is also deliberate because KDA startup may JIT-compile CUDA code and needs
# nvcc, which is absent from a slim Debian image.
CUDA_IMAGE = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-devel-ubuntu22.04",
        add_python="3.12",
    )
    .pip_install(
        "fastapi>=0.115",
        "httpx>=0.27",
        "ninja>=1.11",
        "numpy>=2.0",
        "pydantic>=2.7",
        "tiktoken>=0.9",
        "tokenizers>=0.21",
        "torch>=2.5",
        "transformers>=4.51",
        "uvicorn[standard]>=0.30",
    )
    .add_local_dir(
        Path(__file__).parent / "klinear",
        remote_path="/root/engine/klinear",
    )
    .add_local_dir(
        Path(__file__).parent / "serve",
        remote_path="/root/engine/serve",
    )
)

SMOKE_IMAGE = modal.Image.debian_slim(python_version="3.12").pip_install(
    "httpx>=0.27"
)


def _build_app():
    import torch

    from engine.klinear.model import KLinearModel
    from engine.serve.api import ServerConfig, create_app
    from engine.serve.klinear_engine import KimiChatTokenizer, KLinearEngine

    model_directory = _model_directory()
    if not os.path.isdir(model_directory):
        raise FileNotFoundError(
            f"checkpoint directory is missing for K3_CHECKPOINT={CHECKPOINT_KIND}: "
            f"{model_directory}"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("Kimi-Linear serving requires a CUDA GPU")

    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats(device)
    load_started = time.perf_counter()
    # The tokenizer always comes from the BF16 checkpoint. The quantizer copies
    # support files, but the tokenizer is not a weight and there is no reason
    # for two copies of it to drift apart.
    tokenizer = KimiChatTokenizer.from_directory(BF16_DIRECTORY)
    model = KLinearModel.from_directory(
        model_directory,
        device=device,
        dtype=torch.bfloat16,
        expert_cache_entries=256,
    )
    torch.cuda.synchronize(device)
    load_seconds = time.perf_counter() - load_started
    load_peak_gpu_memory_bytes = int(torch.cuda.max_memory_allocated(device))

    engine = KLinearEngine(
        model,
        tokenizer.eos_token_ids,
        device=device,
        load_seconds=load_seconds,
        load_peak_gpu_memory_bytes=load_peak_gpu_memory_bytes,
    )
    return create_app(
        engine,
        tokenizer,
        ServerConfig(
            model=MODEL_NAME,
            default_max_tokens=512,
            serialize_engine=True,
        ),
    )


@app.function(
    image=CUDA_IMAGE,
    gpu="H200",
    memory=65536,
    min_containers=1,
    scaledown_window=20 * 60,
    timeout=4 * 60 * 60,
    volumes={WEIGHTS_MOUNT: WEIGHTS, QUANTIZED_MOUNT: QUANTIZED},
)
@modal.concurrent(max_inputs=1)
@modal.asgi_app()
def api():
    return _build_app()


@app.function(image=SMOKE_IMAGE, timeout=60 * 60)
def smoke(
    base_url: str,
    *,
    max_tokens: int = 64,
    expected_text: str = "KIMI_LINEAR_OK",
) -> dict[str, Any]:
    """Prove non-streaming and streaming OpenAI chat completions."""

    import httpx

    base = base_url.rstrip("/")
    if not base:
        result = {"verdict": "FAIL", "error": "base_url is required"}
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return result

    request = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "user",
                "content": (
                    f"Reply with exactly {expected_text} and no other visible text."
                ),
            }
        ],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "reasoning_effort": "low",
    }

    try:
        timeout = httpx.Timeout(60 * 60, connect=60.0)
        with httpx.Client(timeout=timeout) as client:
            non_stream_response = client.post(
                f"{base}/v1/chat/completions",
                json={**request, "stream": False},
            )
            non_stream_response.raise_for_status()
            non_stream_body = non_stream_response.json()
            non_stream_choice = non_stream_body["choices"][0]
            non_stream_result = {
                "text": non_stream_choice["message"].get("content", ""),
                "reasoning_text": non_stream_choice["message"].get(
                    "reasoning_content",
                    "",
                ),
                "finish_reason": non_stream_choice.get("finish_reason"),
                "usage": non_stream_body.get("usage"),
            }

            stream_result = _stream_completion(
                client,
                base,
                {**request, "stream": True, "stream_options": {"include_usage": True}},
            )
            health_response = client.get(f"{base}/health")
            health_response.raise_for_status()
            health_body = health_response.json()

        engine_metrics = _engine_metrics(health_body)
        peak_bytes = engine_metrics.get("peak_gpu_memory_bytes")
        non_stream_exact = non_stream_result["text"].strip() == expected_text
        stream_exact = stream_result["text"].strip() == expected_text
        passed = all(
            (
                non_stream_exact,
                stream_exact,
                non_stream_result["finish_reason"] == "stop",
                stream_result["finish_reason"] == "stop",
                stream_result["done_sentinel"],
                _valid_usage(non_stream_result["usage"]),
                _valid_usage(stream_result["usage"]),
                isinstance(engine_metrics.get("load_seconds"), (int, float)),
                isinstance(peak_bytes, int) and peak_bytes > 0,
            )
        )
        result = {
            "verdict": "PASS" if passed else "FAIL",
            "instruction_followed": {
                "non_streaming": non_stream_exact,
                "streaming": stream_exact,
                "expected_visible_text": expected_text,
            },
            "non_streaming": non_stream_result,
            "streaming": stream_result,
            "load_seconds": engine_metrics.get("load_seconds"),
            "load_peak_gpu_memory_bytes": engine_metrics.get(
                "load_peak_gpu_memory_bytes"
            ),
            "peak_gpu_memory_bytes": peak_bytes,
            "peak_gpu_memory_gib": (
                round(peak_bytes / (1024**3), 3)
                if isinstance(peak_bytes, int)
                else None
            ),
            "health": health_body,
        }
    except Exception as exc:
        result = {
            "verdict": "FAIL",
            "error": f"{type(exc).__name__}: {exc}",
        }

    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def _stream_completion(
    client: Any,
    base_url: str,
    request: dict[str, Any],
) -> dict[str, Any]:
    reasoning: list[str] = []
    content: list[str] = []
    finish_reason = None
    usage = None
    done_sentinel = False

    with client.stream(
        "POST",
        f"{base_url}/v1/chat/completions",
        json=request,
    ) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if not line.startswith("data: "):
                continue
            payload = line[len("data: ") :]
            if payload == "[DONE]":
                done_sentinel = True
                break
            event = json.loads(payload)
            choices = event.get("choices") or []
            if choices:
                choice = choices[0]
                delta = choice.get("delta") or {}
                reasoning.append(delta.get("reasoning_content", ""))
                content.append(delta.get("content", ""))
                if choice.get("finish_reason") is not None:
                    finish_reason = choice["finish_reason"]
            elif event.get("usage") is not None:
                usage = event["usage"]

    return {
        "text": "".join(content),
        "reasoning_text": "".join(reasoning),
        "finish_reason": finish_reason,
        "usage": usage,
        "done_sentinel": done_sentinel,
    }


def _engine_metrics(health_body: dict[str, Any]) -> dict[str, Any]:
    detail = health_body.get("engine")
    if not isinstance(detail, str):
        return {}
    try:
        parsed = json.loads(detail)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _valid_usage(value: Any) -> bool:
    return (
        isinstance(value, dict)
        and isinstance(value.get("prompt_tokens"), int)
        and value["prompt_tokens"] > 0
        and isinstance(value.get("completion_tokens"), int)
        and value["completion_tokens"] > 0
        and value.get("total_tokens")
        == value["prompt_tokens"] + value["completion_tokens"]
    )


@app.local_entrypoint()
def main(
    base_url: str = "",
    max_tokens: int = 64,
    expected_text: str = "KIMI_LINEAR_OK",
) -> None:
    resolved_url = base_url or api.get_web_url() or ""
    result = smoke.remote(
        resolved_url,
        max_tokens=max_tokens,
        expected_text=expected_text,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
