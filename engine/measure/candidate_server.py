"""Launch the repository KLinear engine for matched serving measurement."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import time
import uuid
from collections.abc import AsyncIterator, Mapping, Sequence
from pathlib import Path
from typing import Any

from engine.measure.harness import RuntimeSpec

DEFAULT_RUNTIME_NAME = "engine.klinear"
DEFAULT_QUANTIZATION_FORMAT = "selective W4A16 INT4"
DEFAULT_INT4_WEIGHTS_PATH = "/quantized/Kimi-Linear-48B-A3B-Instruct-W4A16"
DEFAULT_COMMAND_TEMPLATE = (
    "python",
    "-m",
    "engine.measure.candidate_server",
    "--model-id",
    "{model_id}",
    "--tokenizer-path",
    "{model_path}",
    "--weights-path",
    "{weights_path}",
    "--host",
    "127.0.0.1",
    "--port",
    "{port}",
    "--served-model-name",
    "{served_model_name}",
    "--max-model-len",
    "{max_model_len}",
    "--tensor-parallel-size",
    "{tensor_parallel_size}",
    "--max-num-seqs",
    "{max_num_seqs}",
)
DEFAULT_VERSION_COMMAND = (
    "python",
    "-m",
    "engine.measure.candidate_server",
    "--version",
)


def candidate_runtime_spec(
    *,
    runtime_name: str,
    quantization_format: str,
    command: list[str],
    version_command: list[str],
    weights_path: str,
    model_id: str,
    requested_revision: str,
    resolved_revision: str,
    served_model_name: str,
    max_model_len: int,
    port: int,
) -> RuntimeSpec:
    """Describe the one repository candidate consumed by the harness."""

    return RuntimeSpec(
        side="candidate",
        name=runtime_name,
        quantization_format=quantization_format,
        command=command,
        version_command=version_command,
        weights_path=weights_path,
        compute_weights_digest=True,
        model_id=model_id,
        requested_revision=requested_revision,
        resolved_revision=resolved_revision,
        served_model_name=served_model_name,
        tensor_parallel_size=1,
        max_model_len=max_model_len,
        port=port,
        disclosures=(
            "engine.klinear serializes generation on one model instance, so "
            "higher HTTP concurrency includes request queueing.",
        ),
    )


def source_version() -> str:
    """Fingerprint the measure launcher and the engine sources it executes."""

    repository_root = Path(__file__).resolve().parents[2]
    engine_root = repository_root / "engine"
    paths = [
        Path(__file__).resolve(),
        repository_root / "k3" / "toolcalls.py",
    ]
    for directory in (engine_root / "klinear", engine_root / "serve"):
        paths.extend(sorted(directory.rglob("*.py")))
    digest = hashlib.sha256()
    for path in sorted(set(paths)):
        relative = path.relative_to(repository_root).as_posix().encode("utf-8")
        payload = path.read_bytes()
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return f"engine.klinear-source-sha256:{digest.hexdigest()}"


def _error_body(message: str, *, param: str | None = None) -> dict[str, Any]:
    return {
        "error": {
            "message": message,
            "type": "invalid_request_error",
            "param": param,
            "code": "invalid_request_error",
        }
    }


def _sse(payload: Mapping[str, Any]) -> bytes:
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return f"data: {encoded}\n\n".encode("utf-8")


def _token_ids(value: Any) -> tuple[int, ...]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or not value
        or any(
            isinstance(token_id, bool)
            or not isinstance(token_id, int)
            or token_id < 0
            for token_id in value
        )
    ):
        raise ValueError("prompt must be a nonempty array of nonnegative token IDs")
    return tuple(value)


def _positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _completion_request(
    body: Any,
    *,
    served_model_name: str,
    max_model_len: int,
) -> dict[str, Any]:
    if not isinstance(body, Mapping):
        raise ValueError("request body must be a JSON object")
    if body.get("model") != served_model_name:
        raise ValueError(f"model must equal {served_model_name!r}")
    prompt = _token_ids(body.get("prompt"))
    max_tokens = _positive_int(body.get("max_tokens"), "max_tokens")
    if len(prompt) + max_tokens > max_model_len:
        raise ValueError("prompt and completion exceed max_model_len")
    if body.get("min_tokens") != max_tokens:
        raise ValueError("min_tokens must equal max_tokens for matched measurement")
    if body.get("ignore_eos") is not True:
        raise ValueError("ignore_eos must be true for matched measurement")
    if body.get("stream") is not True:
        raise ValueError("stream must be true for matched measurement")
    stream_options = body.get("stream_options")
    if not isinstance(stream_options, Mapping) or stream_options.get("include_usage") is not True:
        raise ValueError("stream_options.include_usage must be true")
    temperature = body.get("temperature", 0.0)
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
        raise ValueError("temperature must be numeric")
    if float(temperature) != 0.0:
        raise ValueError("the matched measurement requires greedy temperature 0")
    top_p = body.get("top_p", 1.0)
    if isinstance(top_p, bool) or not isinstance(top_p, (int, float)):
        raise ValueError("top_p must be numeric")
    if float(top_p) != 1.0:
        raise ValueError("the matched measurement requires top_p 1")
    seed = body.get("seed")
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
        raise ValueError("seed must be an integer")
    return {
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": float(temperature),
        "top_p": float(top_p),
    }


def build_app(
    *,
    model_id: str,
    tokenizer_path: str,
    weights_path: str,
    served_model_name: str,
    max_model_len: int,
    tensor_parallel_size: int,
    max_num_seqs: int,
):
    """Build the existing serving app plus token-ID completions for the harness."""

    import torch
    from fastapi import Request
    from fastapi.responses import JSONResponse, StreamingResponse

    from engine.klinear.model import KLinearModel
    from engine.serve.api import ServerConfig, create_app
    from engine.serve.contracts import SamplingParams, TokenEvent, UsageEvent
    from engine.serve.klinear_engine import KimiChatTokenizer, KLinearEngine

    if not model_id.strip() or not served_model_name.strip():
        raise ValueError("model identifiers must not be empty")
    if tensor_parallel_size != 1:
        raise ValueError("engine.klinear measurement requires tensor_parallel_size=1")
    if max_model_len < 2:
        raise ValueError("max_model_len must be at least two")
    if max_num_seqs < 1:
        raise ValueError("max_num_seqs must be positive")
    if not Path(tokenizer_path).is_dir():
        raise FileNotFoundError(f"tokenizer directory does not exist: {tokenizer_path}")
    if not Path(weights_path).is_dir():
        raise FileNotFoundError(f"candidate weight directory does not exist: {weights_path}")
    if not torch.cuda.is_available():
        raise RuntimeError("engine.klinear measurement requires a CUDA GPU")

    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats(device)
    load_started = time.perf_counter()
    tokenizer = KimiChatTokenizer.from_directory(tokenizer_path)
    model = KLinearModel.from_directory(
        weights_path,
        device=device,
        dtype=torch.bfloat16,
        expert_cache_entries=256,
    )
    torch.cuda.synchronize(device)
    load_seconds = time.perf_counter() - load_started
    load_peak_gpu_memory_bytes = int(torch.cuda.max_memory_allocated(device))

    chat_engine = KLinearEngine(
        model,
        tokenizer.eos_token_ids,
        device=device,
        load_seconds=load_seconds,
        load_peak_gpu_memory_bytes=load_peak_gpu_memory_bytes,
    )
    app = create_app(
        chat_engine,
        tokenizer,
        ServerConfig(
            model=served_model_name,
            default_max_tokens=512,
            serialize_engine=True,
        ),
    )

    # The measure protocol requires ignore_eos=true and an exact token count.
    # Token IDs are nonnegative, so -1 is an unreachable stop token that lets
    # the existing incremental KLinearEngine run to max_tokens unchanged.
    measurement_engine = KLinearEngine(
        model,
        {-1},
        device=device,
        load_seconds=load_seconds,
        load_peak_gpu_memory_bytes=load_peak_gpu_memory_bytes,
    )
    generation_lock = asyncio.Lock()
    pending_gate = asyncio.Semaphore(max_num_seqs)
    app.state.measurement_engine = measurement_engine
    app.state.measurement_model_id = model_id
    app.state.measurement_weights_path = weights_path
    app.state.measurement_max_num_seqs = max_num_seqs

    @app.post("/v1/completions")
    async def completions(request: Request) -> Any:
        try:
            body = await request.json()
            parsed = _completion_request(
                body,
                served_model_name=served_model_name,
                max_model_len=max_model_len,
            )
        except Exception as exc:
            return JSONResponse(
                status_code=400,
                content=_error_body(f"{type(exc).__name__}: {exc}"),
            )

        response_id = f"cmpl-{uuid.uuid4().hex[:24]}"
        created = int(time.time())

        async def stream_body() -> AsyncIterator[bytes]:
            source = measurement_engine.generate(
                parsed["prompt"],
                SamplingParams(
                    max_tokens=parsed["max_tokens"],
                    temperature=parsed["temperature"],
                    top_p=parsed["top_p"],
                ),
            )
            completion_tokens = 0
            usage = None
            try:
                async with pending_gate:
                    async with generation_lock:
                        async for event in source:
                            if isinstance(event, TokenEvent):
                                completion_tokens += 1
                                token_text = tokenizer.token_bytes(event.token_id).decode(
                                    "utf-8",
                                    errors="replace",
                                )
                                if not token_text:
                                    raise RuntimeError(
                                        f"token {event.token_id} decoded to an empty byte string"
                                    )
                                yield _sse(
                                    {
                                        "id": response_id,
                                        "object": "text_completion",
                                        "created": created,
                                        "model": served_model_name,
                                        "choices": [
                                            {
                                                "index": 0,
                                                "text": token_text,
                                                "finish_reason": None,
                                            }
                                        ],
                                        "usage": None,
                                    }
                                )
                            elif isinstance(event, UsageEvent):
                                if usage is not None:
                                    raise RuntimeError("engine emitted duplicate usage")
                                usage = event
                            else:
                                raise RuntimeError(
                                    f"engine emitted unsupported event {type(event).__name__}"
                                )
            finally:
                await source.aclose()

            if usage is None:
                raise RuntimeError("engine ended without usage")
            if usage.prompt_tokens != len(parsed["prompt"]):
                raise RuntimeError("engine prompt token usage mismatch")
            if usage.completion_tokens != parsed["max_tokens"]:
                raise RuntimeError("engine did not produce the exact requested token count")
            if completion_tokens != usage.completion_tokens:
                raise RuntimeError("streamed token count does not match engine usage")

            yield _sse(
                {
                    "id": response_id,
                    "object": "text_completion",
                    "created": created,
                    "model": served_model_name,
                    "choices": [
                        {"index": 0, "text": "", "finish_reason": "length"}
                    ],
                    "usage": None,
                }
            )
            yield _sse(
                {
                    "id": response_id,
                    "object": "text_completion",
                    "created": created,
                    "model": served_model_name,
                    "choices": [],
                    "usage": {
                        "prompt_tokens": usage.prompt_tokens,
                        "completion_tokens": usage.completion_tokens,
                        "total_tokens": usage.prompt_tokens + usage.completion_tokens,
                    },
                }
            )
            yield b"data: [DONE]\n\n"

        return StreamingResponse(
            stream_body(),
            media_type="text/event-stream",
            headers={"cache-control": "no-cache", "connection": "keep-alive"},
        )

    return app


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", action="store_true")
    parser.add_argument("--model-id", default="")
    parser.add_argument("--tokenizer-path", default="")
    parser.add_argument("--weights-path", default="")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--served-model-name", default="")
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--max-num-seqs", type=int, default=64)
    return parser


def main() -> None:
    arguments = _parser().parse_args()
    if arguments.version:
        print(source_version())
        return
    import uvicorn

    app = build_app(
        model_id=arguments.model_id,
        tokenizer_path=arguments.tokenizer_path,
        weights_path=arguments.weights_path,
        served_model_name=arguments.served_model_name,
        max_model_len=arguments.max_model_len,
        tensor_parallel_size=arguments.tensor_parallel_size,
        max_num_seqs=arguments.max_num_seqs,
    )
    uvicorn.run(app, host=arguments.host, port=arguments.port, log_level="info")


if __name__ == "__main__":
    main()
