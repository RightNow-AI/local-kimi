"""Single-invocation Modal entrypoint for matched Kimi-Linear measurement.

The candidate command must start the repository's real OpenAI-compatible
inference server. This package does not substitute the partial correctness
adapter from ``engine.bench.candidate`` and has no reference fallback.

The command is a JSON string array with required placeholders. Example shape:

    modal run engine/measure/modal_measure.py \
      --candidate-runtime-name local-kimi-engine \
      --candidate-quantization-format "INT4 weight-only" \
      --candidate-weights-dir optimized/kimi-linear \
      --candidate-command-json '["python","-m","REAL_SERVER_MODULE",...]' \
      --candidate-version-command-json '["python","-m","REAL_SERVER_MODULE","--version"]'

The real command must include ``{model_id}``, ``{model_path}``,
``{weights_path}``, ``{port}``, ``{served_model_name}``, ``{max_model_len}``,
``{tensor_parallel_size}``, and ``{max_num_seqs}``. Missing support fails
before GPU measurement.
"""

from __future__ import annotations

import importlib.metadata
import json
import os
from pathlib import Path
from typing import Any

import modal

from engine.measure.harness import RuntimeSpec, load_download_manifest, run_both_sides
from engine.measure.prompts import build_prompt_set
from engine.measure.record import MIN_REPETITIONS, validate_concurrency_levels
from engine.measure.runtime import (
    expand_command_template,
    python_package_version_command,
)

APP = modal.App("kimi-linear-both-sides-measure")

IMAGE_REQUIREMENTS = (
    # vllm is PINNED, not ranged. Left as a range next to other pins, pip has
    # already resolved this project down to 0.19.1, which predates
    # KimiLinearForCausalLM support. A measurement taken against a version that
    # cannot serve the model is worse than no measurement.
    "vllm==0.26.0",
    "torch>=2.5",
    "transformers>=4.56",
    "huggingface_hub>=0.34",
    "httpx>=0.27",
    "nvidia-ml-py>=12.0",
    "numpy>=2.0",
    "safetensors>=0.4",
    "fla-core",
    "einops>=0.8",
)

# A CUDA devel base, not debian_slim. On debian_slim this model loads all 20
# shards and then dies at engine core init with "Could not find nvcc and
# default cuda_home='/usr/local/cuda' doesn't exist", because Kimi-Linear's KDA
# path JIT-compiles kernels at startup and needs a toolchain present at RUN
# time. That failure has already cost this project one H200 run, and it happens
# only AFTER a multi-minute checkpoint load, so it is expensive every time.
IMAGE = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12"
    )
    .entrypoint([])
    .apt_install("git")
    .pip_install(*IMAGE_REQUIREMENTS)
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "VLLM_USE_V1": "1",
            "CUDA_HOME": "/usr/local/cuda",
        }
    )
    .add_local_dir("engine", remote_path="/root/engine")
)

KIMI_LINEAR_WEIGHTS = modal.Volume.from_name(
    "kimi-linear-weights", create_if_missing=True
)
VOL = "/kimi-linear"
HF_CACHE = f"{VOL}/huggingface"
DOWNLOAD_MANIFEST_PATH = f"{VOL}/bench/kimi-linear-48b-a3b/download.json"
MEASUREMENT_DIR = f"{VOL}/measurements"
MODEL_ID = "moonshotai/Kimi-Linear-48B-A3B-Instruct"
DEFAULT_GPU = "H200"
ALLOWED_SINGLE_GPUS = {"H200", "B200"}
PORT = 8000


def _json_string_array(raw: str, field: str) -> list[str]:
    if not raw.strip():
        raise ValueError(f"{field} is required")
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{field} must be valid JSON") from exc
    if (
        not isinstance(value, list)
        or not value
        or not all(isinstance(item, str) and item for item in value)
    ):
        raise ValueError(f"{field} must be a nonempty JSON string array")
    return value


def _parse_concurrencies(raw: str) -> tuple[int, ...]:
    try:
        values = tuple(int(item.strip()) for item in raw.split(",") if item.strip())
    except ValueError as exc:
        raise ValueError("concurrencies must be comma-separated integers") from exc
    return validate_concurrency_levels(values)


def _package_versions() -> dict[str, str]:
    packages = (
        "vllm",
        "torch",
        "transformers",
        "huggingface-hub",
        "httpx",
        "nvidia-ml-py",
        "safetensors",
    )
    versions = {}
    for package in packages:
        versions[package] = importlib.metadata.version(package)
    return versions


def _atomic_json(path: str, value: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, destination)


@APP.function(
    image=IMAGE,
    gpu=DEFAULT_GPU,
    volumes={VOL: KIMI_LINEAR_WEIGHTS},
    cpu=16.0,
    memory=65536,
    timeout=60 * 60 * 12,
)
def measure_both(
    *,
    revision: str,
    candidate_runtime_name: str,
    candidate_quantization_format: str,
    candidate_weights_dir: str,
    candidate_command_template: list[str],
    candidate_version_command_template: list[str],
    concurrency_levels: tuple[int, ...],
    repetitions: int,
    warmup_requests: int,
    max_model_len: int,
    max_prompt_tokens: int,
    max_output_tokens: int,
    startup_timeout_seconds: float,
    request_timeout_seconds: float,
    requested_gpu: str,
) -> dict[str, Any]:
    """Measure tuned vLLM and the real candidate sequentially on this GPU."""

    from transformers import AutoTokenizer

    if requested_gpu not in ALLOWED_SINGLE_GPUS:
        raise ValueError(f"GPU must be one of {sorted(ALLOWED_SINGLE_GPUS)}")
    if not candidate_runtime_name.strip():
        raise ValueError("candidate runtime name is required")
    if candidate_runtime_name.strip().lower() == "vllm":
        raise ValueError("candidate runtime must identify the repository engine, not vLLM")
    if not candidate_quantization_format.strip():
        raise ValueError("candidate quantization format is required")
    if "int4" not in candidate_quantization_format.strip().lower():
        raise ValueError("this lane requires the candidate INT4 weight format")
    if not candidate_weights_dir.strip():
        raise ValueError("candidate weights directory is required")
    if repetitions < MIN_REPETITIONS:
        raise ValueError(f"repetitions must be at least {MIN_REPETITIONS}")
    if warmup_requests < 1:
        raise ValueError("warmup_requests must be positive")
    if max_model_len < max_prompt_tokens + max_output_tokens:
        raise ValueError("max_model_len must cover prompt and generated tokens")

    concurrency_levels = validate_concurrency_levels(concurrency_levels)
    KIMI_LINEAR_WEIGHTS.reload()
    download = load_download_manifest(
        DOWNLOAD_MANIFEST_PATH,
        model_id=MODEL_ID,
        requested_revision=revision,
    )
    snapshot_path = download["snapshot_path"]
    resolved_revision = download["resolved_revision"]
    candidate_weights_path = (
        candidate_weights_dir
        if os.path.isabs(candidate_weights_dir)
        else f"{VOL}/{candidate_weights_dir.strip('/')}"
    )
    if Path(candidate_weights_path).resolve() == Path(snapshot_path).resolve():
        raise ValueError("candidate weights must be a distinct quantized artifact")
    max_num_seqs = max(concurrency_levels)
    served_model_name = MODEL_ID
    values = {
        "model_id": MODEL_ID,
        "model_path": snapshot_path,
        "weights_path": candidate_weights_path,
        "resolved_revision": resolved_revision,
        "port": PORT,
        "served_model_name": served_model_name,
        "max_model_len": max_model_len,
        "tensor_parallel_size": 1,
        "max_num_seqs": max_num_seqs,
        "quantization_format": candidate_quantization_format,
    }
    candidate_command = expand_command_template(
        candidate_command_template,
        values,
        required_placeholders=(
            "model_id",
            "model_path",
            "weights_path",
            "port",
            "served_model_name",
            "max_model_len",
            "tensor_parallel_size",
            "max_num_seqs",
        ),
    )
    lowered_candidate_command = [item.lower() for item in candidate_command]
    if any(
        item == "vllm" or item.startswith("vllm.") or "vllm.entrypoints" in item
        for item in lowered_candidate_command
    ):
        raise ValueError("candidate command cannot launch the vLLM baseline implementation")
    candidate_version_command = expand_command_template(
        candidate_version_command_template,
        values,
        required_placeholders=(),
    )

    tokenizer = AutoTokenizer.from_pretrained(
        snapshot_path,
        trust_remote_code=True,
        local_files_only=True,
        cache_dir=HF_CACHE,
    )
    prompt_set = build_prompt_set(
        tokenizer,
        model_id=MODEL_ID,
        resolved_revision=resolved_revision,
        max_prompt_tokens=max_prompt_tokens,
        max_output_tokens=max_output_tokens,
        seed=17,
    )
    baseline_command = [
        "vllm",
        "serve",
        snapshot_path,
        "--host",
        "127.0.0.1",
        "--port",
        str(PORT),
        "--served-model-name",
        served_model_name,
        "--trust-remote-code",
        "--dtype",
        "bfloat16",
        "--tensor-parallel-size",
        "1",
        "--max-model-len",
        str(max_model_len),
        "--max-num-seqs",
        str(max_num_seqs),
        "--gpu-memory-utilization",
        "0.95",
        "--enable-chunked-prefill",
    ]
    baseline = RuntimeSpec(
        side="baseline",
        name="vLLM",
        quantization_format="BF16",
        command=baseline_command,
        version_command=python_package_version_command("vllm"),
        weights_path=snapshot_path,
        compute_weights_digest=False,
        model_id=MODEL_ID,
        requested_revision=revision,
        resolved_revision=resolved_revision,
        served_model_name=served_model_name,
        tensor_parallel_size=1,
        max_model_len=max_model_len,
        port=PORT,
    )
    candidate = RuntimeSpec(
        side="candidate",
        name=candidate_runtime_name,
        quantization_format=candidate_quantization_format,
        command=candidate_command,
        version_command=candidate_version_command,
        weights_path=candidate_weights_path,
        compute_weights_digest=True,
        model_id=MODEL_ID,
        requested_revision=revision,
        resolved_revision=resolved_revision,
        served_model_name=served_model_name,
        tensor_parallel_size=1,
        max_model_len=max_model_len,
        port=PORT,
    )
    record = run_both_sides(
        baseline=baseline,
        candidate=candidate,
        prompt_set=prompt_set,
        concurrency_levels=concurrency_levels,
        repetitions=repetitions,
        warmup_requests=warmup_requests,
        startup_timeout_seconds=startup_timeout_seconds,
        request_timeout_seconds=request_timeout_seconds,
    )
    record["environment"]["requested_modal_gpu"] = requested_gpu
    record["environment"]["image_requirements"] = list(IMAGE_REQUIREMENTS)
    record["environment"]["package_versions"] = _package_versions()
    output_path = f"{MEASUREMENT_DIR}/{record['environment']['run_id']}.json"
    record["artifact"] = {
        "volume": "kimi-linear-weights",
        "path": output_path,
    }
    _atomic_json(output_path, record)
    KIMI_LINEAR_WEIGHTS.commit()
    return record


@APP.local_entrypoint()
def main(
    revision: str = "main",
    gpu: str = DEFAULT_GPU,
    candidate_runtime_name: str = "",
    candidate_quantization_format: str = "INT4 weight-only",
    candidate_weights_dir: str = "",
    candidate_command_json: str = "",
    candidate_version_command_json: str = "",
    concurrencies: str = "1,4,16,64",
    repetitions: int = MIN_REPETITIONS,
    warmup_requests: int = 4,
    max_model_len: int = 4096,
    max_prompt_tokens: int = 256,
    max_output_tokens: int = 64,
    startup_timeout_seconds: float = 1800.0,
    request_timeout_seconds: float = 600.0,
    output_path: str = "",
) -> None:
    """Run both sides once and emit exactly one machine-readable JSON object."""

    if gpu not in ALLOWED_SINGLE_GPUS:
        raise ValueError(f"gpu must be one of {sorted(ALLOWED_SINGLE_GPUS)}")
    if not candidate_weights_dir.strip():
        raise ValueError("--candidate-weights-dir is required")
    command_template = _json_string_array(
        candidate_command_json,
        "candidate_command_json",
    )
    version_template = _json_string_array(
        candidate_version_command_json,
        "candidate_version_command_json",
    )
    levels = _parse_concurrencies(concurrencies)
    record = measure_both.with_options(gpu=gpu).remote(
        revision=revision,
        candidate_runtime_name=candidate_runtime_name,
        candidate_quantization_format=candidate_quantization_format,
        candidate_weights_dir=candidate_weights_dir,
        candidate_command_template=command_template,
        candidate_version_command_template=version_template,
        concurrency_levels=levels,
        repetitions=repetitions,
        warmup_requests=warmup_requests,
        max_model_len=max_model_len,
        max_prompt_tokens=max_prompt_tokens,
        max_output_tokens=max_output_tokens,
        startup_timeout_seconds=startup_timeout_seconds,
        request_timeout_seconds=request_timeout_seconds,
        requested_gpu=gpu,
    )
    encoded = json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    if output_path:
        Path(output_path).write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
