"""Run one side of the controlled experiment in a fresh vLLM process."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from engine.accuracy.prompts import (
    build_prompt_set,
    prompt_set_sha256,
    teacher_forced_text,
    teacher_text_sha256,
)
from engine.accuracy.thresholds import ACCURACY_SCREEN_V1

VLLM_VERSION = "0.26.0"
SEED = 20260728
MAX_MODEL_LEN = 8192
MAX_NUM_SEQS = 16
MAX_NUM_BATCHED_TOKENS = 8192
GREEDY_MAX_TOKENS = 256

BASE_ENGINE_ARGUMENTS = {
    "trust_remote_code": True,
    "dtype": "bfloat16",
    "tensor_parallel_size": 1,
    "max_model_len": MAX_MODEL_LEN,
    "max_num_seqs": MAX_NUM_SEQS,
    "max_num_batched_tokens": MAX_NUM_BATCHED_TOKENS,
    "gpu_memory_utilization": 0.90,
    "enforce_eager": True,
    "disable_log_stats": True,
    "max_logprobs": -1,
    "logprobs_mode": "raw_logprobs",
    "seed": SEED,
}

GREEDY_SAMPLING = {
    "temperature": 0.0,
    "top_p": 1.0,
    "top_k": 0,
    "max_tokens": GREEDY_MAX_TOKENS,
    "seed": SEED,
}

TEACHER_SAMPLING = {
    "temperature": 0.0,
    "top_p": 1.0,
    "top_k": 0,
    "max_tokens": 1,
    "ignore_eos": True,
    "prompt_logprobs": 1,
    "flat_logprobs": True,
    "detokenize": False,
    "seed": SEED,
}

DISTRIBUTION_SAMPLING = TEACHER_SAMPLING | {"prompt_logprobs": -1}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_json(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return value


def _engine_arguments(*, router_capture_enabled: bool) -> dict[str, Any]:
    return BASE_ENGINE_ARGUMENTS | {
        "enable_return_routed_experts": router_capture_enabled,
    }


def _gpu_environment() -> dict[str, Any]:
    import torch

    output: dict[str, Any] = {
        "torch_device_name": torch.cuda.get_device_name(0),
        "torch_total_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
        "torch_version": torch.__version__,
        "cuda_runtime_version": torch.version.cuda,
    }
    try:
        line = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=uuid,name,driver_version,memory.total",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        ).strip().splitlines()[0]
        gpu_uuid, gpu_name, driver, memory_mib = [part.strip() for part in line.split(",")]
        output.update(
            {
                "gpu_uuid": gpu_uuid,
                "gpu_name": gpu_name,
                "driver_version": driver,
                "nvidia_smi_memory_mib": int(memory_mib),
            }
        )
    except Exception as exc:
        raise RuntimeError("nvidia-smi GPU identity capture failed") from exc
    return output


def _flat_position_values(container, position: int) -> tuple[list[int], list[float]]:
    if hasattr(container, "start_indices"):
        start = container.start_indices[position]
        end = container.end_indices[position]
        return list(container.token_ids[start:end]), list(container.logprobs[start:end])
    row = container[position]
    if row is None:
        return [], []
    return list(row), [float(row[token_id].logprob) for token_id in row]


def _gold_logprobs(prompt_logprobs, token_ids: list[int]) -> list[float]:
    if prompt_logprobs is None or len(prompt_logprobs) != len(token_ids):
        raise ValueError("vLLM returned an unexpected prompt-logprob length")
    values = []
    for position in range(1, len(token_ids)):
        ids, logprobs = _flat_position_values(prompt_logprobs, position)
        matches = [
            logprob
            for token_id, logprob in zip(ids, logprobs, strict=True)
            if token_id == token_ids[position]
        ]
        if not matches:
            raise ValueError(f"gold token is absent from prompt logprobs at position {position}")
        if max(matches) - min(matches) > 1e-6:
            raise ValueError(f"duplicate gold-token logprobs disagree at position {position}")
        values.append(float(matches[0]))
    return values


def _dense_distribution_logprobs(
    prompt_logprobs,
    *,
    token_count: int,
    vocab_size: int,
) -> np.ndarray:
    if prompt_logprobs is None or len(prompt_logprobs) != token_count:
        raise ValueError("vLLM returned an unexpected full-logprob length")
    rows = np.empty((token_count - 1, vocab_size), dtype=np.float32)
    for output_index, position in enumerate(range(1, token_count)):
        ids, values = _flat_position_values(prompt_logprobs, position)
        token_ids = np.asarray(ids, dtype=np.int64)
        logprobs = np.asarray(values, dtype=np.float32)
        if token_ids.size < vocab_size:
            raise ValueError(
                f"full-logprob row {position} has {token_ids.size} entries for vocab {vocab_size}"
            )
        if token_ids.min(initial=0) < 0 or token_ids.max(initial=-1) >= vocab_size:
            raise ValueError(f"full-logprob row {position} contains an invalid token id")
        unique_ids = np.unique(token_ids)
        if unique_ids.size != vocab_size:
            raise ValueError(
                f"full-logprob row {position} covers {unique_ids.size} unique tokens, expected {vocab_size}"
            )
        row = np.full(vocab_size, np.nan, dtype=np.float32)
        row[token_ids] = logprobs
        if not np.isfinite(row).all():
            raise ValueError(f"full-logprob row {position} is incomplete or non-finite")
        rows[output_index] = row
    return rows


def _moe_layer_indices(text_config, routed_shape: tuple[int, ...]) -> list[int]:
    num_hidden_layers = int(getattr(text_config, "num_hidden_layers"))
    first_dense = int(getattr(text_config, "first_k_dense_replace", 0))
    frequency = int(getattr(text_config, "moe_layer_freq", 1))
    indices = list(range(first_dense, num_hidden_layers, frequency))
    if routed_shape[1] != num_hidden_layers:
        raise ValueError(
            f"routed-expert capture has {routed_shape[1]} layers, expected {num_hidden_layers}"
        )
    return indices


def _validate_routes(routes: np.ndarray, *, text_config) -> tuple[list[int], int, int]:
    if routes.ndim != 3:
        raise ValueError(f"routed experts must have shape [tokens,layers,topk], got {routes.shape}")
    expected_topk = int(getattr(text_config, "num_experts_per_token"))
    num_experts = int(getattr(text_config, "num_experts"))
    if routes.shape[2] != expected_topk:
        raise ValueError(f"routed top-k is {routes.shape[2]}, expected {expected_topk}")
    layer_indices = _moe_layer_indices(text_config, tuple(routes.shape))
    for layer_index in layer_indices:
        layer = routes[:, layer_index, :]
        if int(layer.min()) < 0 or int(layer.max()) >= num_experts:
            raise ValueError(f"router layer {layer_index} emitted an invalid expert id")
        unique_counts = np.apply_along_axis(lambda row: len(set(row.tolist())), 1, layer)
        if not np.all(unique_counts == expected_topk):
            raise ValueError(f"router layer {layer_index} contains duplicate expert ids")
    return layer_indices, expected_topk, num_experts


def run_side(
    *,
    model_path: Path,
    side: str,
    output_dir: Path,
    router_capture_enabled: bool,
    router_unavailable_reason: dict[str, Any] | None = None,
) -> dict[str, Any]:
    os.environ["VLLM_USE_V1"] = "1"

    from engine.accuracy.router_compat import validate_vllm_router_capture_config

    if side not in {"bf16", "int4_dequantized"}:
        raise ValueError("side must be bf16 or int4_dequantized")
    if not model_path.is_dir():
        raise FileNotFoundError(model_path)
    config_path = model_path / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    router_config_preflight = validate_vllm_router_capture_config(
        _read_json(config_path),
        config_path=config_path,
    )
    served_config_sha256 = _sha256_file(config_path)
    if not router_capture_enabled and router_unavailable_reason is None:
        raise ValueError(
            "router capture may be disabled only with a recorded unavailability reason"
        )

    import torch
    import vllm
    from vllm import LLM, SamplingParams

    if vllm.__version__ != VLLM_VERSION:
        raise RuntimeError(f"expected vLLM {VLLM_VERSION}, got {vllm.__version__}")
    output_dir.mkdir(parents=True, exist_ok=True)

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    environment = _gpu_environment()
    engine_arguments = _engine_arguments(router_capture_enabled=router_capture_enabled)
    llm = LLM(model=str(model_path), **engine_arguments)
    tokenizer = llm.get_tokenizer()
    text_config = llm.model_config.hf_text_config
    vocab_size = int(getattr(text_config, "vocab_size"))
    if vocab_size != 163840:
        raise ValueError(f"expected vocabulary size 163840, got {vocab_size}")

    prompts = build_prompt_set()
    messages = [[{"role": "user", "content": prompt.text}] for prompt in prompts]
    greedy_outputs = llm.chat(
        messages,
        sampling_params=SamplingParams(**GREEDY_SAMPLING),
        use_tqdm=False,
        add_generation_prompt=True,
    )
    if len(greedy_outputs) != len(prompts):
        raise ValueError("vLLM returned the wrong number of greedy outputs")
    greedy_records = []
    for prompt, result in zip(prompts, greedy_outputs, strict=True):
        completion = result.outputs[0]
        greedy_records.append(
            {
                "prompt_id": prompt.prompt_id,
                "category": prompt.category,
                "prompt_token_ids": list(result.prompt_token_ids),
                "output_token_ids": list(completion.token_ids),
                "output_text": completion.text,
                "finish_reason": completion.finish_reason,
            }
        )

    teacher_text = teacher_forced_text()
    teacher_token_ids = list(tokenizer.encode(teacher_text, add_special_tokens=True))
    teacher_token_ids = teacher_token_ids[: ACCURACY_SCREEN_V1.teacher_forced_max_tokens]
    if len(teacher_token_ids) < 512:
        raise ValueError(f"teacher-forced sample produced only {len(teacher_token_ids)} tokens")
    teacher_prompt = {"prompt_token_ids": teacher_token_ids}
    teacher_result = llm.generate(
        [teacher_prompt],
        SamplingParams(**TEACHER_SAMPLING),
        use_tqdm=False,
    )[0]
    gold_logprobs = _gold_logprobs(teacher_result.prompt_logprobs, teacher_token_ids)
    experts_per_token = int(getattr(text_config, "num_experts_per_token"))
    num_experts = int(getattr(text_config, "num_experts"))
    if router_capture_enabled:
        routes = getattr(teacher_result.outputs[0], "routed_experts", None)
        if routes is None:
            raise ValueError("vLLM returned no routed-expert capture")
        routes = np.asarray(routes)
        layer_indices, captured_topk, captured_num_experts = _validate_routes(
            routes,
            text_config=text_config,
        )
        if captured_topk != experts_per_token or captured_num_experts != num_experts:
            raise ValueError("router capture metadata disagrees with the model config")
        router_capture = {
            "available": True,
            "requested": True,
            "reason": None,
            "preflight": router_config_preflight,
        }
        routed_experts = routes.tolist()
    else:
        layer_indices = []
        routed_experts = None
        router_capture = {
            "available": False,
            "requested": False,
            "reason": router_unavailable_reason,
            "preflight": router_config_preflight,
        }

    distribution_token_count = ACCURACY_SCREEN_V1.distribution_positions + 1
    distribution_token_ids = teacher_token_ids[:distribution_token_count]
    distribution_result = llm.generate(
        [{"prompt_token_ids": distribution_token_ids}],
        SamplingParams(**DISTRIBUTION_SAMPLING),
        use_tqdm=False,
    )[0]
    distribution = _dense_distribution_logprobs(
        distribution_result.prompt_logprobs,
        token_count=len(distribution_token_ids),
        vocab_size=vocab_size,
    )
    distribution_path = output_dir / f"{side}-distribution-logprobs.npy"
    np.save(distribution_path, distribution, allow_pickle=False)

    tokenizer_template = str(getattr(tokenizer, "chat_template", ""))
    protocol = {
        "vllm_version": vllm.__version__,
        "engine_arguments": engine_arguments,
        "greedy_sampling": GREEDY_SAMPLING,
        "teacher_sampling": TEACHER_SAMPLING,
        "distribution_sampling": DISTRIBUTION_SAMPLING,
        "prompt_set_sha256": prompt_set_sha256(prompts),
        "teacher_text_sha256": teacher_text_sha256(),
        "teacher_token_ids_sha256": _sha256_json(teacher_token_ids),
        "distribution_token_ids_sha256": _sha256_json(distribution_token_ids),
        "served_config_sha256": served_config_sha256,
        "chat_template_sha256": hashlib.sha256(tokenizer_template.encode("utf-8")).hexdigest(),
        "apply_chat_template": True,
        "add_generation_prompt": True,
        "seed": SEED,
    }
    result = {
        "schema_version": "runinfra.kimi_linear.vllm_side.v2",
        "created_at": _utc_now(),
        "side": side,
        "model_path": str(model_path),
        "environment": environment
        | {
            "vllm_version": vllm.__version__,
            "python_version": platform.python_version(),
        },
        "protocol": protocol,
        "protocol_fingerprint": _sha256_json(protocol),
        "model_config": {
            "num_hidden_layers": int(getattr(text_config, "num_hidden_layers")),
            "num_experts": num_experts,
            "num_experts_per_token": experts_per_token,
            "first_k_dense_replace": int(getattr(text_config, "first_k_dense_replace", 0)),
            "vocab_size": vocab_size,
        },
        "greedy": greedy_records,
        "teacher_forced": {
            "text_sha256": teacher_text_sha256(),
            "token_ids": teacher_token_ids,
            "gold_logprobs": gold_logprobs,
            "router_capture": router_capture,
            "router_layer_indices": layer_indices,
            "routed_experts": routed_experts,
        },
        "distribution": {
            "token_ids": distribution_token_ids,
            "positions": int(distribution.shape[0]),
            "vocab_size": int(distribution.shape[1]),
            "artifact_path": str(distribution_path),
            "artifact_sha256": _sha256_file(distribution_path),
            "artifact_dtype": str(distribution.dtype),
            "artifact_shape": list(distribution.shape),
        },
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--side", required=True, choices=("bf16", "int4_dequantized"))
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--result-json", required=True, type=Path)
    parser.add_argument(
        "--router-unavailable-reason-json",
        type=Path,
        help=(
            "Evidence from a failed router-enabled attempt. Supplying it selects "
            "the matched metrics-only retry and records router agreement unavailable."
        ),
    )
    args = parser.parse_args()

    router_unavailable_reason = None
    if args.router_unavailable_reason_json is not None:
        router_unavailable_reason = _read_json(args.router_unavailable_reason_json)
    result = run_side(
        model_path=args.model_path,
        side=args.side,
        output_dir=args.output_dir,
        router_capture_enabled=args.router_unavailable_reason_json is None,
        router_unavailable_reason=router_unavailable_reason,
    )
    args.result_json.parent.mkdir(parents=True, exist_ok=True)
    with args.result_json.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(
        json.dumps(
            {
                "side": result["side"],
                "protocol_fingerprint": result["protocol_fingerprint"],
                "gpu_uuid": result["environment"]["gpu_uuid"],
            }
        )
    )


if __name__ == "__main__":
    main()
