"""Run one implementation side of the BF16 engine-parity experiment."""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import random
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from engine.validate.protocol import (
    MAX_MODEL_LEN,
    SEED,
    checkpoint_record,
    read_json,
    sha256_file,
    utc_now,
    write_json,
)

VLLM_VERSION = "0.26.0"
VOCAB_SIZE = 163840


def _gpu_environment() -> dict[str, Any]:
    import torch

    try:
        line = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=uuid,name,driver_version,memory.total",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        ).strip().splitlines()[0]
        gpu_uuid, gpu_name, driver_version, memory_mib = [
            part.strip() for part in line.split(",")
        ]
    except Exception as exc:
        raise RuntimeError("nvidia-smi GPU identity capture failed") from exc
    return {
        "gpu_uuid": gpu_uuid,
        "gpu_name": gpu_name,
        "driver_version": driver_version,
        "nvidia_smi_memory_mib": int(memory_mib),
        "torch_device_name": torch.cuda.get_device_name(0),
        "torch_total_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
        "torch_version": torch.__version__,
        "cuda_runtime_version": torch.version.cuda,
        "python_version": platform.python_version(),
    }


def _seed_everything() -> None:
    import torch

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)


def _validated_protocol(protocol_path: Path, model_path: Path) -> dict[str, Any]:
    protocol = read_json(protocol_path)
    measurement = protocol.get("measurement")
    prompts = protocol.get("prompts")
    fingerprint = protocol.get("protocol_fingerprint")
    if not isinstance(measurement, dict) or not isinstance(prompts, list):
        raise ValueError("validation protocol is missing measurement or prompts")
    if not isinstance(fingerprint, str) or not fingerprint:
        raise ValueError("validation protocol has no fingerprint")
    current_checkpoint = checkpoint_record(model_path)
    if measurement.get("checkpoint") != current_checkpoint:
        raise ValueError(
            "runtime checkpoint identity differs from the predeclared protocol"
        )
    threshold = measurement.get("threshold")
    if not isinstance(threshold, dict):
        raise ValueError("validation protocol has no threshold")
    if len(prompts) != threshold.get("prompt_count"):
        raise ValueError("validation protocol prompt count differs from its threshold")
    for prompt in prompts:
        token_ids = prompt.get("token_ids") if isinstance(prompt, dict) else None
        if (
            not isinstance(token_ids, list)
            or not token_ids
            or any(isinstance(token_id, bool) or not isinstance(token_id, int) for token_id in token_ids)
        ):
            raise ValueError("validation protocol contains invalid prompt token IDs")
    return protocol


def _side_result(
    *,
    side: str,
    implementation: str,
    protocol: dict[str, Any],
    checkpoint: dict[str, Any],
    environment: dict[str, Any],
    model_config: dict[str, int],
    greedy_records: list[dict[str, Any]],
    first_token_records: list[dict[str, Any]],
    distribution_path: Path,
    distribution: np.ndarray,
    hidden_state_reason: str,
) -> dict[str, Any]:
    return {
        "schema_version": "runinfra.kimi_linear.engine_parity_side.v1",
        "created_at": utc_now(),
        "side": side,
        "implementation": implementation,
        "checkpoint": checkpoint,
        "environment": environment,
        "protocol_fingerprint": protocol["protocol_fingerprint"],
        "model_config": model_config,
        "greedy": greedy_records,
        "first_token_distribution": {
            "records": first_token_records,
            "artifact_path": str(distribution_path),
            "artifact_sha256": sha256_file(distribution_path),
            "artifact_dtype": str(distribution.dtype),
            "artifact_shape": list(distribution.shape),
            "scope": "exact full vocabulary",
        },
        "per_layer_hidden_states": {
            "available": False,
            "reason": hidden_state_reason,
        },
    }


def run_candidate(
    *,
    model_path: Path,
    protocol_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    import torch

    from engine.klinear.generate import decode, prefill, sample_logits
    from engine.klinear.model import KLinearModel

    if not torch.cuda.is_available():
        raise RuntimeError("engine validation requires CUDA")
    protocol = _validated_protocol(protocol_path, model_path)
    checkpoint = checkpoint_record(model_path)
    measurement = protocol["measurement"]
    max_tokens = int(measurement["greedy"]["max_tokens"])

    _seed_everything()
    environment = _gpu_environment()
    device = torch.device("cuda:0")
    model = KLinearModel.from_directory(
        model_path,
        device=device,
        dtype=torch.bfloat16,
    )
    if model.checkpoint_kind != "bf16":
        raise ValueError("candidate did not load the BF16 checkpoint")

    greedy_records: list[dict[str, Any]] = []
    first_token_records: list[dict[str, Any]] = []
    first_token_logprobs: list[np.ndarray] = []
    for prompt in protocol["prompts"]:
        prompt_token_ids = list(prompt["token_ids"])
        input_ids = torch.tensor(
            [prompt_token_ids],
            dtype=torch.long,
            device=device,
        )
        output = prefill(model, input_ids)
        first_logits = output.logits[:, -1].float()
        first_logprobs = torch.log_softmax(first_logits, dim=-1)
        first_top1 = int(first_logits.argmax(dim=-1).item())
        first_token_logprobs.append(
            first_logprobs[0].detach().cpu().numpy().astype(np.float32, copy=False)
        )

        generated_token_ids: list[int] = []
        for _ in range(max_tokens):
            next_token = sample_logits(
                output.logits[:, -1],
                temperature=0.0,
                top_p=1.0,
            )
            generated_token_ids.append(int(next_token.item()))
            output = decode(model, next_token.unsqueeze(1), output.state)

        greedy_records.append(
            {
                "prompt_id": prompt["prompt_id"],
                "category": prompt["category"],
                "prompt_token_ids": prompt_token_ids,
                "output_token_ids": generated_token_ids,
                "finish_reason": "length",
            }
        )
        first_token_records.append(
            {
                "prompt_id": prompt["prompt_id"],
                "prompt_token_ids": prompt_token_ids,
                "top1_token_id": first_top1,
            }
        )
        del output, first_logits, first_logprobs, input_ids
        gc.collect()

    torch.cuda.synchronize(device)
    distribution = np.stack(first_token_logprobs, axis=0)
    if distribution.shape != (len(protocol["prompts"]), VOCAB_SIZE):
        raise ValueError(f"candidate first-token artifact has shape {distribution.shape}")
    output_dir.mkdir(parents=True, exist_ok=True)
    distribution_path = output_dir / "candidate-first-token-logprobs.npy"
    np.save(distribution_path, distribution, allow_pickle=False)
    config = model.config
    return _side_result(
        side="klinear_candidate",
        implementation=(
            "direct engine.klinear using the same prefill, sample_logits, and decode "
            "functions used by generate_tokens"
        ),
        protocol=protocol,
        checkpoint=checkpoint,
        environment=environment,
        model_config={
            "num_hidden_layers": int(config.num_hidden_layers),
            "num_experts": int(config.num_experts),
            "num_experts_per_token": int(config.num_experts_per_token),
            "vocab_size": int(config.vocab_size),
        },
        greedy_records=greedy_records,
        first_token_records=first_token_records,
        distribution_path=distribution_path,
        distribution=distribution,
        hidden_state_reason=(
            "The candidate exposes layer states, but vLLM 0.26.0 does not expose "
            "matching intermediate states through this unmodified runtime path. "
            "Hooks would perturb the comparison, so the optional layer probe is skipped."
        ),
    )


def _flat_position_values(container, position: int) -> tuple[list[int], list[float]]:
    if hasattr(container, "start_indices"):
        start = container.start_indices[position]
        end = container.end_indices[position]
        return list(container.token_ids[start:end]), list(container.logprobs[start:end])
    row = container[position]
    if row is None:
        return [], []
    return list(row), [float(row[token_id].logprob) for token_id in row]


def _dense_first_token_logprobs(completion, *, vocab_size: int) -> np.ndarray:
    logprobs = getattr(completion, "logprobs", None)
    if logprobs is None or len(logprobs) != 1:
        raise ValueError("vLLM did not return exactly one output logprob position")
    ids, values = _flat_position_values(logprobs, 0)
    token_ids = np.asarray(ids, dtype=np.int64)
    token_logprobs = np.asarray(values, dtype=np.float32)
    if token_ids.size < vocab_size:
        raise ValueError(
            f"vLLM returned {token_ids.size} first-token logprobs for vocab {vocab_size}"
        )
    if token_ids.min(initial=0) < 0 or token_ids.max(initial=-1) >= vocab_size:
        raise ValueError("vLLM returned an invalid token ID in first-token logprobs")
    if np.unique(token_ids).size != vocab_size:
        raise ValueError("vLLM first-token logprobs do not cover the full vocabulary")
    row = np.full(vocab_size, np.nan, dtype=np.float32)
    row[token_ids] = token_logprobs
    if not np.isfinite(row).all():
        raise ValueError("vLLM first-token logprobs are incomplete or non-finite")
    return row


def run_reference(
    *,
    model_path: Path,
    protocol_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    os.environ["VLLM_USE_V1"] = "1"

    import torch
    import vllm
    from vllm import LLM, SamplingParams

    if not torch.cuda.is_available():
        raise RuntimeError("engine validation requires CUDA")
    if vllm.__version__ != VLLM_VERSION:
        raise RuntimeError(f"expected vLLM {VLLM_VERSION}, got {vllm.__version__}")
    protocol = _validated_protocol(protocol_path, model_path)
    checkpoint = checkpoint_record(model_path)
    measurement = protocol["measurement"]
    max_tokens = int(measurement["greedy"]["max_tokens"])

    _seed_everything()
    environment = _gpu_environment() | {"vllm_version": vllm.__version__}
    llm = LLM(
        model=str(model_path),
        trust_remote_code=True,
        dtype="bfloat16",
        tensor_parallel_size=1,
        max_model_len=MAX_MODEL_LEN,
        max_num_seqs=16,
        max_num_batched_tokens=8192,
        gpu_memory_utilization=0.90,
        enforce_eager=True,
        disable_log_stats=True,
        max_logprobs=-1,
        logprobs_mode="raw_logprobs",
        seed=SEED,
    )
    text_config = llm.model_config.hf_text_config
    vocab_size = int(getattr(text_config, "vocab_size"))
    if vocab_size != VOCAB_SIZE:
        raise ValueError(f"expected vocabulary size {VOCAB_SIZE}, got {vocab_size}")
    prompt_inputs = [
        {"prompt_token_ids": list(prompt["token_ids"])}
        for prompt in protocol["prompts"]
    ]

    greedy_outputs = llm.generate(
        prompt_inputs,
        SamplingParams(
            temperature=0.0,
            top_p=1.0,
            top_k=0,
            max_tokens=max_tokens,
            ignore_eos=True,
            detokenize=False,
            seed=SEED,
        ),
        use_tqdm=False,
    )
    if len(greedy_outputs) != len(protocol["prompts"]):
        raise ValueError("vLLM returned the wrong number of greedy outputs")
    greedy_records = []
    for prompt, result in zip(protocol["prompts"], greedy_outputs, strict=True):
        if list(result.prompt_token_ids) != list(prompt["token_ids"]):
            raise ValueError(f"vLLM changed prompt token IDs for {prompt['prompt_id']}")
        completion = result.outputs[0]
        greedy_records.append(
            {
                "prompt_id": prompt["prompt_id"],
                "category": prompt["category"],
                "prompt_token_ids": list(result.prompt_token_ids),
                "output_token_ids": list(completion.token_ids),
                "finish_reason": completion.finish_reason,
            }
        )

    first_outputs = llm.generate(
        prompt_inputs,
        SamplingParams(
            temperature=0.0,
            top_p=1.0,
            top_k=0,
            max_tokens=1,
            ignore_eos=True,
            logprobs=-1,
            flat_logprobs=True,
            detokenize=False,
            seed=SEED,
        ),
        use_tqdm=False,
    )
    if len(first_outputs) != len(protocol["prompts"]):
        raise ValueError("vLLM returned the wrong number of first-token outputs")
    first_token_records = []
    first_token_logprobs = []
    for prompt, result in zip(protocol["prompts"], first_outputs, strict=True):
        if list(result.prompt_token_ids) != list(prompt["token_ids"]):
            raise ValueError(f"vLLM changed prompt token IDs for {prompt['prompt_id']}")
        completion = result.outputs[0]
        row = _dense_first_token_logprobs(completion, vocab_size=vocab_size)
        first_token_logprobs.append(row)
        first_token_records.append(
            {
                "prompt_id": prompt["prompt_id"],
                "prompt_token_ids": list(result.prompt_token_ids),
                "top1_token_id": int(np.argmax(row)),
            }
        )

    distribution = np.stack(first_token_logprobs, axis=0)
    output_dir.mkdir(parents=True, exist_ok=True)
    distribution_path = output_dir / "reference-first-token-logprobs.npy"
    np.save(distribution_path, distribution, allow_pickle=False)
    return _side_result(
        side="vllm_reference",
        implementation="vLLM 0.26.0 offline LLM engine",
        protocol=protocol,
        checkpoint=checkpoint,
        environment=environment,
        model_config={
            "num_hidden_layers": int(getattr(text_config, "num_hidden_layers")),
            "num_experts": int(getattr(text_config, "num_experts")),
            "num_experts_per_token": int(
                getattr(text_config, "num_experts_per_token")
            ),
            "vocab_size": vocab_size,
        },
        greedy_records=greedy_records,
        first_token_records=first_token_records,
        distribution_path=distribution_path,
        distribution=distribution,
        hidden_state_reason=(
            "vLLM 0.26.0 does not expose per-layer hidden states through the measured "
            "offline generation path. Adding hooks or a model fork would change that path."
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--side",
        required=True,
        choices=("vllm_reference", "klinear_candidate"),
    )
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--protocol-json", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--result-json", required=True, type=Path)
    args = parser.parse_args()

    if args.side == "vllm_reference":
        result = run_reference(
            model_path=args.model_path,
            protocol_path=args.protocol_json,
            output_dir=args.output_dir,
        )
    else:
        result = run_candidate(
            model_path=args.model_path,
            protocol_path=args.protocol_json,
            output_dir=args.output_dir,
        )
    write_json(args.result_json, result)
    print(
        json.dumps(
            {
                "side": result["side"],
                "checkpoint": result["checkpoint"]["directory"],
                "gpu_uuid": result["environment"]["gpu_uuid"],
                "protocol_fingerprint": result["protocol_fingerprint"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
