"""Load the packed Kimi-Linear checkpoint on one H100 and generate tokens.

    modal run engine/modal_klinear_int4.py
"""

from __future__ import annotations

import json
from pathlib import Path

import modal

app = modal.App("kimi-linear-int4-load")

VOLUME = modal.Volume.from_name("kimi-linear-quantized", create_if_missing=False)
MOUNT = "/weights"
MODEL_DIR = f"{MOUNT}/Kimi-Linear-48B-A3B-Instruct-W4A16"

IMAGE = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12"
    )
    .entrypoint([])
    .apt_install("git")
    .pip_install(
        "torch>=2.5",
        "triton>=3.1",
        "safetensors>=0.4.5",
        "transformers>=4.48",
        "numpy>=2.0",
    )
    .env({"CUDA_HOME": "/usr/local/cuda"})
    .add_local_dir(Path(__file__).parent, remote_path="/root/engine")
)


@app.function(
    image=IMAGE,
    gpu="H100",
    cpu=16.0,
    memory=65536,
    timeout=60 * 60,
    volumes={MOUNT: VOLUME},
)
def load_and_generate(
    prompt: str = "The main benefit of lower model memory is",
    max_new_tokens: int = 8,
) -> dict:
    import time

    import torch
    from transformers import AutoTokenizer

    from engine.klinear.generate import generate
    from engine.klinear.model import KLinearModel

    if not torch.cuda.is_available():
        raise RuntimeError("the INT4 load proof requires CUDA")
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")

    device = torch.device("cuda:0")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_DIR,
        trust_remote_code=True,
        local_files_only=True,
    )
    encoded_prompt = tokenizer(prompt, return_tensors="pt")
    input_ids = encoded_prompt["input_ids"].to(device=device)

    torch.cuda.empty_cache()
    baseline_allocated = torch.cuda.memory_allocated(device)
    baseline_reserved = torch.cuda.memory_reserved(device)
    torch.cuda.reset_peak_memory_stats(device)

    load_started = time.perf_counter()
    model = KLinearModel.from_directory(
        MODEL_DIR,
        device=device,
        dtype=torch.bfloat16,
    )
    torch.cuda.synchronize(device)
    load_seconds = time.perf_counter() - load_started

    resident_weight_bytes = model.resident_weight_bytes
    checkpoint_tensor_storage_bytes = model.checkpoint_tensor_storage_bytes
    if resident_weight_bytes != checkpoint_tensor_storage_bytes:
        raise ValueError(
            "resident weight bytes do not match the W4A16 checkpoint: "
            f"resident={resident_weight_bytes}, "
            f"checkpoint={checkpoint_tensor_storage_bytes}"
        )
    allocated_after_load = torch.cuda.memory_allocated(device)
    reserved_after_load = torch.cuda.memory_reserved(device)
    peak_allocated_after_load = torch.cuda.max_memory_allocated(device)
    peak_reserved_after_load = torch.cuda.max_memory_reserved(device)

    generation = generate(
        model,
        input_ids,
        max_new_tokens=max_new_tokens,
        temperature=0.0,
    )
    torch.cuda.synchronize(device)
    generated_ids = generation.generated_ids[0].detach().cpu().tolist()
    continuation = tokenizer.decode(generated_ids, skip_special_tokens=True)
    full_text = tokenizer.decode(
        generation.token_ids[0].detach().cpu().tolist(),
        skip_special_tokens=True,
    )

    result = {
        "checkpoint_kind": model.checkpoint_kind,
        "checkpoint_tensor_storage_bytes": checkpoint_tensor_storage_bytes,
        "resident_weight_bytes": resident_weight_bytes,
        "resident_matches_checkpoint": True,
        "load_seconds": load_seconds,
        "gpu_name": torch.cuda.get_device_name(device),
        "gpu_total_bytes": torch.cuda.get_device_properties(device).total_memory,
        "baseline_allocated_bytes": baseline_allocated,
        "baseline_reserved_bytes": baseline_reserved,
        "allocated_bytes_after_load": allocated_after_load,
        "reserved_bytes_after_load": reserved_after_load,
        "load_allocated_delta_bytes": allocated_after_load - baseline_allocated,
        "load_reserved_delta_bytes": reserved_after_load - baseline_reserved,
        "peak_allocated_bytes_after_load": peak_allocated_after_load,
        "peak_reserved_bytes_after_load": peak_reserved_after_load,
        "peak_allocated_bytes_after_generation": torch.cuda.max_memory_allocated(
            device
        ),
        "peak_reserved_bytes_after_generation": torch.cuda.max_memory_reserved(device),
        "prompt": prompt,
        "prompt_token_count": int(input_ids.shape[1]),
        "generated_token_ids": generated_ids,
        "generated_token_count": len(generated_ids),
        "continuation": continuation,
        "full_text": full_text,
        "coherence_assessed": False,
        "correctness_note": (
            "Token production is measured here. Coherence and model correctness "
            "require review of the printed continuation."
        ),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


@app.local_entrypoint()
def main(
    prompt: str = "The main benefit of lower model memory is",
    max_new_tokens: int = 8,
) -> None:
    result = load_and_generate.remote(
        prompt=prompt,
        max_new_tokens=max_new_tokens,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
