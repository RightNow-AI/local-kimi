"""Compare Kimi-Linear router behavior with the repository's own engine.

Each invocation compares the BF16 source checkpoint with exactly one existing
W4A16 artifact. The checkpoints load sequentially on one H200, and only compact
CPU router traces remain between loads.

    modal run engine/modal_router_analysis.py
    modal run engine/modal_router_analysis.py --candidate shared-experts-bf16
"""

from __future__ import annotations

import gc
import hashlib
import json
import time
from pathlib import Path

import modal

app = modal.App("kimi-linear-router-analysis")

QUANTIZED = modal.Volume.from_name("kimi-linear-quantized", create_if_missing=False)
WEIGHTS = modal.Volume.from_name("kimi-linear-weights", create_if_missing=False)
QUANTIZED_MOUNT = "/quantized"
WEIGHTS_MOUNT = "/weights"
BF16_DIR = f"{WEIGHTS_MOUNT}/Kimi-Linear-48B-A3B-Instruct"
CANDIDATE_DIRS = {
    "int4": f"{QUANTIZED_MOUNT}/Kimi-Linear-48B-A3B-Instruct-W4A16",
    "shared-experts-bf16": (
        f"{QUANTIZED_MOUNT}/"
        "Kimi-Linear-48B-A3B-Instruct-W4A16-shared-experts-bf16"
    ),
}

# There are 26 MoE layers and 256 routed experts per layer. Retaining the full
# BF16 expert cache avoids re-reading experts for every one of the 130 prompts.
# It still keeps only one checkpoint resident at a time on the H200.
BF16_EXPERT_CACHE_ENTRIES = 26 * 256

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
        "tiktoken>=0.9",
        "blobfile>=3.0",
        "fastapi>=0.115",
        "pydantic>=2.7",
    )
    .env({"CUDA_HOME": "/usr/local/cuda"})
    .add_local_dir(Path(__file__).parent, remote_path="/root/engine")
    .add_local_dir(Path(__file__).parent.parent / "k3", remote_path="/root/k3")
)


@app.function(
    image=IMAGE,
    gpu="H200",
    cpu=16.0,
    memory=131072,
    timeout=6 * 60 * 60,
    volumes={QUANTIZED_MOUNT: QUANTIZED, WEIGHTS_MOUNT: WEIGHTS},
)
def analyze_router(candidate: str = "int4") -> dict:
    import torch

    from engine.accuracy.prompts import build_prompt_set, prompt_set_sha256
    from engine.klinear.model import KLinearModel
    from engine.router_analysis.capture import capture_routing_run
    from engine.router_analysis.metrics import compare_routing_runs
    from engine.router_analysis.records import EncodedPrompt
    from engine.router_analysis.report import render_markdown
    from engine.serve.contracts import ChatPrompt
    from engine.serve.klinear_engine import KimiChatTokenizer

    if candidate not in CANDIDATE_DIRS:
        raise ValueError(
            f"unknown candidate {candidate!r}; choose one of {sorted(CANDIDATE_DIRS)}"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("router analysis requires CUDA")

    device = torch.device("cuda:0")
    tokenizer = KimiChatTokenizer.from_directory(BF16_DIR)
    prompts = build_prompt_set()
    encoded_prompts = tuple(
        EncodedPrompt(
            prompt_id=prompt.prompt_id,
            category=prompt.category,
            token_ids=tuple(
                tokenizer.encode_prompt(
                    ChatPrompt(messages=({"role": "user", "content": prompt.text},))
                )
            ),
        )
        for prompt in prompts
    )
    token_ids_sha256 = _sha256_json(
        [list(prompt.token_ids) for prompt in encoded_prompts]
    )
    prompt_fingerprint = prompt_set_sha256(prompts)

    reference, reference_runtime = _capture_checkpoint(
        KLinearModel=KLinearModel,
        capture_routing_run=capture_routing_run,
        checkpoint="bf16",
        directory=BF16_DIR,
        prompts=encoded_prompts,
        prompt_fingerprint=prompt_fingerprint,
        device=device,
        expert_cache_entries=BF16_EXPERT_CACHE_ENTRIES,
    )
    candidate_run, candidate_runtime = _capture_checkpoint(
        KLinearModel=KLinearModel,
        capture_routing_run=capture_routing_run,
        checkpoint=candidate,
        directory=CANDIDATE_DIRS[candidate],
        prompts=encoded_prompts,
        prompt_fingerprint=prompt_fingerprint,
        device=device,
        expert_cache_entries=256,
    )

    report = compare_routing_runs(reference, candidate_run)
    report["prompt_token_ids_sha256"] = token_ids_sha256
    report["execution"] = {
        "engine": "engine.klinear.KLinearModel on both checkpoints",
        "device": torch.cuda.get_device_name(device),
        "load_strategy": (
            "sequential: capture BF16 to CPU, release BF16, then load and capture "
            "the selected W4A16 candidate"
        ),
        "reason": (
            "holding only compact CPU routing traces between runs avoids keeping "
            "the 91.51 GiB BF16 side and the W4A16 side resident together"
        ),
        "reference": reference_runtime,
        "candidate": candidate_runtime,
    }
    markdown = render_markdown(report)
    print(markdown)
    print(json.dumps(report, indent=2, sort_keys=True))
    return {"report": report, "markdown": markdown}


def _capture_checkpoint(
    *,
    KLinearModel,
    capture_routing_run,
    checkpoint: str,
    directory: str,
    prompts,
    prompt_fingerprint: str,
    device,
    expert_cache_entries: int,
):
    import torch

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    load_started = time.perf_counter()
    model = KLinearModel.from_directory(
        directory,
        device=device,
        dtype=torch.bfloat16,
        expert_cache_entries=expert_cache_entries,
    )
    torch.cuda.synchronize(device)
    load_seconds = time.perf_counter() - load_started
    resident_after_load = int(model.resident_weight_bytes)

    capture_started = time.perf_counter()
    run = capture_routing_run(
        model,
        checkpoint=checkpoint,
        prompt_set_sha256=prompt_fingerprint,
        prompts=prompts,
        device=device,
    )
    torch.cuda.synchronize(device)
    capture_seconds = time.perf_counter() - capture_started
    runtime = {
        "directory": directory,
        "checkpoint_kind": model.checkpoint_kind,
        "checkpoint_tensor_storage_bytes": model.checkpoint_tensor_storage_bytes,
        "resident_weight_bytes_after_load": resident_after_load,
        "resident_weight_bytes_after_capture": int(model.resident_weight_bytes),
        "load_seconds": load_seconds,
        "capture_seconds": capture_seconds,
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
    }

    provider = getattr(model, "_expert_provider", None)
    clear = getattr(provider, "clear", None)
    if callable(clear):
        clear()
    del provider, clear, model
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize(device)
    runtime["allocated_bytes_after_release"] = int(torch.cuda.memory_allocated(device))
    runtime["reserved_bytes_after_release"] = int(torch.cuda.memory_reserved(device))
    return run, runtime


def _sha256_json(value: object) -> str:
    payload = json.dumps(value, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@app.local_entrypoint()
def main(candidate: str = "int4") -> None:
    result = analyze_router.remote(candidate=candidate)
    print(result["markdown"])
