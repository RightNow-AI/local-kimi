"""Profile the real fixed-shape Kimi-Linear decode graph on an L40S.

    modal run engine/modal_decode_profile.py
"""

from __future__ import annotations

import json
from pathlib import Path

import modal

app = modal.App("kimi-linear-decode-profile")

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
        "tiktoken>=0.9",
        "blobfile>=3.0",
    )
    .env({"CUDA_HOME": "/usr/local/cuda"})
    .add_local_dir(Path(__file__).parent, remote_path="/root/engine")
    .add_local_dir(Path(__file__).parent.parent / "k3", remote_path="/root/k3")
)


def _event_device_name(event) -> str:
    return str(getattr(event, "device_type", "")).lower()


@app.function(
    image=IMAGE,
    gpu="L40S",
    cpu=8.0,
    memory=65536,
    timeout=60 * 60,
    volumes={MOUNT: VOLUME},
)
def profile_decode(
    prompt: str = "Explain why memory bandwidth limits autoregressive decoding.",
    max_new_tokens: int = 64,
    budget_gib: float = 32.0,
) -> dict:
    import time

    import torch
    from transformers import AutoTokenizer

    from engine.klinear.generate import CUDAGraphDecodeRunner, prefill
    from engine.klinear.model import KLinearModel

    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    device = torch.device("cuda:0")
    properties = torch.cuda.get_device_properties(device)
    budget_bytes = int(budget_gib * 1024**3)
    if budget_bytes > properties.total_memory:
        raise ValueError("the requested budget exceeds the GPU memory")
    torch.cuda.set_per_process_memory_fraction(
        budget_bytes / properties.total_memory, device
    )

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_DIR, trust_remote_code=True, local_files_only=True
    )
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
    model = KLinearModel.from_directory(
        MODEL_DIR, device=device, dtype=torch.bfloat16
    )
    with torch.inference_mode():
        prefetched = prefill(model, input_ids)
        torch.cuda.synchronize(device)
        setup_started = time.perf_counter()
        runner = CUDAGraphDecodeRunner(
            model, input_ids, prefetched, max_new_tokens
        )
        torch.cuda.synchronize(device)
        graph_setup_seconds = time.perf_counter() - setup_started
        runner.reset()
        torch.cuda.synchronize(device)
        activities = [
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
        with torch.profiler.profile(
            activities=activities,
            record_shapes=True,
            profile_memory=True,
        ) as profiler:
            started = time.perf_counter()
            runner.replay()
            torch.cuda.synchronize(device)
            wall_seconds = time.perf_counter() - started
        result = runner.result()

    averages = list(profiler.key_averages())
    averages.sort(
        key=lambda event: getattr(event, "self_device_time_total", 0.0),
        reverse=True,
    )
    top_costs = [
        {
            "name": event.key,
            "calls": int(event.count),
            "self_cuda_us": round(
                float(getattr(event, "self_device_time_total", 0.0)), 3
            ),
            "total_cuda_us": round(
                float(getattr(event, "device_time_total", 0.0)), 3
            ),
            "self_cpu_us": round(float(event.self_cpu_time_total), 3),
        }
        for event in averages[:10]
    ]
    events = list(profiler.events())
    cuda_events = [event for event in events if "cuda" in _event_device_name(event)]
    synchronization_names = (
        "synchronize",
        "_local_scalar_dense",
        "item",
        "dtoh",
        "device-to-host",
    )
    synchronization_events = [
        event.name
        for event in events
        if any(name in event.name.lower() for name in synchronization_names)
    ]
    torch_cuda_us = sum(
        float(getattr(event, "self_device_time_total", 0.0))
        for event in averages
    )
    generated_ids = result.generated_ids[0].detach().cpu().tolist()
    record = {
        "gpu": properties.name,
        "budget_gib": budget_gib,
        "prompt_tokens": int(input_ids.shape[1]),
        "generated_tokens": max_new_tokens,
        "decode_backend": result.decode_backend,
        "cuda_graph_setup_seconds": round(graph_setup_seconds, 6),
        "total_wall_seconds": round(wall_seconds, 6),
        "wall_ms_per_token": round(wall_seconds * 1000 / max_new_tokens, 4),
        "tokens_per_second": round(max_new_tokens / wall_seconds, 4),
        "torch_cuda_ms_per_token": round(
            torch_cuda_us / 1000 / max_new_tokens, 4
        ),
        "cuda_kernel_events": len(cuda_events),
        "cuda_kernel_events_per_token": round(
            len(cuda_events) / max_new_tokens, 3
        ),
        "host_device_sync_events": synchronization_events,
        "explicit_boundary_synchronizations": 1,
        "per_token_explicit_synchronizations": 0,
        "top_10_costs": top_costs,
        "generated_token_ids": generated_ids,
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
    }
    print(json.dumps(record, indent=2, sort_keys=True))
    return record


@app.local_entrypoint()
def main(
    prompt: str = "Explain why memory bandwidth limits autoregressive decoding.",
    max_new_tokens: int = 64,
    budget_gib: float = 32.0,
) -> None:
    print(
        json.dumps(
            profile_decode.remote(
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                budget_gib=budget_gib,
            ),
            indent=2,
            sort_keys=True,
        )
    )
