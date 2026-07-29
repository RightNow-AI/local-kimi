"""Repeatable greedy decode benchmark and token-equivalence gate on L40S.

    modal run engine/modal_decode_bench.py
"""

from __future__ import annotations

import json
from pathlib import Path

import modal

app = modal.App("kimi-linear-decode-bench")

VOLUME = modal.Volume.from_name("kimi-linear-quantized", create_if_missing=False)
MOUNT = "/weights"
MODEL_DIR = f"{MOUNT}/Kimi-Linear-48B-A3B-Instruct-W4A16"
ARTIFACT_BYTES = 28_803_304_448

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


@app.function(
    image=IMAGE,
    gpu="L40S",
    cpu=8.0,
    memory=65536,
    timeout=60 * 60,
    volumes={MOUNT: VOLUME},
)
def benchmark_decode(
    prompt: str = "The main benefit of lower model memory is",
    max_new_tokens: int = 64,
    repeats: int = 3,
    budget_gib: float = 32.0,
) -> dict:
    import time

    import torch
    from transformers import AutoTokenizer

    from engine.klinear.generate import (
        CUDAGraphDecodeRunner,
        _eager_generate_from_prefill,
        decode,
        generate_tokens,
        prefill,
    )
    from engine.klinear.model import KLinearModel

    if max_new_tokens <= 0 or repeats <= 0:
        raise ValueError("max_new_tokens and repeats must be positive")
    device = torch.device("cuda:0")
    properties = torch.cuda.get_device_properties(device)
    budget_bytes = int(budget_gib * 1024**3)
    if budget_bytes > properties.total_memory:
        raise ValueError("the requested budget exceeds the GPU memory")
    torch.cuda.set_per_process_memory_fraction(
        budget_bytes / properties.total_memory, device
    )
    torch.cuda.reset_peak_memory_stats(device)

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_DIR, trust_remote_code=True, local_files_only=True
    )
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
    model = KLinearModel.from_directory(
        MODEL_DIR, device=device, dtype=torch.bfloat16
    )
    if model.resident_weight_bytes != ARTIFACT_BYTES:
        raise ValueError("grouped decode changed the resident checkpoint byte count")

    with torch.inference_mode():
        reference_output = prefill(model, input_ids)
        reference_tokens = []
        torch.cuda.synchronize(device)
        reference_started = time.perf_counter()
        for _ in range(max_new_tokens):
            token = reference_output.logits[:, -1].argmax(dim=-1)
            reference_tokens.append(token)
            reference_output = decode(
                model, token.unsqueeze(1), reference_output.state
            )
        torch.cuda.synchronize(device)
        reference_seconds = time.perf_counter() - reference_started
        reference_ids = torch.stack(reference_tokens, dim=1)

        eager_prefill = prefill(model, input_ids)
        torch.cuda.synchronize(device)
        eager_started = time.perf_counter()
        eager = _eager_generate_from_prefill(
            model,
            input_ids,
            eager_prefill,
            max_new_tokens,
            temperature=0.0,
            top_p=1.0,
            generator=None,
        )
        torch.cuda.synchronize(device)
        eager_seconds = time.perf_counter() - eager_started

        graph_prefill = prefill(model, input_ids)
        torch.cuda.synchronize(device)
        graph_setup_started = time.perf_counter()
        runner = CUDAGraphDecodeRunner(
            model, input_ids, graph_prefill, max_new_tokens
        )
        torch.cuda.synchronize(device)
        graph_setup_seconds = time.perf_counter() - graph_setup_started
        graph_runs = []
        graph_ids = None
        for _ in range(repeats):
            runner.reset()
            torch.cuda.synchronize(device)
            started = time.perf_counter()
            runner.replay()
            torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - started
            graph_ids = runner.result().generated_ids.clone()
            graph_runs.append(
                {
                    "seconds": round(elapsed, 6),
                    "tokens_per_second": round(max_new_tokens / elapsed, 4),
                }
            )

        torch.cuda.synchronize(device)
        streaming_started = time.perf_counter()
        stream = generate_tokens(
            model,
            input_ids,
            max_new_tokens,
            temperature=0.0,
        )
        streamed_tokens = []
        while True:
            try:
                streamed_tokens.append(next(stream))
            except StopIteration:
                break
        torch.cuda.synchronize(device)
        streaming_seconds = time.perf_counter() - streaming_started
        streaming_ids = torch.stack(streamed_tokens, dim=1)

    eager_equal = torch.equal(eager.generated_ids, reference_ids)
    graph_equal = torch.equal(graph_ids, reference_ids)
    streaming_equal = torch.equal(streaming_ids, reference_ids)
    if not eager_equal or not graph_equal or not streaming_equal:
        raise AssertionError("optimized greedy token IDs diverged from growing decode")
    record = {
        "gpu": properties.name,
        "budget_gib": budget_gib,
        "budget_bytes": budget_bytes,
        "artifact_bytes": ARTIFACT_BYTES,
        "resident_weight_bytes": model.resident_weight_bytes,
        "prompt": prompt,
        "prompt_tokens": int(input_ids.shape[1]),
        "generated_tokens": max_new_tokens,
        "repeats": repeats,
        "growing_eager": {
            "seconds": round(reference_seconds, 6),
            "tokens_per_second": round(max_new_tokens / reference_seconds, 4),
        },
        "preallocated_eager": {
            "seconds": round(eager_seconds, 6),
            "tokens_per_second": round(max_new_tokens / eager_seconds, 4),
            "same_token_ids": eager_equal,
        },
        "cuda_graph_replay": graph_runs,
        "cuda_graph_setup_seconds": round(graph_setup_seconds, 6),
        "cuda_graph_best_tokens_per_second": max(
            run["tokens_per_second"] for run in graph_runs
        ),
        "cuda_graph_median_tokens_per_second": sorted(
            run["tokens_per_second"] for run in graph_runs
        )[len(graph_runs) // 2],
        "streaming_end_to_end": {
            "seconds": round(streaming_seconds, 6),
            "tokens_per_second": round(max_new_tokens / streaming_seconds, 4),
            "includes_prefill_and_graph_setup": True,
            "same_token_ids": streaming_equal,
        },
        "same_token_ids": graph_equal,
        "generated_token_ids": reference_ids[0].detach().cpu().tolist(),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
    }
    print(json.dumps(record, indent=2, sort_keys=True))
    return record


@app.local_entrypoint()
def main(
    prompt: str = "The main benefit of lower model memory is",
    max_new_tokens: int = 64,
    repeats: int = 3,
    budget_gib: float = 32.0,
) -> None:
    print(
        json.dumps(
            benchmark_decode.remote(
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                repeats=repeats,
                budget_gib=budget_gib,
            ),
            indent=2,
            sort_keys=True,
        )
    )
