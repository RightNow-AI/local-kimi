"""Serve Kimi-Linear-48B inside a 32 GiB budget, where vLLM cannot go at all.

THE CLAIM THIS EXISTS TO PROVE, and it is a capability claim rather than a speed
one. Stock vLLM 0.26.0 refuses to serve this model below BF16: measured on an
H200, the bitsandbytes 4-bit path is rejected at load because
KimiLinearForCausalLM declares no packed_modules_mapping. So on vLLM the floor
is 98,245,528,576 bytes of weights, and a 32 GB card cannot run this model at
any speed. Our INT4 artifact is 28,803,304,448 bytes and our engine loads it.

The difference is therefore binary, not incremental, and a throughput comparison
against vLLM on a datacenter GPU measures a race that is beside the point.
Nobody who owns an H200 is the customer for this.

HOW THE BUDGET IS ENFORCED, because a peak-memory reading on a 48 GB card would
prove nothing about a 32 GB one. torch.cuda.set_per_process_memory_fraction caps
this process at exactly 32 GiB of the device, computed from the device's real
total. Every allocation past that raises rather than spilling, so if the model
does not fit, this job FAILS. It cannot quietly succeed by using memory a
consumer card would not have.

The card is an L40S because it is the cheapest Modal GPU with more than 32 GiB,
and the extra capacity is deliberately made unusable by the cap. This is a
demonstration under a hard budget, not a measurement on consumer silicon, and
the record says so.

    modal run engine/modal_consumer_card.py
    modal run engine/modal_consumer_card.py --budget-gib 24
"""

from __future__ import annotations

import json
from pathlib import Path

import modal

app = modal.App("kimi-linear-consumer-card")

QUANTIZED = modal.Volume.from_name("kimi-linear-quantized", create_if_missing=False)
WEIGHTS = modal.Volume.from_name("kimi-linear-weights", create_if_missing=False)
QUANTIZED_MOUNT = "/quantized"
WEIGHTS_MOUNT = "/weights"
INT4_DIR = f"{QUANTIZED_MOUNT}/Kimi-Linear-48B-A3B-Instruct-W4A16"
TOKENIZER_DIR = f"{WEIGHTS_MOUNT}/Kimi-Linear-48B-A3B-Instruct"

# CUDA devel base: Kimi-Linear's KDA path JIT-compiles kernels at startup and
# dies at engine init without nvcc. tiktoken and blobfile are required by
# Moonshot's tokenizer and transformers does not pull them. numpy is explicit
# because torch degrades quietly without it.
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
        # engine.serve.contracts cannot be imported without these: the package
        # __init__ pulls in api.py, which imports FastAPI. This job needs only
        # the ChatPrompt dataclass and the tokenizer, but Python does not let
        # you take part of a package.
        "fastapi>=0.115",
        "pydantic>=2.7",
    )
    .env({"CUDA_HOME": "/usr/local/cuda"})
    .add_local_dir(Path(__file__).parent, remote_path="/root/engine")
    # k3 is required, not optional. engine/serve/klinear_engine reuses
    # k3/toolcalls.py for the K2-family tool-call parser rather than writing a
    # third implementation of it, so the serving package does not import
    # without this.
    .add_local_dir(Path(__file__).parent.parent / "k3", remote_path="/root/k3")
)

PROMPTS = (
    "The main benefit of running a large model locally is",
    "Write a Python function that reverses a linked list.",
    "Explain in two sentences why memory bandwidth limits decoding speed.",
)


@app.function(
    image=IMAGE,
    gpu="L40S",
    cpu=8.0,
    memory=65536,
    timeout=60 * 60,
    volumes={QUANTIZED_MOUNT: QUANTIZED, WEIGHTS_MOUNT: WEIGHTS},
)
def serve_within_budget(budget_gib: float = 32.0, max_new_tokens: int = 48) -> dict:
    """Load and generate with the process hard-capped at ``budget_gib``."""
    import time

    import torch

    from engine.klinear.model import KLinearModel

    device = torch.device("cuda")
    properties = torch.cuda.get_device_properties(0)
    total_bytes = properties.total_memory
    budget_bytes = int(budget_gib * (1024**3))

    if budget_bytes > total_bytes:
        raise ValueError(
            f"budget {budget_gib} GiB exceeds the device's {total_bytes} bytes; "
            "this job cannot demonstrate a budget larger than the card"
        )

    # The hard cap. Past this the allocator raises instead of spilling, so a
    # model that does not fit produces a failure rather than a flattering number.
    fraction = budget_bytes / total_bytes
    torch.cuda.set_per_process_memory_fraction(fraction, 0)
    torch.cuda.reset_peak_memory_stats(device)

    record: dict[str, object] = {
        "claim": (
            "serve Kimi-Linear-48B-A3B-Instruct from the selective INT4 artifact "
            "inside a hard memory budget a consumer card would provide"
        ),
        "gpu_name": properties.name,
        "gpu_total_bytes": total_bytes,
        "budget_gib": budget_gib,
        "budget_bytes": budget_bytes,
        "enforced_fraction": round(fraction, 6),
        "enforcement": (
            "torch.cuda.set_per_process_memory_fraction; allocations beyond the "
            "budget raise rather than spilling, so this cannot pass by using "
            "memory a consumer card would not have"
        ),
        "is_consumer_silicon": False,
        "honest_scope": (
            "This is a demonstration under a hard budget on a datacenter card, "
            "not a measurement on consumer silicon. It establishes that the "
            "working set fits the budget, not that a specific consumer GPU "
            "reaches any particular speed."
        ),
    }

    started = time.perf_counter()
    try:
        model = KLinearModel.from_directory(
            INT4_DIR,
            device=device,
            dtype=torch.bfloat16,
            expert_cache_entries=256,
        )
    except torch.cuda.OutOfMemoryError as exc:
        record["fits"] = False
        record["failure"] = f"OutOfMemoryError within the budget: {exc}"[:2000]
        record["load_seconds"] = round(time.perf_counter() - started, 2)
        print(json.dumps(record, indent=2, sort_keys=True))
        return record

    torch.cuda.synchronize(device)
    record["load_seconds"] = round(time.perf_counter() - started, 2)
    resident = model.resident_weight_bytes
    record["resident_weight_bytes"] = int(resident() if callable(resident) else resident)
    record["peak_allocated_after_load_bytes"] = int(torch.cuda.max_memory_allocated(device))
    record["peak_reserved_after_load_bytes"] = int(torch.cuda.max_memory_reserved(device))

    from engine.klinear.generate import generate
    from engine.serve.contracts import ChatPrompt
    from engine.serve.klinear_engine import KimiChatTokenizer

    tokenizer = KimiChatTokenizer.from_directory(TOKENIZER_DIR)
    eos_ids = set(tokenizer.eos_token_ids)
    generations = []
    for prompt in PROMPTS:
        chat = ChatPrompt(messages=({"role": "user", "content": prompt},))
        prompt_ids = tokenizer.encode_prompt(chat)
        prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        t0 = time.perf_counter()
        result = generate(model, prompt_tensor, max_new_tokens=max_new_tokens)
        elapsed = time.perf_counter() - t0
        token_ids = [int(t) for t in result.generated_ids.flatten().tolist()]
        # generate() runs to max_new_tokens; trim at the first stop token so the
        # decoded text is what a server would actually return.
        for index, token in enumerate(token_ids):
            if token in eos_ids:
                token_ids = token_ids[:index]
                break
        # Decode through the tokenizer's own byte view so a multi-byte
        # character split across two tokens is not corrupted.
        text = b"".join(tokenizer.token_bytes(t) for t in token_ids).decode(
            "utf-8", errors="replace"
        )
        generations.append(
            {
                "prompt": prompt,
                "prompt_token_count": len(prompt_ids),
                "generated_token_count": len(token_ids),
                "seconds": round(elapsed, 3),
                "tokens_per_second": round(len(token_ids) / elapsed, 2) if elapsed else None,
                "text": text,
            }
        )

    record["fits"] = True
    record["generations"] = generations
    record["peak_allocated_after_generation_bytes"] = int(
        torch.cuda.max_memory_allocated(device)
    )
    record["peak_reserved_after_generation_bytes"] = int(
        torch.cuda.max_memory_reserved(device)
    )
    record["headroom_bytes"] = budget_bytes - record["peak_reserved_after_generation_bytes"]
    record["throughput_note"] = (
        "Tokens per second here is a single-stream reference-implementation "
        "figure under a memory cap. It is NOT a tuned serving throughput claim "
        "and must not be quoted as one."
    )
    record["coherence_assessed"] = False

    print(json.dumps(record, indent=2, sort_keys=True))
    return record


@app.local_entrypoint()
def main(budget_gib: float = 32.0, max_new_tokens: int = 48):
    print(
        json.dumps(
            serve_within_budget.remote(
                budget_gib=budget_gib, max_new_tokens=max_new_tokens
            ),
            indent=2,
            sort_keys=True,
        )
    )
