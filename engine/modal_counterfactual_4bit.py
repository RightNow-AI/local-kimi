"""The experiment that decides whether this product has a reason to exist.

Our commercial argument is footprint: that Kimi-Linear can be served in roughly
24 GB where the shipped BF16 needs 91.51 GiB, putting it on one consumer card
instead of a multi-GPU node. That argument assumed the existing tooling could
not get there.

Inspecting vLLM 0.26.0 undercut the assumption. It exposes 31 quantization
methods including awq_marlin, gptq_marlin and moe_wna16, it registers
KimiLinearForCausalLM, its kimi_linear implementation threads quant_config
through 35 call sites, and it never falls back to UnquantizedFusedMoEMethod. On
paper a buyer can already do this without us.

Paper is not a running process. Threading a config through a constructor is not
the same as the kernels supporting this architecture's fused MoE at 256 experts,
its KDA linear-attention layers, or its MLA. So this asks the hardware.

Three outcomes, and all three are useful:
  - It works and is fast: our footprint claim is dead. We say so and pivot.
  - It loads but is unusably slow or wrong: we have a real, demonstrable gap.
  - It refuses to load: we have the strongest version of the claim, measured.

    modal run engine/modal_counterfactual_4bit.py
"""

from __future__ import annotations

import json

import modal

app = modal.App("k3-counterfactual-4bit")

VOLUME = modal.Volume.from_name("kimi-linear-weights", create_if_missing=False)
MOUNT = "/weights"
MODEL_DIR = f"{MOUNT}/Kimi-Linear-48B-A3B-Instruct"

# Two image decisions, both learned by failing first.
#
# vllm is PINNED. Left unpinned alongside bitsandbytes, pip resolved vllm down to
# 0.19.1, which predates KimiLinearForCausalLM support. A load failure under that
# version would say nothing about 4-bit and everything about the version, so the
# experiment would answer the wrong question convincingly.
#
# The base is a CUDA devel image, not debian_slim. On debian_slim this model
# loaded all 20 shards and then died at engine core init with "Could not find
# nvcc and default cuda_home='/usr/local/cuda' doesn't exist". Kimi-Linear's KDA
# path JIT-compiles kernels at startup, so this architecture needs a toolchain
# present at RUN time, not just at build time. That is a real deployment
# characteristic of the model and it belongs in the notes, not just in the image.
IMAGE = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12"
    )
    .entrypoint([])
    .apt_install("git")
    .pip_install("vllm==0.26.0", "bitsandbytes>=0.45", "huggingface_hub>=0.26")
    .env({"VLLM_USE_V1": "1", "CUDA_HOME": "/usr/local/cuda"})
)

PROMPTS = [
    "Explain in two sentences why memory bandwidth limits decoding speed.",
    "What is 17 * 23? Answer with just the number.",
    "Write a Python function that reverses a linked list.",
]


def _attempt(quantization: str | None, load_format: str | None,
             max_num_seqs: int, max_model_len: int) -> dict:
    """Load Kimi-Linear under one configuration and report what happened."""
    import time

    import torch
    from vllm import LLM, SamplingParams

    label = quantization or "bf16-as-shipped"
    out: dict[str, object] = {
        "config": label,
        "quantization": quantization,
        "load_format": load_format,
        "max_num_seqs": max_num_seqs,
        "max_model_len": max_model_len,
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_total_bytes": torch.cuda.get_device_properties(0).total_memory,
    }

    torch.cuda.reset_peak_memory_stats()
    started = time.time()
    kwargs: dict[str, object] = {
        "model": MODEL_DIR,
        "trust_remote_code": True,
        "max_model_len": max_model_len,
        "max_num_seqs": max_num_seqs,
        "gpu_memory_utilization": 0.92,
    }
    if quantization:
        kwargs["quantization"] = quantization
    if load_format:
        kwargs["load_format"] = load_format

    try:
        llm = LLM(**kwargs)
    except Exception as exc:
        out["loaded"] = False
        out["load_seconds"] = round(time.time() - started, 2)
        out["error_type"] = type(exc).__name__
        out["error"] = repr(exc)[:6000]
        return out

    out["loaded"] = True
    out["load_seconds"] = round(time.time() - started, 2)
    out["peak_allocated_bytes_after_load"] = torch.cuda.max_memory_allocated()
    out["peak_reserved_bytes_after_load"] = torch.cuda.max_memory_reserved()

    try:
        t0 = time.time()
        results = llm.generate(PROMPTS, SamplingParams(temperature=0.0, max_tokens=96))
        elapsed = time.time() - t0
        produced = sum(len(r.outputs[0].token_ids) for r in results)
        out["generate_seconds"] = round(elapsed, 2)
        out["output_tokens"] = produced
        out["tokens_per_second_aggregate"] = round(produced / elapsed, 2) if elapsed else None
        out["outputs"] = [
            {"prompt": r.prompt[:80], "text": r.outputs[0].text[:400]} for r in results
        ]
        out["peak_reserved_bytes_after_gen"] = torch.cuda.max_memory_reserved()
    except Exception as exc:
        out["generated"] = False
        out["generate_error"] = repr(exc)[:4000]

    return out


@app.function(image=IMAGE, gpu="H200", volumes={MOUNT: VOLUME}, timeout=60 * 60, memory=131072)
def four_bit(max_num_seqs: int = 16, max_model_len: int = 8192) -> dict:
    """Can a buyer get Kimi-Linear to 4-bit with stock vLLM and no work.

    bitsandbytes is the honest version of this question because it quantizes in
    flight from the BF16 checkpoint, so it needs no pre-quantized artifact and
    no calibration. If this path works, the footprint argument is over.
    """
    result = _attempt("bitsandbytes", "bitsandbytes", max_num_seqs, max_model_len)
    print(json.dumps(result, indent=2, default=str))
    return result


@app.function(image=IMAGE, gpu="H200", volumes={MOUNT: VOLUME}, timeout=60 * 60, memory=131072)
def baseline(max_num_seqs: int = 16, max_model_len: int = 8192) -> dict:
    """The as-shipped BF16 side, same GPU and settings, for the comparison."""
    result = _attempt(None, None, max_num_seqs, max_model_len)
    print(json.dumps(result, indent=2, default=str))
    return result


@app.local_entrypoint()
def main(max_num_seqs: int = 16, max_model_len: int = 8192):
    out = {
        "four_bit_bitsandbytes": four_bit.remote(
            max_num_seqs=max_num_seqs, max_model_len=max_model_len
        ),
        "baseline_bf16": baseline.remote(
            max_num_seqs=max_num_seqs, max_model_len=max_model_len
        ),
    }
    print("=== COUNTERFACTUAL RESULT ===")
    print(json.dumps(out, indent=2, default=str))
