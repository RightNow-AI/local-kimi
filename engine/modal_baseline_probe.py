"""Boot the vLLM baseline for Kimi-Linear and answer the counterfactual.

Before claiming our engine is worth anything, two questions have to be answered
with a running process rather than a search engine:

1. Does a tuned vLLM actually serve this model, and what does it cost in VRAM?
   That is the baseline every later comparison is measured against.
2. What does a buyer get for free if they skip us entirely? Specifically,
   whether vLLM will serve this architecture QUANTIZED, because our entire
   footprint argument dies if it will.

Question 2 is the one that matters commercially, so it is tested first and its
failure is recorded as a finding rather than swallowed.

    modal run engine/modal_baseline_probe.py

Weights come from the volume, already fetched, so download time stays out of
every number here.
"""

from __future__ import annotations

import json

import modal

app = modal.App("k3-baseline-probe")

VOLUME = modal.Volume.from_name("kimi-linear-weights", create_if_missing=False)
MOUNT = "/weights"
MODEL_DIR = f"{MOUNT}/Kimi-Linear-48B-A3B-Instruct"

# A CUDA devel base, because debian_slim is not enough for this model. There it
# loaded all 20 shards and then died at engine core init with "Could not find
# nvcc and default cuda_home='/usr/local/cuda' doesn't exist": Kimi-Linear's KDA
# path JIT-compiles kernels at startup, so a CUDA toolchain has to be present at
# RUN time. Worth recording as a deployment fact about the architecture rather
# than quietly patching around.
IMAGE = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12"
    )
    .entrypoint([])
    .apt_install("git")
    .pip_install("vllm==0.26.0", "huggingface_hub>=0.26")
    .env({"VLLM_USE_V1": "1", "CUDA_HOME": "/usr/local/cuda"})
)


@app.function(image=IMAGE, volumes={MOUNT: VOLUME}, timeout=60 * 20)
def counterfactual() -> dict:
    """What does vLLM say about quantizing this architecture, without a GPU.

    Import-level inspection: which quantization methods vLLM knows, and what the
    model class advertises. A cheap answer to an expensive question.
    """
    out: dict[str, object] = {}
    try:
        from vllm import __version__ as vllm_version

        out["vllm_version"] = vllm_version
    except Exception as exc:
        out["vllm_version_error"] = repr(exc)

    try:
        from vllm.model_executor.layers.quantization import QUANTIZATION_METHODS

        out["quantization_methods"] = sorted(QUANTIZATION_METHODS)
    except Exception as exc:
        out["quantization_methods_error"] = repr(exc)

    try:
        from vllm.model_executor.models.registry import ModelRegistry

        archs = ModelRegistry.get_supported_archs()
        out["kimi_archs_supported"] = sorted(a for a in archs if "Kimi" in a)
    except Exception as exc:
        out["registry_error"] = repr(exc)

    # The decisive detail: does the kimi_linear implementation thread a
    # quant_config into its MoE and its KDA layers, or ignore it.
    try:
        import inspect

        from vllm.model_executor.models import kimi_linear

        src = inspect.getsource(kimi_linear)
        out["kimi_linear_source_chars"] = len(src)
        out["kimi_linear_quant_config_mentions"] = src.count("quant_config")
        out["kimi_linear_mentions_unquantized_moe"] = "UnquantizedFusedMoEMethod" in src
        interesting = [
            line.strip()
            for line in src.splitlines()
            if "quant" in line.lower() and "config" in line.lower()
        ]
        out["kimi_linear_quant_lines"] = interesting[:40]
    except Exception as exc:
        out["kimi_linear_source_error"] = repr(exc)

    print(json.dumps(out, indent=2))
    return out


@app.function(
    image=IMAGE,
    gpu="H200",
    volumes={MOUNT: VOLUME},
    timeout=60 * 45,
    memory=65536,
)
def boot(max_num_seqs: int = 16, max_model_len: int = 32768) -> dict:
    """Actually load the model and generate, recording what it costs.

    max_num_seqs is passed explicitly and recorded, because on a hybrid linear
    attention model the recurrent state is a fixed pool sized by that number.
    A memory figure without it is not a fact about anything.
    """
    import time

    import torch
    from vllm import LLM, SamplingParams

    out: dict[str, object] = {
        "max_num_seqs": max_num_seqs,
        "max_model_len": max_model_len,
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_total_bytes": torch.cuda.get_device_properties(0).total_memory,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
    }

    started = time.time()
    try:
        llm = LLM(
            model=MODEL_DIR,
            trust_remote_code=True,
            max_model_len=max_model_len,
            max_num_seqs=max_num_seqs,
            gpu_memory_utilization=0.90,
            enforce_eager=False,
        )
        out["load_seconds"] = round(time.time() - started, 2)
        out["loaded"] = True
    except Exception as exc:
        out["loaded"] = False
        out["load_error"] = repr(exc)[:4000]
        out["load_seconds"] = round(time.time() - started, 2)
        print(json.dumps(out, indent=2))
        return out

    out["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
    out["peak_reserved_bytes"] = torch.cuda.max_memory_reserved()

    prompts = [
        "Explain in two sentences why memory bandwidth limits decoding speed.",
        "What is 17 * 23?",
    ]
    t0 = time.time()
    results = llm.generate(
        prompts,
        SamplingParams(temperature=0.0, max_tokens=64),
    )
    out["generate_seconds"] = round(time.time() - t0, 2)
    out["outputs"] = [
        {
            "prompt": r.prompt,
            "text": r.outputs[0].text,
            "n_prompt_tokens": len(r.prompt_token_ids),
            "n_output_tokens": len(r.outputs[0].token_ids),
        }
        for r in results
    ]
    out["peak_allocated_bytes_after_gen"] = torch.cuda.max_memory_allocated()
    out["peak_reserved_bytes_after_gen"] = torch.cuda.max_memory_reserved()

    print(json.dumps(out, indent=2))
    return out


@app.local_entrypoint()
def main(max_num_seqs: int = 16, max_model_len: int = 32768):
    print("=== counterfactual, no GPU ===")
    print(json.dumps(counterfactual.remote(), indent=2))
    print("=== boot on H200 ===")
    print(json.dumps(boot.remote(max_num_seqs=max_num_seqs, max_model_len=max_model_len), indent=2))
