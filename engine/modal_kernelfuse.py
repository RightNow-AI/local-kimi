"""Validate and benchmark the fused K3 MXFP4 expert path on an H100.

The benchmark includes all three expert projections and the SiTU activation.
The unfused baseline decodes each packed tensor through the canonical decoder
inside every timed call, so it measures the materialization cost this kernel
exists to remove.

    modal run engine/modal_kernelfuse.py --layer 12 --expert 0 --batch 32
"""

from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

import modal

app = modal.App("k3-kernelfuse")

IMAGE = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch>=2.5", "numpy>=2.0", "triton>=3.1")
    .add_local_dir(Path(__file__).parent / "kernels", remote_path="/root/kernels")
    .add_local_dir(Path(__file__).parent / "k3ref", remote_path="/root/k3ref")
)

WEIGHTS = modal.Volume.from_name("k3-weights", create_if_missing=True)
VOL = "/weights"

# Both paths round decoded weights to the activation dtype and accumulate in
# FP32. The tolerance allows only reduction-order drift, with a larger absolute
# allowance for BF16 values near zero.
GEMM_TOLERANCES = {
    "float16": {"atol": 0.02, "rtol": 0.015},
    "bfloat16": {"atol": 0.08, "rtol": 0.03},
}
EXPERT_TOLERANCES = {
    "float16": {"atol": 0.05, "rtol": 0.03},
    "bfloat16": {"atol": 0.25, "rtol": 0.06},
}


def _bench(fn, *, warmup: int, iterations: int) -> float:
    import torch

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(iterations):
        torch.cuda.synchronize()
        started = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        samples.append(time.perf_counter() - started)
    return statistics.median(samples)


def _load_expert(store, *, layer: int, expert: int, device: str):
    from k3ref.manifest import K3_EXPERT_CHECKPOINT_MANIFEST
    from kernels.moe_grouped import PackedExpertWeights

    base = f"layers.{layer}.block_sparse_moe.experts.{expert}"

    def load(projection: str, suffix: str):
        tensor_suffix = f"{base}.{projection}.{suffix}"
        manifest_key = f"block_sparse_moe.experts.{{expert}}.{projection}.{suffix}"
        store.validate(tensor_suffix, K3_EXPERT_CHECKPOINT_MANIFEST[manifest_key])
        return store.load(tensor_suffix, device=device)

    return PackedExpertWeights(
        w1_packed=load("w1", "weight_packed"),
        w1_scale=load("w1", "weight_scale"),
        w2_packed=load("w2", "weight_packed"),
        w2_scale=load("w2", "weight_scale"),
        w3_packed=load("w3", "weight_packed"),
        w3_scale=load("w3", "weight_scale"),
    )


@app.function(
    image=IMAGE,
    gpu="H100",
    volumes={VOL: WEIGHTS},
    timeout=60 * 30,
)
def validate_and_benchmark(
    layer: int = 12,
    expert: int = 0,
    batch: int = 32,
    warmup: int = 5,
    iterations: int = 20,
    benchmark_dtype: str = "bfloat16",
) -> dict:
    import os

    import torch
    import torch.nn.functional as F
    from k3ref.dequant import dequantize_mxfp4
    from k3ref.weights import RawTensorStore
    from kernels.moe_grouped import mxfp4_expert_mlp
    from kernels.mxfp4_gemm import mxfp4_linear
    from kernels.reference import situ_reference

    directory = f"{VOL}/layer{layer}"
    if not os.path.isdir(directory):
        raise FileNotFoundError(
            f"{directory} is missing; fetch layer {layer} into k3-weights first"
        )
    if benchmark_dtype not in GEMM_TOLERANCES:
        raise ValueError("benchmark_dtype must be float16 or bfloat16")
    if batch <= 0 or warmup < 0 or iterations <= 0:
        raise ValueError("batch and iterations must be positive; warmup cannot be negative")

    device = "cuda"
    store = RawTensorStore(directory)
    weights = _load_expert(store, layer=layer, expert=expert, device=device)
    dtype_by_name = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }

    def canonical_linear(x, packed, scale):
        decoded = dequantize_mxfp4(packed, scale, dtype=x.dtype)
        return F.linear(x, decoded)

    def canonical_expert(x):
        gate = canonical_linear(x, weights.w1_packed, weights.w1_scale)
        up = canonical_linear(x, weights.w3_packed, weights.w3_scale)
        activated = situ_reference(gate, up, beta=4.0, linear_beta=25.0)
        return canonical_linear(activated, weights.w2_packed, weights.w2_scale)

    torch.manual_seed(20260728)
    correctness = {}
    with torch.inference_mode():
        for dtype_name, dtype in dtype_by_name.items():
            projection_tolerance = GEMM_TOLERANCES[dtype_name]
            projection_cases = {
                "w1": (
                    torch.randn(7, 3584, device=device, dtype=dtype),
                    weights.w1_packed,
                    weights.w1_scale,
                ),
                "w3": (
                    torch.randn(7, 3584, device=device, dtype=dtype),
                    weights.w3_packed,
                    weights.w3_scale,
                ),
                "w2": (
                    torch.randn(7, 3072, device=device, dtype=dtype),
                    weights.w2_packed,
                    weights.w2_scale,
                ),
            }
            projection_results = {}
            for name, (projection_input, packed, scale) in projection_cases.items():
                expected = canonical_linear(projection_input, packed, scale)
                actual = mxfp4_linear(projection_input, packed, scale)
                torch.testing.assert_close(
                    actual,
                    expected,
                    atol=projection_tolerance["atol"],
                    rtol=projection_tolerance["rtol"],
                )
                absolute_error = (actual.float() - expected.float()).abs()
                relative_error = absolute_error / expected.float().abs().clamp_min(1e-6)
                projection_results[name] = {
                    "max_abs_error": float(absolute_error.max()),
                    "max_rel_error": float(relative_error.max()),
                }

            expert_input = torch.randn(7, 3584, device=device, dtype=dtype)
            expected_expert = canonical_expert(expert_input)
            actual_expert = mxfp4_expert_mlp(expert_input, weights)
            expert_tolerance = EXPERT_TOLERANCES[dtype_name]
            torch.testing.assert_close(
                actual_expert,
                expected_expert,
                atol=expert_tolerance["atol"],
                rtol=expert_tolerance["rtol"],
            )
            expert_absolute_error = (
                actual_expert.float() - expected_expert.float()
            ).abs()
            correctness[dtype_name] = {
                "gemm_tolerance": projection_tolerance,
                "expert_tolerance": expert_tolerance,
                "projections": projection_results,
                "expert_max_abs_error": float(expert_absolute_error.max()),
            }

        benchmark_torch_dtype = dtype_by_name[benchmark_dtype]
        benchmark_input = torch.randn(
            batch, 3584, device=device, dtype=benchmark_torch_dtype
        )

        def fused():
            return mxfp4_expert_mlp(benchmark_input, weights)

        def unfused():
            return canonical_expert(benchmark_input)

        fused_seconds = _bench(fused, warmup=warmup, iterations=iterations)
        unfused_seconds = _bench(unfused, warmup=warmup, iterations=iterations)

    output = {
        "gpu": torch.cuda.get_device_name(0),
        "layer": layer,
        "expert": expert,
        "batch": batch,
        "benchmark_dtype": benchmark_dtype,
        "correctness": correctness,
        "benchmark": {
            "scope": "three fused GEMMs plus SiTU versus three canonical decodes plus GEMMs",
            "warmup": warmup,
            "iterations": iterations,
            "fused_median_ms": round(fused_seconds * 1e3, 4),
            "unfused_median_ms": round(unfused_seconds * 1e3, 4),
            "speedup": round(unfused_seconds / fused_seconds, 3),
        },
    }
    print(json.dumps(output, indent=2))
    return output


@app.local_entrypoint()
def main(
    layer: int = 12,
    expert: int = 0,
    batch: int = 32,
    warmup: int = 5,
    iterations: int = 20,
    benchmark_dtype: str = "bfloat16",
):
    validate_and_benchmark.remote(
        layer=layer,
        expert=expert,
        batch=batch,
        warmup=warmup,
        iterations=iterations,
        benchmark_dtype=benchmark_dtype,
    )
