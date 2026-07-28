"""Quantise and benchmark real K3 dense weights on an H100.

Correctness gates run before timing. The fused path decodes signed INT4 values
inside the Triton GEMM and never materializes a BF16 weight matrix.

    modal run engine/modal_w4a16.py --layer 12 --batch-sizes 1,8,32
"""

from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

import modal

app = modal.App("k3-w4a16")

IMAGE = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch>=2.5", "numpy>=2.0", "triton>=3.1")
    .add_local_dir(Path(__file__).parent, remote_path="/root/engine")
)

WEIGHTS = modal.Volume.from_name("k3-weights", create_if_missing=True)
VOL = "/weights"

# Both paths accumulate in FP32 and return BF16. This allowance is only for
# reduction-order drift between cuBLAS and Triton, not quantization error.
FUSED_GEMM_ATOL = 0.08
FUSED_GEMM_RTOL = 0.03


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


def _parse_batch_sizes(value: str) -> tuple[int, ...]:
    try:
        batch_sizes = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as exc:
        raise ValueError("batch_sizes must be a comma-separated list of integers") from exc
    if not batch_sizes or any(batch <= 0 for batch in batch_sizes):
        raise ValueError("batch_sizes must contain positive integers")
    return batch_sizes


@app.function(
    image=IMAGE,
    gpu="H100",
    volumes={VOL: WEIGHTS},
    timeout=60 * 30,
)
def validate_and_benchmark(
    layer: int = 12,
    batch_sizes: str = "1,8,32",
    warmup: int = 5,
    iterations: int = 20,
) -> dict:
    import gc
    import os

    import torch
    import torch.nn.functional as F
    from engine.k3ref.manifest import K3_LAYER_TENSOR_MANIFEST
    from engine.k3ref.weights import RawTensorStore
    from engine.quant.plan import build_quantization_plan
    from engine.quant.triton_w4a16 import w4a16_linear
    from engine.quant.verify import (
        error_statistics,
        run_negative_control_checks,
        verify_round_trip,
    )
    from engine.quant.w4a16 import dequantise, quantise

    if layer < 0:
        raise ValueError("layer must be non-negative")
    if warmup < 0 or iterations <= 0:
        raise ValueError("warmup cannot be negative and iterations must be positive")
    batches = _parse_batch_sizes(batch_sizes)
    directory = f"{VOL}/layer{layer}"
    if not os.path.isdir(directory):
        raise FileNotFoundError(
            f"{directory} is missing; fetch layer {layer} into k3-weights first"
        )

    torch.manual_seed(20260728)
    torch.cuda.manual_seed_all(20260728)
    store = RawTensorStore(directory)
    layer_plan = build_quantization_plan(include_lm_head=False)
    selected = [item for item in layer_plan.tensors if item.quantize]
    negative_controls = run_negative_control_checks()

    tensor_results = []
    class_actual = {}
    aggregate_timings = {
        batch: {"bf16_seconds": 0.0, "w4a16_seconds": 0.0}
        for batch in batches
    }
    with torch.inference_mode():
        for decision in selected:
            suffix = f"layers.{layer}.{decision.name}"
            store.validate(suffix, K3_LAYER_TENSOR_MANIFEST[decision.name])
            weight = store.load(suffix, device="cuda")
            encoded = quantise(weight)
            round_trip = verify_round_trip(decision.tensor_class, weight, encoded)
            actual_original_bytes = weight.numel() * weight.element_size()
            actual_quantized_bytes = encoded.storage_bytes
            if actual_original_bytes != decision.original_bytes:
                raise AssertionError(
                    f"{decision.name} original bytes disagree with the manifest plan"
                )
            if actual_quantized_bytes != decision.planned_bytes:
                raise AssertionError(
                    f"{decision.name} W4A16 bytes disagree with the manifest plan"
                )

            class_entry = class_actual.setdefault(
                decision.tensor_class,
                {"original_bytes": 0, "w4a16_bytes": 0},
            )
            class_entry["original_bytes"] += actual_original_bytes
            class_entry["w4a16_bytes"] += actual_quantized_bytes

            decoded_weight = dequantise(encoded)
            benchmarks = []
            for batch in batches:
                activations = torch.randn(
                    batch,
                    weight.shape[1],
                    device="cuda",
                    dtype=torch.bfloat16,
                )
                explicit_quantized = F.linear(activations, decoded_weight)
                fused_quantized = w4a16_linear(activations, encoded)
                torch.testing.assert_close(
                    fused_quantized,
                    explicit_quantized,
                    atol=FUSED_GEMM_ATOL,
                    rtol=FUSED_GEMM_RTOL,
                )
                bf16_output = F.linear(activations, weight)
                output_error = error_statistics(
                    decision.tensor_class,
                    bf16_output,
                    fused_quantized,
                    max_allowed_abs_error=None,
                )

                bf16_seconds = _bench(
                    lambda: F.linear(activations, weight),
                    warmup=warmup,
                    iterations=iterations,
                )
                w4a16_seconds = _bench(
                    lambda: w4a16_linear(activations, encoded),
                    warmup=warmup,
                    iterations=iterations,
                )
                aggregate_timings[batch]["bf16_seconds"] += bf16_seconds
                aggregate_timings[batch]["w4a16_seconds"] += w4a16_seconds
                benchmarks.append(
                    {
                        "batch": batch,
                        "bf16_median_ms": round(bf16_seconds * 1e3, 4),
                        "w4a16_median_ms": round(w4a16_seconds * 1e3, 4),
                        "speedup": round(bf16_seconds / w4a16_seconds, 4),
                        "bf16_rows_per_second": round(batch / bf16_seconds, 2),
                        "w4a16_rows_per_second": round(batch / w4a16_seconds, 2),
                        "fused_vs_explicit_tolerance": {
                            "atol": FUSED_GEMM_ATOL,
                            "rtol": FUSED_GEMM_RTOL,
                        },
                        "quantized_vs_bf16_output_error": output_error.as_dict(),
                    }
                )

            tensor_results.append(
                {
                    "name": decision.name,
                    "class": decision.tensor_class,
                    "shape": list(weight.shape),
                    "round_trip": round_trip.as_dict(),
                    "storage": {
                        "bf16_bytes": actual_original_bytes,
                        "w4a16_bytes": actual_quantized_bytes,
                        "saved_bytes": actual_original_bytes - actual_quantized_bytes,
                        "saving_fraction": (
                            actual_original_bytes - actual_quantized_bytes
                        )
                        / actual_original_bytes,
                    },
                    "benchmarks": benchmarks,
                }
            )
            del activations, decoded_weight, encoded, weight
            gc.collect()
            torch.cuda.empty_cache()

    for values in class_actual.values():
        values["saved_bytes"] = values["original_bytes"] - values["w4a16_bytes"]
        values["saving_fraction"] = values["saved_bytes"] / values["original_bytes"]

    actual_original = sum(item["original_bytes"] for item in class_actual.values())
    actual_w4a16 = sum(item["w4a16_bytes"] for item in class_actual.values())
    aggregate_benchmarks = []
    for batch, timings in aggregate_timings.items():
        bf16_seconds = timings["bf16_seconds"]
        w4a16_seconds = timings["w4a16_seconds"]
        aggregate_benchmarks.append(
            {
                "batch": batch,
                "scope": "sum of per-tensor medians for selected layer projections",
                "bf16_median_ms_sum": round(bf16_seconds * 1e3, 4),
                "w4a16_median_ms_sum": round(w4a16_seconds * 1e3, 4),
                "speedup": round(bf16_seconds / w4a16_seconds, 4),
            }
        )
    output = {
        "gpu": torch.cuda.get_device_name(0),
        "layer": layer,
        "batch_sizes": list(batches),
        "warmup": warmup,
        "iterations": iterations,
        "correctness_gate": {
            "negative_controls_rejected": negative_controls,
            "weight_tolerance": (
                "per element: stored_scale / 2 + "
                "2 * BF16 epsilon * group_absmax"
            ),
            "fused_gemm_atol": FUSED_GEMM_ATOL,
            "fused_gemm_rtol": FUSED_GEMM_RTOL,
        },
        "actual_layer_storage": {
            "by_class": class_actual,
            "bf16_bytes": actual_original,
            "w4a16_bytes": actual_w4a16,
            "saved_bytes": actual_original - actual_w4a16,
            "saving_fraction": (actual_original - actual_w4a16) / actual_original,
        },
        "aggregate_selected_projection_benchmarks": aggregate_benchmarks,
        "plan": build_quantization_plan().as_dict(),
        "tensors": tensor_results,
    }
    print(json.dumps(output, indent=2))
    return output


@app.local_entrypoint()
def main(
    layer: int = 12,
    batch_sizes: str = "1,8,32",
    warmup: int = 5,
    iterations: int = 20,
):
    validate_and_benchmark.remote(
        layer=layer,
        batch_sizes=batch_sizes,
        warmup=warmup,
        iterations=iterations,
    )
