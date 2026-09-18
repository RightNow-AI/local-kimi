"""Benchmark one-launch fused QKV W4A16 GEMV at the real KDA shape.

Run from the repository root on an L40S or another CUDA GPU:

    python engine/kernels/BENCH-QKV.py

The weight bank is larger than three L40S L2 caches so repeated timings do not
turn the three projection matrices into an L2-resident synthetic workload.
Bandwidth is minimum logical traffic. It credits packed weights, scales,
activation inputs, and outputs, but not split-K partial traffic in the shipped
three-call path.
"""

from __future__ import annotations

import argparse
import math
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from engine.kernels.w4a16_dense_gemv import _select_dense_config  # noqa: E402
from engine.kernels.w4a16_fused_qkv import (  # noqa: E402
    FUSED_QKV_CONFIGS,
    _fused_qkv_grid_shape,
    _launch_fused_qkv_w4a16,
    _select_fused_qkv_config,
    fused_qkv_w4a16_reference,
)
from engine.quant.w4a16 import GROUP_SIZE  # noqa: E402

M = 1
N = 4096
K = 2304
PROJECTIONS = 3
KDA_LAYERS = 20
L40S_PEAK_GBPS = 864.0
L40S_SMS = 142
L40S_L2_BYTES = 96 * 1024 * 1024
CACHE_WORKING_SET_BYTES = 3 * L40S_L2_BYTES


@dataclass(frozen=True)
class Timing:
    name: str
    milliseconds: float
    bandwidth_gbps: float
    peak_fraction: float


def _projection_storage_bytes() -> int:
    packed = N * (K // 2)
    scales = N * (K // GROUP_SIZE) * 2
    return packed + scales


def _bank_count() -> int:
    qkv_storage = PROJECTIONS * _projection_storage_bytes()
    return max(1, math.ceil(CACHE_WORKING_SET_BYTES / qkv_storage))


def _logical_byte_counts() -> tuple[int, int]:
    """Return minimum logical bytes for the three-call and fused operators."""
    packed_weights = PROJECTIONS * N * (K // 2)
    scales = PROJECTIONS * N * (K // GROUP_SIZE) * 2
    outputs = PROJECTIONS * M * N * 2
    activation = M * K * 2
    three_call = packed_weights + scales + outputs + PROJECTIONS * activation
    fused = packed_weights + scales + outputs + activation
    return three_call, fused


def _make_inputs() -> tuple[torch.Tensor, ...]:
    count = _bank_count()
    activations = torch.randn((M, K), device="cuda", dtype=torch.bfloat16)
    packed_shape = (count, N, K // 2)
    scale_shape = (count, N, K // GROUP_SIZE)
    packed = tuple(
        torch.randint(0, 256, packed_shape, device="cuda", dtype=torch.uint8)
        for _ in range(PROJECTIONS)
    )
    scales = tuple(
        (
            torch.rand(scale_shape, device="cuda", dtype=torch.float32) * 0.05
            + 0.001
        ).to(torch.bfloat16)
        for _ in range(PROJECTIONS)
    )
    return activations, packed[0], scales[0], packed[1], scales[1], packed[2], scales[2]


def _measure_ms(
    operation: Callable[[int], tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    *,
    bank_count: int,
    warmup: int,
    iterations: int,
) -> float:
    warmup_launches = max(warmup, bank_count)
    for index in range(warmup_launches):
        operation(index % bank_count)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for index in range(iterations):
        operation(index % bank_count)
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / iterations


def _error_stats(
    reference: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    candidate: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> tuple[float, float]:
    absolute_maxima = []
    relative_maxima = []
    for expected, actual in zip(reference, candidate, strict=True):
        difference = (actual.float() - expected.float()).abs()
        relative = difference / expected.float().abs().clamp_min(1.0e-6)
        absolute_maxima.append(difference.max())
        relative_maxima.append(relative.max())
    values = torch.stack(
        (torch.stack(absolute_maxima).max(), torch.stack(relative_maxima).max())
    ).cpu()
    return tuple(values.tolist())


def _timing(name: str, milliseconds: float, byte_count: int) -> Timing:
    bandwidth = byte_count / (milliseconds * 1.0e6)
    return Timing(
        name=name,
        milliseconds=milliseconds,
        bandwidth_gbps=bandwidth,
        peak_fraction=bandwidth / L40S_PEAK_GBPS,
    )


def _print_timing(row: Timing) -> None:
    print(
        f"  {row.name:<20} {row.milliseconds * 1000.0:>9.3f} us, "
        f"{row.bandwidth_gbps:>7.1f} GB/s, "
        f"{row.peak_fraction * 100.0:>5.1f}% of L40S peak"
    )


def _print_grid_report() -> None:
    dense_config = _select_dense_config(N)
    dense_n_tiles = (N + dense_config.block_n - 1) // dense_config.block_n
    dense_main_programs = dense_n_tiles * dense_config.split_k
    dense_reduction_programs = dense_n_tiles if dense_config.split_k > 1 else 0
    fused_config = _select_fused_qkv_config(N)
    fused_grid = _fused_qkv_grid_shape(N, fused_config.block_n)
    fused_programs = math.prod(fused_grid)

    print("Grid size")
    print(
        f"  shipped three-call config: {dense_config.name}, "
        f"3 x grid=({dense_n_tiles}, {dense_config.split_k})"
    )
    if dense_reduction_programs:
        print(
            f"  shipped reduction grids: 3 x ({dense_reduction_programs},), "
            f"{PROJECTIONS * (dense_main_programs + dense_reduction_programs)} "
            "programs across 6 launches"
        )
    else:
        print(
            f"  shipped total: {PROJECTIONS * dense_main_programs} programs "
            "across 3 launches"
        )
    print(
        f"  fused default: grid={fused_grid}, {fused_programs} programs in 1 launch"
    )
    print()
    print("BLOCK_N candidates for the end-to-end sweep")
    for config in FUSED_QKV_CONFIGS:
        grid = _fused_qkv_grid_shape(N, config.block_n)
        programs = math.prod(grid)
        waves = programs / L40S_SMS
        marker = " default" if config == fused_config else ""
        print(
            f"  {config.name}: grid={grid}, programs={programs}, "
            f"SM waves={waves:.2f}{marker}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260729)
    args = parser.parse_args()
    if args.warmup < 10:
        parser.error("--warmup must be at least 10")
    if args.iterations < 50:
        parser.error("--iterations must be at least 50")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for BENCH-QKV.py")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    inputs = _make_inputs()
    activations, q_packed, q_scales, k_packed, k_scales, v_packed, v_scales = inputs
    count = _bank_count()
    fused_config = _select_fused_qkv_config(N)

    def three_call(index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return fused_qkv_w4a16_reference(
            activations,
            q_packed[index],
            q_scales[index],
            k_packed[index],
            k_scales[index],
            v_packed[index],
            v_scales[index],
        )

    def fused(index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return _launch_fused_qkv_w4a16(
            activations,
            q_packed[index],
            q_scales[index],
            k_packed[index],
            k_scales[index],
            v_packed[index],
            v_scales[index],
            fused_config,
        )

    reference_output = three_call(0)
    fused_output = fused(0)
    repeated_output = fused(0)
    torch.cuda.synchronize()
    if not all(
        torch.equal(first, second)
        for first, second in zip(fused_output, repeated_output, strict=True)
    ):
        raise AssertionError("the fused QKV kernel is not deterministic run to run")
    max_abs, max_rel = _error_stats(reference_output, fused_output)

    three_call_ms = _measure_ms(
        three_call,
        bank_count=count,
        warmup=args.warmup,
        iterations=args.iterations,
    )
    fused_ms = _measure_ms(
        fused,
        bank_count=count,
        warmup=args.warmup,
        iterations=args.iterations,
    )
    three_call_bytes, fused_bytes = _logical_byte_counts()
    three_call_timing = _timing("three dense calls", three_call_ms, three_call_bytes)
    fused_timing = _timing("one fused launch", fused_ms, fused_bytes)

    matrix_mb = _projection_storage_bytes() / 1.0e6
    working_set_mb = count * PROJECTIONS * matrix_mb
    print(f"GPU: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    print(f"Shape: M={M}, N={N}, K={K}, projections={PROJECTIONS}")
    print(f"Warmup: {args.warmup}, iterations: {args.iterations}")
    print(
        f"Cold-weight bank: {count} QKV sets, {working_set_mb:.2f} MB total"
    )
    print("Bandwidth metric: minimum logical traffic, decimal GB/s")
    print()
    print("Accuracy against the shipped three-call path")
    print(f"  max abs error: {max_abs:.8g}")
    print(f"  max rel error: {max_rel:.8g}")
    print("  deterministic run to run: yes")
    print()
    print("Mean GPU time and achieved bandwidth")
    _print_timing(three_call_timing)
    _print_timing(fused_timing)
    print(f"  isolated speedup: {three_call_ms / fused_ms:.3f}x")
    print()
    _print_grid_report()
    print()
    print(f"Scaled linearly to {KDA_LAYERS} KDA layers per decoded token")
    print(f"  three-call sequence: {three_call_ms * KDA_LAYERS:.4f} ms")
    print(f"  fused sequence:      {fused_ms * KDA_LAYERS:.4f} ms")
    print(
        f"  projected saving:    {(three_call_ms - fused_ms) * KDA_LAYERS:.4f} ms"
    )
    print()
    print(
        "This isolated timing is a HYPOTHESIS. The end-to-end decode benchmark "
        "is the gate because three tuning decisions won in isolation and lost "
        "in the engine."
    )


if __name__ == "__main__":
    main()
