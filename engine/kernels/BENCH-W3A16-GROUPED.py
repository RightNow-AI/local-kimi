"""Benchmark grouped W3A16 GEMV at the real Kimi expert shapes.

Run from the repository root on an L40S or another CUDA GPU:

    python engine/kernels/BENCH-W3A16-GROUPED.py

The correctness check compares the Triton kernel with the pure PyTorch W3A16
reference. Timing rotates through disjoint routed expert sets while keeping
expert 256 in the final shared route slot. The bandwidth metric counts useful
packed weights, scales, activation reads per N tile, expert indices, outputs,
and W4 split-K partial traffic.

Every timing in this file is an isolated HYPOTHESIS. Three separate tuning
decisions in this project won an isolated benchmark and then lost end to end.
The decode benchmark is the gate for any performance claim.
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from engine.kernels.w3a16_grouped_gemv import (  # noqa: E402
    GROUPED_W3A16_CONFIG,
    grouped_w3a16_gemv,
    grouped_w3a16_gemv_reference,
)
from engine.kernels.w4a16_gemv import (  # noqa: E402
    _select_config as select_w4_config,
)
from engine.kernels.w4a16_gemv import grouped_w4a16_gemv  # noqa: E402

EXPERTS = 257
ROUTED_EXPERTS = 256
TOKENS = 1
ROUTES = 9
GROUP_SIZE = 32
L40S_PEAK_GBPS = 864.0
ATOL = 0.125
RTOL = 0.05


@dataclass(frozen=True)
class ProjectionShape:
    name: str
    output_size: int
    reduction: int


REAL_SHAPES = (
    ProjectionShape("w1/w3", output_size=1024, reduction=2304),
    ProjectionShape("w2", output_size=2304, reduction=1024),
)


@dataclass(frozen=True)
class ErrorStats:
    max_abs: float
    max_rel: float
    within_tolerance: bool
    deterministic: bool


@dataclass(frozen=True)
class Timing:
    name: str
    milliseconds: float
    bandwidth_gbps: float
    peak_fraction: float


def _make_inputs(shape: ProjectionShape) -> tuple[torch.Tensor, ...]:
    activations = torch.randn(
        (TOKENS, shape.reduction), device="cuda", dtype=torch.bfloat16
    )
    scales = (
        torch.rand(
            (
                EXPERTS,
                shape.output_size,
                shape.reduction // GROUP_SIZE,
            ),
            device="cuda",
            dtype=torch.float32,
        )
        * 0.05
        + 0.001
    ).to(torch.bfloat16)
    w3_packed = torch.randint(
        0,
        256,
        (
            EXPERTS,
            shape.output_size,
            shape.reduction // 8 * 3,
        ),
        device="cuda",
        dtype=torch.uint8,
    )
    w4_packed = torch.randint(
        0,
        256,
        (
            EXPERTS,
            shape.output_size,
            shape.reduction // 2,
        ),
        device="cuda",
        dtype=torch.uint8,
    )
    return activations, w3_packed, w4_packed, scales


def _route_index_bank() -> torch.Tensor:
    routed = torch.arange(
        ROUTED_EXPERTS, device="cuda", dtype=torch.long
    ).reshape(ROUTED_EXPERTS // (ROUTES - 1), TOKENS, ROUTES - 1)
    shared = torch.full(
        (routed.shape[0], TOKENS, 1),
        EXPERTS - 1,
        device="cuda",
        dtype=torch.long,
    )
    return torch.cat((routed, shared), dim=-1)


def _measure_indexed(
    function: Callable[[int], torch.Tensor],
    bank_count: int,
    warmup: int,
    iterations: int,
) -> float:
    warmup_launches = max(warmup, bank_count)
    for index in range(warmup_launches):
        function(index % bank_count)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for index in range(iterations):
        function(index % bank_count)
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / iterations


def _error_stats(reference: torch.Tensor, actual: torch.Tensor) -> ErrorStats:
    reference_float = reference.float()
    absolute = (actual.float() - reference_float).abs()
    relative = absolute / reference_float.abs().clamp_min(1e-6)
    tolerance = ATOL + RTOL * reference_float.abs()
    return ErrorStats(
        max_abs=float(absolute.max().item()),
        max_rel=float(relative.max().item()),
        within_tolerance=bool((absolute <= tolerance).all().item()),
        deterministic=False,
    )


def _effective_w3_bytes(shape: ProjectionShape) -> int:
    config = GROUPED_W3A16_CONFIG
    n_tiles = math.ceil(shape.output_size / config.block_n)
    packed_weights = (
        ROUTES * shape.output_size * (shape.reduction // 8 * 3)
    )
    scales = (
        ROUTES
        * shape.output_size
        * (shape.reduction // GROUP_SIZE)
        * 2
    )
    activations = ROUTES * n_tiles * shape.reduction * 2
    expert_indices = ROUTES * n_tiles * 8
    outputs = ROUTES * shape.output_size * 2
    return packed_weights + scales + activations + expert_indices + outputs


def _effective_w4_bytes(shape: ProjectionShape) -> int:
    config = select_w4_config(shape.output_size, TOKENS * ROUTES)
    n_tiles = math.ceil(shape.output_size / config.block_n)
    packed_weights = ROUTES * shape.output_size * (shape.reduction // 2)
    scales = (
        ROUTES
        * shape.output_size
        * (shape.reduction // GROUP_SIZE)
        * 2
    )
    activations = ROUTES * n_tiles * shape.reduction * 2
    expert_indices = ROUTES * n_tiles * config.split_k * 8
    outputs = ROUTES * shape.output_size * 2
    partials = 0
    if config.split_k > 1:
        partials = ROUTES * shape.output_size * config.split_k * 4 * 2
    return (
        packed_weights
        + scales
        + activations
        + expert_indices
        + outputs
        + partials
    )


def _timing(name: str, milliseconds: float, useful_bytes: int) -> Timing:
    seconds = milliseconds / 1000.0
    bandwidth_gbps = useful_bytes / seconds / 1e9
    return Timing(
        name=name,
        milliseconds=milliseconds,
        bandwidth_gbps=bandwidth_gbps,
        peak_fraction=bandwidth_gbps / L40S_PEAK_GBPS,
    )


def _print_timing_table(rows: list[Timing]) -> None:
    print(
        f"{'kernel':<24} {'ms':>10} {'GB/s':>12} "
        f"{'L40S 864 GB/s':>16}"
    )
    print("-" * 66)
    for row in rows:
        print(
            f"{row.name:<24} {row.milliseconds:>10.4f} "
            f"{row.bandwidth_gbps:>12.2f} "
            f"{row.peak_fraction * 100.0:>15.2f}%"
        )


def _benchmark_shape(shape: ProjectionShape, warmup: int, iterations: int) -> None:
    print()
    print(
        f"{shape.name}: E={EXPERTS}, tokens={TOKENS}, routes={ROUTES}, "
        f"N={shape.output_size}, K={shape.reduction}, group={GROUP_SIZE}"
    )
    activations, w3_packed, w4_packed, scales = _make_inputs(shape)
    route_bank = _route_index_bank()
    expert_indices = route_bank[0]

    reference = grouped_w3a16_gemv_reference(
        activations,
        expert_indices,
        w3_packed,
        scales,
        group_size=GROUP_SIZE,
    )
    actual = grouped_w3a16_gemv(
        activations,
        expert_indices,
        w3_packed,
        scales,
        group_size=GROUP_SIZE,
    )
    repeated = grouped_w3a16_gemv(
        activations,
        expert_indices,
        w3_packed,
        scales,
        group_size=GROUP_SIZE,
    )
    errors = _error_stats(reference, actual)
    errors = ErrorStats(
        max_abs=errors.max_abs,
        max_rel=errors.max_rel,
        within_tolerance=errors.within_tolerance,
        deterministic=torch.equal(actual, repeated),
    )
    if not errors.within_tolerance:
        raise AssertionError(
            f"W3A16 exceeds atol={ATOL} and rtol={RTOL}: "
            f"max_abs={errors.max_abs}, max_rel={errors.max_rel}"
        )
    if not errors.deterministic:
        raise AssertionError("grouped W3A16 is not deterministic run to run")

    bank_count = route_bank.shape[0]
    w3_ms = _measure_indexed(
        lambda index: grouped_w3a16_gemv(
            activations,
            route_bank[index],
            w3_packed,
            scales,
            group_size=GROUP_SIZE,
        ),
        bank_count,
        warmup,
        iterations,
    )
    w4_ms = _measure_indexed(
        lambda index: grouped_w4a16_gemv(
            activations,
            route_bank[index],
            w4_packed,
            scales,
        ),
        bank_count,
        warmup,
        iterations,
    )
    rows = [
        _timing("grouped W3A16", w3_ms, _effective_w3_bytes(shape)),
        _timing("grouped W4A16", w4_ms, _effective_w4_bytes(shape)),
    ]

    print(
        f"reference max abs={errors.max_abs:.8f}, "
        f"max rel={errors.max_rel:.8f}, deterministic={errors.deterministic}"
    )
    _print_timing_table(rows)
    print(f"W3 speed relative to W4: {w4_ms / w3_ms:.4f}x")
    print(
        "HYPOTHESIS: this isolated timing does not select the production path. "
        "The end-to-end decode benchmark is the gate."
    )

    del activations, w3_packed, w4_packed, scales
    del route_bank, reference, actual, repeated
    torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260729)
    args = parser.parse_args()
    if args.warmup < 10:
        parser.error("--warmup must be at least 10")
    if args.iterations < 50:
        parser.error("--iterations must be at least 50")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for BENCH-W3A16-GROUPED.py")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    print(f"device: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    print(f"warmup: {args.warmup}, iterations: {args.iterations}")
    print("bandwidth metric: effective useful traffic, decimal GB/s")
    print("shared expert: bank row 256 in the final route slot")
    print(
        "HYPOTHESIS: isolated kernel timing only. Three prior isolated winners "
        "lost end to end, so the decode benchmark is the gate."
    )
    for shape in REAL_SHAPES:
        _benchmark_shape(shape, args.warmup, args.iterations)


if __name__ == "__main__":
    main()
