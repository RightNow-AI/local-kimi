"""Compare dense W3A16 and W4A16 GEMV at the real expert shapes.

Run from the repository root on a CUDA GPU:

    python engine/kernels/BENCH-W3A16.py

The random normal BF16 weight is quantised once per codec. Timing rotates
through enough copies of each packed matrix to exceed the configured cache
working set. Reported bandwidth credits packed weights, scales, activation,
and output bytes. It does not credit split-K partial-buffer traffic.
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from engine.kernels.w3a16_gemv import w3a16_dense_gemv  # noqa: E402
from engine.kernels.w4a16_dense_gemv import w4a16_dense_gemv  # noqa: E402
from engine.quant.w3a16 import (  # noqa: E402
    W3A16Tensor,
    dequantise as dequantise_w3,
    quantise as quantise_w3,
)
from engine.quant.w4a16 import (  # noqa: E402
    W4A16Tensor,
    dequantise as dequantise_w4,
    quantise as quantise_w4,
)

M = 1
ATOL = 0.125
RTOL = 0.05
DEFAULT_CACHE_WORKING_SET_MIB = 216


@dataclass(frozen=True)
class ProjectionShape:
    name: str
    output_size: int
    reduction: int


REAL_EXPERT_SHAPES = (
    ProjectionShape("expert W1/W3", output_size=1024, reduction=2304),
    ProjectionShape("expert W2", output_size=2304, reduction=1024),
)


@dataclass(frozen=True)
class CodecCase:
    name: str
    group_size: int
    packed: torch.Tensor
    scales: torch.Tensor
    restored: torch.Tensor
    run: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]

    @property
    def matrix_bytes(self) -> int:
        return (
            self.packed.numel() * self.packed.element_size()
            + self.scales.numel() * self.scales.element_size()
        )


@dataclass(frozen=True)
class ErrorStats:
    relative_frobenius: float
    max_abs: float


@dataclass(frozen=True)
class Timing:
    name: str
    matrix_bytes: int
    useful_bytes: int
    milliseconds: float
    bandwidth_gbps: float


def _relative_frobenius(original: torch.Tensor, restored: torch.Tensor) -> float:
    difference = restored.float() - original.float()
    numerator = torch.linalg.vector_norm(difference)
    denominator = torch.linalg.vector_norm(original.float()).clamp_min(1e-12)
    return float((numerator / denominator).item())


def _error_stats(original: torch.Tensor, restored: torch.Tensor) -> ErrorStats:
    difference = (restored.float() - original.float()).abs()
    return ErrorStats(
        relative_frobenius=_relative_frobenius(original, restored),
        max_abs=float(difference.max().item()),
    )


def _codec_cases(weight: torch.Tensor) -> tuple[CodecCase, ...]:
    w4 = quantise_w4(weight)
    w3_group32 = quantise_w3(weight, group_size=32)
    w3_group64 = quantise_w3(weight, group_size=64)

    return (
        CodecCase(
            name="W4A16 group32",
            group_size=32,
            packed=w4.packed,
            scales=w4.scales,
            restored=dequantise_w4(w4),
            run=w4a16_dense_gemv,
        ),
        CodecCase(
            name="W3A16 group32",
            group_size=32,
            packed=w3_group32.packed,
            scales=w3_group32.scales,
            restored=dequantise_w3(w3_group32),
            run=lambda activation, packed, scales: w3a16_dense_gemv(
                activation,
                packed,
                scales,
                group_size=32,
            ),
        ),
        CodecCase(
            name="W3A16 group64",
            group_size=64,
            packed=w3_group64.packed,
            scales=w3_group64.scales,
            restored=dequantise_w3(w3_group64),
            run=lambda activation, packed, scales: w3a16_dense_gemv(
                activation,
                packed,
                scales,
                group_size=64,
            ),
        ),
    )


def _bank_count(matrix_bytes: int, cache_working_set_bytes: int) -> int:
    return max(1, math.ceil(cache_working_set_bytes / matrix_bytes))


def _measure(
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


def _assert_kernel_matches_codec(
    case: CodecCase,
    activation: torch.Tensor,
) -> None:
    encoded: W3A16Tensor | W4A16Tensor
    if case.name.startswith("W3"):
        encoded = W3A16Tensor(
            packed=case.packed,
            scales=case.scales,
            original_shape=(case.packed.shape[0], activation.shape[1]),
            group_size=case.group_size,
        )
        restored = dequantise_w3(encoded)
    else:
        encoded = W4A16Tensor(
            packed=case.packed,
            scales=case.scales,
            original_shape=(case.packed.shape[0], activation.shape[1]),
        )
        restored = dequantise_w4(encoded)

    expected = F.linear(activation, restored)
    actual = case.run(activation, case.packed, case.scales)
    torch.testing.assert_close(actual, expected, atol=ATOL, rtol=RTOL)


def _time_case(
    case: CodecCase,
    activation: torch.Tensor,
    output_size: int,
    cache_working_set_bytes: int,
    warmup: int,
    iterations: int,
) -> Timing:
    count = _bank_count(case.matrix_bytes, cache_working_set_bytes)
    packed_bank = case.packed.unsqueeze(0).repeat(count, 1, 1)
    scale_bank = case.scales.unsqueeze(0).repeat(count, 1, 1)
    useful_bytes = (
        case.matrix_bytes
        + activation.numel() * activation.element_size()
        + M * output_size * torch.tensor([], dtype=torch.bfloat16).element_size()
    )
    milliseconds = _measure(
        lambda index: case.run(
            activation,
            packed_bank[index],
            scale_bank[index],
        ),
        count,
        warmup,
        iterations,
    )
    seconds = milliseconds / 1000.0
    timing = Timing(
        name=case.name,
        matrix_bytes=case.matrix_bytes,
        useful_bytes=useful_bytes,
        milliseconds=milliseconds,
        bandwidth_gbps=useful_bytes / seconds / 1e9,
    )
    del packed_bank, scale_bank
    torch.cuda.empty_cache()
    return timing


def _print_error_table(cases: tuple[CodecCase, ...], weight: torch.Tensor) -> None:
    baseline = _error_stats(weight, cases[0].restored)
    print("round-trip error against original random normal BF16 weight")
    print(f"{'codec':<20} {'relative Frobenius':>20} {'versus INT4':>14} {'max abs':>14}")
    print("-" * 72)
    for case in cases:
        errors = _error_stats(weight, case.restored)
        error_ratio = errors.relative_frobenius / max(baseline.relative_frobenius, 1e-12)
        print(
            f"{case.name:<20} {errors.relative_frobenius:>20.8f} "
            f"{error_ratio:>13.4f}x {errors.max_abs:>14.8f}"
        )


def _print_timing_table(rows: list[Timing]) -> None:
    w4_ms = rows[0].milliseconds
    print(
        f"{'codec':<20} {'matrix bytes':>14} {'useful bytes':>14} "
        f"{'ms':>10} {'GB/s':>12} {'speed vs W4':>14}"
    )
    print("-" * 94)
    for row in rows:
        speedup = w4_ms / row.milliseconds
        print(
            f"{row.name:<20} {row.matrix_bytes:>14,d} {row.useful_bytes:>14,d} "
            f"{row.milliseconds:>10.4f} {row.bandwidth_gbps:>12.2f} "
            f"{speedup:>13.4f}x"
        )


def _benchmark_shape(
    shape: ProjectionShape,
    cache_working_set_bytes: int,
    warmup: int,
    iterations: int,
) -> None:
    print()
    print(f"{shape.name}: M={M}, N={shape.output_size}, K={shape.reduction}")
    activation = torch.randn(
        (M, shape.reduction), device="cuda", dtype=torch.bfloat16
    )
    weight = torch.randn(
        (shape.output_size, shape.reduction),
        device="cuda",
        dtype=torch.bfloat16,
    )
    cases = _codec_cases(weight)
    for case in cases:
        _assert_kernel_matches_codec(case, activation)

    _print_error_table(cases, weight)
    rows = [
        _time_case(
            case,
            activation,
            shape.output_size,
            cache_working_set_bytes,
            warmup,
            iterations,
        )
        for case in cases
    ]
    print()
    print("kernel timing and achieved useful bandwidth")
    _print_timing_table(rows)
    print(
        "W3A16 extraction adds about 1.375 source-level integer operations "
        "per weight versus W4A16. The timing above decides whether fewer bytes win."
    )

    del activation, weight, cases
    torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument(
        "--cache-working-set-mib",
        type=int,
        default=DEFAULT_CACHE_WORKING_SET_MIB,
        help="packed matrix bank size used to avoid repeated L2-resident weights",
    )
    args = parser.parse_args()
    if args.warmup < 10:
        parser.error("--warmup must be at least 10")
    if args.iterations < 50:
        parser.error("--iterations must be at least 50")
    if args.cache_working_set_mib <= 0:
        parser.error("--cache-working-set-mib must be positive")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for BENCH-W3A16.py")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    cache_working_set_bytes = args.cache_working_set_mib * 1024 * 1024
    print(f"device: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    print(f"warmup: {args.warmup}, iterations: {args.iterations}")
    print(f"cache working set: {args.cache_working_set_mib} MiB per codec")
    print("bandwidth metric: cold useful traffic, decimal GB/s")
    print("quality metric: relative Frobenius error, with no acceptability claim")
    for shape in REAL_EXPERT_SHAPES:
        _benchmark_shape(
            shape,
            cache_working_set_bytes,
            args.warmup,
            args.iterations,
        )


if __name__ == "__main__":
    main()
