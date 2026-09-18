"""Benchmark fused W2 route combination at the real Kimi K3 decode shape.

This isolated result is a HYPOTHESIS, not a production result. Two earlier
tuning choices in this project won isolated benchmarks and then lost end to
end. The decode benchmark is the gate for accepting this fusion.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from pathlib import Path

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from engine.kernels.moe_combine import (  # noqa: E402
    LAUNCH_CONFIG,
    fused_w2_combine,
)
from engine.kernels.w4a16_gemv import grouped_w4a16_gemv  # noqa: E402

TOKENS = 1
ROUTES = 9
ROUTED_EXPERTS = 8
EXPERTS = 257
HIDDEN_SIZE = 2304
INTERMEDIATE_SIZE = 1024
MOE_LAYERS = 26

# The shipped split-K W2 path launches a partial kernel and a reduction kernel.
# Eager combination then launches cast, multiply, route sum, cast, and add.
BASELINE_LAUNCHES = 7
FUSED_LAUNCHES = 1
TAIL_ONLY_LAUNCHES_SAVED = 5


def _current_sequence(
    activated: torch.Tensor,
    expert_indices: torch.Tensor,
    combine_weights: torch.Tensor,
    w2_packed: torch.Tensor,
    w2_scales: torch.Tensor,
) -> torch.Tensor:
    expert_outputs = grouped_w4a16_gemv(
        activated.reshape(-1, INTERMEDIATE_SIZE),
        expert_indices.reshape(-1, 1),
        w2_packed,
        w2_scales,
    ).reshape(TOKENS, ROUTES, HIDDEN_SIZE)
    routed = (
        expert_outputs[:, :ROUTED_EXPERTS]
        .float()
        .mul(combine_weights[:, :ROUTED_EXPERTS].float().unsqueeze(-1))
        .sum(dim=1)
        .to(activated.dtype)
    )
    return routed + expert_outputs[:, -1]


def _measure_ms(
    operation: Callable[[], torch.Tensor],
    *,
    warmup: int,
    iterations: int,
) -> float:
    for _ in range(warmup):
        operation()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        operation()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / iterations


def _error_stats(
    reference: torch.Tensor,
    candidate: torch.Tensor,
) -> tuple[float, float]:
    difference = (candidate.float() - reference.float()).abs()
    relative = difference / reference.float().abs().clamp_min(1.0e-6)
    return tuple(torch.stack((difference.max(), relative.max())).cpu().tolist())


def _make_inputs(device: torch.device) -> tuple[torch.Tensor, ...]:
    torch.manual_seed(20260729)
    torch.cuda.manual_seed_all(20260729)
    activated = torch.empty(
        (TOKENS, ROUTES, INTERMEDIATE_SIZE),
        device=device,
        dtype=torch.bfloat16,
    ).normal_(mean=0.0, std=0.5)
    expert_indices = torch.tensor(
        [[0, 1, 2, 3, 4, 5, 6, 7, EXPERTS - 1]],
        device=device,
        dtype=torch.long,
    )
    routed_weights = torch.rand(
        (TOKENS, ROUTED_EXPERTS), device=device, dtype=torch.float32
    )
    routed_weights /= routed_weights.sum(dim=1, keepdim=True)
    combine_weights = torch.cat(
        (
            routed_weights,
            torch.ones((TOKENS, 1), device=device, dtype=torch.float32),
        ),
        dim=1,
    )
    w2_packed = torch.randint(
        0,
        256,
        (EXPERTS, HIDDEN_SIZE, INTERMEDIATE_SIZE // 2),
        device=device,
        dtype=torch.uint8,
    )
    w2_scales = torch.empty(
        (EXPERTS, HIDDEN_SIZE, INTERMEDIATE_SIZE // 32),
        device=device,
        dtype=torch.bfloat16,
    ).uniform_(0.004, 0.02)
    return activated, expert_indices, combine_weights, w2_packed, w2_scales


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()
    if args.warmup < 10:
        parser.error("--warmup must be at least 10")
    if args.iterations < 50:
        parser.error("--iterations must be at least 50")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for BENCH-COMBINE.py")

    device = torch.device("cuda")
    inputs = _make_inputs(device)

    def baseline() -> torch.Tensor:
        return _current_sequence(*inputs)

    def fused() -> torch.Tensor:
        return fused_w2_combine(*inputs)

    baseline_output = baseline()
    fused_output = fused()
    repeated_output = fused()
    torch.cuda.synchronize()
    if not torch.equal(fused_output, repeated_output):
        raise AssertionError("fused W2 combine is not deterministic run to run")
    max_abs, max_rel = _error_stats(baseline_output, fused_output)

    baseline_ms = _measure_ms(
        baseline,
        warmup=args.warmup,
        iterations=args.iterations,
    )
    fused_ms = _measure_ms(
        fused,
        warmup=args.warmup,
        iterations=args.iterations,
    )

    launches_saved = BASELINE_LAUNCHES - FUSED_LAUNCHES
    per_layer_saving_us = (baseline_ms - fused_ms) * 1000.0
    intermediate_bytes = TOKENS * ROUTES * HIDDEN_SIZE * 2

    print(f"GPU: {torch.cuda.get_device_name(device)}")
    print(
        f"Shape: tokens={TOKENS}, routes={ROUTES}, experts={EXPERTS}, "
        f"hidden={HIDDEN_SIZE}, intermediate={INTERMEDIATE_SIZE}"
    )
    print(f"Fused launch configuration: {LAUNCH_CONFIG} adapted to one program")
    print(f"Warmup: {args.warmup}, iterations: {args.iterations}")
    print("Relative error denominator is clamped to 1e-6.")
    print()
    print("Accuracy against the current W2 plus eager combination sequence")
    print(f"  max abs error: {max_abs:.8g}")
    print(f"  max rel error: {max_rel:.8g}")
    print("  deterministic repeated output: yes")
    print()
    print("Mean GPU time")
    print(f"  current sequence: {baseline_ms * 1000.0:.3f} us")
    print(f"  fused kernel:     {fused_ms * 1000.0:.3f} us")
    print(f"  isolated ratio:   {baseline_ms / fused_ms:.3f}x")
    print()
    print("Static launch and traffic model")
    print(
        f"  estimated launches: {BASELINE_LAUNCHES} current, "
        f"{FUSED_LAUNCHES} fused, {launches_saved} saved per layer"
    )
    print(
        f"  estimated launches saved across {MOE_LAYERS} layers: "
        f"{launches_saved * MOE_LAYERS}"
    )
    print(
        f"  conservative tail-only count: {TAIL_ONLY_LAUNCHES_SAVED} saved per "
        f"layer, {TAIL_ONLY_LAUNCHES_SAVED * MOE_LAYERS} across the model"
    )
    print(
        f"  eliminated expert-output tensor: {intermediate_bytes / 1024.0:.1f} KiB"
    )
    print(
        f"  minimum eliminated write plus read: "
        f"{2 * intermediate_bytes / 1024.0:.1f} KiB per layer"
    )
    print()
    print(f"Projected over {MOE_LAYERS} MoE layers for one decoded token")
    print(f"  measured isolated delta per layer: {per_layer_saving_us:.3f} us")
    print(
        f"  arithmetic projection across layers: "
        f"{per_layer_saving_us * MOE_LAYERS:.3f} us"
    )
    print()
    print("HYPOTHESIS ONLY: the end-to-end decode benchmark is the acceptance gate.")
    print(
        "The fusion may lose if the accumulator lifetime across nine routes "
        "reduces occupancy enough to outweigh the removed traffic and launches."
    )


if __name__ == "__main__":
    main()
