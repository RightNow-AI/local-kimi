from __future__ import annotations

import math

import pytest
import torch


def test_fused_qkv_real_shape_arithmetic() -> None:
    from engine.kernels.w4a16_fused_qkv import _fused_qkv_grid_shape

    m = 1
    n = 4096
    k = 2304
    group_size = 32
    block_n = 64

    assert k % group_size == 0
    assert (n, k // 2) == (4096, 1152)
    assert (n, k // group_size) == (4096, 72)
    assert 3 * n == 12288
    assert _fused_qkv_grid_shape(n, block_n) == (3, 64)
    assert math.prod(_fused_qkv_grid_shape(n, block_n)) == 192
    assert (m, n) == (1, 4096)


def _gpu_inputs(m: int) -> tuple[torch.Tensor, ...]:
    output_size = 256
    reduction = 256
    activations = torch.randn(
        (m, reduction), device="cuda", dtype=torch.bfloat16
    )
    projections: list[torch.Tensor] = []
    for _ in range(3):
        packed = torch.randint(
            0,
            256,
            (output_size, reduction // 2),
            device="cuda",
            dtype=torch.uint8,
        )
        scales = (
            torch.rand(
                (output_size, reduction // 32),
                device="cuda",
                dtype=torch.float32,
            )
            * 0.05
            + 0.001
        ).to(torch.bfloat16)
        projections.extend((packed, scales))
    return (activations, *projections)


@pytest.mark.gpu
def test_fused_qkv_matches_shipped_three_call_reference() -> None:
    pytest.importorskip("triton")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")

    from engine.kernels.w4a16_fused_qkv import (
        fused_qkv_w4a16,
        fused_qkv_w4a16_reference,
    )

    torch.manual_seed(20260729)
    inputs = _gpu_inputs(m=1)
    expected = fused_qkv_w4a16_reference(*inputs)
    actual = fused_qkv_w4a16(*inputs)

    for expected_projection, actual_projection in zip(expected, actual, strict=True):
        torch.testing.assert_close(
            actual_projection,
            expected_projection,
            atol=0.125,
            rtol=0.05,
        )


@pytest.mark.gpu
def test_fused_qkv_is_deterministic_run_to_run() -> None:
    pytest.importorskip("triton")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")

    from engine.kernels.w4a16_fused_qkv import fused_qkv_w4a16

    torch.manual_seed(20260730)
    inputs = _gpu_inputs(m=1)
    first = fused_qkv_w4a16(*inputs)
    second = fused_qkv_w4a16(*inputs)

    assert all(
        torch.equal(first_projection, second_projection)
        for first_projection, second_projection in zip(first, second, strict=True)
    )


@pytest.mark.gpu
def test_fused_qkv_prefill_uses_the_unchanged_dense_paths() -> None:
    pytest.importorskip("triton")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")

    from engine.kernels.w4a16_fused_qkv import (
        fused_qkv_w4a16,
        fused_qkv_w4a16_reference,
    )

    torch.manual_seed(20260731)
    inputs = _gpu_inputs(m=3)
    expected = fused_qkv_w4a16_reference(*inputs)
    actual = fused_qkv_w4a16(*inputs)

    assert all(
        torch.equal(expected_projection, actual_projection)
        for expected_projection, actual_projection in zip(expected, actual, strict=True)
    )
