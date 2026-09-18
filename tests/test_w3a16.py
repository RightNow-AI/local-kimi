from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from engine.quant.w3a16 import (
    W3A16Tensor,
    _pack_int3_codes,
    _unpack_int3_codes,
    dequantise,
    quantise,
)


def test_int3_bit_layout_is_exact() -> None:
    codes = torch.tensor([[-4, -3, -2, -1, 0, 1, 2, 3]], dtype=torch.int16)

    packed = _pack_int3_codes(codes)

    # The 24-bit word is 0x688FAC. Storage is little endian by byte.
    assert torch.equal(packed, torch.tensor([[0xAC, 0x8F, 0x68]], dtype=torch.uint8))


def test_int3_sign_extension_covers_the_full_range() -> None:
    expected = torch.tensor([[-4, -3, -2, -1, 0, 1, 2, 3]], dtype=torch.int16)
    packed = torch.tensor([[0xAC, 0x8F, 0x68]], dtype=torch.uint8)

    unpacked = _unpack_int3_codes(packed, (1, 8))
    encoded = W3A16Tensor(
        packed=packed,
        scales=torch.ones((1, 1), dtype=torch.bfloat16),
        original_shape=(1, 8),
        group_size=8,
    )

    assert torch.equal(unpacked, expected)
    assert torch.equal(dequantise(encoded), expected.to(torch.bfloat16))


@pytest.mark.parametrize("group_size", [32, 64])
def test_round_trip_stays_within_half_a_stored_step(group_size: int) -> None:
    torch.manual_seed(20260729 + group_size)
    weight = torch.randn((4, group_size * 3), dtype=torch.bfloat16)

    encoded = quantise(weight, group_size=group_size)
    restored = dequantise(encoded, dtype=torch.float32)

    grouped_error = (restored - weight.float()).abs().reshape(
        weight.shape[0], weight.shape[1] // group_size, group_size
    )
    group_max_error = grouped_error.amax(dim=-1)
    half_step = encoded.scales.float().abs() / 2.0
    arithmetic_slack = torch.finfo(torch.float32).eps * weight.float().abs().amax(
        dim=-1, keepdim=True
    )
    assert bool((group_max_error <= half_step + arithmetic_slack).all())


def test_reduction_axis_validates_packing_and_scale_divisibility() -> None:
    with pytest.raises(ValueError, match="divisible by 8"):
        quantise(torch.zeros((2, 34), dtype=torch.bfloat16), group_size=17)

    with pytest.raises(ValueError, match="group_size=32"):
        quantise(torch.zeros((2, 40), dtype=torch.bfloat16), group_size=32)


def test_all_zero_groups_use_finite_unit_scales() -> None:
    weight = torch.zeros((3, 64), dtype=torch.bfloat16)

    encoded = quantise(weight, group_size=32)

    assert torch.equal(encoded.scales, torch.ones_like(encoded.scales))
    assert torch.equal(dequantise(encoded), weight)


@pytest.mark.gpu
@pytest.mark.parametrize("group_size", [32, 64])
def test_w3a16_gemv_matches_dequantise_then_linear(group_size: int) -> None:
    pytest.importorskip("triton")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")

    from engine.kernels.w3a16_gemv import w3a16_dense_gemv

    torch.manual_seed(20260729 + group_size)
    output_size = 128
    reduction = 256
    activations = torch.randn((1, reduction), device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(
        (output_size, reduction), device="cuda", dtype=torch.bfloat16
    )
    encoded = quantise(weight, group_size=group_size)

    expected = F.linear(activations, dequantise(encoded))
    actual = w3a16_dense_gemv(
        activations,
        encoded.packed,
        encoded.scales,
        group_size=group_size,
    )

    torch.testing.assert_close(actual, expected, atol=0.125, rtol=0.05)


@pytest.mark.gpu
def test_w3a16_prefill_uses_dequantise_then_linear() -> None:
    pytest.importorskip("triton")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")

    from engine.kernels.w3a16_gemv import w3a16_dense_gemv

    torch.manual_seed(20260730)
    activations = torch.randn((3, 256), device="cuda", dtype=torch.bfloat16)
    weight = torch.randn((128, 256), device="cuda", dtype=torch.bfloat16)
    encoded = quantise(weight, group_size=32)

    expected = F.linear(activations, dequantise(encoded))
    actual = w3a16_dense_gemv(
        activations,
        encoded.packed,
        encoded.scales,
        group_size=32,
    )

    assert torch.equal(actual, expected)
