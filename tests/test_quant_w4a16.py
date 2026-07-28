from __future__ import annotations

from math import prod

import pytest
import torch

from engine.k3ref.manifest import K3_LAYER_TENSOR_MANIFEST
from engine.quant.plan import build_quantization_plan
from engine.quant.verify import (
    VerificationError,
    negative_control_quantized_tensor,
    swapped_nibble_dequantise,
    verify_dequantizer,
    verify_round_trip,
    wrong_group_axis_dequantise,
    wrong_scale_dequantise,
)
from engine.quant.w4a16 import GROUP_SIZE, dequantise, quantise


def _grid_weight(scales: torch.Tensor) -> torch.Tensor:
    pattern = torch.tensor(
        [index % 15 - 7 for index in range(GROUP_SIZE)], dtype=torch.bfloat16
    )
    codes = pattern.reshape(1, 1, GROUP_SIZE).expand(
        scales.shape[0], scales.shape[1], GROUP_SIZE
    )
    return (codes * scales.unsqueeze(-1)).reshape(scales.shape[0], -1)


def test_round_trip_is_exact_for_values_on_the_quantization_grid() -> None:
    weight = _grid_weight(
        torch.tensor([[0.25, 2.0], [8.0, 0.5]], dtype=torch.bfloat16)
    )

    restored = dequantise(quantise(weight))

    assert torch.equal(restored, weight)


def test_scales_are_per_group_along_the_reduction_axis() -> None:
    expected_scales = torch.tensor(
        [[0.25, 8.0], [2.0, 0.5]], dtype=torch.bfloat16
    )
    weight = _grid_weight(expected_scales)

    encoded = quantise(weight)

    assert encoded.scales.shape == (2, 2)
    assert torch.equal(encoded.scales, expected_scales)
    assert torch.equal(dequantise(encoded), weight)


def test_verifier_rejects_wrong_group_axis() -> None:
    encoded = negative_control_quantized_tensor()

    with pytest.raises(VerificationError, match="wrong_group_axis"):
        verify_dequantizer(
            encoded,
            wrong_group_axis_dequantise,
            decoder_name="wrong_group_axis",
        )


def test_verifier_rejects_wrong_scale() -> None:
    encoded = negative_control_quantized_tensor()

    with pytest.raises(VerificationError, match="wrong_scale"):
        verify_dequantizer(
            encoded,
            wrong_scale_dequantise,
            decoder_name="wrong_scale",
        )


def test_verifier_rejects_swapped_nibble_order() -> None:
    encoded = negative_control_quantized_tensor()

    with pytest.raises(VerificationError, match="swapped_nibbles"):
        verify_dequantizer(
            encoded,
            swapped_nibble_dequantise,
            decoder_name="swapped_nibbles",
        )


def test_plan_savings_match_independent_manifest_recomputation() -> None:
    plan = build_quantization_plan(include_lm_head=False)
    expected_original = 0
    expected_planned = 0
    expected_by_class = {}

    for decision in plan.tensors:
        spec = K3_LAYER_TENSOR_MANIFEST[decision.name]
        elements = prod(spec.shape)
        original = prod(spec.shape) * (2 if spec.dtype == "BF16" else 4)
        expected_original += original
        if decision.quantize:
            expected_planned += elements // 2 + elements // GROUP_SIZE * 2
        else:
            expected_planned += original
        class_totals = expected_by_class.setdefault(
            decision.tensor_class,
            {"original": 0, "planned": 0},
        )
        class_totals["original"] += original
        class_totals["planned"] += (
            elements // 2 + elements // GROUP_SIZE * 2
            if decision.quantize
            else original
        )

    assert plan.original_bytes == expected_original
    assert plan.planned_bytes == expected_planned
    assert plan.saved_bytes == expected_original - expected_planned
    for report in plan.by_class():
        expected = expected_by_class[report.tensor_class]
        assert report.original_bytes == expected["original"]
        assert report.planned_bytes == expected["planned"]
        assert report.saved_bytes == expected["original"] - expected["planned"]


def test_round_trip_verifier_accepts_the_claimed_scale_aware_bound() -> None:
    torch.manual_seed(20260728)
    weight = torch.randn((4, 96), dtype=torch.bfloat16)

    stats = verify_round_trip("test projection", weight)

    assert stats.elements == weight.numel()
    assert stats.max_allowed_abs_error is not None
    assert stats.max_abs_error <= stats.max_allowed_abs_error


def test_all_zero_tensor_is_finite_and_does_not_divide_by_zero() -> None:
    weight = torch.zeros((3, 64), dtype=torch.bfloat16)

    encoded = quantise(weight)
    restored = dequantise(encoded)

    assert torch.isfinite(encoded.scales).all()
    assert torch.equal(encoded.scales, torch.ones_like(encoded.scales))
    assert torch.equal(restored, weight)


def test_non_finite_weights_are_rejected() -> None:
    weight = torch.zeros((1, 32), dtype=torch.bfloat16)
    weight[0, 0] = float("inf")

    with pytest.raises(ValueError, match="finite"):
        quantise(weight)
