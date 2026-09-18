from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from engine.kernels.w3a16_grouped_gemv import (
    grouped_w3a16_gemv_reference,
)
from engine.quant.w3a16 import W3A16Tensor, dequantise, quantise


@pytest.mark.parametrize(
    (
        "output_size",
        "reduction",
        "expected_packed_shape",
        "expected_scale_shape",
    ),
    (
        (1024, 2304, (257, 1024, 864), (257, 1024, 72)),
        (2304, 1024, (257, 2304, 384), (257, 2304, 32)),
    ),
)
def test_real_grouped_bank_shape_arithmetic(
    output_size: int,
    reduction: int,
    expected_packed_shape: tuple[int, int, int],
    expected_scale_shape: tuple[int, int, int],
) -> None:
    experts = 257
    group_size = 32

    assert (
        experts,
        output_size,
        reduction // 8 * 3,
    ) == expected_packed_shape
    assert (
        experts,
        output_size,
        reduction // group_size,
    ) == expected_scale_shape


def _quantise_grouped_bank(
    weights: torch.Tensor,
    *,
    group_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    experts, output_size, reduction = weights.shape
    encoded = quantise(
        weights.reshape(experts * output_size, reduction).contiguous(),
        group_size=group_size,
    )
    return (
        encoded.packed.reshape(experts, output_size, -1),
        encoded.scales.reshape(experts, output_size, -1),
    )


def test_grouped_reference_matches_explicit_per_expert_loop() -> None:
    torch.manual_seed(20260729)
    group_size = 32
    experts = 5
    output_size = 7
    reduction = 96
    activations = torch.randn((2, reduction), dtype=torch.bfloat16)
    weights = torch.randn(
        (experts, output_size, reduction), dtype=torch.bfloat16
    )
    packed, scales = _quantise_grouped_bank(
        weights, group_size=group_size
    )
    expert_indices = torch.tensor(
        [[0, 2, 4], [3, 1, 4]], dtype=torch.long
    )

    actual = grouped_w3a16_gemv_reference(
        activations,
        expert_indices,
        packed,
        scales,
        group_size=group_size,
    )

    token_outputs: list[torch.Tensor] = []
    for token in range(activations.shape[0]):
        route_outputs: list[torch.Tensor] = []
        for route in range(expert_indices.shape[1]):
            expert = int(expert_indices[token, route])
            encoded = W3A16Tensor(
                packed=packed[expert],
                scales=scales[expert],
                original_shape=(output_size, reduction),
                group_size=group_size,
            )
            route_outputs.append(
                F.linear(
                    activations[token].float(),
                    dequantise(encoded).float(),
                ).to(torch.bfloat16)
            )
        token_outputs.append(torch.stack(route_outputs))
    expected = torch.stack(token_outputs)

    torch.testing.assert_close(actual, expected, atol=0.015625, rtol=0.01)


@pytest.mark.gpu
def test_grouped_triton_matches_reference_with_shared_expert_last() -> None:
    pytest.importorskip("triton")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")

    from engine.kernels.w3a16_grouped_gemv import grouped_w3a16_gemv

    torch.manual_seed(20260730)
    torch.cuda.manual_seed_all(20260730)
    group_size = 32
    experts = 257
    output_size = 64
    reduction = 96
    activations = torch.randn(
        (1, reduction), device="cuda", dtype=torch.bfloat16
    )
    weights = torch.randn(
        (experts, output_size, reduction),
        device="cuda",
        dtype=torch.bfloat16,
    )
    packed, scales = _quantise_grouped_bank(
        weights, group_size=group_size
    )
    expert_indices = torch.tensor(
        [[7, 6, 5, 4, 3, 2, 1, 0, 256]],
        device="cuda",
        dtype=torch.long,
    )

    expected = grouped_w3a16_gemv_reference(
        activations,
        expert_indices,
        packed,
        scales,
        group_size=group_size,
    )
    actual = grouped_w3a16_gemv(
        activations,
        expert_indices,
        packed,
        scales,
        group_size=group_size,
    )
    repeated = grouped_w3a16_gemv(
        activations,
        expert_indices,
        packed,
        scales,
        group_size=group_size,
    )

    torch.testing.assert_close(actual, expected, atol=0.125, rtol=0.05)
    assert torch.equal(actual, repeated)
