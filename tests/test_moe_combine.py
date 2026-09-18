from __future__ import annotations

import pytest
import torch

from engine.kernels.moe_combine import fused_w2_combine_reference


def _constant_packed_weight(
    value: int,
    *,
    output_size: int,
    reduction: int,
) -> torch.Tensor:
    nibble = value & 0xF
    packed_byte = nibble | (nibble << 4)
    return torch.full(
        (output_size, reduction // 2), packed_byte, dtype=torch.uint8
    )


def _decode_weight(
    packed: torch.Tensor,
    scales: torch.Tensor,
) -> torch.Tensor:
    low = packed & 0xF
    high = packed >> 4
    nibbles = torch.stack((low, high), dim=-1).reshape(packed.shape[0], -1)
    signed = nibbles.to(torch.int16)
    signed = torch.where(signed >= 8, signed - 16, signed)
    group_indices = torch.arange(signed.shape[1], dtype=torch.long) // 32
    return (signed.float() * scales.float()[:, group_indices]).to(torch.bfloat16)


def _explicit_current_sequence(
    activated: torch.Tensor,
    expert_indices: torch.Tensor,
    combine_weights: torch.Tensor,
    w2_packed: torch.Tensor,
    w2_scales: torch.Tensor,
) -> torch.Tensor:
    tokens, routes, _ = activated.shape
    output_size = w2_packed.shape[1]
    expert_outputs = torch.empty(
        (tokens, routes, output_size), dtype=torch.bfloat16
    )
    for token in range(tokens):
        for route in range(routes):
            expert = int(expert_indices[token, route])
            decoded = _decode_weight(w2_packed[expert], w2_scales[expert])
            expert_outputs[token, route] = (
                activated[token, route].float().unsqueeze(0) * decoded.float()
            ).sum(dim=-1).to(torch.bfloat16)

    routed = (
        expert_outputs[:, :-1]
        .float()
        .mul(combine_weights[:, :-1].float().unsqueeze(-1))
        .sum(dim=1)
        .to(torch.bfloat16)
    )
    return routed + expert_outputs[:, -1]


def test_shared_expert_is_added_with_weight_one() -> None:
    reduction = 32
    activated = torch.ones((1, 2, reduction), dtype=torch.bfloat16)
    expert_indices = torch.tensor([[0, 1]], dtype=torch.long)
    # The sentinel in the last column proves the reference does not treat the
    # shared expert as another router-weighted route.
    combine_weights = torch.tensor([[0.25, 0.0]], dtype=torch.float32)
    w2_packed = torch.stack(
        (
            _constant_packed_weight(1, output_size=1, reduction=reduction),
            _constant_packed_weight(2, output_size=1, reduction=reduction),
        )
    )
    w2_scales = torch.ones((2, 1, 1), dtype=torch.bfloat16)

    actual = fused_w2_combine_reference(
        activated,
        expert_indices,
        combine_weights,
        w2_packed,
        w2_scales,
    )

    # Routed: 32 * 0.25 = 8. Shared: 32 * 2 = 64. Total: 72.
    expected = torch.tensor([[72.0]], dtype=torch.bfloat16)
    assert torch.equal(actual, expected)


def test_reference_matches_explicit_route_loop() -> None:
    torch.manual_seed(20260729)
    tokens = 2
    routes = 3
    experts = 4
    output_size = 5
    reduction = 64
    activated = torch.randn(
        (tokens, routes, reduction), dtype=torch.bfloat16
    )
    expert_indices = torch.tensor([[0, 1, 3], [2, 0, 3]], dtype=torch.long)
    combine_weights = torch.tensor(
        [[0.25, 0.75, 0.125], [0.6, 0.4, 7.0]], dtype=torch.float32
    )
    w2_packed = torch.randint(
        0,
        256,
        (experts, output_size, reduction // 2),
        dtype=torch.uint8,
    )
    w2_scales = (
        torch.rand((experts, output_size, reduction // 32)) * 0.05 + 0.001
    ).to(torch.bfloat16)

    expected = _explicit_current_sequence(
        activated,
        expert_indices,
        combine_weights,
        w2_packed,
        w2_scales,
    )
    actual = fused_w2_combine_reference(
        activated,
        expert_indices,
        combine_weights,
        w2_packed,
        w2_scales,
    )

    assert torch.equal(actual, expected)


@pytest.mark.gpu
def test_fused_w2_combine_matches_reference_and_is_deterministic() -> None:
    pytest.importorskip("triton")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")

    from engine.kernels.moe_combine import fused_w2_combine

    torch.manual_seed(20260729)
    torch.cuda.manual_seed_all(20260729)
    device = torch.device("cuda")
    tokens = 2
    routes = 9
    experts = 10
    output_size = 96
    reduction = 64
    activated = torch.randn(
        (tokens, routes, reduction), device=device, dtype=torch.bfloat16
    )
    expert_indices = torch.tensor(
        [
            [0, 1, 2, 3, 4, 5, 6, 7, 9],
            [8, 7, 6, 5, 4, 3, 2, 1, 9],
        ],
        device=device,
        dtype=torch.long,
    )
    routed_weights = torch.rand(
        (tokens, routes - 1), device=device, dtype=torch.float32
    )
    routed_weights /= routed_weights.sum(dim=1, keepdim=True)
    combine_weights = torch.cat(
        (
            routed_weights,
            torch.ones((tokens, 1), device=device, dtype=torch.float32),
        ),
        dim=1,
    )
    w2_packed = torch.randint(
        0,
        256,
        (experts, output_size, reduction // 2),
        device=device,
        dtype=torch.uint8,
    )
    w2_scales = (
        torch.rand(
            (experts, output_size, reduction // 32),
            device=device,
            dtype=torch.float32,
        )
        * 0.05
        + 0.001
    ).to(torch.bfloat16)
    arguments = (
        activated,
        expert_indices,
        combine_weights,
        w2_packed,
        w2_scales,
    )

    expected = fused_w2_combine_reference(*arguments)
    first = fused_w2_combine(*arguments)
    second = fused_w2_combine(*arguments)

    assert torch.equal(first, second)
    torch.testing.assert_close(first, expected, atol=0.125, rtol=0.05)
