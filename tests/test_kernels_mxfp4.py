from __future__ import annotations

from collections import Counter

import pytest
import torch
import torch.nn.functional as F

from engine.k3ref.dequant import dequantize_mxfp4
from engine.kernels.moe_grouped import (
    expert_major_moe,
    expert_union_size,
    situ_and_mul,
)
from engine.kernels.reference import (
    dequantize_mxfp4_reference,
    naive_token_major_moe,
    situ_reference,
)


def _pack(codes: list[int]) -> list[int]:
    if len(codes) % 2:
        raise ValueError("codes must contain an even number of values")
    return [
        codes[index] | (codes[index + 1] << 4)
        for index in range(0, len(codes), 2)
    ]


def test_unfused_reference_matches_canonical_decoder_exactly() -> None:
    first = list(range(16)) * 4
    second = list(reversed(range(16))) * 4
    packed = torch.tensor([_pack(first), _pack(second)], dtype=torch.uint8)
    scale = torch.tensor([[126, 130], [127, 125]], dtype=torch.uint8)

    expected = dequantize_mxfp4(packed, scale)
    actual = dequantize_mxfp4_reference(packed, scale)

    assert torch.equal(actual, expected)
    assert actual[0, 0].item() == 0.0
    assert actual[0, 1].item() == 0.25
    assert actual[0, 32].item() == 0.0
    assert actual[0, 33].item() == 4.0


def test_expert_major_matches_naive_token_major_for_random_routing() -> None:
    torch.manual_seed(20260728)
    token_count = 9
    hidden_size = 6
    expert_count = 7
    hidden = torch.randn(token_count, hidden_size)
    expert_indices = torch.randint(0, expert_count, (token_count, 4))
    expert_indices[0, 3] = -1
    expert_indices[5, 1:] = -1
    routing_weights = torch.rand(token_count, 4)
    matrices = torch.randn(expert_count, hidden_size, hidden_size)

    def expert_fn(expert_id: int, tokens: torch.Tensor) -> torch.Tensor:
        return F.linear(tokens, matrices[expert_id])

    expected = naive_token_major_moe(
        hidden, expert_indices, routing_weights, expert_fn
    )
    calls: Counter[int] = Counter()

    def counted_expert_fn(expert_id: int, tokens: torch.Tensor) -> torch.Tensor:
        calls[expert_id] += 1
        return expert_fn(expert_id, tokens)

    actual = expert_major_moe(
        hidden, expert_indices, routing_weights, counted_expert_fn
    )

    torch.testing.assert_close(actual.values, expected, atol=1e-6, rtol=1e-6)
    assert actual.union_size == len(torch.unique(expert_indices[expert_indices >= 0]))
    assert set(calls.values()) == {1}


def test_union_size_and_occurrence_counts_for_known_routes() -> None:
    expert_indices = torch.tensor(
        [
            [1, 2, 2, -1],
            [7, 1, -1, -1],
            [7, 9, 9, 1],
        ]
    )
    routing_weights = torch.ones_like(expert_indices, dtype=torch.float32)
    hidden = torch.ones(3, 2)

    result = expert_major_moe(
        hidden,
        expert_indices,
        routing_weights,
        lambda expert_id, tokens: tokens * (expert_id + 1),
    )

    assert expert_union_size(expert_indices) == 4
    assert result.union_size == 4
    assert result.expert_token_counts == ((1, 3), (2, 2), (7, 2), (9, 2))


def test_zero_routes_and_duplicate_expert_routes_preserve_semantics() -> None:
    hidden = torch.tensor([[2.0, -1.0], [3.0, 5.0]])
    expert_indices = torch.tensor([[-1, -1, -1], [3, 3, -1]])
    routing_weights = torch.tensor([[0.2, 0.8, 1.0], [0.25, 0.75, 4.0]])
    calls: list[tuple[int, int]] = []

    def expert_fn(expert_id: int, tokens: torch.Tensor) -> torch.Tensor:
        calls.append((expert_id, tokens.shape[0]))
        return tokens + float(expert_id)

    result = expert_major_moe(hidden, expert_indices, routing_weights, expert_fn)

    assert result.union_size == 1
    assert result.expert_token_counts == ((3, 2),)
    assert calls == [(3, 2)]
    assert torch.equal(result.values[0], torch.zeros_like(result.values[0]))
    torch.testing.assert_close(result.values[1], hidden[1] + 3.0)


def test_fast_path_uses_exact_k3_situ_math() -> None:
    torch.manual_seed(91)
    gate = torch.randn(4, 7, dtype=torch.float16)
    up = torch.randn(4, 7, dtype=torch.float16)

    expected = situ_reference(gate, up, beta=4.0, linear_beta=25.0)
    actual = situ_and_mul(gate, up, beta=4.0, linear_beta=25.0)

    assert torch.equal(actual, expected)


@pytest.mark.gpu
@pytest.mark.parametrize(
    ("dtype", "atol", "rtol"),
    [
        (torch.float16, 0.02, 0.015),
        (torch.bfloat16, 0.08, 0.03),
    ],
)
def test_fused_mxfp4_gemm_matches_unfused_reference_on_gpu(
    dtype: torch.dtype, atol: float, rtol: float
) -> None:
    pytest.importorskip("triton")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    from engine.kernels.mxfp4_gemm import mxfp4_gemm

    torch.manual_seed(17)
    activations = torch.randn(5, 64, device="cuda", dtype=dtype)
    packed = torch.randint(0, 256, (48, 32), device="cuda", dtype=torch.uint8)
    scale = torch.randint(121, 129, (48, 2), device="cuda", dtype=torch.uint8)

    expected = dequantize_mxfp4_reference(packed, scale, dtype=dtype)
    expected = F.linear(activations, expected)
    actual = mxfp4_gemm(activations, packed, scale)

    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
