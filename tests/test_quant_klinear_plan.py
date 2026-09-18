from __future__ import annotations

from math import prod

import pytest

from engine.quant.klinear_plan import (
    TensorMetadata,
    build_klinear_quantization_plan,
)
from engine.quant.w4a16 import GROUP_SIZE


def _decision(plan, name: str):
    return next(tensor for tensor in plan.tensors if tensor.name == name)


def test_plan_refuses_to_quantize_router_gate_weight() -> None:
    name = "model.layers.1.block_sparse_moe.gate.weight"
    plan = build_klinear_quantization_plan(
        [TensorMetadata(name=name, shape=(256, 2304), dtype="BF16")]
    )

    decision = _decision(plan, name)

    assert decision.tensor_class == "router gate"
    assert decision.quantize is False
    assert decision.planned_bytes == decision.original_bytes


def test_plan_refuses_non_divisible_reduction_instead_of_padding() -> None:
    name = "model.layers.1.block_sparse_moe.experts.0.w1.weight"

    with pytest.raises(ValueError, match="not divisible by group size"):
        build_klinear_quantization_plan(
            [TensorMetadata(name=name, shape=(1024, 2305), dtype="BF16")]
        )


def test_projected_total_is_computed_from_supplied_real_shapes() -> None:
    expert_name = "model.layers.1.block_sparse_moe.experts.0.w1.weight"
    embedding_name = "model.embed_tokens.weight"
    tensors = [
        TensorMetadata(expert_name, (1024, 2304), "BF16"),
        TensorMetadata(embedding_name, (163840, 2304), "BF16"),
    ]

    plan = build_klinear_quantization_plan(tensors)
    expert_elements = prod(tensors[0].shape)
    expert_bytes = expert_elements // 2 + expert_elements // GROUP_SIZE * 2
    embedding_bytes = prod(tensors[1].shape) * 2

    assert plan.planned_bytes == expert_bytes + embedding_bytes

    wider_plan = build_klinear_quantization_plan(
        [
            TensorMetadata(expert_name, (2048, 2304), "BF16"),
            tensors[1],
        ]
    )
    assert wider_plan.planned_bytes == embedding_bytes + expert_bytes * 2


def test_skipped_tensor_retains_its_original_byte_count() -> None:
    name = "model.layers.1.input_layernorm.weight"
    plan = build_klinear_quantization_plan(
        [TensorMetadata(name=name, shape=(2304,), dtype="BF16")]
    )

    decision = _decision(plan, name)

    assert decision.quantize is False
    assert decision.original_bytes == 2304 * 2
    assert decision.planned_bytes == decision.original_bytes


def test_kda_projections_quantize_but_kda_gates_remain_bf16() -> None:
    q_name = "model.layers.1.self_attn.q_proj.weight"
    gate_name = "model.layers.1.self_attn.g_a_proj.weight"
    plan = build_klinear_quantization_plan(
        [
            TensorMetadata(q_name, (4096, 2304), "BF16"),
            TensorMetadata(gate_name, (128, 2304), "BF16"),
        ]
    )

    assert _decision(plan, q_name).quantize is True
    assert _decision(plan, gate_name).quantize is False
    assert _decision(plan, gate_name).tensor_class == "KDA gates, state, and convolutions"


def test_shared_expert_is_not_misclassified_as_a_routed_expert() -> None:
    name = "model.layers.1.block_sparse_moe.shared_experts.gate_proj.weight"
    plan = build_klinear_quantization_plan(
        [TensorMetadata(name, (1024, 2304), "BF16")]
    )

    decision = _decision(plan, name)

    assert decision.quantize is True
    assert decision.tensor_class == "shared expert projections"
