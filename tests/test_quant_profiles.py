from __future__ import annotations

import json
from math import prod

import pytest

from engine.quant.klinear_plan import (
    DEFAULT_PROFILE_NAME,
    SHARED_EXPERTS_BF16_PROFILE_NAME,
    TensorMetadata,
    build_klinear_quantization_plan,
)
from engine.quant.w4a16 import GROUP_SIZE

SOURCE_SHARD = "model-00001-of-00020.safetensors"


def _legacy_policy_manifest() -> tuple[TensorMetadata, ...]:
    return (
        TensorMetadata(
            "model.layers.1.block_sparse_moe.e_score_correction_bias",
            (256,),
            "BF16",
            SOURCE_SHARD,
        ),
        TensorMetadata(
            "model.layers.1.input_layernorm.weight",
            (2304,),
            "BF16",
            SOURCE_SHARD,
        ),
        TensorMetadata(
            "model.embed_tokens.weight",
            (163840, 2304),
            "BF16",
            SOURCE_SHARD,
        ),
        TensorMetadata(
            "model.lm_head.weight",
            (163840, 2304),
            "BF16",
            SOURCE_SHARD,
        ),
        TensorMetadata(
            "model.layers.1.block_sparse_moe.gate.weight",
            (256, 2304),
            "BF16",
            SOURCE_SHARD,
        ),
        TensorMetadata(
            "model.layers.2.self_attn.g_a_proj.weight",
            (128, 2304),
            "BF16",
            SOURCE_SHARD,
        ),
        TensorMetadata(
            "model.layers.1.block_sparse_moe.experts.0.w1.weight",
            (1024, 2304),
            "BF16",
            SOURCE_SHARD,
        ),
        TensorMetadata(
            "model.layers.1.block_sparse_moe.shared_experts.gate_proj.weight",
            (1024, 2304),
            "BF16",
            SOURCE_SHARD,
        ),
        TensorMetadata(
            "model.layers.0.mlp.down_proj.weight",
            (2304, 1024),
            "BF16",
            SOURCE_SHARD,
        ),
        TensorMetadata(
            "model.layers.3.self_attn.kv_a_proj_with_mqa.weight",
            (576, 2304),
            "BF16",
            SOURCE_SHARD,
        ),
        TensorMetadata(
            "model.layers.4.self_attn.q_proj.weight",
            (2304, 2304),
            "BF16",
            SOURCE_SHARD,
        ),
        TensorMetadata(
            "model.rotary_emb.inv_freq",
            (128,),
            "BF16",
            SOURCE_SHARD,
        ),
        TensorMetadata(
            "model.unapproved_projection.weight",
            (128, 2304),
            "BF16",
            SOURCE_SHARD,
        ),
    )


_LEGACY_POLICY = {
    "model.layers.1.block_sparse_moe.e_score_correction_bias": (
        "biases",
        False,
        "Bias tensors are small and remain in source precision.",
    ),
    "model.layers.1.input_layernorm.weight": (
        "normalization",
        False,
        "Normalization tensors are small and numerically sensitive.",
    ),
    "model.embed_tokens.weight": (
        "token embedding",
        False,
        "Keep the input embedding in BF16 to avoid vocabulary representation loss.",
    ),
    "model.lm_head.weight": (
        "lm head",
        False,
        "Keep the untied vocabulary head in BF16 because output logits are quantization sensitive.",
    ),
    "model.layers.1.block_sparse_moe.gate.weight": (
        "router gate",
        False,
        "Router error changes discrete expert selection, so router weights remain in BF16.",
    ),
    "model.layers.2.self_attn.g_a_proj.weight": (
        "KDA gates, state, and convolutions",
        False,
        "Keep KDA recurrent controls and short convolutions in source precision.",
    ),
    "model.layers.1.block_sparse_moe.experts.0.w1.weight": (
        "routed expert projections",
        True,
        "Routed experts dominate resident bytes and are the primary fit target.",
    ),
    "model.layers.1.block_sparse_moe.shared_experts.gate_proj.weight": (
        "shared expert projections",
        True,
        "Shared expert matrices are large token-path projections and must be compressed for fit.",
    ),
    "model.layers.0.mlp.down_proj.weight": (
        "dense layer 0 MLP projections",
        True,
        "The first dense MLP is a large matrix bank and is part of the fit target.",
    ),
    "model.layers.3.self_attn.kv_a_proj_with_mqa.weight": (
        "MLA latent down-projection",
        False,
        (
            "This projection produces the compressed KV latent that is written to "
            "the cache, so error here persists for the whole sequence instead of "
            "perturbing one token's activation. It is also only 576 by 2304 across "
            "seven layers, roughly 18.6 MB in BF16, so quantizing it does nothing "
            "for fit. Quantize for fit, not for its own sake."
        ),
    ),
    "model.layers.4.self_attn.q_proj.weight": (
        "attention projections",
        True,
        "Quantize large attention projection matrices while preserving KDA controls.",
    ),
    "model.rotary_emb.inv_freq": (
        "vectors and non-matrix state",
        False,
        "Only matrix weights in an approved class use W4A16.",
    ),
    "model.unapproved_projection.weight": (
        "other checkpoint matrices",
        False,
        "The matrix is outside the approved fit-driven classes and remains unchanged.",
    ),
}


def _legacy_snapshot_bytes(tensors: tuple[TensorMetadata, ...]) -> bytes:
    records = []
    for tensor in sorted(tensors, key=lambda item: item.name):
        tensor_class, quantize, reason = _LEGACY_POLICY[tensor.name]
        original_bytes = prod(tensor.shape) * 2
        planned_bytes = original_bytes
        if quantize:
            elements = prod(tensor.shape)
            planned_bytes = elements // 2 + elements // GROUP_SIZE * 2
        records.append(
            {
                "dtype": tensor.dtype,
                "name": tensor.name,
                "original_bytes": original_bytes,
                "packed_name": (
                    f"{tensor.name}.w4a16_packed" if quantize else None
                ),
                "planned_bytes": planned_bytes,
                "quantize": quantize,
                "reason": reason,
                "saved_bytes": original_bytes - planned_bytes,
                "scales_name": (
                    f"{tensor.name}.w4a16_scales" if quantize else None
                ),
                "shape": tensor.shape,
                "source_file": tensor.source_file,
                "tensor_class": tensor_class,
            }
        )
    return json.dumps(records, sort_keys=True, separators=(",", ":")).encode()


def _decision_bytes(plan) -> bytes:
    return json.dumps(
        [decision.as_dict() for decision in plan.tensors],
        sort_keys=True,
        separators=(",", ":"),
    ).encode()


def _real_shared_expert_header_census() -> tuple[TensorMetadata, ...]:
    tensors = []
    for layer in range(1, 27):
        prefix = f"model.layers.{layer}.block_sparse_moe.shared_experts"
        tensors.extend(
            (
                TensorMetadata(f"{prefix}.gate_proj.weight", (1024, 2304), "BF16"),
                TensorMetadata(f"{prefix}.up_proj.weight", (1024, 2304), "BF16"),
                TensorMetadata(f"{prefix}.down_proj.weight", (2304, 1024), "BF16"),
            )
        )
    return tuple(tensors)


def test_default_profile_preserves_the_legacy_decisions_byte_for_byte() -> None:
    tensors = _legacy_policy_manifest()
    expected = _legacy_snapshot_bytes(tensors)

    implicit = build_klinear_quantization_plan(tensors)
    explicit = build_klinear_quantization_plan(
        tensors,
        profile=DEFAULT_PROFILE_NAME,
    )

    assert implicit.profile.name == DEFAULT_PROFILE_NAME
    assert _decision_bytes(implicit) == expected
    assert _decision_bytes(explicit) == expected


def test_shared_expert_profile_retains_shared_experts_but_not_routed_experts() -> None:
    routed_name = "model.layers.1.block_sparse_moe.experts.0.w1.weight"
    tensors = _real_shared_expert_header_census() + (
        TensorMetadata(routed_name, (1024, 2304), "BF16"),
    )

    plan = build_klinear_quantization_plan(
        tensors,
        profile=SHARED_EXPERTS_BF16_PROFILE_NAME,
    )
    shared = [
        decision
        for decision in plan.tensors
        if decision.tensor_class == "shared expert projections"
    ]
    routed = next(decision for decision in plan.tensors if decision.name == routed_name)

    assert len(shared) == 78
    assert all(not decision.quantize for decision in shared)
    assert all(decision.planned_bytes == decision.original_bytes for decision in shared)
    assert routed.quantize is True


def test_shared_expert_profile_delta_equals_the_real_shared_tensor_delta() -> None:
    tensors = _real_shared_expert_header_census() + (
        TensorMetadata(
            "model.layers.1.block_sparse_moe.experts.0.w1.weight",
            (1024, 2304),
            "BF16",
        ),
    )
    default = build_klinear_quantization_plan(
        tensors,
        profile=DEFAULT_PROFILE_NAME,
    )
    retained = build_klinear_quantization_plan(
        tensors,
        profile=SHARED_EXPERTS_BF16_PROFILE_NAME,
    )
    shared_delta = sum(
        decision.original_bytes - decision.planned_bytes
        for decision in default.tensors
        if decision.tensor_class == "shared expert projections"
    )

    assert shared_delta == 264_536_064
    assert retained.planned_bytes > default.planned_bytes
    assert retained.planned_bytes - default.planned_bytes == shared_delta


def test_unknown_profile_is_refused() -> None:
    with pytest.raises(ValueError, match="unknown Kimi-Linear quantization profile"):
        build_klinear_quantization_plan(
            _legacy_policy_manifest(),
            profile="not-a-real-profile",
        )
