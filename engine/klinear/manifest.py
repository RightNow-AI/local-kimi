"""Pinned tensor names and shapes from the real Kimi-Linear safetensors headers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

BF16 = "BF16"
F32 = "F32"


@dataclass(frozen=True)
class TensorSpec:
    shape: tuple[int, ...]
    dtype: str


REAL_MODEL_TENSOR_MANIFEST = {
    "model.embed_tokens.weight": TensorSpec((163_840, 2_304), BF16),
    "model.norm.weight": TensorSpec((2_304,), BF16),
    "lm_head.weight": TensorSpec((163_840, 2_304), BF16),
}

REAL_DENSE_MLP_MANIFEST = {
    "mlp.down_proj.weight": TensorSpec((2_304, 9_216), BF16),
    "mlp.gate_proj.weight": TensorSpec((9_216, 2_304), BF16),
    "mlp.up_proj.weight": TensorSpec((9_216, 2_304), BF16),
}

REAL_KDA_ATTENTION_MANIFEST = {
    "self_attn.A_log": TensorSpec((1, 1, 32, 1), F32),
    "self_attn.b_proj.weight": TensorSpec((32, 2_304), BF16),
    "self_attn.dt_bias": TensorSpec((4_096,), F32),
    "self_attn.f_a_proj.weight": TensorSpec((128, 2_304), BF16),
    "self_attn.f_b_proj.weight": TensorSpec((4_096, 128), BF16),
    "self_attn.g_a_proj.weight": TensorSpec((128, 2_304), BF16),
    "self_attn.g_b_proj.weight": TensorSpec((4_096, 128), BF16),
    "self_attn.k_conv1d.weight": TensorSpec((4_096, 1, 4), BF16),
    "self_attn.k_proj.weight": TensorSpec((4_096, 2_304), BF16),
    "self_attn.o_norm.weight": TensorSpec((128,), BF16),
    "self_attn.o_proj.weight": TensorSpec((2_304, 4_096), BF16),
    "self_attn.q_conv1d.weight": TensorSpec((4_096, 1, 4), BF16),
    "self_attn.q_proj.weight": TensorSpec((4_096, 2_304), BF16),
    "self_attn.v_conv1d.weight": TensorSpec((4_096, 1, 4), BF16),
    "self_attn.v_proj.weight": TensorSpec((4_096, 2_304), BF16),
}

REAL_MLA_ATTENTION_MANIFEST = {
    "self_attn.kv_a_layernorm.weight": TensorSpec((512,), BF16),
    "self_attn.kv_a_proj_with_mqa.weight": TensorSpec((576, 2_304), BF16),
    "self_attn.kv_b_proj.weight": TensorSpec((8_192, 512), BF16),
    "self_attn.o_proj.weight": TensorSpec((2_304, 4_096), BF16),
    "self_attn.q_proj.weight": TensorSpec((6_144, 2_304), BF16),
}

REAL_MOE_NON_EXPERT_MANIFEST = {
    "block_sparse_moe.gate.e_score_correction_bias": TensorSpec((256,), BF16),
    "block_sparse_moe.gate.weight": TensorSpec((256, 2_304), BF16),
    "block_sparse_moe.shared_experts.down_proj.weight": TensorSpec(
        (2_304, 1_024), BF16
    ),
    "block_sparse_moe.shared_experts.gate_proj.weight": TensorSpec(
        (1_024, 2_304), BF16
    ),
    "block_sparse_moe.shared_experts.up_proj.weight": TensorSpec(
        (1_024, 2_304), BF16
    ),
}

REAL_EXPERT_TEMPLATE_MANIFEST = {
    "block_sparse_moe.experts.{expert}.w1.weight": TensorSpec(
        (1_024, 2_304), BF16
    ),
    "block_sparse_moe.experts.{expert}.w2.weight": TensorSpec(
        (2_304, 1_024), BF16
    ),
    "block_sparse_moe.experts.{expert}.w3.weight": TensorSpec(
        (1_024, 2_304), BF16
    ),
}

REAL_LAYER_NORM_MANIFEST = {
    "input_layernorm.weight": TensorSpec((2_304,), BF16),
    "post_attention_layernorm.weight": TensorSpec((2_304,), BF16),
}

REAL_UNRESOLVED_TENSORS: tuple[str, ...] = ()

_REAL_KDA_ONE_BASED = {
    1,
    2,
    3,
    5,
    6,
    7,
    9,
    10,
    11,
    13,
    14,
    15,
    17,
    18,
    19,
    21,
    22,
    23,
    25,
    26,
}


def real_layer_manifest(layer_idx: int, *, include_experts: bool = True) -> dict[str, TensorSpec]:
    if not 0 <= layer_idx < 27:
        raise IndexError("real Kimi-Linear layer index must be in 0..26")
    manifest = dict(REAL_LAYER_NORM_MANIFEST)
    if layer_idx + 1 in _REAL_KDA_ONE_BASED:
        manifest.update(REAL_KDA_ATTENTION_MANIFEST)
    else:
        manifest.update(REAL_MLA_ATTENTION_MANIFEST)
    if layer_idx == 0:
        manifest.update(REAL_DENSE_MLP_MANIFEST)
        return manifest
    manifest.update(REAL_MOE_NON_EXPERT_MANIFEST)
    if include_experts:
        for expert_id in range(256):
            for name, spec in REAL_EXPERT_TEMPLATE_MANIFEST.items():
                manifest[name.format(expert=expert_id)] = spec
    return manifest


def real_checkpoint_manifest() -> dict[str, TensorSpec]:
    manifest = dict(REAL_MODEL_TENSOR_MANIFEST)
    for layer_idx in range(27):
        prefix = f"model.layers.{layer_idx}."
        for suffix, spec in real_layer_manifest(layer_idx).items():
            manifest[prefix + suffix] = spec
    return manifest


REAL_CHECKPOINT_MANIFEST = real_checkpoint_manifest()
assert len(REAL_CHECKPOINT_MANIFEST) == 20_493


def validate_real_checkpoint_layout(actual: Mapping[str, TensorSpec]) -> None:
    expected_names = set(REAL_CHECKPOINT_MANIFEST)
    actual_names = set(actual)
    missing = sorted(expected_names - actual_names)
    unexpected = sorted(actual_names - expected_names)
    if missing or unexpected:
        raise ValueError(
            "checkpoint tensor names do not match Kimi-Linear: "
            f"missing={missing}, unexpected={unexpected}"
        )
    mismatches = [
        name
        for name, expected in REAL_CHECKPOINT_MANIFEST.items()
        if actual[name] != expected
    ]
    if mismatches:
        details = [
            f"{name}: expected={REAL_CHECKPOINT_MANIFEST[name]}, actual={actual[name]}"
            for name in mismatches
        ]
        raise ValueError("checkpoint tensor specs disagree: " + "; ".join(details))

