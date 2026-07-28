"""Authoritative Kimi K3 layer-12 tensor manifest.

These shapes and storage dtypes were read from the real checkpoint's
safetensors headers. They must not be derived from module declarations.
"""

from __future__ import annotations

from dataclasses import dataclass


BF16 = "BF16"
F32 = "F32"
U8 = "U8"
MXFP4_GROUP_SIZE = 32


@dataclass(frozen=True)
class TensorSpec:
    shape: tuple[int, ...]
    dtype: str


# Exact non-expert checkpoint tensors for one KDA MoE decoder layer.
K3_LAYER_TENSOR_MANIFEST: dict[str, TensorSpec] = {
    "self_attn.q_proj.weight": TensorSpec((12288, 7168), BF16),
    "self_attn.k_proj.weight": TensorSpec((12288, 7168), BF16),
    "self_attn.v_proj.weight": TensorSpec((12288, 7168), BF16),
    "self_attn.g_proj.weight": TensorSpec((12288, 7168), BF16),
    "self_attn.o_proj.weight": TensorSpec((7168, 12288), BF16),
    "self_attn.b_proj.weight": TensorSpec((96, 7168), BF16),
    "self_attn.f_a_proj.weight": TensorSpec((128, 7168), BF16),
    "self_attn.f_b_proj.weight": TensorSpec((12288, 128), BF16),
    "self_attn.A_log": TensorSpec((128,), F32),
    "self_attn.dt_bias": TensorSpec((12288,), F32),
    "self_attn.q_conv1d.weight": TensorSpec((12288, 1, 4), F32),
    "self_attn.k_conv1d.weight": TensorSpec((12288, 1, 4), F32),
    "self_attn.v_conv1d.weight": TensorSpec((12288, 1, 4), F32),
    "self_attn.o_norm.weight": TensorSpec((128,), BF16),
    "input_layernorm.weight": TensorSpec((7168,), BF16),
    "post_attention_layernorm.weight": TensorSpec((7168,), BF16),
    "self_attention_res_proj.weight": TensorSpec((1, 7168), BF16),
    "mlp_res_proj.weight": TensorSpec((1, 7168), BF16),
    "self_attention_res_norm.weight": TensorSpec((7168,), BF16),
    "mlp_res_norm.weight": TensorSpec((7168,), BF16),
    "block_sparse_moe.gate.weight": TensorSpec((896, 7168), BF16),
    "block_sparse_moe.gate.e_score_correction_bias": TensorSpec((896,), BF16),
    "block_sparse_moe.routed_expert_down_proj.weight": TensorSpec(
        (3584, 7168), BF16
    ),
    "block_sparse_moe.routed_expert_norm.weight": TensorSpec((3584,), BF16),
    "block_sparse_moe.routed_expert_up_proj.weight": TensorSpec(
        (7168, 3584), BF16
    ),
    "block_sparse_moe.shared_experts.gate_proj.weight": TensorSpec(
        (6144, 7168), BF16
    ),
    "block_sparse_moe.shared_experts.up_proj.weight": TensorSpec(
        (6144, 7168), BF16
    ),
    "block_sparse_moe.shared_experts.down_proj.weight": TensorSpec(
        (7168, 6144), BF16
    ),
}


# Raw checkpoint storage for each routed expert. Replace {expert} with 0..895.
K3_EXPERT_CHECKPOINT_MANIFEST: dict[str, TensorSpec] = {
    "block_sparse_moe.experts.{expert}.w1.weight_packed": TensorSpec(
        (3072, 1792), U8
    ),
    "block_sparse_moe.experts.{expert}.w1.weight_scale": TensorSpec(
        (3072, 112), U8
    ),
    "block_sparse_moe.experts.{expert}.w2.weight_packed": TensorSpec(
        (3584, 1536), U8
    ),
    "block_sparse_moe.experts.{expert}.w2.weight_scale": TensorSpec(
        (3584, 96), U8
    ),
    "block_sparse_moe.experts.{expert}.w3.weight_packed": TensorSpec(
        (3072, 1792), U8
    ),
    "block_sparse_moe.experts.{expert}.w3.weight_scale": TensorSpec(
        (3072, 112), U8
    ),
}


K3_EXPERT_RUNTIME_MANIFEST: dict[str, TensorSpec] = {
    "w1.weight": TensorSpec((3072, 3584), BF16),
    "w2.weight": TensorSpec((3584, 3072), BF16),
    "w3.weight": TensorSpec((3072, 3584), BF16),
}


def runtime_parameter_manifest(num_experts: int = 896) -> dict[str, TensorSpec]:
    """Expected named_parameters() contract for an unquantized KDA layer."""
    manifest = dict(K3_LAYER_TENSOR_MANIFEST)
    for expert_id in range(num_experts):
        for name, spec in K3_EXPERT_RUNTIME_MANIFEST.items():
            manifest[f"block_sparse_moe.experts.{expert_id}.{name}"] = spec
    return manifest
