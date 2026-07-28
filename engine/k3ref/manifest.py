"""Safetensors-backed manifests for Kimi-family reference layers.

The legacy K3 constants remain the exact layer-12 checkpoint snapshot. New
models use their own config to define the allowed architecture and their own
safetensors headers as the authority for shapes and storage dtypes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .config import K3LayerConfig

BF16 = "BF16"
F32 = "F32"
U8 = "U8"
MXFP4_GROUP_SIZE = 32


@dataclass(frozen=True)
class TensorSpec:
    shape: tuple[int, ...]
    dtype: str


# Exact layer-12 non-expert checkpoint tensors, in safetensors-header order.
K3_LAYER_TENSOR_MANIFEST: dict[str, TensorSpec] = {
    "block_sparse_moe.gate.e_score_correction_bias": TensorSpec((896,), F32),
    "block_sparse_moe.gate.weight": TensorSpec((896, 7168), BF16),
    "block_sparse_moe.routed_expert_down_proj.weight": TensorSpec(
        (3584, 7168), BF16
    ),
    "block_sparse_moe.routed_expert_norm.weight": TensorSpec((3584,), BF16),
    "block_sparse_moe.routed_expert_up_proj.weight": TensorSpec(
        (7168, 3584), BF16
    ),
    "block_sparse_moe.shared_experts.down_proj.weight": TensorSpec(
        (7168, 6144), BF16
    ),
    "block_sparse_moe.shared_experts.gate_proj.weight": TensorSpec(
        (6144, 7168), BF16
    ),
    "block_sparse_moe.shared_experts.up_proj.weight": TensorSpec(
        (6144, 7168), BF16
    ),
    "input_layernorm.weight": TensorSpec((7168,), BF16),
    "mlp_res_norm.weight": TensorSpec((7168,), BF16),
    "mlp_res_proj.weight": TensorSpec((1, 7168), BF16),
    "post_attention_layernorm.weight": TensorSpec((7168,), BF16),
    "self_attention_res_norm.weight": TensorSpec((7168,), BF16),
    "self_attention_res_proj.weight": TensorSpec((1, 7168), BF16),
    "self_attn.A_log": TensorSpec((128,), F32),
    "self_attn.b_proj.weight": TensorSpec((96, 7168), BF16),
    "self_attn.dt_bias": TensorSpec((12288,), F32),
    "self_attn.f_a_proj.weight": TensorSpec((128, 7168), BF16),
    "self_attn.f_b_proj.weight": TensorSpec((12288, 128), BF16),
    "self_attn.g_proj.weight": TensorSpec((12288, 7168), BF16),
    "self_attn.k_conv1d.weight": TensorSpec((12288, 1, 4), F32),
    "self_attn.k_proj.weight": TensorSpec((12288, 7168), BF16),
    "self_attn.o_norm.weight": TensorSpec((128,), F32),
    "self_attn.o_proj.weight": TensorSpec((7168, 12288), BF16),
    "self_attn.q_conv1d.weight": TensorSpec((12288, 1, 4), F32),
    "self_attn.q_proj.weight": TensorSpec((12288, 7168), BF16),
    "self_attn.v_conv1d.weight": TensorSpec((12288, 1, 4), F32),
    "self_attn.v_proj.weight": TensorSpec((12288, 7168), BF16),
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


def _shape_options(*shapes: tuple[int, ...]) -> tuple[tuple[int, ...], ...]:
    return shapes


def expected_layer_tensor_shapes(
    config: K3LayerConfig,
    layer_idx: int,
) -> dict[str, tuple[tuple[int, ...], ...]]:
    """Config-derived shape constraints for one decoder layer.

    The returned shapes validate architecture axes. Storage dtype and the final
    shape choice come from the safetensors header. KDA ``A_log`` accepts the
    two Moonshot layouts observed in the family: per-head and per-head-dimension.
    """
    hidden = config.hidden_size
    shapes: dict[str, tuple[tuple[int, ...], ...]] = {}

    def add(name: str, *allowed: tuple[int, ...]) -> None:
        shapes[name] = _shape_options(*allowed)

    if config.has_moe_layer(layer_idx):
        add("block_sparse_moe.gate.e_score_correction_bias", (config.num_experts,))
        add("block_sparse_moe.gate.weight", (config.num_experts, hidden))
        if config.routed_expert_hidden_size is not None:
            latent = config.routed_expert_hidden_size
            add("block_sparse_moe.routed_expert_down_proj.weight", (latent, hidden))
            if config.latent_moe_use_norm:
                add("block_sparse_moe.routed_expert_norm.weight", (latent,))
            add("block_sparse_moe.routed_expert_up_proj.weight", (hidden, latent))
        if config.num_shared_experts:
            shared = config.moe_intermediate_size * config.num_shared_experts
            add("block_sparse_moe.shared_experts.down_proj.weight", (hidden, shared))
            add("block_sparse_moe.shared_experts.gate_proj.weight", (shared, hidden))
            add("block_sparse_moe.shared_experts.up_proj.weight", (shared, hidden))
    else:
        add("mlp.down_proj.weight", (hidden, config.intermediate_size))
        add("mlp.gate_proj.weight", (config.intermediate_size, hidden))
        add("mlp.up_proj.weight", (config.intermediate_size, hidden))

    add("input_layernorm.weight", (hidden,))
    add("post_attention_layernorm.weight", (hidden,))
    if config.attn_res_block_size is not None:
        add("mlp_res_norm.weight", (hidden,))
        add("mlp_res_proj.weight", (1, hidden))
        add("self_attention_res_norm.weight", (hidden,))
        add("self_attention_res_proj.weight", (1, hidden))

    if config.is_kda_layer(layer_idx):
        projection = config.kda_projection_size
        add(
            "self_attn.A_log",
            (config.kda_head_dim,),
            (config.kda_num_heads,),
        )
        add("self_attn.b_proj.weight", (config.kda_num_heads, hidden))
        add("self_attn.dt_bias", (projection,))
        add("self_attn.f_a_proj.weight", (config.kda_head_dim, hidden))
        add("self_attn.f_b_proj.weight", (projection, config.kda_head_dim))
        if config.kda_use_full_rank_gate:
            add("self_attn.g_proj.weight", (projection, hidden))
        else:
            add("self_attn.g_a_proj.weight", (config.kda_head_dim, hidden))
            add("self_attn.g_b_proj.weight", (projection, config.kda_head_dim))
        for name in ("k_conv1d", "q_conv1d", "v_conv1d"):
            add(
                f"self_attn.{name}.weight",
                (projection, 1, config.short_conv_kernel_size),
            )
        for name in ("k_proj", "q_proj", "v_proj"):
            add(f"self_attn.{name}.weight", (projection, hidden))
        add("self_attn.o_norm.weight", (config.kda_head_dim,))
        add("self_attn.o_proj.weight", (hidden, projection))
    else:
        q_projection = config.num_attention_heads * config.mla_q_head_dim
        if config.q_lora_rank is None:
            add("self_attn.q_proj.weight", (q_projection, hidden))
        else:
            add("self_attn.q_a_layernorm.weight", (config.q_lora_rank,))
            add("self_attn.q_a_proj.weight", (config.q_lora_rank, hidden))
            add("self_attn.q_b_proj.weight", (q_projection, config.q_lora_rank))
        add("self_attn.kv_a_layernorm.weight", (config.kv_lora_rank,))
        add(
            "self_attn.kv_a_proj_with_mqa.weight",
            (config.kv_lora_rank + config.qk_rope_head_dim, hidden),
        )
        add(
            "self_attn.kv_b_proj.weight",
            (
                config.num_attention_heads
                * (config.qk_nope_head_dim + config.v_head_dim),
                config.kv_lora_rank,
            ),
        )
        if config.mla_use_output_gate:
            add(
                "self_attn.g_proj.weight",
                (config.num_attention_heads * config.v_head_dim, hidden),
            )
        add(
            "self_attn.o_proj.weight",
            (hidden, config.num_attention_heads * config.v_head_dim),
        )
    return shapes


def _layer_suffix(name: str, layer_idx: int) -> str | None:
    marker = f"layers.{layer_idx}."
    position = name.find(marker)
    if position < 0:
        return None
    return name[position + len(marker) :]


def _tensor_spec(name: str, entry: Mapping[str, Any]) -> TensorSpec:
    shape = entry.get("shape")
    dtype = entry.get("dtype")
    if not isinstance(shape, (list, tuple)) or not all(
        isinstance(dimension, int) and dimension >= 0 for dimension in shape
    ):
        raise ValueError(f"safetensors header has an invalid shape for {name}")
    if not isinstance(dtype, str) or not dtype:
        raise ValueError(f"safetensors header has an invalid dtype for {name}")
    return TensorSpec(tuple(shape), dtype)


def layer_tensor_manifest(
    config: K3LayerConfig,
    safetensors_header: Mapping[str, Any],
    layer_idx: int,
) -> dict[str, TensorSpec]:
    """Build one non-expert layer manifest from a merged safetensors header."""
    expected = expected_layer_tensor_shapes(config, layer_idx)
    manifest: dict[str, TensorSpec] = {}
    for name, entry in safetensors_header.items():
        if name == "__metadata__" or not isinstance(entry, Mapping):
            continue
        suffix = _layer_suffix(name, layer_idx)
        if suffix is None or "block_sparse_moe.experts." in suffix:
            continue
        manifest[suffix] = _tensor_spec(name, entry)

    missing = sorted(set(expected) - set(manifest))
    unexpected = sorted(set(manifest) - set(expected))
    if missing or unexpected:
        raise ValueError(
            "config and safetensors layer tensors disagree: "
            f"missing={missing}, unexpected={unexpected}"
        )
    for name, spec in manifest.items():
        if spec.shape not in expected[name]:
            raise ValueError(
                f"config and safetensors shape disagree for {name}: "
                f"header={spec.shape}, allowed={expected[name]}"
            )
    return manifest


def expert_checkpoint_manifest(
    config: K3LayerConfig,
    safetensors_header: Mapping[str, Any],
    layer_idx: int,
    expert_id: int = 0,
) -> dict[str, TensorSpec]:
    """Build a reusable one-expert checkpoint template from the real header."""
    marker = f"block_sparse_moe.experts.{expert_id}."
    manifest: dict[str, TensorSpec] = {}
    for name, entry in safetensors_header.items():
        if not isinstance(entry, Mapping):
            continue
        suffix = _layer_suffix(name, layer_idx)
        if suffix is None or marker not in suffix:
            continue
        normalized = suffix.replace(marker, "block_sparse_moe.experts.{expert}.", 1)
        manifest[normalized] = _tensor_spec(name, entry)
    if not manifest:
        raise ValueError(f"safetensors header has no layer {layer_idx} expert {expert_id}")

    runtime = expert_runtime_manifest(config)
    for name, spec in runtime.items():
        checkpoint_name = f"block_sparse_moe.experts.{{expert}}.{name}"
        if checkpoint_name in manifest and manifest[checkpoint_name].shape != spec.shape:
            raise ValueError(
                f"config and safetensors expert shape disagree for {checkpoint_name}: "
                f"header={manifest[checkpoint_name].shape}, expected={spec.shape}"
            )
    return manifest


def expert_runtime_manifest(
    config: K3LayerConfig,
    dtype: str = BF16,
) -> dict[str, TensorSpec]:
    """Config-derived unquantized expert parameters used by the reference."""
    hidden = config.expert_hidden_size
    intermediate = config.moe_intermediate_size
    return {
        "w1.weight": TensorSpec((intermediate, hidden), dtype),
        "w2.weight": TensorSpec((hidden, intermediate), dtype),
        "w3.weight": TensorSpec((intermediate, hidden), dtype),
    }


def runtime_parameter_manifest(
    num_experts: int = 896,
    *,
    layer_manifest: Mapping[str, TensorSpec] | None = None,
    expert_manifest: Mapping[str, TensorSpec] | None = None,
) -> dict[str, TensorSpec]:
    """Expected named_parameters() contract for an unquantized KDA layer."""
    manifest = dict(layer_manifest or K3_LAYER_TENSOR_MANIFEST)
    experts = expert_manifest or K3_EXPERT_RUNTIME_MANIFEST
    for expert_id in range(num_experts):
        for name, spec in experts.items():
            manifest[f"block_sparse_moe.experts.{expert_id}.{name}"] = spec
    return manifest
