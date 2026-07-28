"""Configuration values needed by the Kimi architecture reference."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

_FULL_ATTENTION_LAYERS = tuple(range(4, 94, 4)) + (93,)


@dataclass(frozen=True)
class K3LayerConfig:
    model_type: str = "kimi_linear"
    vocab_size: int = 163840
    hidden_size: int = 7168
    intermediate_size: int = 33792
    num_hidden_layers: int = 93
    num_attention_heads: int = 96
    num_key_value_heads: int = 96
    q_lora_rank: int | None = 1536
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128
    mla_use_output_gate: bool = True
    kda_head_dim: int = 128
    kda_num_heads: int = 96
    short_conv_kernel_size: int = 4
    kda_gate_lower_bound: float | None = -5.0
    kda_use_full_rank_gate: bool = True
    routed_expert_hidden_size: int | None = 3584
    moe_intermediate_size: int = 3072
    num_experts: int = 896
    num_experts_per_token: int = 16
    num_shared_experts: int = 2
    num_expert_group: int = 1
    topk_group: int = 1
    first_k_dense_replace: int = 1
    moe_layer_freq: int = 1
    latent_moe_use_norm: bool = True
    moe_renormalize: bool = True
    routed_scaling_factor: float = 1.0
    rms_norm_eps: float = 1e-5
    activation_situ_beta: float | None = 4.0
    activation_situ_linear_beta: float | None = 25.0
    attn_res_block_size: int | None = 12
    full_attention_layers: tuple[int, ...] = field(
        default_factory=lambda: _FULL_ATTENTION_LAYERS
    )

    @classmethod
    def from_json(cls, path: str | Path) -> "K3LayerConfig":
        with Path(path).open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return cls.from_mapping(payload)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "K3LayerConfig":
        """Load either K3's nested config or a direct Kimi-Linear config."""
        text = payload.get("text_config", payload)
        linear = text["linear_attn_config"]
        return cls(
            model_type=text.get("model_type", "kimi_linear"),
            vocab_size=text["vocab_size"],
            hidden_size=text["hidden_size"],
            intermediate_size=text["intermediate_size"],
            num_hidden_layers=text["num_hidden_layers"],
            num_attention_heads=text["num_attention_heads"],
            num_key_value_heads=text["num_key_value_heads"],
            q_lora_rank=text.get("q_lora_rank"),
            kv_lora_rank=text["kv_lora_rank"],
            qk_nope_head_dim=text["qk_nope_head_dim"],
            qk_rope_head_dim=text["qk_rope_head_dim"],
            v_head_dim=text["v_head_dim"],
            mla_use_output_gate=text.get("mla_use_output_gate", False),
            kda_head_dim=linear["head_dim"],
            kda_num_heads=linear["num_heads"],
            short_conv_kernel_size=linear["short_conv_kernel_size"],
            kda_gate_lower_bound=linear.get("gate_lower_bound"),
            kda_use_full_rank_gate=linear.get("use_full_rank_gate", False),
            routed_expert_hidden_size=text.get("routed_expert_hidden_size"),
            moe_intermediate_size=text["moe_intermediate_size"],
            num_experts=text["num_experts"],
            num_experts_per_token=text["num_experts_per_token"],
            num_shared_experts=text.get("num_shared_experts") or 0,
            num_expert_group=text.get("num_expert_group", 1),
            topk_group=text.get("topk_group", 1),
            first_k_dense_replace=text.get("first_k_dense_replace", 0),
            moe_layer_freq=text.get("moe_layer_freq", 1),
            latent_moe_use_norm=text.get("latent_moe_use_norm", False),
            moe_renormalize=text["moe_renormalize"],
            routed_scaling_factor=text["routed_scaling_factor"],
            rms_norm_eps=text["rms_norm_eps"],
            activation_situ_beta=text.get("activation_situ_beta"),
            activation_situ_linear_beta=text.get("activation_situ_linear_beta"),
            attn_res_block_size=text.get("attn_res_block_size"),
            full_attention_layers=tuple(linear["full_attn_layers"]),
        )

    def is_kda_layer(self, layer_idx: int) -> bool:
        # Moonshot stores attention layer numbers as one-based values.
        return (layer_idx + 1) not in self.full_attention_layers

    @property
    def kda_projection_size(self) -> int:
        return self.kda_num_heads * self.kda_head_dim

    @property
    def mla_q_head_dim(self) -> int:
        return self.qk_nope_head_dim + self.qk_rope_head_dim

    @property
    def expert_hidden_size(self) -> int:
        return self.routed_expert_hidden_size or self.hidden_size

    def has_moe_layer(self, layer_idx: int) -> bool:
        return (
            self.num_experts > 0
            and layer_idx >= self.first_k_dense_replace
            and layer_idx % self.moe_layer_freq == 0
        )
