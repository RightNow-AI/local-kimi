"""Configuration values needed by the single-layer reference."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

_FULL_ATTENTION_LAYERS = tuple(range(4, 94, 4)) + (93,)


@dataclass(frozen=True)
class K3LayerConfig:
    hidden_size: int = 7168
    num_attention_heads: int = 96
    num_key_value_heads: int = 96
    q_lora_rank: int = 1536
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128
    mla_use_output_gate: bool = True
    kda_head_dim: int = 128
    kda_num_heads: int = 96
    short_conv_kernel_size: int = 4
    kda_gate_lower_bound: float | None = -5.0
    routed_expert_hidden_size: int = 3584
    moe_intermediate_size: int = 3072
    num_experts: int = 896
    num_experts_per_token: int = 16
    num_shared_experts: int = 2
    num_expert_group: int = 1
    topk_group: int = 1
    moe_renormalize: bool = True
    routed_scaling_factor: float = 1.0
    rms_norm_eps: float = 1e-5
    activation_situ_beta: float = 4.0
    activation_situ_linear_beta: float | None = 25.0
    attn_res_block_size: int | None = 12
    full_attention_layers: tuple[int, ...] = field(
        default_factory=lambda: _FULL_ATTENTION_LAYERS
    )

    @classmethod
    def from_json(cls, path: str | Path) -> "K3LayerConfig":
        with Path(path).open("r", encoding="utf-8") as handle:
            text = json.load(handle)["text_config"]
        linear = text["linear_attn_config"]
        return cls(
            hidden_size=text["hidden_size"],
            num_attention_heads=text["num_attention_heads"],
            num_key_value_heads=text["num_key_value_heads"],
            q_lora_rank=text["q_lora_rank"],
            kv_lora_rank=text["kv_lora_rank"],
            qk_nope_head_dim=text["qk_nope_head_dim"],
            qk_rope_head_dim=text["qk_rope_head_dim"],
            v_head_dim=text["v_head_dim"],
            mla_use_output_gate=text["mla_use_output_gate"],
            kda_head_dim=linear["head_dim"],
            kda_num_heads=linear["num_heads"],
            short_conv_kernel_size=linear["short_conv_kernel_size"],
            kda_gate_lower_bound=linear.get("gate_lower_bound"),
            routed_expert_hidden_size=text["routed_expert_hidden_size"],
            moe_intermediate_size=text["moe_intermediate_size"],
            num_experts=text["num_experts"],
            num_experts_per_token=text["num_experts_per_token"],
            num_shared_experts=text["num_shared_experts"],
            num_expert_group=text["num_expert_group"],
            topk_group=text["topk_group"],
            moe_renormalize=text["moe_renormalize"],
            routed_scaling_factor=text["routed_scaling_factor"],
            rms_norm_eps=text["rms_norm_eps"],
            activation_situ_beta=text["activation_situ_beta"],
            activation_situ_linear_beta=text["activation_situ_linear_beta"],
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
