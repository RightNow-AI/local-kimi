"""Configuration and total layer classification for Kimi-Linear."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Mapping

LayerKind = Literal["dense", "kda", "mla"]
AttentionKind = Literal["kda", "mla"]

_REAL_FULL_ATTENTION_LAYERS = (4, 8, 12, 16, 20, 24, 27)
_REAL_KDA_LAYERS = (
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
)


@dataclass(frozen=True)
class KLinearConfig:
    model_type: str = "kimi_linear"
    vocab_size: int = 163_840
    hidden_size: int = 2_304
    head_dim: int = 72
    intermediate_size: int = 9_216
    num_hidden_layers: int = 27
    num_attention_heads: int = 32
    num_key_value_heads: int = 32
    hidden_act: str = "silu"
    rms_norm_eps: float = 1e-5
    q_lora_rank: int | None = None
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128
    mla_use_nope: bool = True
    kda_num_heads: int = 32
    kda_head_dim: int = 128
    short_conv_kernel_size: int = 4
    num_experts: int = 256
    num_experts_per_token: int = 8
    num_shared_experts: int = 1
    moe_intermediate_size: int = 1_024
    first_k_dense_replace: int = 1
    moe_layer_freq: int = 1
    moe_renormalize: bool = True
    moe_router_activation_func: str = "sigmoid"
    routed_scaling_factor: float = 2.446
    use_grouped_topk: bool = True
    num_expert_group: int = 1
    topk_group: int = 1
    bos_token_id: int = 163_584
    eos_token_id: int = 163_586
    pad_token_id: int = 163_839
    tie_word_embeddings: bool = False
    rope_theta: float = 10_000.0
    rope_scaling: Mapping[str, Any] | None = None
    full_attention_layers: tuple[int, ...] = field(
        default_factory=lambda: _REAL_FULL_ATTENTION_LAYERS
    )
    kda_layers: tuple[int, ...] = field(default_factory=lambda: _REAL_KDA_LAYERS)

    def __post_init__(self) -> None:
        if self.model_type != "kimi_linear":
            raise ValueError("model_type must be kimi_linear")
        if self.hidden_act != "silu":
            raise ValueError("the Kimi-Linear engine implements SwiGLU with silu")
        if self.q_lora_rank is not None:
            raise ValueError("Kimi-Linear requires direct MLA q_proj weights")
        if not self.mla_use_nope:
            raise ValueError("this checkpoint requires MLA no-position-embedding slices")
        if self.tie_word_embeddings:
            raise ValueError("the real Kimi-Linear checkpoint has an untied LM head")
        if self.num_hidden_layers <= 0:
            raise ValueError("num_hidden_layers must be positive")
        if not 0 <= self.first_k_dense_replace <= self.num_hidden_layers:
            raise ValueError("first_k_dense_replace is outside the layer range")
        if self.moe_layer_freq <= 0:
            raise ValueError("moe_layer_freq must be positive")
        if not 0 < self.num_experts_per_token <= self.num_experts:
            raise ValueError("num_experts_per_token must be in the expert range")
        if self.num_experts % self.num_expert_group:
            raise ValueError("num_experts must be divisible by num_expert_group")
        if not 0 < self.topk_group <= self.num_expert_group:
            raise ValueError("topk_group must be in the expert-group range")

        one_based_layers = set(range(1, self.num_hidden_layers + 1))
        kda = set(self.kda_layers)
        mla = set(self.full_attention_layers)
        overlap = kda & mla
        if overlap:
            raise ValueError(f"attention layer lists overlap: {sorted(overlap)}")
        missing = one_based_layers - (kda | mla)
        unexpected = (kda | mla) - one_based_layers
        if missing or unexpected:
            raise ValueError(
                "attention layer lists must classify every layer exactly once: "
                f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
            )
        for layer_idx in range(self.num_hidden_layers):
            self.attention_kind(layer_idx)
            self.layer_kind(layer_idx)

    @classmethod
    def from_json(cls, path: str | Path) -> "KLinearConfig":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError("config JSON must contain an object")
        return cls.from_mapping(payload)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "KLinearConfig":
        linear = payload["linear_attn_config"]
        return cls(
            model_type=payload.get("model_type", "kimi_linear"),
            vocab_size=payload["vocab_size"],
            hidden_size=payload["hidden_size"],
            head_dim=payload.get(
                "head_dim", payload["hidden_size"] // payload["num_attention_heads"]
            ),
            intermediate_size=payload["intermediate_size"],
            num_hidden_layers=payload["num_hidden_layers"],
            num_attention_heads=payload["num_attention_heads"],
            num_key_value_heads=payload.get(
                "num_key_value_heads", payload["num_attention_heads"]
            ),
            hidden_act=payload.get("hidden_act", "silu"),
            rms_norm_eps=payload.get("rms_norm_eps", 1e-6),
            q_lora_rank=payload.get("q_lora_rank"),
            kv_lora_rank=payload["kv_lora_rank"],
            qk_nope_head_dim=payload["qk_nope_head_dim"],
            qk_rope_head_dim=payload["qk_rope_head_dim"],
            v_head_dim=payload["v_head_dim"],
            mla_use_nope=payload.get("mla_use_nope", False),
            kda_num_heads=linear["num_heads"],
            kda_head_dim=linear["head_dim"],
            short_conv_kernel_size=linear["short_conv_kernel_size"],
            num_experts=payload["num_experts"],
            num_experts_per_token=payload["num_experts_per_token"],
            num_shared_experts=payload.get("num_shared_experts") or 0,
            moe_intermediate_size=payload["moe_intermediate_size"],
            first_k_dense_replace=payload.get("first_k_dense_replace", 0),
            moe_layer_freq=payload.get("moe_layer_freq", 1),
            moe_renormalize=payload.get("moe_renormalize", True),
            moe_router_activation_func=payload.get(
                "moe_router_activation_func", "sigmoid"
            ),
            routed_scaling_factor=payload.get("routed_scaling_factor", 1.0),
            use_grouped_topk=payload.get("use_grouped_topk", True),
            num_expert_group=payload.get("num_expert_group", 1),
            topk_group=payload.get("topk_group", 1),
            bos_token_id=payload.get("bos_token_id", 1),
            eos_token_id=payload.get("eos_token_id", 2),
            pad_token_id=payload.get("pad_token_id", 0),
            tie_word_embeddings=payload.get("tie_word_embeddings", False),
            rope_theta=payload.get("rope_theta", 10_000.0),
            rope_scaling=payload.get("rope_scaling"),
            full_attention_layers=tuple(linear["full_attn_layers"]),
            kda_layers=tuple(linear["kda_layers"]),
        )

    def _check_layer_index(self, layer_idx: int) -> None:
        if not 0 <= layer_idx < self.num_hidden_layers:
            raise IndexError(
                f"layer index {layer_idx} is outside 0..{self.num_hidden_layers - 1}"
            )

    def attention_kind(self, layer_idx: int) -> AttentionKind:
        self._check_layer_index(layer_idx)
        one_based = layer_idx + 1
        in_kda = one_based in self.kda_layers
        in_mla = one_based in self.full_attention_layers
        if in_kda == in_mla:
            raise ValueError(f"layer {layer_idx} is not classified exactly once")
        return "kda" if in_kda else "mla"

    def has_moe_layer(self, layer_idx: int) -> bool:
        self._check_layer_index(layer_idx)
        return (
            self.num_experts > 0
            and layer_idx >= self.first_k_dense_replace
            and layer_idx % self.moe_layer_freq == 0
        )

    def layer_kind(self, layer_idx: int) -> LayerKind:
        attention = self.attention_kind(layer_idx)
        if not self.has_moe_layer(layer_idx):
            return "dense"
        return attention

    @property
    def kda_projection_size(self) -> int:
        return self.kda_num_heads * self.kda_head_dim

    @property
    def mla_q_head_dim(self) -> int:
        return self.qk_nope_head_dim + self.qk_rope_head_dim
