"""One decoder layer and its real-checkpoint loader."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from engine.quant.w4a16 import W4A16Tensor

from .attention import KDAAttention, MLAAttention
from .config import KLinearConfig, LayerKind
from .manifest import TensorSpec, real_layer_manifest
from .moe import DenseMLP, ExpertProvider, KLinearMoE
from .norm import RMSNorm
from .quantized import LinearFactory, W4A16Linear
from .state import KDALayerState, LayerState, MLALayerState
from .weights import SafetensorIndexStore


def _replace_parameter(module: nn.Module, name: str, tensor: torch.Tensor) -> None:
    old = getattr(module, name)
    if tuple(old.shape) != tuple(tensor.shape):
        raise ValueError(
            f"shape mismatch for {module.__class__.__name__}.{name}: "
            f"expected {tuple(old.shape)}, got {tuple(tensor.shape)}"
        )
    module._parameters[name] = nn.Parameter(tensor, requires_grad=False)


@dataclass
class KLinearLayerOutput:
    hidden_states: torch.Tensor
    attention_state: LayerState
    router_indices: torch.Tensor
    router_weights: torch.Tensor


class KLinearDecoderLayer(nn.Module):
    def __init__(
        self,
        config: KLinearConfig,
        layer_idx: int,
        *,
        expert_provider: ExpertProvider | None = None,
        linear_factory: LinearFactory | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.kind: LayerKind = config.layer_kind(layer_idx)
        self.attention_kind = config.attention_kind(layer_idx)
        self.is_kda = self.attention_kind == "kda"
        if self.is_kda:
            self.self_attn = KDAAttention(
                config.hidden_size,
                config.kda_num_heads,
                config.kda_head_dim,
                conv_size=config.short_conv_kernel_size,
                rms_norm_eps=config.rms_norm_eps,
                tensor_prefix=f"model.layers.{layer_idx}.self_attn",
                linear_factory=linear_factory,
                device=device,
                dtype=dtype,
            )
        else:
            self.self_attn = MLAAttention(
                config.hidden_size,
                config.num_attention_heads,
                config.num_key_value_heads,
                config.kv_lora_rank,
                config.qk_nope_head_dim,
                config.qk_rope_head_dim,
                config.v_head_dim,
                rms_norm_eps=1e-6,
                tensor_prefix=f"model.layers.{layer_idx}.self_attn",
                linear_factory=linear_factory,
                device=device,
                dtype=dtype,
            )

        if config.has_moe_layer(layer_idx):
            self.block_sparse_moe = KLinearMoE(
                layer_idx,
                config.hidden_size,
                config.moe_intermediate_size,
                config.num_experts,
                config.num_experts_per_token,
                num_shared_experts=config.num_shared_experts,
                use_grouped_topk=config.use_grouped_topk,
                num_expert_group=config.num_expert_group,
                topk_group=config.topk_group,
                renormalize=config.moe_renormalize,
                routed_scaling_factor=config.routed_scaling_factor,
                router_activation=config.moe_router_activation_func,
                expert_provider=expert_provider,
                tensor_prefix=f"model.layers.{layer_idx}.block_sparse_moe",
                linear_factory=linear_factory,
                device=device,
                dtype=dtype,
            )
            self.mlp = None
        else:
            self.block_sparse_moe = None
            self.mlp = DenseMLP(
                config.hidden_size,
                config.intermediate_size,
                tensor_prefix=f"model.layers.{layer_idx}.mlp",
                linear_factory=linear_factory,
                device=device,
                dtype=dtype,
            )
        self.input_layernorm = RMSNorm(
            config.hidden_size,
            config.rms_norm_eps,
            device=device,
            dtype=dtype,
        )
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size,
            config.rms_norm_eps,
            device=device,
            dtype=dtype,
        )

    def _run_attention(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None,
        state: LayerState,
    ) -> tuple[torch.Tensor, LayerState]:
        if self.is_kda:
            if state is not None and not isinstance(state, KDALayerState):
                raise TypeError(f"layer {self.layer_idx} requires KDALayerState")
        elif state is not None and not isinstance(state, MLALayerState):
            raise TypeError(f"layer {self.layer_idx} requires MLALayerState")
        output, next_state = self.self_attn(
            hidden_states,
            attention_mask=attention_mask,
            state=state,
            return_state=True,
        )
        return output, next_state

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        state: LayerState = None,
        *,
        return_aux: bool = False,
    ) -> torch.Tensor | KLinearLayerOutput:
        residual = hidden_states
        attended, next_state = self._run_attention(
            self.input_layernorm(hidden_states), attention_mask, state
        )
        hidden_states = residual + attended
        residual = hidden_states
        feed_forward_input = self.post_attention_layernorm(hidden_states)
        if self.block_sparse_moe is not None:
            feed_forward, indices, weights = self.block_sparse_moe(
                feed_forward_input, return_router=True
            )
        else:
            feed_forward = self.mlp(feed_forward_input)
            tokens = hidden_states.shape[0] * hidden_states.shape[1]
            indices = torch.empty(
                tokens, 0, device=hidden_states.device, dtype=torch.long
            )
            weights = hidden_states.new_empty(tokens, 0)
        hidden_states = residual + feed_forward
        if return_aux:
            return KLinearLayerOutput(hidden_states, next_state, indices, weights)
        return hidden_states

    def load_checkpoint_weights(
        self,
        store: SafetensorIndexStore,
        *,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> None:
        prefix = f"model.layers.{self.layer_idx}."
        expected = real_layer_manifest(self.layer_idx, include_experts=False)
        loaded: set[str] = set()

        def load_parameter(
            module: nn.Module,
            parameter: str,
            suffix: str,
            cast: torch.dtype | None = dtype,
        ) -> None:
            spec: TensorSpec = expected[suffix]
            store.validate(prefix + suffix, spec)
            tensor = store.load(prefix + suffix, device=device, dtype=cast)
            _replace_parameter(module, parameter, tensor)
            loaded.add(suffix)

        def load_linear(module: nn.Module, suffix: str) -> None:
            spec = expected[suffix]
            if len(spec.shape) != 2 or spec.dtype != "BF16":
                raise ValueError(f"linear tensor has an invalid source spec: {suffix}")
            payload = store.load_linear_weight(
                prefix + suffix,
                spec.shape,
                device=device,
                dtype=dtype,
            )
            if isinstance(payload, W4A16Tensor):
                if not isinstance(module, W4A16Linear):
                    raise TypeError(
                        f"{prefix + suffix} is packed but the model built a BF16 linear"
                    )
                module.load_encoded(payload)
            else:
                if isinstance(module, W4A16Linear):
                    raise TypeError(
                        f"{prefix + suffix} is retained BF16 but the model built W4A16"
                    )
                _replace_parameter(module, "weight", payload)
            loaded.add(suffix)

        if self.is_kda:
            for name in (
                "q_proj",
                "k_proj",
                "v_proj",
                "f_a_proj",
                "f_b_proj",
                "b_proj",
                "g_a_proj",
                "g_b_proj",
                "o_proj",
            ):
                load_linear(
                    getattr(self.self_attn, name), f"self_attn.{name}.weight"
                )
            for name in ("q_conv1d", "k_conv1d", "v_conv1d"):
                load_parameter(
                    getattr(self.self_attn, name),
                    "weight",
                    f"self_attn.{name}.weight",
                )
            load_parameter(self.self_attn, "A_log", "self_attn.A_log", cast=None)
            load_parameter(self.self_attn, "dt_bias", "self_attn.dt_bias", cast=None)
            load_parameter(
                self.self_attn.o_norm, "weight", "self_attn.o_norm.weight"
            )
        else:
            for name in ("q_proj", "kv_a_proj_with_mqa", "kv_b_proj", "o_proj"):
                load_linear(
                    getattr(self.self_attn, name), f"self_attn.{name}.weight"
                )
            load_parameter(
                self.self_attn.kv_a_layernorm,
                "weight",
                "self_attn.kv_a_layernorm.weight",
            )

        if self.block_sparse_moe is not None:
            moe = self.block_sparse_moe
            load_parameter(moe.gate, "weight", "block_sparse_moe.gate.weight")
            load_parameter(
                moe.gate,
                "e_score_correction_bias",
                "block_sparse_moe.gate.e_score_correction_bias",
            )
            if moe.shared_experts is not None:
                for name in ("gate_proj", "up_proj", "down_proj"):
                    load_linear(
                        getattr(moe.shared_experts, name),
                        f"block_sparse_moe.shared_experts.{name}.weight",
                    )
        else:
            for name in ("gate_proj", "up_proj", "down_proj"):
                load_linear(getattr(self.mlp, name), f"mlp.{name}.weight")

        load_parameter(self.input_layernorm, "weight", "input_layernorm.weight")
        load_parameter(
            self.post_attention_layernorm,
            "weight",
            "post_attention_layernorm.weight",
        )
        if loaded != set(expected):
            raise ValueError(
                f"layer {self.layer_idx} loader and manifest disagree: "
                f"missing={sorted(set(expected) - loaded)}, "
                f"unexpected={sorted(loaded - set(expected))}"
            )

