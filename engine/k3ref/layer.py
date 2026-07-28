"""Assembly and raw-checkpoint loading for one Kimi K3 decoder layer."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from .attention import KDAAttention, KDAState, MLAAttention, MLAState
from .config import K3LayerConfig
from .manifest import K3_LAYER_TENSOR_MANIFEST
from .moe import LatentMoE
from .norm import RMSNorm, apply_attention_residual
from .weights import MXFP4ExpertProvider, RawTensorStore

AttentionState = KDAState | MLAState | None


@dataclass
class K3LayerOutput:
    hidden_states: torch.Tensor
    attention_state: AttentionState
    block_residual: torch.Tensor | None
    router_indices: torch.Tensor
    router_weights: torch.Tensor


def _replace_parameter(
    module: nn.Module,
    name: str,
    tensor: torch.Tensor,
) -> None:
    old = getattr(module, name)
    if tuple(old.shape) != tuple(tensor.shape):
        raise ValueError(
            f"shape mismatch for {module.__class__.__name__}.{name}: "
            f"expected {tuple(old.shape)}, got {tuple(tensor.shape)}"
        )
    module._parameters[name] = nn.Parameter(tensor, requires_grad=False)


class K3ReferenceLayer(nn.Module):
    def __init__(
        self,
        config: K3LayerConfig,
        layer_idx: int,
        *,
        expert_provider=None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if layer_idx == 0:
            raise ValueError("layer 0 is dense; this reference lane implements a K3 MoE layer")
        self.config = config
        self.layer_idx = layer_idx
        self.is_kda = config.is_kda_layer(layer_idx)
        if self.is_kda:
            self.self_attn = KDAAttention(
                config.hidden_size,
                config.kda_num_heads,
                config.kda_head_dim,
                conv_size=config.short_conv_kernel_size,
                gate_lower_bound=config.kda_gate_lower_bound,
                rms_norm_eps=config.rms_norm_eps,
                device=device,
                dtype=dtype,
            )
        else:
            self.self_attn = MLAAttention(
                config.hidden_size,
                config.num_attention_heads,
                config.num_key_value_heads,
                config.q_lora_rank,
                config.kv_lora_rank,
                config.qk_nope_head_dim,
                config.qk_rope_head_dim,
                config.v_head_dim,
                use_output_gate=config.mla_use_output_gate,
                # Moonshot constructs the MLA low-rank norms with the 1e-6 default.
                rms_norm_eps=1e-6,
                device=device,
                dtype=dtype,
            )

        self.block_sparse_moe = LatentMoE(
            config.hidden_size,
            config.routed_expert_hidden_size,
            config.moe_intermediate_size,
            config.num_experts,
            config.num_experts_per_token,
            num_shared_experts=config.num_shared_experts,
            num_expert_group=config.num_expert_group,
            topk_group=config.topk_group,
            renormalize=config.moe_renormalize,
            routed_scaling_factor=config.routed_scaling_factor,
            rms_norm_eps=config.rms_norm_eps,
            situ_beta=config.activation_situ_beta,
            situ_linear_beta=config.activation_situ_linear_beta,
            expert_provider=expert_provider,
            device=device,
            dtype=dtype,
        )
        self.input_layernorm = RMSNorm(
            config.hidden_size, config.rms_norm_eps, device=device, dtype=dtype
        )
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, config.rms_norm_eps, device=device, dtype=dtype
        )

        self.use_attn_residuals = config.attn_res_block_size is not None
        if self.use_attn_residuals:
            self.self_attention_res_norm = RMSNorm(
                config.hidden_size, config.rms_norm_eps, device=device, dtype=dtype
            )
            self.mlp_res_norm = RMSNorm(
                config.hidden_size, config.rms_norm_eps, device=device, dtype=dtype
            )
            self.self_attention_res_proj = nn.Linear(
                config.hidden_size, 1, bias=False, device=device, dtype=dtype
            )
            self.mlp_res_proj = nn.Linear(
                config.hidden_size, 1, bias=False, device=device, dtype=dtype
            )

    def _attention(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None,
        state: AttentionState,
    ) -> tuple[torch.Tensor, AttentionState]:
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
        attention_state: AttentionState = None,
        block_residual: torch.Tensor | None = None,
        *,
        return_aux: bool = False,
    ) -> torch.Tensor | K3LayerOutput:
        if not self.use_attn_residuals:
            residual = hidden_states
            normalized = self.input_layernorm(hidden_states)
            attended, next_attention_state = self._attention(
                normalized, attention_mask, attention_state
            )
            hidden_states = residual + attended
            residual = hidden_states
            moe_output, indices, weights = self.block_sparse_moe(
                self.post_attention_layernorm(hidden_states), return_router=True
            )
            hidden_states = residual + moe_output
            next_block_residual = None
        else:
            batch, sequence, hidden_size = hidden_states.shape
            prefix_sum = hidden_states
            if block_residual is None:
                block_residual = hidden_states.new_zeros(
                    batch * sequence, 0, hidden_size
                )
            if block_residual.shape[1] > 0:
                hidden_states = apply_attention_residual(
                    prefix_sum.view(-1, hidden_size),
                    block_residual,
                    self.self_attention_res_proj.weight,
                    self.self_attention_res_norm.weight,
                    self.config.rms_norm_eps,
                ).view(batch, sequence, hidden_size)

            if self.layer_idx % self.config.attn_res_block_size == 0:
                block_residual = torch.cat(
                    (
                        block_residual,
                        prefix_sum.view(-1, hidden_size).unsqueeze(1),
                    ),
                    dim=1,
                )
                prefix_sum = None

            normalized = self.input_layernorm(hidden_states)
            attended, next_attention_state = self._attention(
                normalized, attention_mask, attention_state
            )
            prefix_sum = attended if prefix_sum is None else prefix_sum + attended
            moe_input = apply_attention_residual(
                prefix_sum.view(-1, hidden_size),
                block_residual,
                self.mlp_res_proj.weight,
                self.mlp_res_norm.weight,
                self.config.rms_norm_eps,
            ).view(batch, sequence, hidden_size)
            moe_output, indices, weights = self.block_sparse_moe(
                self.post_attention_layernorm(moe_input), return_router=True
            )
            hidden_states = prefix_sum + moe_output
            next_block_residual = block_residual

        if return_aux:
            return K3LayerOutput(
                hidden_states,
                next_attention_state,
                next_block_residual,
                indices,
                weights,
            )
        return hidden_states

    @classmethod
    def from_directory(
        cls,
        directory: str | Path,
        layer_idx: int,
        *,
        config: K3LayerConfig | None = None,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.bfloat16,
    ) -> "K3ReferenceLayer":
        config = config or K3LayerConfig()
        store = RawTensorStore(directory)
        provider = MXFP4ExpertProvider(store, layer_idx)
        layer = cls(
            config,
            layer_idx,
            expert_provider=provider,
            device="meta",
            dtype=dtype,
        )
        layer._load_raw_weights(store, device=device, dtype=dtype)
        layer.eval()
        return layer

    def _load_raw_weights(
        self,
        store: RawTensorStore,
        *,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> None:
        prefix = f"layers.{self.layer_idx}."
        loaded_manifest_names: set[str] = set()

        def load_parameter(
            module: nn.Module,
            parameter: str,
            suffix: str,
            cast: torch.dtype | None = dtype,
        ) -> None:
            if self.is_kda and suffix not in K3_LAYER_TENSOR_MANIFEST:
                raise KeyError(f"KDA tensor is absent from the layer-12 manifest: {suffix}")
            checkpoint_spec = K3_LAYER_TENSOR_MANIFEST.get(suffix)
            if checkpoint_spec is not None:
                store.validate(prefix + suffix, checkpoint_spec)
                if self.is_kda:
                    loaded_manifest_names.add(suffix)
            tensor = store.load(prefix + suffix, device=device, dtype=cast)
            _replace_parameter(module, parameter, tensor)

        if self.is_kda:
            for name in ("q_proj", "k_proj", "v_proj", "f_a_proj", "f_b_proj", "b_proj", "g_proj", "o_proj"):
                load_parameter(
                    getattr(self.self_attn, name),
                    "weight",
                    f"self_attn.{name}.weight",
                )
            for name in ("q_conv1d", "k_conv1d", "v_conv1d"):
                load_parameter(
                    getattr(self.self_attn, name),
                    "weight",
                    f"self_attn.{name}.weight",
                    cast=None,
                )
            load_parameter(self.self_attn, "A_log", "self_attn.A_log", cast=None)
            load_parameter(self.self_attn, "dt_bias", "self_attn.dt_bias", cast=None)
            load_parameter(
                self.self_attn.o_norm,
                "weight",
                "self_attn.o_norm.weight",
                cast=None,
            )
        else:
            for name in (
                "q_a_proj",
                "q_b_proj",
                "kv_a_proj_with_mqa",
                "kv_b_proj",
                "o_proj",
                "g_proj",
            ):
                module = getattr(self.self_attn, name)
                if module is not None:
                    load_parameter(module, "weight", f"self_attn.{name}.weight")
            load_parameter(
                self.self_attn.q_a_layernorm,
                "weight",
                "self_attn.q_a_layernorm.weight",
            )
            load_parameter(
                self.self_attn.kv_a_layernorm,
                "weight",
                "self_attn.kv_a_layernorm.weight",
            )

        moe = self.block_sparse_moe
        load_parameter(moe.gate, "weight", "block_sparse_moe.gate.weight")
        load_parameter(
            moe.gate,
            "e_score_correction_bias",
            "block_sparse_moe.gate.e_score_correction_bias",
            cast=None,
        )
        load_parameter(
            moe.routed_expert_down_proj,
            "weight",
            "block_sparse_moe.routed_expert_down_proj.weight",
        )
        load_parameter(
            moe.routed_expert_norm,
            "weight",
            "block_sparse_moe.routed_expert_norm.weight",
        )
        load_parameter(
            moe.routed_expert_up_proj,
            "weight",
            "block_sparse_moe.routed_expert_up_proj.weight",
        )
        if moe.shared_experts is not None:
            for name in ("gate_proj", "up_proj", "down_proj"):
                load_parameter(
                    getattr(moe.shared_experts, name),
                    "weight",
                    f"block_sparse_moe.shared_experts.{name}.weight",
                )

        load_parameter(self.input_layernorm, "weight", "input_layernorm.weight")
        load_parameter(
            self.post_attention_layernorm,
            "weight",
            "post_attention_layernorm.weight",
        )
        if self.use_attn_residuals:
            load_parameter(
                self.self_attention_res_norm,
                "weight",
                "self_attention_res_norm.weight",
            )
            load_parameter(
                self.mlp_res_norm,
                "weight",
                "mlp_res_norm.weight",
            )
            load_parameter(
                self.self_attention_res_proj,
                "weight",
                "self_attention_res_proj.weight",
            )
            load_parameter(
                self.mlp_res_proj,
                "weight",
                "mlp_res_proj.weight",
            )

        if self.is_kda:
            expected_manifest_names = set(K3_LAYER_TENSOR_MANIFEST)
            if loaded_manifest_names != expected_manifest_names:
                missing = sorted(expected_manifest_names - loaded_manifest_names)
                unexpected = sorted(loaded_manifest_names - expected_manifest_names)
                raise ValueError(
                    "KDA layer loader and checkpoint manifest disagree: "
                    f"missing={missing}, unexpected={unexpected}"
                )
