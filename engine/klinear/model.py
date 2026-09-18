"""Complete 27-layer pure PyTorch Kimi-Linear causal language model."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from .config import KLinearConfig
from .layer import KLinearDecoderLayer, _replace_parameter
from .manifest import REAL_MODEL_TENSOR_MANIFEST
from .norm import RMSNorm
from .quantized import LinearFactory
from .state import KLinearDecodeState, LayerState, MLALayerState
from .weights import (
    CheckpointKind,
    SafetensorExpertProvider,
    SafetensorIndexStore,
    W4A16ExpertProvider,
)


@dataclass
class KLinearModelOutput:
    hidden_states: torch.Tensor
    logits: torch.Tensor
    state: KLinearDecodeState
    layer_hidden_states: tuple[torch.Tensor, ...]
    router_indices: tuple[torch.Tensor, ...]
    router_weights: tuple[torch.Tensor, ...]


def _validate_real_config(config: KLinearConfig) -> None:
    real = KLinearConfig()
    fields = (
        "vocab_size",
        "hidden_size",
        "head_dim",
        "intermediate_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "hidden_act",
        "rms_norm_eps",
        "q_lora_rank",
        "kv_lora_rank",
        "qk_nope_head_dim",
        "qk_rope_head_dim",
        "v_head_dim",
        "mla_use_nope",
        "kda_num_heads",
        "kda_head_dim",
        "short_conv_kernel_size",
        "num_experts",
        "num_experts_per_token",
        "num_shared_experts",
        "moe_intermediate_size",
        "first_k_dense_replace",
        "moe_layer_freq",
        "moe_renormalize",
        "moe_router_activation_func",
        "routed_scaling_factor",
        "use_grouped_topk",
        "num_expert_group",
        "topk_group",
        "bos_token_id",
        "eos_token_id",
        "pad_token_id",
        "tie_word_embeddings",
        "full_attention_layers",
        "kda_layers",
    )
    mismatches = [name for name in fields if getattr(config, name) != getattr(real, name)]
    if mismatches:
        raise ValueError(
            "checkpoint loading is pinned to moonshotai/Kimi-Linear-48B-A3B-Instruct; "
            f"config fields disagree: {mismatches}"
        )


class KLinearModel(nn.Module):
    def __init__(
        self,
        config: KLinearConfig,
        *,
        expert_provider=None,
        linear_factory: LinearFactory | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
            padding_idx=config.pad_token_id,
            device=device,
            dtype=dtype,
        )
        self.layers = nn.ModuleList(
            [
                KLinearDecoderLayer(
                    config,
                    layer_idx,
                    expert_provider=expert_provider,
                    linear_factory=linear_factory,
                    device=device,
                    dtype=dtype,
                )
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(
            config.hidden_size,
            config.rms_norm_eps,
            device=device,
            dtype=dtype,
        )
        self.lm_head = nn.Linear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
            device=device,
            dtype=dtype,
        )
        self._weight_store = None
        self._expert_provider = expert_provider

    @classmethod
    def from_directory(
        cls,
        directory: str | Path,
        *,
        device: torch.device | str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        expert_cache_entries: int = 256,
    ) -> "KLinearModel":
        directory = Path(directory)
        config = KLinearConfig.from_json(directory / "config.json")
        _validate_real_config(config)
        store = SafetensorIndexStore(directory, validate_real_layout=True)
        if store.checkpoint_kind is CheckpointKind.W4A16:
            if dtype != torch.bfloat16:
                raise TypeError("W4A16 checkpoint loading requires torch.bfloat16")
            expert_provider = W4A16ExpertProvider(store, device=device)
        else:
            expert_provider = SafetensorExpertProvider(
                store, cache_entries=expert_cache_entries
            )
        model = cls(
            config,
            expert_provider=expert_provider,
            linear_factory=store.linear_factory(),
            device="meta",
            dtype=dtype,
        )
        model.load_checkpoint_weights(store, device=device, dtype=dtype)
        model._weight_store = store
        model._expert_provider = expert_provider
        if store.checkpoint_kind is CheckpointKind.W4A16:
            model.prepare_grouped_decode_weights()
        if (
            store.checkpoint_kind is CheckpointKind.W4A16
            and model.resident_weight_bytes != store.tensor_storage_bytes
        ):
            raise ValueError(
                "resident W4A16 bytes disagree with checkpoint tensor storage: "
                f"resident={model.resident_weight_bytes}, "
                f"checkpoint={store.tensor_storage_bytes}"
            )
        model.eval()
        return model

    def load_checkpoint_weights(
        self,
        store: SafetensorIndexStore,
        *,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> None:
        for name, module in (
            ("model.embed_tokens.weight", self.embed_tokens),
            ("model.norm.weight", self.norm),
            ("lm_head.weight", self.lm_head),
        ):
            store.validate(name, REAL_MODEL_TENSOR_MANIFEST[name])
            tensor = store.load(name, device=device, dtype=dtype)
            _replace_parameter(module, "weight", tensor)
        for layer in self.layers:
            layer.load_checkpoint_weights(store, device=device, dtype=dtype)

    @property
    def resident_weight_bytes(self) -> int:
        """Return bytes held by loaded parameters, buffers, and expert weights."""
        tensors = tuple(self.parameters()) + tuple(self.buffers())
        meta_names = [
            name
            for name, tensor in (
                tuple(self.named_parameters()) + tuple(self.named_buffers())
            )
            if tensor.device.type == "meta"
        ]
        if meta_names:
            raise RuntimeError(f"model still has unloaded meta tensors: {meta_names}")
        module_bytes = sum(
            tensor.numel() * tensor.element_size() for tensor in tensors
        )
        provider_bytes = getattr(self._expert_provider, "resident_bytes", 0)
        return module_bytes + provider_bytes

    @property
    def checkpoint_tensor_storage_bytes(self) -> int | None:
        if self._weight_store is None:
            return None
        return self._weight_store.tensor_storage_bytes

    @property
    def checkpoint_kind(self) -> str | None:
        if self._weight_store is None:
            return None
        return self._weight_store.checkpoint_kind.value

    def empty_state(self) -> KLinearDecodeState:
        return KLinearDecodeState.empty(self.config.num_hidden_layers)

    def prepare_grouped_decode_weights(self) -> None:
        """Move resident W4A16 experts into layer-contiguous grouped banks."""
        for layer in self.layers:
            if layer.block_sparse_moe is not None:
                layer.block_sparse_moe.prepare_grouped_w4a16()

    def _attention_masks(
        self,
        attention_mask: torch.Tensor | None,
        state: KLinearDecodeState,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        batch, sequence, _ = hidden_states.shape
        if state.is_static:
            if sequence != 1:
                raise ValueError("fixed-capacity decode state accepts one token")
            current = torch.ones(
                batch,
                1,
                device=hidden_states.device,
                dtype=(
                    state.attention_mask.dtype
                    if state.attention_mask is not None
                    else torch.long
                ),
            )
            if attention_mask is not None:
                if tuple(attention_mask.shape) != (batch, 1):
                    raise ValueError("static decode attention_mask must cover one token")
                current = attention_mask.to(device=hidden_states.device)
            if state.attention_mask is None:
                return None, None, None
            state.attention_mask.index_copy_(
                1,
                state.position.reshape(1),
                current.to(dtype=state.attention_mask.dtype),
            )
            return current, state.attention_mask, state.attention_mask
        if attention_mask is None:
            if state.attention_mask is None:
                return None, None, None
            current = torch.ones(
                batch,
                sequence,
                device=state.attention_mask.device,
                dtype=state.attention_mask.dtype,
            )
            full = torch.cat((state.attention_mask, current), dim=1)
            return current, full, full
        if attention_mask.ndim != 2 or attention_mask.shape[0] != batch:
            raise ValueError("attention_mask must have shape [batch, sequence]")
        total_length = state.tokens_seen + sequence
        if attention_mask.shape[1] == sequence and state.tokens_seen:
            prefix = state.attention_mask
            if prefix is None:
                prefix = torch.ones(
                    batch,
                    state.tokens_seen,
                    device=attention_mask.device,
                    dtype=attention_mask.dtype,
                )
            else:
                prefix = prefix.to(
                    device=attention_mask.device, dtype=attention_mask.dtype
                )
            full = torch.cat((prefix, attention_mask), dim=1)
        elif attention_mask.shape[1] == total_length:
            full = attention_mask
        elif state.tokens_seen == 0 and attention_mask.shape[1] == sequence:
            full = attention_mask
        else:
            raise ValueError("attention mask does not cover the current or complete sequence")
        return full[:, -sequence:], full, full

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        *,
        hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        state: KLinearDecodeState | None = None,
    ) -> KLinearModelOutput:
        if (input_ids is None) == (hidden_states is None):
            raise ValueError("provide exactly one of input_ids or hidden_states")
        if input_ids is not None:
            if input_ids.ndim != 2:
                raise ValueError("input_ids must have shape [batch, sequence]")
            hidden_states = self.embed_tokens(input_ids)
        if hidden_states.ndim != 3:
            raise ValueError("hidden_states must have shape [batch, sequence, hidden]")
        if hidden_states.shape[-1] != self.config.hidden_size:
            raise ValueError("hidden_states have the wrong hidden size")
        if hidden_states.shape[1] == 0:
            raise ValueError("the input sequence cannot be empty")

        if state is None:
            state = self.empty_state()
        state.validate_for(self.config.num_hidden_layers)
        kda_mask, mla_mask, cached_mask = self._attention_masks(
            attention_mask, state, hidden_states
        )

        next_layer_states: list[LayerState] = []
        layer_hidden_states: list[torch.Tensor] = []
        router_indices: list[torch.Tensor] = []
        router_weights: list[torch.Tensor] = []
        for layer, layer_state in zip(self.layers, state.layer_states, strict=True):
            if (
                isinstance(layer_state, MLALayerState)
                and not layer_state.is_static
                and layer_state.sequence_length != state.tokens_seen
            ):
                raise ValueError(
                    f"MLA cache length for layer {layer.layer_idx} does not match tokens_seen"
                )
            layer_mask = kda_mask if layer.is_kda else mla_mask
            output = layer(
                hidden_states,
                attention_mask=layer_mask,
                state=layer_state,
                return_aux=True,
            )
            hidden_states = output.hidden_states
            next_layer_states.append(output.attention_state)
            layer_hidden_states.append(hidden_states)
            router_indices.append(output.router_indices)
            router_weights.append(output.router_weights)

        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)
        next_state = state.advanced(
            next_layer_states,
            hidden_states.shape[1],
            cached_mask,
        )
        return KLinearModelOutput(
            hidden_states,
            logits,
            next_state,
            tuple(layer_hidden_states),
            tuple(router_indices),
            tuple(router_weights),
        )
