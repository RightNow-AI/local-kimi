"""Slow, explicit multi-layer Kimi K3 reference model."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from .attention import KDAAttention, MLAAttention
from .config import K3LayerConfig
from .embed import K3EmbeddingHead
from .layer import K3ReferenceLayer, _replace_parameter
from .moe import K3SharedMLP
from .norm import RMSNorm, apply_attention_residual
from .state import K3DecodeState, KDALayerState, LayerState, MLALayerState
from .weights import RawTensorStore


@dataclass
class K3ModelOutput:
    hidden_states: torch.Tensor
    logits: torch.Tensor | None
    state: K3DecodeState
    layer_hidden_states: tuple[torch.Tensor, ...]
    router_indices: tuple[torch.Tensor, ...]
    router_weights: tuple[torch.Tensor, ...]


class K3DenseReferenceLayer(nn.Module):
    """Layer-zero variant, which uses a dense SiTU MLP instead of latent MoE."""

    def __init__(
        self,
        config: K3LayerConfig,
        layer_idx: int = 0,
        *,
        intermediate_size: int = 33792,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if layer_idx != 0:
            raise ValueError("the dense K3 reference layer is only valid for layer 0")
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
                rms_norm_eps=1e-6,
                device=device,
                dtype=dtype,
            )
        self.mlp = K3SharedMLP(
            config.hidden_size,
            intermediate_size,
            situ_beta=config.activation_situ_beta,
            situ_linear_beta=config.activation_situ_linear_beta,
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

    @classmethod
    def from_directory(
        cls,
        directory: str | Path,
        *,
        config: K3LayerConfig,
        intermediate_size: int = 33792,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.bfloat16,
    ) -> "K3DenseReferenceLayer":
        store = RawTensorStore(directory)
        layer = cls(
            config,
            intermediate_size=intermediate_size,
            device="meta",
            dtype=dtype,
        )
        prefix = "layers.0."

        def load_parameter(
            module: nn.Module,
            parameter: str,
            suffix: str,
            cast: torch.dtype | None = dtype,
        ) -> None:
            tensor = store.load(prefix + suffix, device=device, dtype=cast)
            _replace_parameter(module, parameter, tensor)

        if layer.is_kda:
            for name in (
                "q_proj",
                "k_proj",
                "v_proj",
                "f_a_proj",
                "f_b_proj",
                "b_proj",
                "g_proj",
                "o_proj",
            ):
                load_parameter(
                    getattr(layer.self_attn, name),
                    "weight",
                    f"self_attn.{name}.weight",
                )
            for name in ("q_conv1d", "k_conv1d", "v_conv1d"):
                load_parameter(
                    getattr(layer.self_attn, name),
                    "weight",
                    f"self_attn.{name}.weight",
                    cast=None,
                )
            load_parameter(layer.self_attn, "A_log", "self_attn.A_log", cast=None)
            load_parameter(layer.self_attn, "dt_bias", "self_attn.dt_bias", cast=None)
            load_parameter(
                layer.self_attn.o_norm,
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
                module = getattr(layer.self_attn, name)
                if module is not None:
                    load_parameter(module, "weight", f"self_attn.{name}.weight")
            load_parameter(
                layer.self_attn.q_a_layernorm,
                "weight",
                "self_attn.q_a_layernorm.weight",
            )
            load_parameter(
                layer.self_attn.kv_a_layernorm,
                "weight",
                "self_attn.kv_a_layernorm.weight",
            )
        for name in ("gate_proj", "up_proj", "down_proj"):
            load_parameter(getattr(layer.mlp, name), "weight", f"mlp.{name}.weight")
        load_parameter(layer.input_layernorm, "weight", "input_layernorm.weight")
        load_parameter(
            layer.post_attention_layernorm,
            "weight",
            "post_attention_layernorm.weight",
        )
        if layer.use_attn_residuals:
            load_parameter(
                layer.self_attention_res_norm,
                "weight",
                "self_attention_res_norm.weight",
            )
            load_parameter(layer.mlp_res_norm, "weight", "mlp_res_norm.weight")
            load_parameter(
                layer.self_attention_res_proj,
                "weight",
                "self_attention_res_proj.weight",
            )
            load_parameter(layer.mlp_res_proj, "weight", "mlp_res_proj.weight")
        layer.eval()
        return layer


ReferenceLayer = K3ReferenceLayer | K3DenseReferenceLayer


def _mla_attention(
    attention: MLAAttention,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor | None,
    state: MLALayerState | None,
) -> tuple[torch.Tensor, MLALayerState]:
    """Run MLA while retaining only the latent and rotary slices in the cache."""

    batch, sequence, _ = hidden_states.shape
    query = attention.q_b_proj(
        attention.q_a_layernorm(attention.q_a_proj(hidden_states))
    )
    query = query.view(
        batch, sequence, attention.num_heads, attention.q_head_dim
    ).transpose(1, 2)
    query_pass, query_rotary = torch.split(
        query,
        [attention.qk_nope_head_dim, attention.qk_rope_head_dim],
        dim=-1,
    )

    current = attention.kv_a_proj_with_mqa(hidden_states)
    current_latent, current_rotary = torch.split(
        current,
        [attention.kv_lora_rank, attention.qk_rope_head_dim],
        dim=-1,
    )
    if state is None:
        latent = current_latent
        rotary = current_rotary
        past_length = 0
    else:
        if state.compressed_kv.shape[:1] != (batch,):
            raise ValueError("MLA cache batch size does not match the current input")
        if state.compressed_kv.shape[-1] != attention.kv_lora_rank:
            raise ValueError("MLA cache has the wrong latent rank")
        if state.rotary_key.shape[-1] != attention.qk_rope_head_dim:
            raise ValueError("MLA cache has the wrong rotary width")
        past_length = state.sequence_length
        latent = torch.cat((state.compressed_kv, current_latent), dim=1)
        rotary = torch.cat((state.rotary_key, current_rotary), dim=1)

    key_pass = attention.kv_b_proj(attention.kv_a_layernorm(latent)).view(
        batch,
        latent.shape[1],
        attention.num_heads,
        attention.qk_nope_head_dim + attention.v_head_dim,
    ).transpose(1, 2)
    key_pass, value = torch.split(
        key_pass,
        [attention.qk_nope_head_dim, attention.v_head_dim],
        dim=-1,
    )
    key_rotary = rotary.view(
        batch, 1, latent.shape[1], attention.qk_rope_head_dim
    ).expand(*key_pass.shape[:-1], -1)
    query = torch.cat((query_pass, query_rotary), dim=-1)
    key = torch.cat((key_pass, key_rotary), dim=-1)

    mask = attention._attention_mask(
        attention_mask,
        batch,
        sequence,
        latent.shape[1],
        past_length,
        hidden_states.device,
    )
    scores = torch.einsum("bhqd,bhkd->bhqk", query, key) * attention.scaling
    probabilities = (scores.float() + mask).softmax(dim=-1).to(query.dtype)
    output = torch.einsum("bhqk,bhkd->bhqd", probabilities, value)
    output = output.transpose(1, 2).reshape(
        batch, sequence, attention.num_heads * attention.v_head_dim
    )
    if attention.g_proj is not None:
        output = output * attention.g_proj(hidden_states).sigmoid()
    output = attention.o_proj(output)
    return output, MLALayerState(latent, rotary)


def _run_attention(
    layer: ReferenceLayer,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor | None,
    state: LayerState,
) -> tuple[torch.Tensor, LayerState]:
    if layer.is_kda:
        if state is not None and not isinstance(state, KDALayerState):
            raise TypeError(f"layer {layer.layer_idx} requires KDALayerState")
        attention_state = state.as_attention_state() if state is not None else None
        output, next_state = layer.self_attn(
            hidden_states,
            attention_mask=attention_mask,
            state=attention_state,
            return_state=True,
        )
        return output, KDALayerState.from_attention_state(next_state)
    if state is not None and not isinstance(state, MLALayerState):
        raise TypeError(f"layer {layer.layer_idx} requires MLALayerState")
    return _mla_attention(layer.self_attn, hidden_states, attention_mask, state)


def _run_feed_forward(
    layer: ReferenceLayer,
    hidden_states: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if isinstance(layer, K3DenseReferenceLayer):
        output = layer.mlp(hidden_states)
        tokens = hidden_states.shape[0] * hidden_states.shape[1]
        indices = torch.empty((tokens, 0), device=hidden_states.device, dtype=torch.long)
        weights = hidden_states.new_empty((tokens, 0))
        return output, indices, weights
    return layer.block_sparse_moe(hidden_states, return_router=True)


def _forward_layer(
    layer: ReferenceLayer,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor | None,
    state: LayerState,
    block_residual: torch.Tensor | None,
) -> tuple[
    torch.Tensor,
    LayerState,
    torch.Tensor | None,
    torch.Tensor,
    torch.Tensor,
]:
    if not layer.use_attn_residuals:
        residual = hidden_states
        attended, next_state = _run_attention(
            layer,
            layer.input_layernorm(hidden_states),
            attention_mask,
            state,
        )
        hidden_states = residual + attended
        feed_forward, indices, weights = _run_feed_forward(
            layer,
            layer.post_attention_layernorm(hidden_states),
        )
        return hidden_states + feed_forward, next_state, None, indices, weights

    batch, sequence, hidden_size = hidden_states.shape
    prefix_sum = hidden_states
    if block_residual is None:
        block_residual = hidden_states.new_zeros(batch * sequence, 0, hidden_size)
    if block_residual.shape[0] != batch * sequence:
        raise ValueError("block residual does not match the current token count")
    if block_residual.shape[1] > 0:
        hidden_states = apply_attention_residual(
            prefix_sum.reshape(-1, hidden_size),
            block_residual,
            layer.self_attention_res_proj.weight,
            layer.self_attention_res_norm.weight,
            layer.config.rms_norm_eps,
        ).view(batch, sequence, hidden_size)

    if layer.layer_idx % layer.config.attn_res_block_size == 0:
        block_residual = torch.cat(
            (block_residual, prefix_sum.reshape(-1, hidden_size).unsqueeze(1)),
            dim=1,
        )
        prefix_sum = None

    attended, next_state = _run_attention(
        layer,
        layer.input_layernorm(hidden_states),
        attention_mask,
        state,
    )
    prefix_sum = attended if prefix_sum is None else prefix_sum + attended
    feed_forward_input = apply_attention_residual(
        prefix_sum.reshape(-1, hidden_size),
        block_residual,
        layer.mlp_res_proj.weight,
        layer.mlp_res_norm.weight,
        layer.config.rms_norm_eps,
    ).view(batch, sequence, hidden_size)
    feed_forward, indices, weights = _run_feed_forward(
        layer,
        layer.post_attention_layernorm(feed_forward_input),
    )
    return (
        prefix_sum + feed_forward,
        next_state,
        block_residual,
        indices,
        weights,
    )


class K3Model(nn.Module):
    """A partial or complete ordered stack of real K3 decoder layers."""

    def __init__(
        self,
        config: K3LayerConfig,
        layer_indices: list[int] | tuple[int, ...],
        *,
        layers: list[ReferenceLayer] | tuple[ReferenceLayer, ...] | None = None,
        embedding_head: K3EmbeddingHead | None = None,
        dense_intermediate_size: int = 33792,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        indices = tuple(layer_indices)
        if not indices:
            raise ValueError("K3Model requires at least one explicit layer index")
        if any(index < 0 for index in indices):
            raise ValueError("layer indices cannot be negative")
        if tuple(sorted(set(indices))) != indices:
            raise ValueError("layer indices must be unique and strictly increasing")
        if layers is None:
            built: list[ReferenceLayer] = []
            for index in indices:
                if index == 0:
                    built.append(
                        K3DenseReferenceLayer(
                            config,
                            intermediate_size=dense_intermediate_size,
                            device=device,
                            dtype=dtype,
                        )
                    )
                else:
                    built.append(
                        K3ReferenceLayer(
                            config,
                            index,
                            device=device,
                            dtype=dtype,
                        )
                    )
            layers = built
        if len(layers) != len(indices):
            raise ValueError("layers and layer_indices must have the same length")
        for index, layer in zip(indices, layers, strict=True):
            if layer.layer_idx != index:
                raise ValueError(
                    f"layer object index {layer.layer_idx} does not match {index}"
                )
            if layer.is_kda != config.is_kda_layer(index):
                raise ValueError(f"layer {index} has the wrong attention implementation")
        if embedding_head is not None and embedding_head.hidden_size != config.hidden_size:
            raise ValueError("embedding hidden size does not match the model config")

        self.config = config
        self.layer_indices = indices
        self.layers = nn.ModuleList(layers)
        self.embedding_head = embedding_head

    @classmethod
    def from_directories(
        cls,
        root: str | Path,
        layer_indices: list[int] | tuple[int, ...],
        *,
        config: K3LayerConfig,
        embedding_head: K3EmbeddingHead | None = None,
        model_tensor_directory: str | Path | None = None,
        dense_intermediate_size: int = 33792,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.bfloat16,
    ) -> "K3Model":
        root = Path(root)
        layers: list[ReferenceLayer] = []
        for index in layer_indices:
            directory = root / f"layer{index}"
            if index == 0:
                layer = K3DenseReferenceLayer.from_directory(
                    directory,
                    config=config,
                    intermediate_size=dense_intermediate_size,
                    device=device,
                    dtype=dtype,
                )
            else:
                layer = K3ReferenceLayer.from_directory(
                    directory,
                    index,
                    config=config,
                    device=device,
                    dtype=dtype,
                )
            layers.append(layer)
        if embedding_head is None and model_tensor_directory is not None:
            embedding_head = K3EmbeddingHead.from_directory(
                model_tensor_directory,
                device=device,
                dtype=dtype,
                rms_norm_eps=config.rms_norm_eps,
            )
        model = cls(
            config,
            layer_indices,
            layers=layers,
            embedding_head=embedding_head,
            dense_intermediate_size=dense_intermediate_size,
        )
        model.eval()
        return model

    def empty_state(self) -> K3DecodeState:
        return K3DecodeState.empty(self.layer_indices)

    def _attention_masks(
        self,
        attention_mask: torch.Tensor | None,
        state: K3DecodeState,
        hidden_states: torch.Tensor,
    ) -> tuple[
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        if attention_mask is None:
            if state.attention_mask is None:
                return None, None, None
            batch, sequence, _ = hidden_states.shape
            current = torch.ones(
                batch,
                sequence,
                device=state.attention_mask.device,
                dtype=state.attention_mask.dtype,
            )
            full_mask = torch.cat((state.attention_mask, current), dim=1)
            return current, full_mask, full_mask
        batch, sequence, _ = hidden_states.shape
        if attention_mask.shape[0] != batch:
            raise ValueError("attention mask batch size does not match the input")
        if attention_mask.ndim == 4:
            cached_mask = None
            if state.attention_mask is not None:
                current = torch.ones(
                    batch,
                    sequence,
                    device=state.attention_mask.device,
                    dtype=state.attention_mask.dtype,
                )
                cached_mask = torch.cat((state.attention_mask, current), dim=1)
            return None, attention_mask, cached_mask
        if attention_mask.ndim != 2:
            raise ValueError("attention_mask must be two-dimensional or four-dimensional")
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
                    device=attention_mask.device,
                    dtype=attention_mask.dtype,
                )
            full_mask = torch.cat((prefix, attention_mask), dim=1)
        elif attention_mask.shape[1] == total_length:
            full_mask = attention_mask
        else:
            raise ValueError("attention mask does not cover the current or complete sequence")
        return full_mask[:, -sequence:], full_mask, full_mask

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        *,
        hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        state: K3DecodeState | None = None,
    ) -> K3ModelOutput:
        if (input_ids is None) == (hidden_states is None):
            raise ValueError("provide exactly one of input_ids or hidden_states")
        if input_ids is not None:
            if self.embedding_head is None:
                raise ValueError("input_ids require an embedding head")
            hidden_states = self.embedding_head.embed(input_ids)
        if hidden_states.ndim != 3:
            raise ValueError("hidden states must have shape [batch, sequence, hidden]")
        if hidden_states.shape[-1] != self.config.hidden_size:
            raise ValueError("hidden states have the wrong hidden size")

        if state is None:
            state = self.empty_state()
        state.validate_for(self.layer_indices)
        kda_mask, mla_mask, cached_attention_mask = self._attention_masks(
            attention_mask,
            state,
            hidden_states,
        )
        batch, sequence, hidden_size = hidden_states.shape
        block_residual = None
        if self.config.attn_res_block_size is not None:
            block_residual = hidden_states.new_zeros(
                batch * sequence,
                0,
                hidden_size,
            )

        next_layer_states: list[LayerState] = []
        layer_hidden_states: list[torch.Tensor] = []
        router_indices: list[torch.Tensor] = []
        router_weights: list[torch.Tensor] = []
        for layer, layer_state in zip(self.layers, state.layer_states, strict=True):
            if (
                isinstance(layer_state, MLALayerState)
                and layer_state.sequence_length != state.tokens_seen
            ):
                raise ValueError(
                    f"MLA cache length for layer {layer.layer_idx} "
                    "does not match tokens_seen"
                )
            layer_mask = kda_mask if layer.is_kda else mla_mask
            (
                hidden_states,
                next_layer_state,
                block_residual,
                indices,
                weights,
            ) = _forward_layer(
                layer,
                hidden_states,
                layer_mask,
                layer_state,
                block_residual,
            )
            next_layer_states.append(next_layer_state)
            layer_hidden_states.append(hidden_states)
            router_indices.append(indices)
            router_weights.append(weights)

        if self.embedding_head is not None:
            hidden_states = self.embedding_head.finish_hidden(
                hidden_states,
                block_residual,
            )
            logits = self.embedding_head.logits(hidden_states)
        else:
            logits = None
        next_state = state.advanced(
            next_layer_states,
            sequence,
            cached_attention_mask,
        )
        return K3ModelOutput(
            hidden_states,
            logits,
            next_state,
            tuple(layer_hidden_states),
            tuple(router_indices),
            tuple(router_weights),
        )
