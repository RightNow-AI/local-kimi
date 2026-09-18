"""Decode-cache types for the hybrid Kimi-Linear stack."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class KDALayerState:
    q_conv: torch.Tensor
    k_conv: torch.Tensor
    v_conv: torch.Tensor
    recurrent: torch.Tensor
    is_static: bool = False

    @property
    def batch_size(self) -> int:
        return self.recurrent.shape[0]


@dataclass(frozen=True)
class MLALayerState:
    compressed_kv: torch.Tensor
    rotary_key: torch.Tensor
    key_pass: torch.Tensor | None = None
    value: torch.Tensor | None = None
    position: torch.Tensor | None = None

    @property
    def batch_size(self) -> int:
        return self.compressed_kv.shape[0]

    @property
    def sequence_length(self) -> int:
        return self.compressed_kv.shape[1]

    @property
    def is_static(self) -> bool:
        return self.position is not None

    @property
    def capacity(self) -> int:
        return self.compressed_kv.shape[1]


LayerState = KDALayerState | MLALayerState | None


@dataclass(frozen=True)
class KLinearDecodeState:
    layer_states: tuple[LayerState, ...]
    tokens_seen: int = 0
    attention_mask: torch.Tensor | None = None
    position: torch.Tensor | None = None

    @classmethod
    def empty(cls, num_layers: int) -> "KLinearDecodeState":
        return cls((None,) * num_layers)

    def validate_for(self, num_layers: int) -> None:
        if len(self.layer_states) != num_layers:
            raise ValueError("decode state has the wrong number of layer states")
        if self.tokens_seen < 0:
            raise ValueError("decode state tokens_seen cannot be negative")
        if self.attention_mask is not None:
            if self.attention_mask.ndim != 2:
                raise ValueError("cached attention mask must be two-dimensional")
            if self.position is None and self.attention_mask.shape[1] != self.tokens_seen:
                raise ValueError("cached attention mask length does not match tokens_seen")

    @property
    def is_static(self) -> bool:
        return self.position is not None

    def reserve_decode_capacity(self, additional_tokens: int) -> "KLinearDecodeState":
        """Copy growing caches into fixed-capacity, in-place decode buffers."""
        if additional_tokens < 0:
            raise ValueError("additional_tokens cannot be negative")
        capacity = self.tokens_seen + additional_tokens
        static_states: list[LayerState] = []
        position_device: torch.device | None = None
        for layer_state in self.layer_states:
            if layer_state is None:
                static_states.append(None)
                continue
            if isinstance(layer_state, KDALayerState):
                position_device = layer_state.recurrent.device
                static_states.append(
                    KDALayerState(
                        layer_state.q_conv.clone(),
                        layer_state.k_conv.clone(),
                        layer_state.v_conv.clone(),
                        layer_state.recurrent.clone(),
                        is_static=True,
                    )
                )
                continue
            position_device = layer_state.compressed_kv.device
            if layer_state.key_pass is None or layer_state.value is None:
                raise ValueError("MLA state has no projected key/value cache")
            batch, _, latent_width = layer_state.compressed_kv.shape
            rotary_width = layer_state.rotary_key.shape[-1]
            heads = layer_state.key_pass.shape[1]
            key_width = layer_state.key_pass.shape[-1]
            value_width = layer_state.value.shape[-1]
            compressed = layer_state.compressed_kv.new_zeros(
                batch, capacity, latent_width
            )
            rotary = layer_state.rotary_key.new_zeros(batch, capacity, rotary_width)
            key_pass = layer_state.key_pass.new_zeros(
                batch, heads, capacity, key_width
            )
            value = layer_state.value.new_zeros(
                batch, heads, capacity, value_width
            )
            if self.tokens_seen:
                compressed[:, : self.tokens_seen].copy_(layer_state.compressed_kv)
                rotary[:, : self.tokens_seen].copy_(layer_state.rotary_key)
                key_pass[:, :, : self.tokens_seen].copy_(layer_state.key_pass)
                value[:, :, : self.tokens_seen].copy_(layer_state.value)
            layer_position = torch.full(
                (), self.tokens_seen, dtype=torch.long, device=position_device
            )
            static_states.append(
                MLALayerState(
                    compressed,
                    rotary,
                    key_pass,
                    value,
                    layer_position,
                )
            )

        if position_device is None:
            if self.attention_mask is None:
                position_device = torch.device("cpu")
            else:
                position_device = self.attention_mask.device
        position = torch.full(
            (), self.tokens_seen, dtype=torch.long, device=position_device
        )
        static_mask = None
        if self.attention_mask is not None:
            batch = self.attention_mask.shape[0]
            static_mask = self.attention_mask.new_zeros(batch, capacity)
            if self.tokens_seen:
                static_mask[:, : self.tokens_seen].copy_(self.attention_mask)
        return KLinearDecodeState(
            tuple(static_states),
            self.tokens_seen,
            static_mask,
            position,
        )

    def clone_static(self) -> "KLinearDecodeState":
        if not self.is_static:
            raise ValueError("clone_static requires fixed-capacity state")
        states: list[LayerState] = []
        for layer_state in self.layer_states:
            if layer_state is None:
                states.append(None)
            elif isinstance(layer_state, KDALayerState):
                states.append(
                    KDALayerState(
                        layer_state.q_conv.clone(),
                        layer_state.k_conv.clone(),
                        layer_state.v_conv.clone(),
                        layer_state.recurrent.clone(),
                        is_static=True,
                    )
                )
            else:
                states.append(
                    MLALayerState(
                        layer_state.compressed_kv.clone(),
                        layer_state.rotary_key.clone(),
                        layer_state.key_pass.clone(),
                        layer_state.value.clone(),
                        layer_state.position.clone(),
                    )
                )
        return KLinearDecodeState(
            tuple(states),
            self.tokens_seen,
            None if self.attention_mask is None else self.attention_mask.clone(),
            self.position.clone(),
        )

    def copy_from_(self, source: "KLinearDecodeState") -> None:
        """Restore one static state without changing any captured addresses."""
        if not self.is_static or not source.is_static:
            raise ValueError("copy_from_ requires fixed-capacity states")
        if len(self.layer_states) != len(source.layer_states):
            raise ValueError("static state layer counts disagree")
        for target, value in zip(self.layer_states, source.layer_states, strict=True):
            if target is None or value is None:
                if target is not value:
                    raise ValueError("static state layer kinds disagree")
            elif isinstance(target, KDALayerState) and isinstance(value, KDALayerState):
                target.q_conv.copy_(value.q_conv)
                target.k_conv.copy_(value.k_conv)
                target.v_conv.copy_(value.v_conv)
                target.recurrent.copy_(value.recurrent)
            elif isinstance(target, MLALayerState) and isinstance(value, MLALayerState):
                target.compressed_kv.copy_(value.compressed_kv)
                target.rotary_key.copy_(value.rotary_key)
                target.key_pass.copy_(value.key_pass)
                target.value.copy_(value.value)
                target.position.copy_(value.position)
            else:
                raise ValueError("static state layer kinds disagree")
        if self.attention_mask is not None:
            self.attention_mask.copy_(source.attention_mask)
        self.position.copy_(source.position)

    def with_tokens_seen(self, tokens_seen: int) -> "KLinearDecodeState":
        return KLinearDecodeState(
            self.layer_states,
            tokens_seen,
            self.attention_mask,
            self.position,
        )

    def advanced(
        self,
        layer_states: list[LayerState] | tuple[LayerState, ...],
        token_count: int,
        attention_mask: torch.Tensor | None,
    ) -> "KLinearDecodeState":
        states = tuple(layer_states)
        if len(states) != len(self.layer_states):
            raise ValueError("next decode state has the wrong number of layers")
        if token_count < 0:
            raise ValueError("token_count cannot be negative")
        next_tokens_seen = self.tokens_seen + token_count
        if attention_mask is not None:
            expected_length = (
                attention_mask.shape[1] if self.is_static else next_tokens_seen
            )
            if attention_mask.ndim != 2 or attention_mask.shape[1] != expected_length:
                raise ValueError("next attention mask has the wrong sequence length")
        if self.position is not None:
            self.position.add_(token_count)
        return KLinearDecodeState(
            states,
            next_tokens_seen,
            attention_mask,
            self.position,
        )

