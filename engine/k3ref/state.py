"""Explicit decode-cache types for the hybrid Kimi K3 stack."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .attention import KDAState


@dataclass(frozen=True)
class KDALayerState:
    """Fixed-size KDA convolution windows and recurrent delta-rule state."""

    q_conv: torch.Tensor
    k_conv: torch.Tensor
    v_conv: torch.Tensor
    recurrent: torch.Tensor

    @classmethod
    def from_attention_state(cls, state: KDAState) -> "KDALayerState":
        return cls(state.q_conv, state.k_conv, state.v_conv, state.recurrent)

    def as_attention_state(self) -> KDAState:
        return KDAState(self.q_conv, self.k_conv, self.v_conv, self.recurrent)

    @property
    def batch_size(self) -> int:
        return self.recurrent.shape[0]


@dataclass(frozen=True)
class MLALayerState:
    """Growing MLA cache stored in latent form rather than expanded per head."""

    compressed_kv: torch.Tensor
    rotary_key: torch.Tensor

    @property
    def batch_size(self) -> int:
        return self.compressed_kv.shape[0]

    @property
    def sequence_length(self) -> int:
        return self.compressed_kv.shape[1]


LayerState = KDALayerState | MLALayerState | None


@dataclass(frozen=True)
class K3DecodeState:
    """Cache aligned to the model's explicit, possibly partial layer list."""

    layer_indices: tuple[int, ...]
    layer_states: tuple[LayerState, ...]
    tokens_seen: int = 0
    attention_mask: torch.Tensor | None = None

    @classmethod
    def empty(cls, layer_indices: tuple[int, ...] | list[int]) -> "K3DecodeState":
        indices = tuple(layer_indices)
        return cls(indices, (None,) * len(indices), 0, None)

    def validate_for(self, layer_indices: tuple[int, ...]) -> None:
        if self.layer_indices != layer_indices:
            raise ValueError(
                "decode state layer indices do not match the model: "
                f"state={self.layer_indices}, model={layer_indices}"
            )
        if len(self.layer_states) != len(layer_indices):
            raise ValueError("decode state has the wrong number of layer states")
        if self.tokens_seen < 0:
            raise ValueError("decode state tokens_seen cannot be negative")
        if self.attention_mask is not None:
            if self.attention_mask.ndim != 2:
                raise ValueError("cached attention mask must be two-dimensional")
            if self.attention_mask.shape[1] != self.tokens_seen:
                raise ValueError("cached attention mask length does not match tokens_seen")

    def advanced(
        self,
        layer_states: tuple[LayerState, ...] | list[LayerState],
        token_count: int,
        attention_mask: torch.Tensor | None = None,
    ) -> "K3DecodeState":
        states = tuple(layer_states)
        if len(states) != len(self.layer_indices):
            raise ValueError("next decode state has the wrong number of layers")
        if token_count < 0:
            raise ValueError("token_count cannot be negative")
        next_tokens_seen = self.tokens_seen + token_count
        if attention_mask is not None:
            if attention_mask.ndim != 2 or attention_mask.shape[1] != next_tokens_seen:
                raise ValueError("next attention mask has the wrong sequence length")
        return K3DecodeState(
            self.layer_indices,
            states,
            next_tokens_seen,
            attention_mask,
        )
