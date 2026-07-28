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

    @property
    def batch_size(self) -> int:
        return self.recurrent.shape[0]


@dataclass(frozen=True)
class MLALayerState:
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
class KLinearDecodeState:
    layer_states: tuple[LayerState, ...]
    tokens_seen: int = 0
    attention_mask: torch.Tensor | None = None

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
            if self.attention_mask.shape[1] != self.tokens_seen:
                raise ValueError("cached attention mask length does not match tokens_seen")

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
            if attention_mask.ndim != 2 or attention_mask.shape[1] != next_tokens_seen:
                raise ValueError("next attention mask has the wrong sequence length")
        return KLinearDecodeState(states, next_tokens_seen, attention_mask)

