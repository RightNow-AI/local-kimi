"""Compact in-memory records for router traces captured on the CPU."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class EncodedPrompt:
    prompt_id: str
    category: str
    token_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.prompt_id:
            raise ValueError("prompt_id cannot be empty")
        if not self.category:
            raise ValueError("prompt category cannot be empty")
        if not self.token_ids:
            raise ValueError(f"prompt {self.prompt_id} has no token IDs")
        if any(isinstance(token, bool) or not isinstance(token, int) for token in self.token_ids):
            raise TypeError(f"prompt {self.prompt_id} contains a non-integer token ID")


@dataclass(frozen=True)
class LayerRoutingTrace:
    layer_index: int
    expert_ids: np.ndarray
    expert_weights: np.ndarray

    def __post_init__(self) -> None:
        expert_ids = np.asarray(self.expert_ids)
        expert_weights = np.asarray(self.expert_weights)
        if self.layer_index < 0:
            raise ValueError("layer_index cannot be negative")
        if expert_ids.ndim != 2 or expert_weights.ndim != 2:
            raise ValueError("router IDs and weights must have shape [tokens, top_k]")
        if expert_ids.shape != expert_weights.shape:
            raise ValueError("router ID and weight shapes differ")
        if expert_ids.shape[0] == 0 or expert_ids.shape[1] == 0:
            raise ValueError("a routed layer trace cannot be empty")
        if not np.issubdtype(expert_ids.dtype, np.integer):
            raise TypeError("expert IDs must be integers")
        if not np.issubdtype(expert_weights.dtype, np.floating):
            raise TypeError("expert weights must be floating point")
        if np.any(expert_ids < 0):
            raise ValueError("expert IDs cannot be negative")
        if not np.all(np.isfinite(expert_weights)) or np.any(expert_weights < 0):
            raise ValueError("expert weights must be finite and non-negative")
        if np.any(expert_weights.sum(axis=1) <= 0):
            raise ValueError("each token must carry positive routing weight")
        sorted_ids = np.sort(expert_ids, axis=1)
        if np.any(sorted_ids[:, 1:] == sorted_ids[:, :-1]):
            raise ValueError("a token cannot select the same expert twice")
        object.__setattr__(self, "expert_ids", expert_ids)
        object.__setattr__(self, "expert_weights", expert_weights)

    @property
    def token_count(self) -> int:
        return int(self.expert_ids.shape[0])

    @property
    def top_k(self) -> int:
        return int(self.expert_ids.shape[1])


@dataclass(frozen=True)
class PromptRoutingTrace:
    prompt_id: str
    token_ids: tuple[int, ...]
    layers: tuple[LayerRoutingTrace, ...]

    def __post_init__(self) -> None:
        if not self.prompt_id:
            raise ValueError("prompt_id cannot be empty")
        if not self.token_ids:
            raise ValueError(f"prompt {self.prompt_id} has no token IDs")
        if not self.layers:
            raise ValueError(f"prompt {self.prompt_id} has no routed layers")
        layer_indices = [layer.layer_index for layer in self.layers]
        if len(set(layer_indices)) != len(layer_indices):
            raise ValueError(f"prompt {self.prompt_id} repeats a routed layer")
        if layer_indices != sorted(layer_indices):
            raise ValueError(f"prompt {self.prompt_id} layers are not ordered")
        for layer in self.layers:
            if layer.token_count != len(self.token_ids):
                raise ValueError(
                    f"prompt {self.prompt_id} layer {layer.layer_index} captured "
                    "a different token count"
                )


@dataclass(frozen=True)
class RoutingRun:
    checkpoint: str
    prompt_set_sha256: str
    prompts: tuple[PromptRoutingTrace, ...]
    router_config: Mapping[str, object]

    def __post_init__(self) -> None:
        if not self.checkpoint:
            raise ValueError("checkpoint cannot be empty")
        if not self.prompt_set_sha256:
            raise ValueError("prompt_set_sha256 cannot be empty")
        if not self.prompts:
            raise ValueError("a routing run must contain prompts")
        prompt_ids = [prompt.prompt_id for prompt in self.prompts]
        if len(set(prompt_ids)) != len(prompt_ids):
            raise ValueError("a routing run contains duplicate prompt IDs")
        first_layers = tuple(layer.layer_index for layer in self.prompts[0].layers)
        first_top_k = tuple(layer.top_k for layer in self.prompts[0].layers)
        for prompt in self.prompts[1:]:
            if tuple(layer.layer_index for layer in prompt.layers) != first_layers:
                raise ValueError("routed layer indices differ between prompts")
            if tuple(layer.top_k for layer in prompt.layers) != first_top_k:
                raise ValueError("top-k widths differ between prompts")
        object.__setattr__(self, "router_config", dict(self.router_config))

    @property
    def prompt_token_count(self) -> int:
        return sum(len(prompt.token_ids) for prompt in self.prompts)
