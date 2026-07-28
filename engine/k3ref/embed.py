"""Token embeddings, final normalization, and the Kimi K3 language-model head."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from .norm import apply_attention_residual, rms_norm
from .weights import RawTensorStore


class K3EmbeddingHead(nn.Module):
    """Small wrapper around the model-level tensors outside layer directories."""

    def __init__(
        self,
        embedding_weight: torch.Tensor,
        lm_head_weight: torch.Tensor | None = None,
        *,
        norm_weight: torch.Tensor | None = None,
        output_residual_norm_weight: torch.Tensor | None = None,
        output_residual_projection_weight: torch.Tensor | None = None,
        rms_norm_eps: float = 1e-5,
    ) -> None:
        super().__init__()
        if embedding_weight.ndim != 2:
            raise ValueError("embedding_weight must have shape [vocab, hidden]")
        if lm_head_weight is None:
            # Tests can exercise generation before the untied checkpoint head lands.
            lm_head_weight = embedding_weight
        if lm_head_weight.ndim != 2:
            raise ValueError("lm_head_weight must have shape [vocab, hidden]")
        if lm_head_weight.shape[1] != embedding_weight.shape[1]:
            raise ValueError("embedding and LM head hidden sizes do not match")

        hidden_size = embedding_weight.shape[1]
        if norm_weight is None:
            norm_weight = embedding_weight.new_ones(hidden_size)
        if norm_weight.shape != (hidden_size,):
            raise ValueError("norm_weight has the wrong hidden size")
        residual_weights = (
            output_residual_norm_weight,
            output_residual_projection_weight,
        )
        if (residual_weights[0] is None) != (residual_weights[1] is None):
            raise ValueError("both output residual weights must be supplied together")
        if output_residual_norm_weight is not None:
            if output_residual_norm_weight.shape != (hidden_size,):
                raise ValueError("output residual norm has the wrong hidden size")
            if output_residual_projection_weight.shape != (1, hidden_size):
                raise ValueError("output residual projection has the wrong shape")

        self.embedding = nn.Embedding.from_pretrained(
            embedding_weight,
            freeze=True,
        )
        self.lm_head_weight = nn.Parameter(lm_head_weight, requires_grad=False)
        self.norm_weight = nn.Parameter(norm_weight, requires_grad=False)
        self.output_residual_norm_weight = (
            nn.Parameter(output_residual_norm_weight, requires_grad=False)
            if output_residual_norm_weight is not None
            else None
        )
        self.output_residual_projection_weight = (
            nn.Parameter(output_residual_projection_weight, requires_grad=False)
            if output_residual_projection_weight is not None
            else None
        )
        self.rms_norm_eps = rms_norm_eps

    @property
    def hidden_size(self) -> int:
        return self.embedding.embedding_dim

    @property
    def vocab_size(self) -> int:
        return self.embedding.num_embeddings

    def embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding(input_ids)

    def finish_hidden(
        self,
        hidden_states: torch.Tensor,
        block_residual: torch.Tensor | None,
    ) -> torch.Tensor:
        if self.output_residual_norm_weight is not None:
            if block_residual is None:
                raise ValueError("output residual weights require a block residual")
            batch, sequence, hidden_size = hidden_states.shape
            hidden_states = apply_attention_residual(
                hidden_states.reshape(-1, hidden_size),
                block_residual,
                self.output_residual_projection_weight,
                self.output_residual_norm_weight,
                self.rms_norm_eps,
            ).view(batch, sequence, hidden_size)
        return rms_norm(hidden_states, self.norm_weight, self.rms_norm_eps)

    def logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return F.linear(hidden_states, self.lm_head_weight)

    @classmethod
    def from_directory(
        cls,
        directory: str | Path,
        *,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.bfloat16,
        rms_norm_eps: float = 1e-5,
    ) -> "K3EmbeddingHead":
        store = RawTensorStore(directory)

        def load(name: str, cast: torch.dtype | None = dtype) -> torch.Tensor:
            return store.load(name, device=device, dtype=cast)

        embedding = load("language_model.model.embed_tokens.weight")
        lm_head = load("language_model.lm_head.weight")
        norm = load("language_model.model.norm.weight")
        available = set(store.names)

        def has_suffix(suffix: str) -> bool:
            return any(name.endswith(suffix) for name in available)

        output_norm_name = "language_model.model.output_attn_res_norm.weight"
        output_proj_name = "language_model.model.output_attn_res_proj.weight"
        has_output_norm = has_suffix(output_norm_name)
        has_output_proj = has_suffix(output_proj_name)
        if has_output_norm != has_output_proj:
            raise FileNotFoundError("model-level output residual tensors are incomplete")
        output_norm = load(output_norm_name) if has_output_norm else None
        output_proj = load(output_proj_name) if has_output_proj else None
        return cls(
            embedding,
            lm_head,
            norm_weight=norm,
            output_residual_norm_weight=output_norm,
            output_residual_projection_weight=output_proj,
            rms_norm_eps=rms_norm_eps,
        )
