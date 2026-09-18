"""Small pure-PyTorch reference implementations for correctness tests."""

from __future__ import annotations

from typing import Protocol

import torch

from .draft import propose
from .verify import verify_greedy


class CausalLogitsModel(Protocol):
    def __call__(self, input_ids: torch.Tensor) -> torch.Tensor: ...


def _model_logits(model: CausalLogitsModel, input_ids: torch.Tensor) -> torch.Tensor:
    logits = model(input_ids)
    if not isinstance(logits, torch.Tensor) or logits.ndim != 3:
        raise ValueError("reference model must return [batch, sequence, vocab] logits")
    if tuple(logits.shape[:2]) != tuple(input_ids.shape):
        raise ValueError("reference logits must align with every input position")
    return logits


def _validate_inputs(prompt_ids: torch.Tensor, max_new_tokens: int) -> None:
    if prompt_ids.ndim != 2 or prompt_ids.shape[0] != 1 or prompt_ids.shape[1] == 0:
        raise ValueError("reference decoding requires prompt shape [1, sequence > 0]")
    if prompt_ids.device.type != "cpu":
        raise ValueError("the prompt-lookup reference path is CPU-only")
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens cannot be negative")


@torch.inference_mode()
def ordinary_greedy(
    model: CausalLogitsModel,
    prompt_ids: torch.Tensor,
    max_new_tokens: int,
) -> torch.Tensor:
    """Generate tokens by ordinary full-context greedy decoding."""

    _validate_inputs(prompt_ids, max_new_tokens)
    context = prompt_ids.clone()
    for _ in range(max_new_tokens):
        next_token = _model_logits(model, context)[:, -1].argmax(dim=-1, keepdim=True)
        context = torch.cat((context, next_token), dim=1)
    return context[:, prompt_ids.shape[1] :]


@torch.inference_mode()
def speculative_greedy(
    model: CausalLogitsModel,
    prompt_ids: torch.Tensor,
    max_new_tokens: int,
    *,
    k: int,
    ngram: int,
) -> torch.Tensor:
    """Generate with prompt lookup and the exact greedy accept rule.

    This intentionally recomputes the full context. It is a compact semantic
    reference, not a performance path. The production engine keeps recurrent
    state and uses :class:`DecodeCheckpoint` for rollback.
    """

    _validate_inputs(prompt_ids, max_new_tokens)
    if k <= 0:
        raise ValueError("k must be positive")
    if ngram <= 0:
        raise ValueError("ngram must be positive")

    context = prompt_ids.clone()
    prompt_length = prompt_ids.shape[1]
    while context.shape[1] - prompt_length < max_new_tokens:
        remaining = max_new_tokens - (context.shape[1] - prompt_length)
        draft = propose(context[0].tolist(), min(k, remaining), ngram)
        if not draft:
            next_token = _model_logits(model, context)[:, -1].argmax(
                dim=-1, keepdim=True
            )
            context = torch.cat((context, next_token), dim=1)
            continue

        draft_ids = torch.tensor(draft, dtype=context.dtype).unsqueeze(0)
        candidate = torch.cat((context, draft_ids), dim=1)
        candidate_logits = _model_logits(model, candidate)
        start = context.shape[1] - 1
        aligned_logits = candidate_logits[:, start : start + len(draft)]
        verification = verify_greedy(draft_ids, aligned_logits)
        emitted_count = int(verification.emitted_count.tolist()[0])
        emitted = verification.emitted_ids[:, :emitted_count]
        context = torch.cat((context, emitted), dim=1)

    return context[:, prompt_length : prompt_length + max_new_tokens]
