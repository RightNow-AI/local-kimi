"""Tensor-only greedy speculative verification."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class GreedyVerification:
    """Fixed-shape verification result suitable for CUDA graph capture."""

    model_ids: torch.Tensor
    accepted_prefix_length: torch.Tensor
    emitted_ids: torch.Tensor
    emitted_count: torch.Tensor
    emitted_mask: torch.Tensor


def align_verification_logits(
    next_logits: torch.Tensor,
    drafted_forward_logits: torch.Tensor,
) -> torch.Tensor:
    """Align model logits with the drafted tokens they verify.

    A causal forward over drafted tokens returns logits predicting the tokens
    after each draft input. The first draft token is instead predicted by the
    logits already available before the round. Prepending those logits and
    dropping the final forward row produces one verifier row per draft token.
    """

    if next_logits.ndim != 2:
        raise ValueError("next_logits must have shape [batch, vocab]")
    if drafted_forward_logits.ndim != 3:
        raise ValueError(
            "drafted_forward_logits must have shape [batch, draft, vocab]"
        )
    if drafted_forward_logits.shape[1] == 0:
        raise ValueError("drafted_forward_logits must contain at least one token")
    if (
        next_logits.shape[0] != drafted_forward_logits.shape[0]
        or next_logits.shape[1] != drafted_forward_logits.shape[2]
    ):
        raise ValueError("next and drafted logits must agree on batch and vocabulary")
    return torch.cat(
        (next_logits.unsqueeze(1), drafted_forward_logits[:, :-1]),
        dim=1,
    )


def verify_greedy(
    draft_ids: torch.Tensor,
    aligned_logits: torch.Tensor,
) -> GreedyVerification:
    """Apply the exact greedy speculative accept rule.

    Under greedy decoding this produces exactly the same output as ordinary
    greedy decoding, token for token. Every accepted draft token equals the
    model's own argmax at that position. At the first mismatch, the draft token
    is replaced by that same model argmax. The fixed-shape result guarantees at
    least one emitted token and contains no host scalar extraction.

    ``aligned_logits[:, i]`` must predict ``draft_ids[:, i]``. Use
    :func:`align_verification_logits` when a stateful causal model was run once
    over the full draft.
    """

    if draft_ids.ndim != 2 or draft_ids.shape[1] == 0:
        raise ValueError("draft_ids must have shape [batch, draft > 0]")
    if aligned_logits.ndim != 3:
        raise ValueError("aligned_logits must have shape [batch, draft, vocab]")
    if tuple(aligned_logits.shape[:2]) != tuple(draft_ids.shape):
        raise ValueError("aligned logits must provide one row per draft token")
    if aligned_logits.shape[2] == 0:
        raise ValueError("aligned_logits must contain a nonempty vocabulary")
    if draft_ids.device != aligned_logits.device:
        raise ValueError("draft_ids and aligned_logits must share one device")

    batch, draft_length = draft_ids.shape
    model_ids = aligned_logits.argmax(dim=-1)
    prefix_matches = torch.cumprod(
        model_ids.eq(draft_ids).to(torch.int64),
        dim=1,
    ).bool()
    accepted = prefix_matches.sum(dim=1)

    positions = torch.arange(draft_length, device=draft_ids.device).view(1, -1)
    positions = positions.expand(batch, -1)
    accepted_column = accepted.unsqueeze(1)
    accepted_draft = positions < accepted_column
    first_mismatch = (positions == accepted_column) & (
        accepted_column < draft_length
    )
    emitted_count = (accepted + 1).clamp(max=draft_length)
    emitted_mask = positions < emitted_count.unsqueeze(1)
    emitted_ids = torch.where(
        accepted_draft,
        draft_ids,
        torch.where(first_mismatch, model_ids, torch.zeros_like(draft_ids)),
    )
    return GreedyVerification(
        model_ids,
        accepted,
        emitted_ids,
        emitted_count,
        emitted_mask,
    )
