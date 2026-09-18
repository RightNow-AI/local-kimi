"""Weight-free prompt lookup for speculative decoding."""

from __future__ import annotations

from collections.abc import Sequence


def propose(context_ids: Sequence[int], k: int, ngram: int) -> list[int]:
    """Propose up to ``k`` tokens following the latest matching n-gram.

    The current suffix is not allowed to match itself. The search proceeds from
    newest to oldest, so repeated text uses the most recent available
    continuation. The function is intentionally host-side. Production wiring
    should keep the generated token history in host memory rather than copying
    the model state or logits back merely to perform this lookup.
    """

    if k < 0:
        raise ValueError("k cannot be negative")
    if ngram <= 0:
        raise ValueError("ngram must be positive")
    if k == 0 or len(context_ids) < ngram + 1:
        return []

    suffix_start = len(context_ids) - ngram
    suffix = context_ids[suffix_start:]
    for start in range(suffix_start - 1, -1, -1):
        if context_ids[start : start + ngram] != suffix:
            continue
        continuation_start = start + ngram
        continuation_end = min(continuation_start + k, len(context_ids))
        if continuation_start < continuation_end:
            return [int(token_id) for token_id in context_ids[continuation_start:continuation_end]]
    return []
