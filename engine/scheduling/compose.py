"""Choose pending tokens whose per-layer expert routes overlap.

The objective is the sum of distinct experts over all routed layers. This is
also the average per-layer union multiplied by the layer count. It is the right
objective for routed weight traffic because every distinct expert at every
layer contributes one expert-sized weight read. Optimizing one layer can move
cost into another layer, while minimizing the worst layer can increase total
bytes.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Hashable, Iterable, Literal, Sequence

Strategy = Literal["greedy", "random"]


def _expert_mask(experts: Iterable[int]) -> tuple[frozenset[int], int]:
    normalized = frozenset(experts)
    if not normalized:
        raise ValueError("each layer must route to at least one expert")
    if any(isinstance(expert, bool) or not isinstance(expert, int) for expert in normalized):
        raise TypeError("expert identifiers must be integers")
    if any(expert < 0 for expert in normalized):
        raise ValueError("expert identifiers must be non-negative")
    mask = 0
    for expert in normalized:
        mask |= 1 << expert
    return normalized, mask


@dataclass(frozen=True)
class PendingToken:
    """One schedulable token with a known expert set for every MoE layer."""

    token_id: Hashable
    layer_experts: tuple[frozenset[int], ...]
    layer_masks: tuple[int, ...] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        try:
            hash(self.token_id)
        except TypeError as exc:
            raise TypeError("token_id must be hashable") from exc
        if not self.layer_experts:
            raise ValueError("a pending token must contain at least one routed layer")

        normalized_layers = []
        masks = []
        for experts in self.layer_experts:
            normalized, mask = _expert_mask(experts)
            normalized_layers.append(normalized)
            masks.append(mask)
        object.__setattr__(self, "layer_experts", tuple(normalized_layers))
        object.__setattr__(self, "layer_masks", tuple(masks))


@dataclass(frozen=True)
class CompositionDecision:
    """Greedy choice and the paired random baseline from the same pool."""

    selected: tuple[PendingToken, ...]
    greedy_candidate: tuple[PendingToken, ...]
    random_baseline: tuple[PendingToken, ...]
    selected_total_union: int
    greedy_total_union: int
    random_total_union: int
    used_random_guard: bool


def _validated_pool(
    pool: Sequence[PendingToken], max_batch_size: int
) -> tuple[tuple[PendingToken, ...], int]:
    if isinstance(max_batch_size, bool) or not isinstance(max_batch_size, int):
        raise TypeError("max_batch_size must be an integer")
    if max_batch_size <= 0:
        raise ValueError("max_batch_size must be positive")
    normalized = tuple(pool)
    if not normalized:
        return normalized, 0
    layer_count = len(normalized[0].layer_masks)
    if any(len(token.layer_masks) != layer_count for token in normalized):
        raise ValueError("all pending tokens must describe the same number of layers")
    token_ids = [token.token_id for token in normalized]
    if len(set(token_ids)) != len(token_ids):
        raise ValueError("pending token identifiers must be unique")
    return normalized, min(max_batch_size, len(normalized))


def _total_union(tokens: Sequence[PendingToken]) -> int:
    if not tokens:
        return 0
    unions = [0] * len(tokens[0].layer_masks)
    for token in tokens:
        for layer, mask in enumerate(token.layer_masks):
            unions[layer] |= mask
    return sum(mask.bit_count() for mask in unions)


def _random_candidate(
    pool: tuple[PendingToken, ...], batch_size: int, seed: int | None
) -> tuple[PendingToken, ...]:
    if batch_size == 0:
        return ()
    rng = random.Random(seed)
    return tuple(rng.sample(pool, batch_size))


def _greedy_candidate(
    pool: tuple[PendingToken, ...], batch_size: int
) -> tuple[PendingToken, ...]:
    if batch_size == 0:
        return ()
    layer_unions = [0] * len(pool[0].layer_masks)
    remaining = list(enumerate(pool))
    selected: list[PendingToken] = []

    while len(selected) < batch_size:
        best_position = 0
        best_key: tuple[int, int] | None = None
        uncovered = tuple(~mask for mask in layer_unions)
        for position, (original_position, token) in enumerate(remaining):
            added_experts = sum(
                (mask & uncovered[layer]).bit_count()
                for layer, mask in enumerate(token.layer_masks)
            )
            key = (added_experts, original_position)
            if best_key is None or key < best_key:
                best_key = key
                best_position = position

        _, chosen = remaining.pop(best_position)
        selected.append(chosen)
        for layer, mask in enumerate(chosen.layer_masks):
            layer_unions[layer] |= mask

    return tuple(selected)


def compare_composers(
    pool: Sequence[PendingToken],
    max_batch_size: int,
    *,
    seed: int | None = 0,
) -> CompositionDecision:
    """Compare greedy with random and fail closed if greedy is locally worse.

    Greedy set-union minimization is a heuristic, not a global optimum. The
    paired random candidate is already needed for measurement, so it is also a
    cheap safety guard. The public greedy result therefore never has a larger
    union than random on the same pool and seed.
    """

    normalized, batch_size = _validated_pool(pool, max_batch_size)
    random_baseline = _random_candidate(normalized, batch_size, seed)
    greedy_candidate = _greedy_candidate(normalized, batch_size)
    random_total = _total_union(random_baseline)
    greedy_total = _total_union(greedy_candidate)
    use_random = greedy_total > random_total
    selected = random_baseline if use_random else greedy_candidate
    return CompositionDecision(
        selected=selected,
        greedy_candidate=greedy_candidate,
        random_baseline=random_baseline,
        selected_total_union=min(greedy_total, random_total),
        greedy_total_union=greedy_total,
        random_total_union=random_total,
        used_random_guard=use_random,
    )


def compose_batch(
    pool: Sequence[PendingToken],
    max_batch_size: int,
    *,
    strategy: Strategy = "greedy",
    seed: int | None = 0,
) -> tuple[PendingToken, ...]:
    """Return at most ``max_batch_size`` unique tokens from ``pool``."""

    if strategy == "greedy":
        return compare_composers(pool, max_batch_size, seed=seed).selected
    if strategy == "random":
        normalized, batch_size = _validated_pool(pool, max_batch_size)
        return _random_candidate(normalized, batch_size, seed)
    raise ValueError(f"unknown composition strategy: {strategy}")
