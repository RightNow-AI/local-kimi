"""Offline-safe quality metrics for reference and candidate model outputs.

The public functions accept ordinary Python sequences. When called with torch
tensors inside the Modal benchmark, they use torch operations without making
torch a local development dependency.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from numbers import Integral, Real
from typing import Any


def _is_torch_tensor(value: Any) -> bool:
    module = value.__class__.__module__
    return module == "torch" or module.startswith("torch.")


def _as_python(value: Any) -> Any:
    if _is_torch_tensor(value):
        return value.detach().cpu().tolist()
    return value


def _logit_rows(logits: Any) -> list[list[float]]:
    """Flatten leading dimensions while preserving the vocabulary dimension."""
    value = _as_python(logits)
    rows: list[list[float]] = []

    def visit(node: Any) -> None:
        node = _as_python(node)
        if not isinstance(node, Sequence) or isinstance(node, (str, bytes)):
            raise TypeError("logits must be a sequence with a final vocabulary dimension")
        if len(node) == 0:
            return
        if all(isinstance(item, Real) for item in node):
            row = [float(item) for item in node]
            if not all(math.isfinite(item) for item in row):
                raise ValueError("logits must all be finite")
            rows.append(row)
            return
        for child in node:
            visit(child)

    visit(value)
    if not rows:
        raise ValueError("logits must contain at least one token row")
    width = len(rows[0])
    if width == 0 or any(len(row) != width for row in rows):
        raise ValueError("all logit rows must have the same nonzero vocabulary size")
    return rows


def _same_tensor_shape(reference: Any, candidate: Any) -> None:
    if tuple(reference.shape) != tuple(candidate.shape):
        raise ValueError(
            f"shape mismatch: reference {tuple(reference.shape)} != candidate {tuple(candidate.shape)}"
        )
    if reference.numel() == 0 or reference.shape[-1] == 0:
        raise ValueError("logits must contain at least one token and one vocabulary item")


def _logsumexp(row: Sequence[float]) -> float:
    peak = max(row)
    return peak + math.log(math.fsum(math.exp(item - peak) for item in row))


def token_kl_divergence(reference_logits: Any, candidate_logits: Any) -> float:
    """Return mean token KL(reference || candidate) in nats.

    Each token contributes sum(p_ref * (log(p_ref) - log(p_candidate))).
    The final value is the arithmetic mean over token positions.
    """
    if _is_torch_tensor(reference_logits) and _is_torch_tensor(candidate_logits):
        import torch

        _same_tensor_shape(reference_logits, candidate_logits)
        reference_log_probs = torch.log_softmax(reference_logits.float(), dim=-1)
        candidate_log_probs = torch.log_softmax(candidate_logits.float(), dim=-1)
        per_token = torch.sum(
            reference_log_probs.exp() * (reference_log_probs - candidate_log_probs),
            dim=-1,
        )
        # Roundoff can produce a tiny negative for mathematically identical inputs.
        return max(0.0, float(per_token.mean().item()))

    reference = _logit_rows(reference_logits)
    candidate = _logit_rows(candidate_logits)
    if len(reference) != len(candidate) or len(reference[0]) != len(candidate[0]):
        raise ValueError("reference and candidate logits must have identical shapes")

    token_values: list[float] = []
    for ref_row, cand_row in zip(reference, candidate, strict=True):
        ref_norm = _logsumexp(ref_row)
        cand_norm = _logsumexp(cand_row)
        value = math.fsum(
            math.exp(ref_logit - ref_norm)
            * ((ref_logit - ref_norm) - (cand_logit - cand_norm))
            for ref_logit, cand_logit in zip(ref_row, cand_row, strict=True)
        )
        token_values.append(value)
    return max(0.0, math.fsum(token_values) / len(token_values))


def top1_agreement(reference_logits: Any, candidate_logits: Any) -> float:
    """Return the fraction of token positions with the same argmax token."""
    if _is_torch_tensor(reference_logits) and _is_torch_tensor(candidate_logits):
        _same_tensor_shape(reference_logits, candidate_logits)
        reference_top1 = reference_logits.argmax(dim=-1)
        candidate_top1 = candidate_logits.argmax(dim=-1)
        return float((reference_top1 == candidate_top1).float().mean().item())

    reference = _logit_rows(reference_logits)
    candidate = _logit_rows(candidate_logits)
    if len(reference) != len(candidate) or len(reference[0]) != len(candidate[0]):
        raise ValueError("reference and candidate logits must have identical shapes")
    matches = sum(
        max(range(len(ref_row)), key=ref_row.__getitem__)
        == max(range(len(cand_row)), key=cand_row.__getitem__)
        for ref_row, cand_row in zip(reference, candidate, strict=True)
    )
    return matches / len(reference)


def _target_ids(target_token_ids: Any) -> list[int]:
    value = _as_python(target_token_ids)
    targets: list[int] = []

    def visit(node: Any) -> None:
        node = _as_python(node)
        if isinstance(node, Integral):
            targets.append(int(node))
            return
        if not isinstance(node, Sequence) or isinstance(node, (str, bytes)):
            raise TypeError("target_token_ids must be an integer sequence")
        for child in node:
            visit(child)

    visit(value)
    return targets


def perplexity(logits: Any, target_token_ids: Any, *, ignore_index: int = -100) -> float:
    """Return exp(mean next-token negative log likelihood).

    Every logit row must correspond directly to the target at the same flattened
    position. Callers perform the causal shift before invoking this function.
    """
    if _is_torch_tensor(logits) and _is_torch_tensor(target_token_ids):
        import torch.nn.functional as functional

        if tuple(logits.shape[:-1]) != tuple(target_token_ids.shape):
            raise ValueError("logit leading dimensions must match target_token_ids")
        loss = functional.cross_entropy(
            logits.float().reshape(-1, logits.shape[-1]),
            target_token_ids.reshape(-1),
            ignore_index=ignore_index,
        )
        if not loss.isfinite():
            raise ValueError("perplexity requires at least one non-ignored finite target")
        return float(loss.exp().item())

    rows = _logit_rows(logits)
    targets = _target_ids(target_token_ids)
    if len(rows) != len(targets):
        raise ValueError("each logit row must have exactly one target token")
    losses: list[float] = []
    for row, target in zip(rows, targets, strict=True):
        if target == ignore_index:
            continue
        if target < 0 or target >= len(row):
            raise ValueError(f"target token {target} is outside vocabulary size {len(row)}")
        losses.append(_logsumexp(row) - row[target])
    if not losses:
        raise ValueError("perplexity requires at least one non-ignored target")
    return math.exp(math.fsum(losses) / len(losses))


def _routing_rows(routes: Any) -> list[tuple[int, ...]]:
    value = _as_python(routes)
    rows: list[tuple[int, ...]] = []

    def visit(node: Any) -> None:
        node = _as_python(node)
        if isinstance(node, Mapping):
            for key in sorted(node):
                visit(node[key])
            return
        if not isinstance(node, Sequence) or isinstance(node, (str, bytes)):
            raise TypeError("routing decisions must be sequences or layer mappings")
        if len(node) == 0:
            return
        if all(isinstance(item, Integral) for item in node):
            row = tuple(int(item) for item in node)
            if len(set(row)) != len(row):
                raise ValueError("a token routing decision contains a duplicate expert")
            rows.append(row)
            return
        for child in node:
            visit(child)

    visit(value)
    if not rows:
        raise ValueError("routing decisions must contain at least one token")
    width = len(rows[0])
    if width == 0 or any(len(row) != width for row in rows):
        raise ValueError("all routing decisions must select the same nonzero expert count")
    return rows


def routing_agreement(
    reference_routes: Any,
    candidate_routes: Any,
    *,
    expected_experts_per_token: int | None = None,
) -> float:
    """Return the fraction of tokens with an identical selected expert set."""
    reference = _routing_rows(reference_routes)
    candidate = _routing_rows(candidate_routes)
    if len(reference) != len(candidate):
        raise ValueError("reference and candidate routing must contain the same token count")
    if len(reference[0]) != len(candidate[0]):
        raise ValueError("reference and candidate routing must select the same expert count")
    if expected_experts_per_token is not None and len(reference[0]) != expected_experts_per_token:
        raise ValueError(
            f"routing selected {len(reference[0])} experts, expected {expected_experts_per_token}"
        )
    matches = sum(
        frozenset(ref_row) == frozenset(cand_row)
        for ref_row, cand_row in zip(reference, candidate, strict=True)
    )
    return matches / len(reference)
