"""Fixed, stratified prompt selection for the engine validation lane."""

from __future__ import annotations

import hashlib
import json

from engine.accuracy.prompts import AccuracyPrompt, build_prompt_set

_SELECTED_PROMPT_IDS = (
    "factual-000",
    "factual-004",
    "factual-008",
    "factual-012",
    "factual-016",
    "factual-020",
    "factual-025",
    "reasoning-000",
    "reasoning-004",
    "reasoning-008",
    "reasoning-012",
    "reasoning-016",
    "reasoning-020",
    "reasoning-025",
    "code-000",
    "code-004",
    "code-008",
    "code-012",
    "code-016",
    "code-020",
    "code-025",
    "instruction-000",
    "instruction-004",
    "instruction-008",
    "instruction-012",
    "instruction-016",
    "instruction-020",
    "instruction-025",
    "long_context_recall-000",
    "long_context_recall-008",
    "long_context_recall-017",
    "long_context_recall-025",
)


def build_validation_prompt_set() -> tuple[AccuracyPrompt, ...]:
    by_id = {prompt.prompt_id: prompt for prompt in build_prompt_set()}
    missing = [prompt_id for prompt_id in _SELECTED_PROMPT_IDS if prompt_id not in by_id]
    if missing:
        raise AssertionError(f"validation prompts are absent from engine.accuracy: {missing}")
    prompts = tuple(by_id[prompt_id] for prompt_id in _SELECTED_PROMPT_IDS)
    if len(prompts) != 32 or len({prompt.prompt_id for prompt in prompts}) != 32:
        raise AssertionError("the validation prompt set must contain 32 unique prompts")
    return prompts


def validation_prompt_set_sha256(
    prompts: tuple[AccuracyPrompt, ...] | None = None,
) -> str:
    selected = build_validation_prompt_set() if prompts is None else prompts
    payload = json.dumps(
        [prompt.as_dict() for prompt in selected],
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


__all__ = ["build_validation_prompt_set", "validation_prompt_set_sha256"]
