"""One immutable prompt plan shared by both measured runtimes."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from typing import Any


def default_prompts() -> tuple[dict[str, str], ...]:
    """Reuse the repository's fixed Kimi-Linear quality prompt corpus."""

    from engine.modal_bench import PROMPTS

    return tuple({"id": prompt["id"], "text": prompt["text"]} for prompt in PROMPTS)


def build_prompt_set(
    tokenizer: Any,
    *,
    model_id: str,
    resolved_revision: str,
    max_prompt_tokens: int,
    max_output_tokens: int,
    seed: int,
    prompts: Sequence[dict[str, str]] | None = None,
) -> dict[str, Any]:
    """Tokenize once so both servers receive identical token ID arrays."""

    if max_prompt_tokens < 1:
        raise ValueError("max_prompt_tokens must be positive")
    if max_output_tokens < 2:
        raise ValueError("max_output_tokens must be at least two")
    selected = tuple(prompts or default_prompts())
    if not selected:
        raise ValueError("prompt set must not be empty")

    records: list[dict[str, Any]] = []
    for prompt in selected:
        prompt_id = prompt.get("id")
        text = prompt.get("text")
        if not isinstance(prompt_id, str) or not prompt_id:
            raise ValueError("every prompt must have a nonempty id")
        if not isinstance(text, str) or not text:
            raise ValueError(f"prompt {prompt_id!r} must have nonempty text")
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": text}],
            tokenize=False,
            add_generation_prompt=True,
        )
        encoded = tokenizer(
            rendered,
            add_special_tokens=False,
            truncation=True,
            max_length=max_prompt_tokens,
        )
        token_ids = encoded.get("input_ids")
        if (
            not isinstance(token_ids, list)
            or not token_ids
            or not all(isinstance(token, int) and token >= 0 for token in token_ids)
        ):
            raise RuntimeError(f"tokenizer returned invalid token IDs for {prompt_id}")
        records.append(
            {
                "id": prompt_id,
                "text": text,
                "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                "rendered_prompt_sha256": hashlib.sha256(
                    rendered.encode("utf-8")
                ).hexdigest(),
                "token_ids": token_ids,
                "prompt_tokens": len(token_ids),
            }
        )

    identity_payload = {
        "model_id": model_id,
        "resolved_revision": resolved_revision,
        "max_prompt_tokens": max_prompt_tokens,
        "max_output_tokens": max_output_tokens,
        "seed": seed,
        "prompts": records,
        "request_parameters": {
            "temperature": 0.0,
            "min_tokens": max_output_tokens,
            "ignore_eos": True,
            "stream": True,
            "stream_options": {"include_usage": True},
        },
    }
    encoded_identity = json.dumps(
        identity_payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return {
        "id": "sha256:" + hashlib.sha256(encoded_identity).hexdigest(),
        "model_id": model_id,
        "resolved_revision": resolved_revision,
        "tokenizer_class": type(tokenizer).__name__,
        "max_prompt_tokens": max_prompt_tokens,
        "max_output_tokens": max_output_tokens,
        "seed": seed,
        "request_parameters": identity_payload["request_parameters"],
        "prompts": records,
    }
