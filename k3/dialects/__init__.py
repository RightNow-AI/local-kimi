"""Client dialects.

Each module lowers one client wire format into the canonical IR and raises the
canonical IR back out of it. Nothing in here talks to the engine.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .base import DialectContext, Dialect, sse

if TYPE_CHECKING:  # pragma: no cover
    pass


def get_dialect(name: str) -> "Dialect":
    from . import anthropic_messages, openai_chat, openai_responses

    table = {
        "anthropic_messages": anthropic_messages,
        "openai_chat": openai_chat,
        "openai_responses": openai_responses,
    }
    try:
        return table[name]  # type: ignore[return-value]
    except KeyError:
        raise ValueError(
            f"unknown dialect {name!r}; known: {', '.join(sorted(table))}"
        ) from None


__all__ = ["DialectContext", "Dialect", "get_dialect", "sse"]
