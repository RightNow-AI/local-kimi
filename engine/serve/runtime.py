"""Request-local generation accounting and engine serialization."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from typing import Literal

from .contracts import (
    ChatTokenizer,
    DecodedFragment,
    InferenceEngine,
    SamplingParams,
    TokenEvent,
    UsageEvent,
)


class GenerationContractError(RuntimeError):
    """The engine or tokenizer violated the serving contract."""


@dataclass(frozen=True, slots=True)
class CompletionDelta:
    channel: Literal["reasoning", "content"]
    text: str


@dataclass(frozen=True, slots=True)
class CompletionEnd:
    finish_reason: Literal["stop", "length"]
    prompt_tokens: int
    completion_tokens: int

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


CompletionEvent = CompletionDelta | CompletionEnd


class GenerationRuntime:
    """Drive one engine safely and verify exact token accounting.

    The default lock is intentional. The reference engine is single-threaded,
    so concurrent HTTP requests wait here instead of sharing mutable decode
    state. A deployment with an engine that owns request isolation may set
    ``serialize_engine=False`` without changing the engine or tokenizer APIs.
    """

    def __init__(
        self,
        engine: InferenceEngine,
        tokenizer: ChatTokenizer,
        *,
        serialize_engine: bool = True,
    ) -> None:
        self.engine = engine
        self.tokenizer = tokenizer
        self._lock = asyncio.Lock() if serialize_engine else None

    async def run(
        self,
        prompt_token_ids: Sequence[int],
        params: SamplingParams,
    ) -> AsyncIterator[CompletionEvent]:
        prompt_ids = tuple(prompt_token_ids)
        if not prompt_ids:
            raise GenerationContractError("tokenizer produced an empty prompt")
        if any(
            isinstance(token_id, bool) or not isinstance(token_id, int)
            for token_id in prompt_ids
        ):
            raise GenerationContractError("tokenizer produced a non-integer prompt token ID")
        if params.max_tokens < 1:
            raise GenerationContractError("max_tokens must be at least 1")

        decoder = self.tokenizer.new_decoder()
        source = self._serialized_events(prompt_ids, params)
        completion_tokens = 0
        usage: UsageEvent | None = None
        saw_eos = False

        try:
            async for event in source:
                if isinstance(event, UsageEvent):
                    if usage is not None:
                        raise GenerationContractError("engine emitted more than one usage record")
                    usage = event
                    continue

                if not isinstance(event, TokenEvent):
                    raise GenerationContractError(
                        f"engine emitted unsupported event {type(event).__name__}"
                    )
                if usage is not None:
                    raise GenerationContractError("engine emitted a token after its usage record")
                if isinstance(event.token_id, bool) or not isinstance(event.token_id, int):
                    raise GenerationContractError("engine emitted a non-integer token ID")
                if saw_eos:
                    raise GenerationContractError("engine emitted a token after EOS")

                completion_tokens += 1
                if completion_tokens > params.max_tokens:
                    raise GenerationContractError("engine exceeded max_tokens")
                if event.token_id in self.tokenizer.eos_token_ids:
                    saw_eos = True
                    continue

                for fragment in decoder.push(event.token_id):
                    yield _checked_fragment(fragment)

            if usage is None:
                raise GenerationContractError("engine ended without a final usage record")
            if usage.prompt_tokens != len(prompt_ids):
                raise GenerationContractError(
                    "engine prompt token usage does not match the encoded prompt"
                )
            if usage.completion_tokens != completion_tokens:
                raise GenerationContractError(
                    "engine completion token usage does not match emitted token IDs"
                )

            for fragment in decoder.finish():
                yield _checked_fragment(fragment)

            if saw_eos:
                finish_reason: Literal["stop", "length"] = "stop"
            elif completion_tokens == params.max_tokens:
                finish_reason = "length"
            else:
                raise GenerationContractError(
                    "engine ended before EOS without reaching max_tokens"
                )

            yield CompletionEnd(
                finish_reason=finish_reason,
                prompt_tokens=usage.prompt_tokens,
                completion_tokens=usage.completion_tokens,
            )
        finally:
            await source.aclose()

    async def _serialized_events(
        self,
        prompt_token_ids: Sequence[int],
        params: SamplingParams,
    ) -> AsyncIterator[TokenEvent | UsageEvent]:
        if self._lock is None:
            async for event in self._engine_events(prompt_token_ids, params):
                yield event
            return

        async with self._lock:
            async for event in self._engine_events(prompt_token_ids, params):
                yield event

    async def _engine_events(
        self,
        prompt_token_ids: Sequence[int],
        params: SamplingParams,
    ) -> AsyncIterator[TokenEvent | UsageEvent]:
        source = self.engine.generate(prompt_token_ids, params)
        try:
            async for event in source:
                yield event
        finally:
            await source.aclose()


def _checked_fragment(fragment: DecodedFragment) -> CompletionDelta:
    if fragment.channel not in ("reasoning", "content"):
        raise GenerationContractError(f"tokenizer emitted unknown channel {fragment.channel!r}")
    if not isinstance(fragment.text, str):
        raise GenerationContractError("tokenizer emitted non-string text")
    return CompletionDelta(channel=fragment.channel, text=fragment.text)


__all__ = [
    "CompletionDelta",
    "CompletionEnd",
    "CompletionEvent",
    "GenerationContractError",
    "GenerationRuntime",
]
