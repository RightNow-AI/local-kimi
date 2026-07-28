"""Interfaces shared by the HTTP layer, tokenizers, and inference engines."""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, Protocol, TypeAlias


@dataclass(frozen=True, slots=True)
class SamplingParams:
    """Sampling controls supported by the reference generator."""

    max_tokens: int
    temperature: float = 0.0
    top_p: float = 1.0


@dataclass(frozen=True, slots=True)
class ChatPrompt:
    """Structured chat input presented to the tokenizer boundary."""

    messages: tuple[Mapping[str, object], ...]
    tools: tuple[Mapping[str, object], ...] = ()
    tool_choice: object = None
    reasoning_effort: str | None = None


@dataclass(frozen=True, slots=True)
class TokenEvent:
    """One token sampled by the engine."""

    token_id: int


@dataclass(frozen=True, slots=True)
class UsageEvent:
    """The final exact token counts for one generation."""

    prompt_tokens: int
    completion_tokens: int


GenerationEvent: TypeAlias = TokenEvent | UsageEvent


class InferenceEngine(Protocol):
    """Minimal engine surface needed by the server.

    Implementations must return a new async iterator per request. The iterator
    yields token IDs in generation order, then exactly one final UsageEvent.
    Closing the iterator must cancel that request and release its resources.
    """

    def generate(
        self,
        prompt_token_ids: Sequence[int],
        params: SamplingParams,
    ) -> AsyncIterator[GenerationEvent]: ...


Channel = Literal["reasoning", "content"]


@dataclass(frozen=True, slots=True)
class DecodedFragment:
    """Decoded text together with its OpenAI output channel."""

    channel: Channel
    text: str


class IncrementalDecoder(Protocol):
    """A request-local decoder for generated token IDs."""

    def push(self, token_id: int) -> Sequence[DecodedFragment]: ...

    def finish(self) -> Sequence[DecodedFragment]: ...


class ChatTokenizer(Protocol):
    """Explicit boundary between OpenAI messages and engine token IDs.

    A Kimi tokenizer implementation is responsible for the real chat template
    and for mapping its think and response control tokens to separate output
    channels. No HTTP code assumes a character-to-token ratio.
    """

    @property
    def eos_token_ids(self) -> frozenset[int]: ...

    def encode_prompt(self, prompt: ChatPrompt) -> Sequence[int]: ...

    def new_decoder(self) -> IncrementalDecoder: ...


__all__ = [
    "Channel",
    "ChatPrompt",
    "ChatTokenizer",
    "DecodedFragment",
    "GenerationEvent",
    "IncrementalDecoder",
    "InferenceEngine",
    "SamplingParams",
    "TokenEvent",
    "UsageEvent",
]
