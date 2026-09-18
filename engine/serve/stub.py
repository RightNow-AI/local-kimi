"""CPU-only tokenizer and echo engine used by tests and local smoke checks."""

from __future__ import annotations

import asyncio
import codecs
import json
from collections.abc import AsyncIterator, Mapping, Sequence
from typing import Any

from .contracts import (
    ChatPrompt,
    DecodedFragment,
    GenerationEvent,
    SamplingParams,
    TokenEvent,
    UsageEvent,
)


class ByteChatTokenizer:
    """A deterministic byte tokenizer with explicit output channel tokens.

    It is deliberately simple, but its counts are real token counts. UTF-8
    bytes are token IDs 0 through 255. Three control IDs mark reasoning,
    visible response text, and EOS.
    """

    REASONING = 256
    RESPONSE = 257
    EOS = 258

    @property
    def eos_token_ids(self) -> frozenset[int]:
        return frozenset({self.EOS})

    def encode_prompt(self, prompt: ChatPrompt) -> list[int]:
        rendered = json.dumps(
            {
                "messages": [dict(message) for message in prompt.messages],
                "tools": [dict(tool) for tool in prompt.tools],
                "tool_choice": prompt.tool_choice,
                "reasoning_effort": prompt.reasoning_effort,
            },
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        return list(rendered.encode("utf-8"))

    def decode_prompt(self, prompt_token_ids: Sequence[int]) -> dict[str, Any]:
        raw = bytes(prompt_token_ids).decode("utf-8")
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else {}

    def encode_generation(self, reasoning: str, content: str, *, eos: bool = True) -> list[int]:
        token_ids = [self.REASONING]
        token_ids.extend(reasoning.encode("utf-8"))
        token_ids.append(self.RESPONSE)
        token_ids.extend(content.encode("utf-8"))
        if eos:
            token_ids.append(self.EOS)
        return token_ids

    def new_decoder(self) -> "ByteIncrementalDecoder":
        return ByteIncrementalDecoder(self)


class ByteIncrementalDecoder:
    def __init__(self, tokenizer: ByteChatTokenizer) -> None:
        self.tokenizer = tokenizer
        self.channel = "content"
        self._decoder = codecs.getincrementaldecoder("utf-8")("replace")

    def push(self, token_id: int) -> list[DecodedFragment]:
        if token_id == self.tokenizer.REASONING:
            return self._switch("reasoning")
        if token_id == self.tokenizer.RESPONSE:
            return self._switch("content")
        if not 0 <= token_id <= 255:
            raise ValueError(f"unknown stub token ID {token_id}")
        text = self._decoder.decode(bytes([token_id]), final=False)
        return [DecodedFragment(self.channel, text)] if text else []

    def finish(self) -> list[DecodedFragment]:
        text = self._decoder.decode(b"", final=True)
        self._reset_decoder()
        return [DecodedFragment(self.channel, text)] if text else []

    def _switch(self, channel: str) -> list[DecodedFragment]:
        text = self._decoder.decode(b"", final=True)
        previous = self.channel
        self.channel = channel
        self._reset_decoder()
        return [DecodedFragment(previous, text)] if text else []

    def _reset_decoder(self) -> None:
        self._decoder = codecs.getincrementaldecoder("utf-8")("replace")


class EchoEngine:
    """A cancellable zero-weight engine that echoes the final user message."""

    def __init__(
        self,
        tokenizer: ByteChatTokenizer,
        *,
        reasoning: str = "I will echo the final user message.",
        response_prefix: str = "echo: ",
        delay: float = 0.0,
        emit_eos: bool = True,
    ) -> None:
        self.tokenizer = tokenizer
        self.reasoning = reasoning
        self.response_prefix = response_prefix
        self.delay = delay
        self.emit_eos = emit_eos
        self.prompt_token_ids: list[tuple[int, ...]] = []
        self.generated_token_ids: list[tuple[int, ...]] = []
        self.active_generations = 0
        self.max_active_generations = 0
        self.cancelled_generations = 0

    async def generate(
        self,
        prompt_token_ids: Sequence[int],
        params: SamplingParams,
    ) -> AsyncIterator[GenerationEvent]:
        completed = False
        self.active_generations += 1
        self.max_active_generations = max(
            self.max_active_generations,
            self.active_generations,
        )
        try:
            prompt_ids = tuple(prompt_token_ids)
            self.prompt_token_ids.append(prompt_ids)
            prompt = self.tokenizer.decode_prompt(prompt_ids)
            content = self.response_prefix + _last_user_text(prompt.get("messages"))
            generated = self.tokenizer.encode_generation(
                self.reasoning,
                content,
                eos=self.emit_eos,
            )
            emitted = tuple(generated[: params.max_tokens])
            self.generated_token_ids.append(emitted)

            for token_id in emitted:
                if self.delay:
                    await asyncio.sleep(self.delay)
                yield TokenEvent(token_id)

            yield UsageEvent(
                prompt_tokens=len(prompt_ids),
                completion_tokens=len(emitted),
            )
            completed = True
        finally:
            self.active_generations -= 1
            if not completed:
                self.cancelled_generations += 1

    async def health(self) -> tuple[bool, str]:
        return True, "cpu echo engine"


def _last_user_text(messages: object) -> str:
    if not isinstance(messages, list):
        return ""
    for message in reversed(messages):
        if not isinstance(message, Mapping) or message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            pieces: list[str] = []
            for part in content:
                if isinstance(part, str):
                    pieces.append(part)
                elif isinstance(part, Mapping) and isinstance(part.get("text"), str):
                    pieces.append(part["text"])
            return "".join(pieces)
        return str(content) if content is not None else ""
    return ""


__all__ = ["ByteChatTokenizer", "ByteIncrementalDecoder", "EchoEngine"]
