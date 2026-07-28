"""Kimi-Linear engine and Moonshot tokenizer adapters for ``engine.serve``."""

from __future__ import annotations

import asyncio
import codecs
import json
from collections.abc import AsyncIterator, Collection, Mapping, Sequence
from typing import Any, Literal

import torch

from ..klinear.generate import generate_tokens
from ..klinear.model import KLinearModel
from .contracts import (
    ChatPrompt,
    DecodedFragment,
    GenerationEvent,
    SamplingParams,
    TokenEvent,
    UsageEvent,
)

_OPEN = "<|open|>"
_CLOSE = "<|close|>"
_SEPARATOR = "<|sep|>"
_END_OF_MESSAGE = "<|end_of_msg|>"


class KimiChatTokenizer:
    """Adapt Moonshot's remote-code tiktoken tokenizer to the serve contract."""

    def __init__(self, tokenizer: Any) -> None:
        self.tokenizer = tokenizer
        self.open_token_id = _special_token_id(tokenizer, _OPEN)
        self.close_token_id = _special_token_id(tokenizer, _CLOSE)
        self.separator_token_id = _special_token_id(tokenizer, _SEPARATOR)
        self.end_of_message_token_id = _special_token_id(
            tokenizer,
            _END_OF_MESSAGE,
        )
        token_byte_decoder = getattr(
            getattr(tokenizer, "model", None),
            "decode_single_token_bytes",
            None,
        )
        if not callable(token_byte_decoder):
            raise TypeError(
                "Moonshot tokenizer must expose model.decode_single_token_bytes"
            )
        self._decode_single_token_bytes = token_byte_decoder

        eos_ids = _token_id_set(getattr(tokenizer, "eos_token_id", None))
        eos_ids.add(self.end_of_message_token_id)
        self._eos_token_ids = frozenset(eos_ids)

    @classmethod
    def from_directory(cls, directory: str) -> "KimiChatTokenizer":
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            directory,
            trust_remote_code=True,
            local_files_only=True,
        )
        return cls(tokenizer)

    @property
    def eos_token_ids(self) -> frozenset[int]:
        return self._eos_token_ids

    def encode_prompt(self, prompt: ChatPrompt) -> list[int]:
        messages = [dict(message) for message in prompt.messages]
        kwargs: dict[str, Any] = {
            "tokenize": True,
            "add_generation_prompt": True,
            "thinking": True,
        }
        if prompt.tools:
            kwargs["tools"] = [dict(tool) for tool in prompt.tools]
        if prompt.tool_choice is not None:
            kwargs["tool_choice"] = prompt.tool_choice
        if prompt.reasoning_effort is not None:
            kwargs["thinking_effort"] = _thinking_effort(prompt.reasoning_effort)

        token_ids = self.tokenizer.apply_chat_template(messages, **kwargs)
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        if not isinstance(token_ids, Sequence) or isinstance(token_ids, (str, bytes)):
            raise TypeError("Moonshot chat template did not return token IDs")
        if token_ids and isinstance(token_ids[0], Sequence):
            raise ValueError("Moonshot chat template unexpectedly returned a batch")
        result = list(token_ids)
        if not result:
            raise ValueError("Moonshot chat template returned an empty prompt")
        if any(
            isinstance(token_id, bool) or not isinstance(token_id, int)
            for token_id in result
        ):
            raise TypeError("Moonshot chat template returned a non-integer token ID")
        return result

    def new_decoder(self) -> "KimiIncrementalDecoder":
        return KimiIncrementalDecoder(self)

    def token_bytes(self, token_id: int) -> bytes:
        raw = self._decode_single_token_bytes(token_id)
        if not isinstance(raw, (bytes, bytearray)):
            raise TypeError("Moonshot tokenizer returned non-byte token payload")
        return bytes(raw)


class KimiIncrementalDecoder:
    """Decode exact token bytes while treating XTML tags as channel controls."""

    def __init__(self, tokenizer: KimiChatTokenizer) -> None:
        self.tokenizer = tokenizer
        self.channel: Literal["reasoning", "content"] | None = "reasoning"
        self._control: Literal["open", "close"] | None = None
        self._control_text: list[str] = []
        self._decoder = codecs.getincrementaldecoder("utf-8")("replace")

    def push(self, token_id: int) -> list[DecodedFragment]:
        if token_id == self.tokenizer.open_token_id:
            return self._start_control("open")
        if token_id == self.tokenizer.close_token_id:
            return self._start_control("close")
        if token_id == self.tokenizer.separator_token_id:
            fragments = self._flush_decoder()
            if self._control is not None:
                descriptor = "".join(self._control_text).strip()
                control = self._control
                self._control = None
                self._control_text.clear()
                self._apply_control(control, descriptor)
            return fragments
        if token_id == self.tokenizer.end_of_message_token_id:
            fragments = self._flush_decoder()
            self.channel = None
            return fragments

        text = self._decoder.decode(self.tokenizer.token_bytes(token_id), final=False)
        if not text:
            return []
        if self._control is not None:
            self._control_text.append(text)
            return []
        return self._fragment(text)

    def finish(self) -> list[DecodedFragment]:
        fragments = self._flush_decoder()
        self._control = None
        self._control_text.clear()
        return fragments

    def _start_control(
        self,
        control: Literal["open", "close"],
    ) -> list[DecodedFragment]:
        fragments = self._flush_decoder()
        if self._control is not None:
            raise ValueError("Moonshot output started a nested XTML control tag")
        self._control = control
        self._control_text.clear()
        return fragments

    def _flush_decoder(self) -> list[DecodedFragment]:
        text = self._decoder.decode(b"", final=True)
        self._decoder = codecs.getincrementaldecoder("utf-8")("replace")
        if not text:
            return []
        if self._control is not None:
            self._control_text.append(text)
            return []
        return self._fragment(text)

    def _fragment(self, text: str) -> list[DecodedFragment]:
        if self.channel is None or not text:
            return []
        return [DecodedFragment(self.channel, text)]

    def _apply_control(self, control: Literal["open", "close"], descriptor: str) -> None:
        tag = descriptor.split(maxsplit=1)[0] if descriptor else ""
        if control == "open" and tag == "think":
            self.channel = "reasoning"
        elif control == "close" and tag == "think":
            self.channel = None
        elif control == "open" and tag == "response":
            self.channel = "content"
        elif control == "close" and tag == "response":
            self.channel = None


class KLinearEngine:
    """Incremental, cancellable serving adapter around ``engine.klinear``."""

    def __init__(
        self,
        model: KLinearModel,
        eos_token_ids: Collection[int],
        *,
        device: torch.device | str | None = None,
        load_seconds: float | None = None,
        load_peak_gpu_memory_bytes: int | None = None,
    ) -> None:
        self.model = model
        self.eos_token_ids = frozenset(int(token_id) for token_id in eos_token_ids)
        if not self.eos_token_ids:
            raise ValueError("KLinearEngine requires at least one EOS token ID")
        self.device = torch.device(device) if device is not None else _model_device(model)
        if self.device.type == "meta":
            raise ValueError("KLinearEngine cannot serve a model on the meta device")
        self.load_seconds = load_seconds
        self.load_peak_gpu_memory_bytes = load_peak_gpu_memory_bytes
        self.active_generations = 0
        self.cancelled_generations = 0

    async def generate(
        self,
        prompt_token_ids: Sequence[int],
        params: SamplingParams,
    ) -> AsyncIterator[GenerationEvent]:
        prompt_ids = tuple(prompt_token_ids)
        prompt = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)
        stream = None
        completed = False
        completion_tokens = 0
        self.active_generations += 1
        try:
            stream = generate_tokens(
                self.model,
                prompt,
                params.max_tokens,
                temperature=params.temperature,
                top_p=params.top_p,
            )
            for sampled in stream:
                if sampled.ndim != 1 or sampled.numel() != 1:
                    raise ValueError("KLinearEngine serves exactly one sequence per request")
                token_id = int(sampled.item())
                completion_tokens += 1
                yield TokenEvent(token_id)
                await asyncio.sleep(0)
                if token_id in self.eos_token_ids or completion_tokens >= params.max_tokens:
                    break

            completed = True
            yield UsageEvent(
                prompt_tokens=len(prompt_ids),
                completion_tokens=completion_tokens,
            )
        finally:
            if stream is not None:
                close = getattr(stream, "close", None)
                if callable(close):
                    close()
            self.active_generations -= 1
            if not completed:
                self.cancelled_generations += 1

    def health(self) -> tuple[bool, str]:
        detail = {
            "engine": "engine.klinear",
            "device": str(self.device),
            "load_seconds": self.load_seconds,
            "load_peak_gpu_memory_bytes": self.load_peak_gpu_memory_bytes,
            "peak_gpu_memory_bytes": self._peak_gpu_memory_bytes(),
            "active_generations": self.active_generations,
            "cancelled_generations": self.cancelled_generations,
        }
        return True, json.dumps(detail, separators=(",", ":"), sort_keys=True)

    def _peak_gpu_memory_bytes(self) -> int | None:
        if self.device.type != "cuda" or not torch.cuda.is_available():
            return None
        return int(torch.cuda.max_memory_allocated(self.device))


def _model_device(model: KLinearModel) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration as exc:
        raise ValueError("KLinearEngine model has no parameters") from exc


def _special_token_id(tokenizer: Any, token: str) -> int:
    mapping = getattr(tokenizer, "special_tokens", None)
    if isinstance(mapping, Mapping):
        candidate = mapping.get(token)
        if isinstance(candidate, int) and not isinstance(candidate, bool):
            return candidate
    convert = getattr(tokenizer, "convert_tokens_to_ids", None)
    if callable(convert):
        candidate = convert(token)
        unknown = getattr(tokenizer, "unk_token_id", None)
        if (
            isinstance(candidate, int)
            and not isinstance(candidate, bool)
            and candidate != unknown
        ):
            return candidate
    raise ValueError(f"Moonshot tokenizer is missing required special token {token}")


def _token_id_set(value: Any) -> set[int]:
    if isinstance(value, int) and not isinstance(value, bool):
        return {value}
    if isinstance(value, Collection) and not isinstance(value, (str, bytes)):
        result = {
            token_id
            for token_id in value
            if isinstance(token_id, int) and not isinstance(token_id, bool)
        }
        return result
    return set()


def _thinking_effort(value: str) -> str:
    normalized = value.strip().lower()
    mapped = {
        "minimal": "low",
        "low": "low",
        "medium": "high",
        "high": "high",
        "max": "max",
    }.get(normalized)
    if mapped is None:
        raise ValueError(
            "reasoning_effort must be one of minimal, low, medium, high, or max"
        )
    return mapped


__all__ = ["KimiChatTokenizer", "KimiIncrementalDecoder", "KLinearEngine"]
