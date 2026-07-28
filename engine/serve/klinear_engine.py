"""Kimi-Linear engine and Moonshot tokenizer adapters for ``engine.serve``."""

from __future__ import annotations

import asyncio
import codecs
import json
from collections.abc import AsyncIterator, Collection, Mapping, Sequence
from typing import Any

import torch

from k3.toolcalls import KimiToolParser, ParsedText, ParsedToolCall, parse_all

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

_EOS = "[EOS]"
_END_OF_TURN = "[EOT]"
_IM_END = "<|im_end|>"
_IM_USER = "<|im_user|>"
_IM_ASSISTANT = "<|im_assistant|>"
_IM_SYSTEM = "<|im_system|>"
_IM_MIDDLE = "<|im_middle|>"
_TOOL_SECTION_BEGIN = "<|tool_calls_section_begin|>"
_TOOL_SECTION_END = "<|tool_calls_section_end|>"
_TOOL_CALL_BEGIN = "<|tool_call_begin|>"
_TOOL_ARGUMENT_BEGIN = "<|tool_call_argument_begin|>"
_TOOL_CALL_END = "<|tool_call_end|>"

_REQUIRED_CONTROL_TOKENS = (
    _EOS,
    _END_OF_TURN,
    _IM_END,
    _IM_USER,
    _IM_ASSISTANT,
    _IM_SYSTEM,
    _IM_MIDDLE,
    _TOOL_SECTION_BEGIN,
    _TOOL_SECTION_END,
    _TOOL_CALL_BEGIN,
    _TOOL_ARGUMENT_BEGIN,
    _TOOL_CALL_END,
)


class _KimiLinearToolParser(KimiToolParser):
    """Reuse the K2 envelope parser while retaining Kimi-Linear call IDs."""

    def _split_id(self, raw: str) -> tuple[str, str]:
        normalized = raw.strip()
        match = self._ID_RE.match(normalized)
        if match is None:
            return super()._split_id(raw)
        return match.group("name") or "unknown", normalized


class KimiChatTokenizer:
    """Adapt Moonshot's remote-code tiktoken tokenizer to the serve contract."""

    def __init__(self, tokenizer: Any) -> None:
        self.tokenizer = tokenizer
        self.control_token_ids = {
            token: _special_token_id(tokenizer, token)
            for token in _REQUIRED_CONTROL_TOKENS
        }
        self.im_end_token_id = self.control_token_ids[_IM_END]
        self.end_of_turn_token_id = self.control_token_ids[_END_OF_TURN]
        self.eos_token_id = self.control_token_ids[_EOS]
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
        eos_ids.update(
            (
                self.eos_token_id,
                self.im_end_token_id,
                self.end_of_turn_token_id,
            )
        )
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
        }
        if prompt.tools:
            kwargs["tools"] = [dict(tool) for tool in prompt.tools]
        if prompt.tool_choice is not None:
            kwargs["tool_choice"] = prompt.tool_choice

        token_ids = self.tokenizer.apply_chat_template(messages, **kwargs)
        # apply_chat_template returns different shapes across tokenizer classes:
        # a plain list, a tensor, or a BatchEncoding-like object carrying
        # input_ids. Normalise all three rather than assuming one, and if it is
        # none of them, say WHAT arrived, because the previous message did not
        # and cost a GPU round trip to diagnose.
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        elif hasattr(token_ids, "input_ids"):
            token_ids = token_ids.input_ids
            if isinstance(token_ids, torch.Tensor):
                token_ids = token_ids.tolist()
        elif isinstance(token_ids, Mapping) and "input_ids" in token_ids:
            token_ids = token_ids["input_ids"]
            if isinstance(token_ids, torch.Tensor):
                token_ids = token_ids.tolist()
        if not isinstance(token_ids, Sequence) or isinstance(token_ids, (str, bytes)):
            raise TypeError(
                "Moonshot chat template did not return token IDs; got "
                f"{type(token_ids).__name__}"
            )
        # A single-element batch is a normal shape for some tokenizers; unwrap it
        # rather than refusing, and only refuse a genuine multi-row batch.
        if (
            len(token_ids) == 1
            and isinstance(token_ids[0], Sequence)
            and not isinstance(token_ids[0], (str, bytes))
        ):
            token_ids = token_ids[0]
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

    def parse_assistant_output(self, text: str) -> dict[str, Any]:
        """Convert a Kimi-Linear tool envelope into OpenAI message fields."""

        if not isinstance(text, str):
            raise TypeError("Moonshot assistant output must be text")
        content: list[str] = []
        tool_calls: list[dict[str, Any]] = []
        for event in parse_all(_KimiLinearToolParser(), text):
            if isinstance(event, ParsedText):
                content.append(event.text)
            elif isinstance(event, ParsedToolCall):
                tool_calls.append(
                    {
                        "id": event.id,
                        "type": "function",
                        "function": {
                            "name": event.name,
                            "arguments": event.arguments,
                        },
                    }
                )
        message: dict[str, Any] = {"content": "".join(content) or None}
        if tool_calls:
            message["tool_calls"] = tool_calls
        return message

    def token_bytes(self, token_id: int) -> bytes:
        raw = self._decode_single_token_bytes(token_id)
        if not isinstance(raw, (bytes, bytearray)):
            raise TypeError("Moonshot tokenizer returned non-byte token payload")
        return bytes(raw)


class KimiIncrementalDecoder:
    """Decode Kimi-Linear output as visible content with no reasoning channel."""

    def __init__(self, tokenizer: KimiChatTokenizer) -> None:
        self.tokenizer = tokenizer
        self._finished = False
        self._decoder = codecs.getincrementaldecoder("utf-8")("replace")

    def push(self, token_id: int) -> list[DecodedFragment]:
        if token_id in self.tokenizer.eos_token_ids:
            fragments = self._flush_decoder()
            self._finished = True
            return fragments
        if self._finished:
            return []

        text = self._decoder.decode(self.tokenizer.token_bytes(token_id), final=False)
        return [DecodedFragment("content", text)] if text else []

    def finish(self) -> list[DecodedFragment]:
        self._finished = True
        return self._flush_decoder()

    def _flush_decoder(self) -> list[DecodedFragment]:
        text = self._decoder.decode(b"", final=True)
        self._decoder = codecs.getincrementaldecoder("utf-8")("replace")
        return [DecodedFragment("content", text)] if text else []


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


__all__ = ["KimiChatTokenizer", "KimiIncrementalDecoder", "KLinearEngine"]
