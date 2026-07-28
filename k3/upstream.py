"""The engine adapter — the only thing in k3 that talks to K3.

K3 is served behind an OpenAI-compatible ``/v1/chat/completions`` endpoint
(vLLM, SGLang, or the reference server), emits ``reasoning_content``, and wants
the complete assistant message back verbatim on the next turn. Everything above
this module is dialect translation; everything below it is somebody else's
inference server.
"""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Optional

import httpx

from .ir import CanonicalRequest, ImagePart, Message, TextPart, ToolResultPart
from .presets import Preset
from .reasoning import ReasoningLedger, build_upstream_assistant, restore_assistant
from .template import (
    build_system_prompt,
    collapse_messages,
    native_tools_payload,
    render_kimi_k3_tool_call,
)


class UpstreamError(RuntimeError):
    def __init__(self, status: int, body: str, kind: str = "upstream_error") -> None:
        super().__init__(f"engine returned {status}: {body[:500]}")
        self.status = status
        self.body = body
        self.kind = kind


@dataclass(slots=True)
class UpstreamConfig:
    base_url: str = "http://127.0.0.1:8000/v1"
    api_key: Optional[str] = None
    model: str = "k3"
    #: Optional cheaper model for clients that route background work elsewhere.
    small_model: Optional[str] = None
    #: Field the engine expects reasoning in. ``none`` drops it, ``inline``
    #: wraps it in <think> tags inside content.
    reasoning_field: str = "reasoning_content"
    timeout: float = 600.0
    connect_timeout: float = 10.0
    #: Some engines 400 on unknown sampling params; turn this off for those.
    send_reasoning_effort: bool = True
    extra_body: dict[str, Any] = field(default_factory=dict)
    headers: dict[str, str] = field(default_factory=dict)

    @property
    def chat_url(self) -> str:
        return self.base_url.rstrip("/") + "/chat/completions"

    @property
    def models_url(self) -> str:
        return self.base_url.rstrip("/") + "/models"


def resolve_model(requested: str, preset: Preset, cfg: UpstreamConfig) -> str:
    """Whatever model string the client asks for resolves to K3."""
    if requested and preset.models.is_passthrough(requested):
        return requested
    if requested and cfg.small_model and preset.models.is_small(requested):
        return cfg.small_model
    return cfg.model


# --------------------------------------------------------------------------
# payload assembly
# --------------------------------------------------------------------------


def _content_for_user(msg: Message) -> Any:
    """String when it is only text, OpenAI part list when it is not."""
    if all(isinstance(p, TextPart) for p in msg.parts):
        return "".join(p.text for p in msg.parts if isinstance(p, TextPart))
    parts: list[dict[str, Any]] = []
    for p in msg.parts:
        if isinstance(p, TextPart):
            if p.text:
                parts.append({"type": "text", "text": p.text})
        elif isinstance(p, ImagePart):
            url = p.data if p.is_url else f"data:{p.media_type};base64,{p.data}"
            parts.append({"type": "image_url", "image_url": {"url": url}})
    return parts or ""


def build_payload(
    req: CanonicalRequest,
    preset: Preset,
    ledger: ReasoningLedger,
    cfg: UpstreamConfig,
    upstream_model: str,
) -> dict[str, Any]:
    """Lower a CanonicalRequest into an OpenAI chat-completions body for K3."""
    messages: list[dict[str, Any]] = []

    system = build_system_prompt(req, preset.template)
    if system:
        messages.append({"role": "system", "content": system})

    for msg in req.messages:
        if msg.role == "system":
            messages.append({"role": "system", "content": msg.text()})
        elif msg.role == "user":
            messages.append({"role": "user", "content": _content_for_user(msg)})
        elif msg.role == "assistant":
            restored = restore_assistant(msg, ledger)
            built = build_upstream_assistant(msg, restored, cfg.reasoning_field)
            if preset.template.drop_empty_assistant and not (
                built.get("content") or built.get("tool_calls")
            ):
                continue
            messages.append(built)
        elif msg.role == "tool":
            for part in msg.parts:
                if isinstance(part, ToolResultPart):
                    entry: dict[str, Any] = {
                        "role": "tool",
                        "tool_call_id": part.tool_call_id,
                        "content": part.content,
                    }
                    if part.name:
                        entry["name"] = part.name
                    messages.append(entry)

    if preset.template.collapse_consecutive:
        messages = collapse_messages(messages)

    payload: dict[str, Any] = {
        "model": upstream_model,
        "messages": messages,
        "stream": bool(req.stream),
    }

    if preset.template.tool_mode == "native" and req.tools:
        payload["tools"] = native_tools_payload(req.tools)
        if req.tool_choice is not None:
            payload["tool_choice"] = req.tool_choice
        if req.parallel_tool_calls is not None:
            payload["parallel_tool_calls"] = req.parallel_tool_calls

    if req.max_tokens is not None:
        payload["max_tokens"] = req.max_tokens
    if req.temperature is not None:
        payload["temperature"] = req.temperature
    if req.top_p is not None:
        payload["top_p"] = req.top_p
    if req.stop:
        payload["stop"] = req.stop
    if req.reasoning_effort and cfg.send_reasoning_effort:
        payload["reasoning_effort"] = req.reasoning_effort
    if req.stream and preset.defaults.stream_include_usage:
        payload["stream_options"] = {"include_usage": True}

    for key, value in cfg.extra_body.items():
        payload.setdefault(key, value)
    for key, value in req.passthrough.items():
        payload.setdefault(key, value)

    return payload


# --------------------------------------------------------------------------
# transport
# --------------------------------------------------------------------------


class Upstream:
    """Thin async client over the engine's chat-completions endpoint."""

    def __init__(self, cfg: UpstreamConfig) -> None:
        self.cfg = cfg
        self._client: Optional[httpx.AsyncClient] = None

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            headers = {"content-type": "application/json", **self.cfg.headers}
            if self.cfg.api_key:
                headers["authorization"] = f"Bearer {self.cfg.api_key}"
            self._client = httpx.AsyncClient(
                headers=headers,
                timeout=httpx.Timeout(self.cfg.timeout, connect=self.cfg.connect_timeout),
            )
        return self._client

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def chat(self, payload: dict[str, Any]) -> dict[str, Any]:
        body = dict(payload)
        body["stream"] = False
        try:
            resp = await self.client.post(self.cfg.chat_url, json=body)
        except httpx.RequestError as exc:
            raise UpstreamError(502, f"cannot reach engine at {self.cfg.chat_url}: {exc}", "connection") from exc
        if resp.status_code >= 400:
            raise UpstreamError(resp.status_code, resp.text)
        return resp.json()

    async def chat_stream(self, payload: dict[str, Any]) -> AsyncIterator[dict[str, Any]]:
        body = dict(payload)
        body["stream"] = True
        try:
            async with self.client.stream("POST", self.cfg.chat_url, json=body) as resp:
                if resp.status_code >= 400:
                    text = (await resp.aread()).decode("utf-8", "replace")
                    raise UpstreamError(resp.status_code, text)
                async for chunk in _iter_sse_json(resp):
                    yield chunk
        except httpx.RequestError as exc:
            raise UpstreamError(502, f"cannot reach engine at {self.cfg.chat_url}: {exc}", "connection") from exc

    async def models(self) -> Optional[dict[str, Any]]:
        try:
            resp = await self.client.get(self.cfg.models_url)
        except httpx.RequestError:
            return None
        if resp.status_code >= 400:
            return None
        try:
            return resp.json()
        except ValueError:
            return None

    async def health(self) -> tuple[bool, str]:
        try:
            resp = await self.client.get(self.cfg.models_url)
        except httpx.RequestError as exc:
            return False, str(exc)
        return resp.status_code < 400, f"HTTP {resp.status_code}"


async def _iter_sse_json(resp: httpx.Response) -> AsyncIterator[dict[str, Any]]:
    async for line in resp.aiter_lines():
        line = line.strip()
        if not line or line.startswith(":"):
            continue
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            return
        try:
            yield json.loads(data)
        except json.JSONDecodeError:
            # A malformed chunk is the engine's problem; dropping it beats
            # tearing down a stream the client is already rendering.
            continue


# --------------------------------------------------------------------------
# mock engine
# --------------------------------------------------------------------------


class MockUpstream:
    """A scripted stand-in for K3 so the whole path runs without a GPU.

    It deliberately emits tool calls in the selected K2 or K3 text format
    rather than native ``tool_calls`` deltas, so mock traffic exercises the text
    parser too — the part most likely to be wrong.
    """

    def __init__(
        self,
        cfg: Optional[UpstreamConfig] = None,
        delay: float = 0.0,
        tool_parser: str = "kimi_k3",
    ) -> None:
        self.cfg = cfg or UpstreamConfig()
        self.delay = delay
        self.tool_parser = tool_parser
        self.calls: list[dict[str, Any]] = []

    async def aclose(self) -> None:  # pragma: no cover - nothing to close
        return None

    async def chat(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.calls.append(payload)
        reasoning, content = _mock_generate(payload, self.tool_parser)
        return {
            "id": "chatcmpl-mock",
            "object": "chat.completion",
            "created": 0,
            "model": payload.get("model", "k3"),
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": content,
                        "reasoning_content": reasoning,
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": _rough_prompt_tokens(payload),
                "completion_tokens": max(1, (len(reasoning) + len(content)) // 4),
                "total_tokens": 0,
            },
        }

    async def chat_stream(self, payload: dict[str, Any]) -> AsyncIterator[dict[str, Any]]:
        self.calls.append(payload)
        reasoning, content = _mock_generate(payload, self.tool_parser)
        model = payload.get("model", "k3")

        def chunk(delta: dict[str, Any], finish: Optional[str] = None) -> dict[str, Any]:
            return {
                "id": "chatcmpl-mock",
                "object": "chat.completion.chunk",
                "created": 0,
                "model": model,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
            }

        yield chunk({"role": "assistant", "content": ""})
        for piece in _slice(reasoning, 24):
            if self.delay:
                await asyncio.sleep(self.delay)
            yield chunk({"reasoning_content": piece})
        for piece in _slice(content, 24):
            if self.delay:
                await asyncio.sleep(self.delay)
            yield chunk({"content": piece})
        yield chunk({}, finish="stop")
        yield {
            "id": "chatcmpl-mock",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": model,
            "choices": [],
            "usage": {
                "prompt_tokens": _rough_prompt_tokens(payload),
                "completion_tokens": max(1, (len(reasoning) + len(content)) // 4),
                "total_tokens": 0,
            },
        }

    async def models(self) -> Optional[dict[str, Any]]:
        return {"object": "list", "data": [{"id": self.cfg.model, "object": "model"}]}

    async def health(self) -> tuple[bool, str]:
        return True, "mock"


def _slice(text: str, size: int) -> list[str]:
    return [text[i : i + size] for i in range(0, len(text), size)] if text else []


def _rough_prompt_tokens(payload: dict[str, Any]) -> int:
    total = 0
    for msg in payload.get("messages") or []:
        content = msg.get("content")
        if isinstance(content, str):
            total += len(content)
        elif isinstance(content, list):
            total += sum(len(str(p)) for p in content)
    return max(1, total // 4)


_MOCK_K2_TOOL_CALL = (
    "<|tool_calls_section_begin|>"
    "<|tool_call_begin|>functions.{name}:0"
    '<|tool_call_argument_begin|>{args}'
    "<|tool_call_end|>"
    "<|tool_calls_section_end|>"
)


def _mock_generate(
    payload: dict[str, Any], tool_parser: str = "kimi_k3"
) -> tuple[str, str]:
    messages = payload.get("messages") or []
    tools = payload.get("tools") or []
    last_user = ""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            content = msg.get("content")
            last_user = content if isinstance(content, str) else json.dumps(content)[:200]
            break
    already_called = any(m.get("role") == "tool" for m in messages)

    if tools and not already_called:
        tool = tools[0].get("function", tools[0])
        name = tool.get("name", "unknown")
        args = _mock_args(tool.get("parameters") or {}, last_user)
        reasoning = (
            f"The user asked: {last_user.strip()[:160]!r}. "
            f"I have {len(tools)} tool(s) available; {name} is the right one here, "
            "so I will call it before answering."
        )
        if tool_parser == "kimi_k3":
            return reasoning, _mock_k3_tool_call(name, args)
        return reasoning, _MOCK_K2_TOOL_CALL.format(
            name=name,
            args=json.dumps(args, ensure_ascii=False),
        )

    reasoning = (
        f"Reviewing the conversation. The user's request was {last_user.strip()[:160]!r}. "
        + ("The tool result is in hand, so I can answer directly now." if already_called else "No tools are needed.")
    )
    content = (
        "[k3 mock engine] "
        + ("Done — I used the tool result to answer. " if already_called else "")
        + "Start the real engine and point --upstream at it to get actual output."
    )
    return reasoning, content


def _mock_args(schema: dict[str, Any], hint: str) -> dict[str, Any]:
    props = (schema or {}).get("properties") or {}
    required = (schema or {}).get("required") or list(props)[:1]
    out: dict[str, Any] = {}
    for key in required:
        spec = props.get(key) or {}
        kind = spec.get("type", "string")
        if kind == "integer":
            out[key] = 1
        elif kind == "number":
            out[key] = 1.0
        elif kind == "boolean":
            out[key] = True
        elif kind == "array":
            out[key] = []
        elif kind == "object":
            out[key] = {}
        else:
            out[key] = re.sub(r"\s+", " ", hint).strip()[:80] or "k3"
    return out


def _mock_k3_tool_call(name: str, arguments: dict[str, Any]) -> str:
    return "".join(
        [
            "<|open|>tools<|sep|>",
            render_kimi_k3_tool_call(name, arguments),
            "<|close|>tools<|sep|>",
        ]
    )


__all__ = [
    "UpstreamConfig",
    "Upstream",
    "MockUpstream",
    "UpstreamError",
    "build_payload",
    "resolve_model",
]
