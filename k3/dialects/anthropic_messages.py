"""Anthropic Messages API — the Claude Code dialect.

Two things in here carry the whole preset:

*Reasoning.* K3's ``reasoning_content`` becomes a ``thinking`` content block,
and the block's ``signature`` carries the exact bytes back. Claude Code echoes
``thinking`` blocks verbatim — text *and* signature — which makes this the
highest-fidelity client of the lot.

*Tool-call ids.* Anthropic ``tool_use`` ids and OpenAI ``tool_calls`` ids live
in different namespaces, and the id has to survive the round trip or the
``tool_result`` will not match the assistant turn we restore from the ledger. We
use a deterministic, reversible prefix rather than minting a fresh id and
keeping a map, because a map does not survive a restart and a mismatch here
silently breaks every agent loop.
"""

from __future__ import annotations

import json
from typing import Any, AsyncIterator, Optional

from ..ir import (
    CanonicalRequest,
    CanonicalResponse,
    ImagePart,
    Message,
    ReasoningDelta,
    ReasoningEnd,
    ReasoningPart,
    StreamEnd,
    StreamEvent,
    StreamStart,
    TextDelta,
    TextPart,
    ToolCallArgsDelta,
    ToolCallEnd,
    ToolCallPart,
    ToolCallStart,
    ToolDef,
    ToolResultPart,
    Usage,
)
from ..reasoning import ReasoningPolicy, strip_inline_think
from .base import DialectContext, estimate_tokens, sse

TOOL_ID_PREFIX = "toolu_k3_"

_STOP_REASON = {
    "stop": "end_turn",
    "length": "max_tokens",
    "tool_calls": "tool_use",
    "content_filter": "end_turn",
    "error": "end_turn",
}


def to_anthropic_tool_id(upstream_id: str) -> str:
    """Reversible: the engine's id, namespaced so it looks like an Anthropic one.

    Always prefixes, even when the input already starts with the prefix. Skipping
    it there would make the pair non-injective — ``toolu_k3_abc`` would come back
    as ``abc`` — and a tool_result that no longer matches its tool_use silently
    breaks the agent loop.
    """
    return TOOL_ID_PREFIX + upstream_id


def from_anthropic_tool_id(client_id: str) -> str:
    """Inverse of :func:`to_anthropic_tool_id`; ids we did not mint pass through."""
    if client_id.startswith(TOOL_ID_PREFIX):
        return client_id[len(TOOL_ID_PREFIX) :]
    return client_id


# --------------------------------------------------------------------------
# ingress
# --------------------------------------------------------------------------


def _blocks(content: Any) -> list[dict[str, Any]]:
    if content is None:
        return []
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    if isinstance(content, list):
        return [b for b in content if isinstance(b, dict)]
    return []


def _stringify_tool_result(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks: list[str] = []
        for block in content:
            if not isinstance(block, dict):
                chunks.append(str(block))
            elif block.get("type") == "text":
                chunks.append(str(block.get("text") or ""))
            else:
                # Images and other rich results have no representation in an
                # OpenAI tool message; describe them rather than drop them.
                chunks.append(json.dumps(block, ensure_ascii=False))
        return "\n".join(c for c in chunks if c)
    return json.dumps(content, ensure_ascii=False)


def _image_part(source: dict[str, Any]) -> Optional[ImagePart]:
    kind = source.get("type")
    if kind == "url":
        url = source.get("url")
        return ImagePart(media_type="image/*", data=str(url), is_url=True) if url else None
    if kind == "base64":
        return ImagePart(
            media_type=str(source.get("media_type") or "image/png"),
            data=str(source.get("data") or ""),
        )
    return None


def _system_text(system: Any) -> list[str]:
    if not system:
        return []
    if isinstance(system, str):
        return [system]
    out: list[str] = []
    for block in _blocks(system):
        if block.get("type") == "text" and block.get("text"):
            out.append(str(block["text"]))
    return out


def _tool_defs(tools: Any) -> list[ToolDef]:
    out: list[ToolDef] = []
    if not isinstance(tools, list):
        return out
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        name = tool.get("name")
        if not name:
            continue
        schema = tool.get("input_schema") or tool.get("parameters") or {
            "type": "object",
            "properties": {},
        }
        out.append(
            ToolDef(
                name=str(name),
                description=str(tool.get("description") or ""),
                parameters=schema if isinstance(schema, dict) else {"type": "object", "properties": {}},
            )
        )
    return out


def _tool_choice(raw: Any) -> Any:
    if not isinstance(raw, dict):
        return None
    kind = raw.get("type")
    if kind == "auto":
        return "auto"
    if kind == "any":
        return "required"
    if kind == "none":
        return "none"
    if kind == "tool" and raw.get("name"):
        return {"type": "function", "function": {"name": raw["name"]}}
    return None


def _effort_from_budget(budget: Optional[int]) -> Optional[str]:
    """Claude Code sends a thinking budget; K3 wants an effort level."""
    if not budget:
        return None
    if budget <= 4096:
        return "low"
    if budget <= 16384:
        return "medium"
    return "high"


def ingress(body: dict[str, Any], ctx: DialectContext) -> CanonicalRequest:
    body = body or {}
    defaults = ctx.preset.defaults
    inline_reasoning = ctx.preset.reasoning is ReasoningPolicy.INLINE_TAGS

    req = CanonicalRequest(model=str(body.get("model") or ctx.client_model), raw=body)
    req.system = _system_text(body.get("system"))

    for raw_msg in body.get("messages") or []:
        if not isinstance(raw_msg, dict):
            continue
        role = raw_msg.get("role")
        blocks = _blocks(raw_msg.get("content"))

        if role == "assistant":
            msg = Message(role="assistant")
            for block in blocks:
                kind = block.get("type")
                if kind == "text":
                    text = str(block.get("text") or "")
                    if inline_reasoning:
                        reasoning, text = strip_inline_think(text)
                        if reasoning:
                            msg.parts.append(ReasoningPart(text=reasoning))
                    if text:
                        msg.parts.append(TextPart(text))
                elif kind == "thinking":
                    msg.parts.append(
                        ReasoningPart(
                            text=str(block.get("thinking") or ""),
                            signature=block.get("signature"),
                        )
                    )
                elif kind == "redacted_thinking":
                    msg.parts.append(
                        ReasoningPart(text="", signature=block.get("data"), redacted=True)
                    )
                elif kind == "tool_use":
                    msg.parts.append(
                        ToolCallPart(
                            id=from_anthropic_tool_id(str(block.get("id") or "")),
                            name=str(block.get("name") or ""),
                            arguments=json.dumps(block.get("input") or {}, ensure_ascii=False),
                        )
                    )
            if msg.parts:
                req.messages.append(msg)
            continue

        # A user turn carrying tool_result blocks is really a tool turn plus,
        # possibly, some user text. Tool results have to come first so they land
        # next to the assistant turn that requested them.
        results = [b for b in blocks if b.get("type") == "tool_result"]
        others = [b for b in blocks if b.get("type") != "tool_result"]

        if results:
            tool_msg = Message(role="tool")
            for block in results:
                tool_msg.parts.append(
                    ToolResultPart(
                        tool_call_id=from_anthropic_tool_id(str(block.get("tool_use_id") or "")),
                        content=_stringify_tool_result(block.get("content")),
                        is_error=bool(block.get("is_error")),
                    )
                )
            req.messages.append(tool_msg)

        if others:
            user_msg = Message(role="user" if role != "system" else "system")
            for block in others:
                kind = block.get("type")
                if kind == "text" and block.get("text"):
                    user_msg.parts.append(TextPart(str(block["text"])))
                elif kind == "image":
                    part = _image_part(block.get("source") or {})
                    if part:
                        user_msg.parts.append(part)
                elif kind == "document":
                    source = block.get("source") or {}
                    if source.get("type") == "text":
                        user_msg.parts.append(TextPart(str(source.get("data") or "")))
            if user_msg.parts:
                req.messages.append(user_msg)

    req.tools = _tool_defs(body.get("tools"))
    req.tool_choice = _tool_choice(body.get("tool_choice"))

    max_tokens = body.get("max_tokens")
    req.max_tokens = int(max_tokens) if isinstance(max_tokens, (int, float)) else defaults.max_tokens
    req.temperature = body.get("temperature") if body.get("temperature") is not None else defaults.temperature
    req.top_p = body.get("top_p") if body.get("top_p") is not None else defaults.top_p

    stop = body.get("stop_sequences")
    if isinstance(stop, str):
        req.stop = [stop]
    elif isinstance(stop, list):
        req.stop = [str(s) for s in stop]

    req.stream = bool(body.get("stream"))

    thinking = body.get("thinking")
    if isinstance(thinking, dict):
        # Only "disabled" is a refusal. Claude Code also sends "adaptive", and
        # treating anything-but-"enabled" as off would hide reasoning from the
        # client that asked for it.
        req.thinking_enabled = thinking.get("type") != "disabled"
        budget = thinking.get("budget_tokens")
        req.thinking_budget = int(budget) if isinstance(budget, (int, float)) else None
        req.reasoning_effort = _effort_from_budget(req.thinking_budget) or defaults.reasoning_effort
    else:
        req.reasoning_effort = defaults.reasoning_effort

    if defaults.parallel_tool_calls is not None:
        req.parallel_tool_calls = defaults.parallel_tool_calls

    metadata = body.get("metadata")
    if isinstance(metadata, dict):
        req.metadata = metadata

    return req


# --------------------------------------------------------------------------
# egress
# --------------------------------------------------------------------------


def _show_thinking(ctx: DialectContext) -> bool:
    """Thinking blocks require both a preset that emits them and a client that
    asked for them; the real API returns none when thinking is disabled, and
    rejects an assistant turn carrying them on the way back in."""
    return (
        ctx.preset.reasoning is ReasoningPolicy.THINKING_BLOCKS
        and ctx.thinking_enabled is not False
    )


def _reasoning_blocks(response: CanonicalResponse, ctx: DialectContext) -> list[dict[str, Any]]:
    if not _show_thinking(ctx):
        return []
    blocks: list[dict[str, Any]] = []
    for part in response.reasoning_parts():
        if part.redacted:
            blocks.append({"type": "redacted_thinking", "data": part.signature or ""})
        else:
            blocks.append(
                {
                    "type": "thinking",
                    "thinking": part.text,
                    "signature": part.signature or "",
                }
            )
    return blocks


def _text_block(response: CanonicalResponse, ctx: DialectContext) -> list[dict[str, Any]]:
    text = response.text()
    if ctx.preset.reasoning is ReasoningPolicy.INLINE_TAGS:
        reasoning = "".join(p.text for p in response.reasoning_parts())
        if reasoning:
            text = f"<think>{reasoning}</think>{text}"
    return [{"type": "text", "text": text}] if text else []


def _tool_blocks(response: CanonicalResponse) -> list[dict[str, Any]]:
    return [
        {
            "type": "tool_use",
            "id": to_anthropic_tool_id(tc.id),
            "name": tc.name,
            "input": tc.arguments_obj(),
        }
        for tc in response.tool_calls()
    ]


def _usage(usage: Usage, ctx: DialectContext) -> dict[str, Any]:
    return {
        "input_tokens": usage.input_tokens or ctx.input_tokens_hint,
        "output_tokens": usage.output_tokens,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": usage.cached_input_tokens,
    }


def egress(response: CanonicalResponse, ctx: DialectContext) -> dict[str, Any]:
    content = _reasoning_blocks(response, ctx) + _text_block(response, ctx) + _tool_blocks(response)
    if not content:
        content = [{"type": "text", "text": ""}]
    return {
        "id": ctx.request_id or ctx.new_id("msg"),
        "type": "message",
        "role": "assistant",
        "model": ctx.client_model,
        "content": content,
        "stop_reason": _STOP_REASON.get(response.stop_reason, "end_turn"),
        "stop_sequence": None,
        "usage": _usage(response.usage, ctx),
    }


# --------------------------------------------------------------------------
# streaming
# --------------------------------------------------------------------------


class _BlockWriter:
    """Tracks the one content block Anthropic allows to be open at a time."""

    def __init__(self) -> None:
        self.index = -1
        self.kind: Optional[str] = None

    def open(self, kind: str, block: dict[str, Any]) -> list[bytes]:
        out = self.close()
        self.index += 1
        self.kind = kind
        out.append(
            sse(
                {"type": "content_block_start", "index": self.index, "content_block": block},
                "content_block_start",
            )
        )
        return out

    def delta(self, delta: dict[str, Any]) -> bytes:
        return sse(
            {"type": "content_block_delta", "index": self.index, "delta": delta},
            "content_block_delta",
        )

    def close(self) -> list[bytes]:
        if self.kind is None:
            return []
        self.kind = None
        return [sse({"type": "content_block_stop", "index": self.index}, "content_block_stop")]


async def egress_stream(
    events: AsyncIterator[StreamEvent], ctx: DialectContext
) -> AsyncIterator[bytes]:
    policy = ctx.preset.reasoning
    show_thinking = _show_thinking(ctx)
    inline = policy is ReasoningPolicy.INLINE_TAGS

    writer = _BlockWriter()
    message_id = ctx.request_id or ctx.new_id("msg")
    started = False
    inline_open = False
    final: Optional[StreamEnd] = None

    def message_start() -> bytes:
        return sse(
            {
                "type": "message_start",
                "message": {
                    "id": message_id,
                    "type": "message",
                    "role": "assistant",
                    "model": ctx.client_model,
                    "content": [],
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {
                        "input_tokens": ctx.input_tokens_hint,
                        "output_tokens": 0,
                        "cache_creation_input_tokens": 0,
                        "cache_read_input_tokens": 0,
                    },
                },
            },
            "message_start",
        )

    async for event in events:
        if not started:
            started = True
            yield message_start()
            if ctx.preset.defaults.stream_keepalive:
                yield sse({"type": "ping"}, "ping")

        if isinstance(event, StreamStart):
            continue

        if isinstance(event, ReasoningDelta):
            if show_thinking:
                if writer.kind != "thinking":
                    for chunk in writer.open("thinking", {"type": "thinking", "thinking": ""}):
                        yield chunk
                yield writer.delta({"type": "thinking_delta", "thinking": event.text})
            elif inline:
                if writer.kind != "text":
                    for chunk in writer.open("text", {"type": "text", "text": ""}):
                        yield chunk
                if not inline_open:
                    inline_open = True
                    yield writer.delta({"type": "text_delta", "text": "<think>"})
                yield writer.delta({"type": "text_delta", "text": event.text})

        elif isinstance(event, ReasoningEnd):
            if show_thinking and writer.kind == "thinking":
                # The signature is the vehicle that gets these exact bytes back
                # to K3 next turn, so it has to land before the block closes.
                yield writer.delta(
                    {"type": "signature_delta", "signature": event.signature or ""}
                )
                for chunk in writer.close():
                    yield chunk
            elif inline and inline_open:
                inline_open = False
                yield writer.delta({"type": "text_delta", "text": "</think>"})

        elif isinstance(event, TextDelta):
            if writer.kind != "text":
                for chunk in writer.open("text", {"type": "text", "text": ""}):
                    yield chunk
            yield writer.delta({"type": "text_delta", "text": event.text})

        elif isinstance(event, ToolCallStart):
            for chunk in writer.open(
                "tool_use",
                {
                    "type": "tool_use",
                    "id": to_anthropic_tool_id(event.id),
                    "name": event.name,
                    "input": {},
                },
            ):
                yield chunk

        elif isinstance(event, ToolCallArgsDelta):
            if writer.kind == "tool_use":
                yield writer.delta({"type": "input_json_delta", "partial_json": event.text})

        elif isinstance(event, ToolCallEnd):
            if writer.kind == "tool_use":
                for chunk in writer.close():
                    yield chunk

        elif isinstance(event, StreamEnd):
            final = event

    if not started:
        yield message_start()

    for chunk in writer.close():
        yield chunk

    stop_reason = _STOP_REASON.get(final.stop_reason, "end_turn") if final else "end_turn"
    usage = final.usage if final else Usage()
    yield sse(
        {
            "type": "message_delta",
            "delta": {"stop_reason": stop_reason, "stop_sequence": None},
            "usage": {
                "input_tokens": usage.input_tokens or ctx.input_tokens_hint,
                "output_tokens": usage.output_tokens,
            },
        },
        "message_delta",
    )
    yield sse({"type": "message_stop"}, "message_stop")


# --------------------------------------------------------------------------
# /v1/messages/count_tokens
# --------------------------------------------------------------------------


def count_tokens(body: dict[str, Any], ctx: DialectContext) -> dict[str, Any]:
    """Claude Code calls this before every request; it must never 404.

    The engine has the real tokenizer, but asking it costs a round trip on the
    hot path for a number the client only uses to decide when to compact. An
    estimate is the right trade here, and it is labelled as one.
    """
    req = ingress(body, ctx)
    total = sum(estimate_tokens(s) for s in req.system)
    for msg in req.messages:
        total += estimate_tokens(msg.text())
        total += estimate_tokens(msg.reasoning())
        for tc in msg.tool_calls():
            total += estimate_tokens(tc.name) + estimate_tokens(tc.arguments)
        for tr in msg.tool_results():
            total += estimate_tokens(tr.content)
        total += 4  # per-message framing
    for tool in req.tools:
        total += estimate_tokens(tool.name + tool.description)
        total += estimate_tokens(json.dumps(tool.parameters, ensure_ascii=False))
    return {"input_tokens": max(1, total)}


__all__ = [
    "ingress",
    "egress",
    "egress_stream",
    "count_tokens",
    "to_anthropic_tool_id",
    "from_anthropic_tool_id",
    "TOOL_ID_PREFIX",
]
