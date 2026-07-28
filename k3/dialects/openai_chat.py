"""OpenAI Chat Completions — ``POST /v1/chat/completions``.

The lowest common denominator dialect: most clients speak it, and the ones that
don't have an opinion about reasoning at all still land here. What differs
between the presets sharing this dialect is *only* how reasoning surfaces —
``reasoning_content`` for the clients that render it, ``<think>`` tags for the
ones whose own filter already knows to hide them, nothing at all for the rest —
so the reasoning policy is the single branch that runs through every function
below.

The one thing that is not negotiable is tool-call ``arguments``: whatever string
the model produced goes back to the client byte for byte, and comes back in the
same way. Re-serialising it shifts whitespace and key order, and K3 notices.
"""

from __future__ import annotations

import json
import uuid
from typing import Any, AsyncIterator, Optional

from ..ir import (
    CanonicalRequest,
    CanonicalResponse,
    ImagePart,
    Message,
    Part,
    ReasoningDelta,
    ReasoningEnd,
    ReasoningPart,
    StreamEnd,
    StreamEvent,
    StreamStart,
    TextDelta,
    TextPart,
    ToolCallArgsDelta,
    ToolCallStart,
    ToolCallPart,
    ToolDef,
    ToolResultPart,
    Usage,
)
from ..reasoning import ReasoningPolicy, strip_inline_think
from ..toolcalls import new_call_id
from .base import DialectContext, sse

#: Top-level body keys ingress either maps into the IR or deliberately drops.
#: Everything else becomes ``passthrough`` so an engine-specific knob the client
#: set survives the round trip instead of being silently eaten.
_KNOWN_KEYS = frozenset(
    {
        # mapped
        "max_completion_tokens",
        "max_tokens",
        "parallel_tool_calls",
        "reasoning_effort",
        "stop",
        "temperature",
        "tool_choice",
        "top_p",
        # handled structurally, or meaningless to a single-choice proxy
        "messages",
        "metadata",
        "model",
        "n",
        "stream",
        "stream_options",
        "tools",
        "user",
    }
)

_IMAGE_TYPES = {
    "png": "image/png",
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "gif": "image/gif",
    "webp": "image/webp",
}


# --------------------------------------------------------------------------
# ingress
# --------------------------------------------------------------------------


def ingress(body: dict[str, Any], ctx: DialectContext) -> CanonicalRequest:
    """Lower a Chat Completions body into the canonical request."""
    body = body if isinstance(body, dict) else {}
    inline = ctx.preset.reasoning is ReasoningPolicy.INLINE_TAGS

    req = CanonicalRequest(
        model=_as_str(body.get("model")) or ctx.client_model or "k3",
        raw=body,
    )

    for raw in _as_list(body.get("messages")):
        if not isinstance(raw, dict):
            continue
        role = raw.get("role")
        # `developer` is the current name for what everyone still writes as
        # `system`; both become system text rather than a turn in the history.
        if role in ("system", "developer"):
            text = _joined_text(_content_parts(raw.get("content")))
            if text:
                req.system.append(text)
        elif role == "tool":
            req.messages.append(_tool_message(raw))
        elif role == "assistant":
            req.messages.append(_assistant_message(raw, inline))
        else:
            # Unknown roles are rarer than clients that invent one, so treat
            # anything left as a user turn instead of dropping content.
            req.messages.append(
                Message(role="user", parts=_content_parts(raw.get("content")), name=_as_str(raw.get("name")))
            )

    for raw in _as_list(body.get("tools")):
        tool = _tool_def(raw)
        if tool is not None:
            req.tools.append(tool)

    # The newer field wins when a client sends both; it is the one it means.
    req.max_tokens = _as_int(body.get("max_completion_tokens"))
    if req.max_tokens is None:
        req.max_tokens = _as_int(body.get("max_tokens"))
    req.temperature = _as_float(body.get("temperature"))
    req.top_p = _as_float(body.get("top_p"))
    req.stop = _stop_sequences(body.get("stop"))
    req.stream = bool(body.get("stream"))
    req.reasoning_effort = _as_str(body.get("reasoning_effort"))
    if "parallel_tool_calls" in body:
        req.parallel_tool_calls = bool(body.get("parallel_tool_calls"))
    req.tool_choice = body.get("tool_choice")
    if isinstance(body.get("metadata"), dict):
        req.metadata = dict(body["metadata"])

    # Preset defaults fill gaps only — a client that stated a value owns it,
    # including a deliberate 0.0 temperature.
    defaults = ctx.preset.defaults
    if req.max_tokens is None:
        req.max_tokens = defaults.max_tokens
    if req.temperature is None:
        req.temperature = defaults.temperature
    if req.top_p is None:
        req.top_p = defaults.top_p
    if req.reasoning_effort is None:
        req.reasoning_effort = defaults.reasoning_effort
    if req.parallel_tool_calls is None:
        req.parallel_tool_calls = defaults.parallel_tool_calls

    req.passthrough = {k: v for k, v in body.items() if k not in _KNOWN_KEYS}
    return req


def _tool_message(raw: dict[str, Any]) -> Message:
    return Message(
        role="tool",
        parts=[
            ToolResultPart(
                tool_call_id=_as_str(raw.get("tool_call_id")) or "",
                content=_stringify(raw.get("content")),
                name=_as_str(raw.get("name")),
            )
        ],
    )


def _assistant_message(raw: dict[str, Any], inline: bool) -> Message:
    parts = _content_parts(raw.get("content"))
    reasoning = _as_str(raw.get("reasoning_content")) or _as_str(raw.get("reasoning")) or ""

    if inline:
        # Under this policy the reasoning went out inside the content, so that
        # is where it comes back. Split it out even when an explicit field is
        # also present, or the raw tags would reach the prompt.
        text = _joined_text(parts)
        if "<think>" in text:
            thought, rest = strip_inline_think(text)
            reasoning = reasoning or thought
            parts = [p for p in parts if not isinstance(p, TextPart)]
            if rest:
                parts.insert(0, TextPart(rest))

    msg = Message(role="assistant", name=_as_str(raw.get("name")))
    if reasoning:
        msg.parts.append(ReasoningPart(text=reasoning))
    msg.parts.extend(parts)

    for tc in _as_list(raw.get("tool_calls")):
        if not isinstance(tc, dict):
            continue
        fn = tc.get("function") if isinstance(tc.get("function"), dict) else {}
        name = _as_str(fn.get("name"))
        if not name:
            continue
        msg.parts.append(
            ToolCallPart(
                id=_as_str(tc.get("id")) or new_call_id(),
                name=name,
                arguments=_raw_arguments(fn.get("arguments")),
            )
        )
    return msg


def _raw_arguments(value: Any) -> str:
    """Verbatim when the client sent a string — the invariant this file exists for."""
    if isinstance(value, str):
        return value
    if value is None:
        return "{}"
    # Some clients hand back the parsed object they were given. Nothing to
    # preserve at that point, so serialising is the only option left.
    try:
        return json.dumps(value, ensure_ascii=False)
    except (TypeError, ValueError):
        return "{}"


def _tool_def(raw: Any) -> Optional[ToolDef]:
    if not isinstance(raw, dict):
        return None
    fn = raw.get("function")
    if not isinstance(fn, dict):
        # A few clients flatten the function onto the tool itself.
        fn = raw
    name = _as_str(fn.get("name"))
    if not name:
        return None
    params = fn.get("parameters")
    return ToolDef(
        name=name,
        description=_as_str(fn.get("description")) or "",
        parameters=params if isinstance(params, dict) else {"type": "object", "properties": {}},
    )


def _content_parts(content: Any) -> list[Part]:
    if content is None:
        return []
    if isinstance(content, str):
        return [TextPart(content)] if content else []
    if not isinstance(content, list):
        return [TextPart(_stringify(content))]

    parts: list[Part] = []
    for item in content:
        if isinstance(item, str):
            if item:
                parts.append(TextPart(item))
            continue
        if not isinstance(item, dict):
            continue
        if item.get("type") == "image_url":
            image = _image_part(item.get("image_url"))
            if image is not None:
                parts.append(image)
            continue
        # Everything else that carries a `text` field is text to us, which also
        # covers the input_text/output_text spellings clients borrow from the
        # Responses API.
        text = item.get("text")
        if isinstance(text, str) and text:
            parts.append(TextPart(text))
    return parts


def _image_part(spec: Any) -> Optional[ImagePart]:
    url = spec.get("url") if isinstance(spec, dict) else spec
    if not isinstance(url, str) or not url:
        return None
    if url.startswith("http://") or url.startswith("https://"):
        return ImagePart(media_type=_media_type_from_url(url), data=url, is_url=True)
    if url.startswith("data:"):
        header, _, payload = url[len("data:") :].partition(",")
        return ImagePart(
            media_type=header.split(";", 1)[0] or "image/png",
            data=payload,
            is_url=False,
        )
    # A bare payload with no header at all — assume base64 rather than lose it.
    return ImagePart(media_type="image/png", data=url, is_url=False)


def _media_type_from_url(url: str) -> str:
    ext = url.split("?", 1)[0].rsplit(".", 1)[-1].lower()
    return _IMAGE_TYPES.get(ext, "image/jpeg")


def _stop_sequences(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [s for s in value if isinstance(s, str)]
    return []


# --------------------------------------------------------------------------
# egress
# --------------------------------------------------------------------------


def egress(response: CanonicalResponse, ctx: DialectContext) -> dict[str, Any]:
    """Raise a canonical response into a Chat Completions object."""
    tool_calls = response.tool_calls()
    message = _assistant_payload(
        text=response.text(),
        reasoning="".join(p.text for p in response.reasoning_parts() if not p.redacted),
        tool_calls=tool_calls,
        policy=ctx.preset.reasoning,
    )
    finish = "tool_calls" if tool_calls else _finish_reason(response.stop_reason)

    return {
        "id": ctx.request_id or _new_chat_id(),
        "object": "chat.completion",
        "created": ctx.created,
        "model": ctx.client_model,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": finish,
                "logprobs": None,
            }
        ],
        "usage": _usage_payload(response.usage),
    }


def _assistant_payload(
    text: str,
    reasoning: str,
    tool_calls: list[ToolCallPart],
    policy: ReasoningPolicy,
) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": text}
    if reasoning:
        if policy is ReasoningPolicy.REASONING_CONTENT:
            message["reasoning_content"] = reasoning
        elif policy is ReasoningPolicy.INLINE_TAGS:
            message["content"] = f"<think>{reasoning}</think>{text}"
        # STRIP and the block/item policies have no home for it here; the
        # ledger recovers it by fingerprint on the next turn anyway.

    if tool_calls:
        message["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.name, "arguments": tc.arguments},
            }
            for tc in tool_calls
        ]
        # `content` is typed `string | null`, and an empty string reads as a
        # real (blank) assistant reply to clients that render it.
        if not message["content"]:
            message["content"] = None
    return message


def _usage_payload(usage: Optional[Usage]) -> dict[str, Any]:
    usage = usage or Usage()
    return {
        "prompt_tokens": usage.input_tokens,
        "completion_tokens": usage.output_tokens,
        "total_tokens": usage.input_tokens + usage.output_tokens,
        "prompt_tokens_details": {"cached_tokens": usage.cached_input_tokens},
        "completion_tokens_details": {"reasoning_tokens": usage.reasoning_tokens},
    }


def _finish_reason(stop_reason: str) -> str:
    if stop_reason == "tool_calls":
        return "tool_calls"
    if stop_reason == "length":
        return "length"
    return "stop"


def _new_chat_id() -> str:
    # Hyphenated, because clients pattern-match `chatcmpl-` on the id.
    return f"chatcmpl-{uuid.uuid4().hex[:24]}"


# --------------------------------------------------------------------------
# egress_stream
# --------------------------------------------------------------------------


async def egress_stream(
    events: AsyncIterator[StreamEvent], ctx: DialectContext
) -> AsyncIterator[bytes]:
    """Translate the canonical event stream into Chat Completions SSE."""
    policy = ctx.preset.reasoning
    stream_id = ctx.request_id or _new_chat_id()
    created = ctx.created
    model = ctx.client_model

    def chunk(delta: dict[str, Any], finish_reason: Optional[str] = None) -> bytes:
        return sse(
            {
                "id": stream_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
            }
        )

    started = False
    think_open = False
    saw_tool_call = False
    finish = "stop"
    usage = Usage()

    async for ev in events:
        # The role-only opener is what tells the client a choice exists. Any
        # content event can trigger it, so a stream that skips StreamStart is
        # still well formed.
        if not started and not isinstance(ev, StreamEnd):
            started = True
            yield chunk({"role": "assistant", "content": ""})

        if isinstance(ev, StreamStart):
            continue

        if isinstance(ev, ReasoningDelta):
            if not ev.text:
                continue
            if policy is ReasoningPolicy.REASONING_CONTENT:
                yield chunk({"reasoning_content": ev.text})
            elif policy is ReasoningPolicy.INLINE_TAGS:
                if not think_open:
                    think_open = True
                    yield chunk({"content": "<think>"})
                yield chunk({"content": ev.text})

        elif isinstance(ev, ReasoningEnd):
            if think_open:
                think_open = False
                yield chunk({"content": "</think>"})

        elif isinstance(ev, TextDelta):
            if think_open:
                # Visible text before ReasoningEnd means the engine dropped it;
                # close the tag anyway or the rest of the reply is swallowed.
                think_open = False
                yield chunk({"content": "</think>"})
            if ev.text:
                yield chunk({"content": ev.text})

        elif isinstance(ev, ToolCallStart):
            saw_tool_call = True
            yield chunk(
                {
                    "tool_calls": [
                        {
                            "index": ev.index,
                            "id": ev.id,
                            "type": "function",
                            "function": {"name": ev.name, "arguments": ""},
                        }
                    ]
                }
            )

        elif isinstance(ev, ToolCallArgsDelta):
            yield chunk(
                {"tool_calls": [{"index": ev.index, "function": {"arguments": ev.text}}]}
            )

        elif isinstance(ev, StreamEnd):
            usage = ev.usage or usage
            finish = _finish_reason(ev.stop_reason)

    # The terminal frames live outside the loop so they are emitted exactly
    # once, including when the engine cut out before sending StreamEnd.
    if not started:
        yield chunk({"role": "assistant", "content": ""})
    if think_open:
        yield chunk({"content": "</think>"})
    yield chunk({}, finish_reason="tool_calls" if saw_tool_call else finish)

    include_usage = (
        ctx.stream_usage
        if ctx.stream_usage is not None
        else ctx.preset.defaults.stream_include_usage
    )
    if include_usage:
        yield sse(
            {
                "id": stream_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [],
                "usage": _usage_payload(usage),
            }
        )
    yield sse("[DONE]")


# --------------------------------------------------------------------------
# shared coercions — a malformed body must never raise
# --------------------------------------------------------------------------


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _as_str(value: Any) -> Optional[str]:
    return value if isinstance(value, str) and value else None


def _as_int(value: Any) -> Optional[int]:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value.strip())
        except ValueError:
            return None
    return None


def _as_float(value: Any) -> Optional[float]:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return None
    return None


def _joined_text(parts: list[Part]) -> str:
    return "".join(p.text for p in parts if isinstance(p, TextPart))


def _stringify(content: Any) -> str:
    """Flatten whatever a client put in a content slot into one string."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks: list[str] = []
        for item in content:
            if isinstance(item, str):
                chunks.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                chunks.append(text if isinstance(text, str) else _dumps(item))
            else:
                chunks.append(str(item))
        return "".join(chunks)
    if isinstance(content, (dict, bool, int, float)):
        return _dumps(content)
    return str(content)


def _dumps(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(value)


__all__ = ["ingress", "egress", "egress_stream"]
