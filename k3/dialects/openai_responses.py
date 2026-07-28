"""OpenAI Responses API dialect — ``POST /v1/responses``, i.e. what Codex speaks.

The Responses wire format is item-shaped rather than message-shaped: a turn is a
flat list of ``reasoning`` / ``message`` / ``function_call`` items, and the
client replays that whole list back at us next turn. Two consequences drive
everything in this file:

1. ``encrypted_content`` on a ``reasoning`` item is our signature vehicle. Codex
   echoes it verbatim without ever looking inside, so it is the one field that
   survives a full agent loop — it must go out and come back untouched.
2. The items of one assistant turn arrive as *siblings*, not as one message. We
   merge consecutive assistant-ish items back into a single :class:`Message` so
   :func:`k3.reasoning.restore_assistant` sees the reasoning, the text and the
   tool calls together and can fingerprint the turn as a unit.

Streaming is named-event SSE (``event:`` *and* ``data:``), every payload carries
a monotonic ``sequence_number``, and there is no ``[DONE]`` sentinel.
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
    StopReason,
    StreamEnd,
    StreamEvent,
    StreamStart,
    TextDelta,
    TextEnd,
    TextPart,
    ToolCallArgsDelta,
    ToolCallEnd,
    ToolCallPart,
    ToolCallStart,
    ToolDef,
    ToolResultPart,
    Usage,
)
from ..reasoning import ReasoningPolicy
from .base import DialectContext, sse

#: Top-level body keys we consume ourselves. Everything else is a leftover we
#: hand to the engine untouched, so anything mapped onto a CanonicalRequest
#: field has to be listed here or it would be sent twice.
_CONSUMED_KEYS = frozenset(
    {
        "input",
        "instructions",
        "tools",
        "model",
        "stream",
        "store",
        "include",
        "reasoning",
        "metadata",
        "previous_response_id",
        "text",
        "max_output_tokens",
        "temperature",
        "top_p",
        "tool_choice",
        "parallel_tool_calls",
    }
)


# --------------------------------------------------------------------------
# ingress
# --------------------------------------------------------------------------


def ingress(body: dict[str, Any], ctx: DialectContext) -> CanonicalRequest:
    """Lower a Responses request into the canonical IR."""
    body = body if isinstance(body, dict) else {}

    req = CanonicalRequest(
        model=str(body.get("model") or ctx.client_model or "k3"),
        stream=bool(body.get("stream")),
        raw=body,
    )

    # `instructions` is the Responses spelling of a system prompt.
    instructions = body.get("instructions")
    if isinstance(instructions, str) and instructions.strip():
        req.system.append(instructions)

    _read_input(body.get("input"), req)

    for raw_tool in body.get("tools") or []:
        tool = _tool_def(raw_tool)
        if tool is not None:
            req.tools.append(tool)

    if body.get("tool_choice") is not None:
        req.tool_choice = body["tool_choice"]

    max_output = _as_int(body.get("max_output_tokens"))
    if max_output is not None:
        req.max_tokens = max_output
    req.temperature = _as_float(body.get("temperature"))
    req.top_p = _as_float(body.get("top_p"))
    if isinstance(body.get("parallel_tool_calls"), bool):
        req.parallel_tool_calls = body["parallel_tool_calls"]

    reasoning = body.get("reasoning")
    if isinstance(reasoning, dict):
        effort = reasoning.get("effort")
        if isinstance(effort, str) and effort:
            req.reasoning_effort = effort

    if isinstance(body.get("metadata"), dict):
        req.metadata = dict(body["metadata"])

    # `previous_response_id` is deliberately dropped: we never `store`, so the
    # client is always replaying the full transcript in `input` anyway.
    for key, value in body.items():
        if key not in _CONSUMED_KEYS:
            req.passthrough[key] = value

    _apply_defaults(req, ctx)
    return req


def _apply_defaults(req: CanonicalRequest, ctx: DialectContext) -> None:
    """Preset defaults fill gaps only — an explicit client value always wins."""
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


def _read_input(raw: Any, req: CanonicalRequest) -> None:
    """Fold the ``input`` items into ``req.messages`` / ``req.system``."""
    if isinstance(raw, str):
        if raw:
            req.messages.append(Message(role="user", parts=[TextPart(raw)]))
        return
    if not isinstance(raw, list):
        return

    for item in raw:
        if isinstance(item, str):
            if item:
                req.messages.append(Message(role="user", parts=[TextPart(item)]))
            continue
        if not isinstance(item, dict):
            continue

        # Some clients omit `type` on plain message items.
        kind = item.get("type") or ("message" if "role" in item else "")
        if kind == "message":
            _read_message_item(item, req)
        elif kind == "reasoning":
            _read_reasoning_item(item, req)
        elif kind == "function_call":
            _read_function_call_item(item, req)
        elif kind == "function_call_output":
            _read_function_output_item(item, req)
        # Anything else (web_search_call, local_shell_call, …) is not ours to
        # translate; skipping beats guessing at a shape K3 cannot execute.


def _read_message_item(item: dict[str, Any], req: CanonicalRequest) -> None:
    role = str(item.get("role") or "user").lower()
    parts = _content_parts(item.get("content"))
    if role in ("system", "developer"):
        text = "".join(p.text for p in parts if isinstance(p, TextPart))
        if text:
            req.system.append(text)
        return
    if role == "assistant":
        _assistant_turn(req).parts.extend(parts)
        return
    if parts:
        req.messages.append(Message(role="user", parts=parts))


def _read_reasoning_item(item: dict[str, Any], req: CanonicalRequest) -> None:
    signature = item.get("encrypted_content")
    text = _summary_text(item)
    if not text and not signature:
        return
    _assistant_turn(req).parts.append(
        ReasoningPart(
            text=text,
            signature=signature if isinstance(signature, str) and signature else None,
        )
    )


def _read_function_call_item(item: dict[str, Any], req: CanonicalRequest) -> None:
    # `call_id` is what `function_call_output` refers back to; `id` is only the
    # item handle, so it is the fallback rather than the other way round.
    call_id = item.get("call_id") or item.get("id") or ""
    arguments = item.get("arguments")
    _assistant_turn(req).parts.append(
        ToolCallPart(
            id=str(call_id),
            name=str(item.get("name") or ""),
            # Never re-serialise: whitespace and key order shift the tokens.
            arguments=arguments if isinstance(arguments, str) else _stringify(arguments),
        )
    )


def _read_function_output_item(item: dict[str, Any], req: CanonicalRequest) -> None:
    call_id = item.get("call_id") or item.get("id") or ""
    req.messages.append(
        Message(
            role="tool",
            parts=[
                ToolResultPart(
                    tool_call_id=str(call_id),
                    content=_stringify(item.get("output")),
                )
            ],
        )
    )


def _assistant_turn(req: CanonicalRequest) -> Message:
    """The assistant message currently being accumulated.

    Reasoning, text and tool calls for one turn arrive as sibling items; they
    have to land on one Message or the reasoning ledger cannot fingerprint the
    turn. Any user/tool item in between ends the run naturally, because it makes
    the last message something other than an assistant one.
    """
    if req.messages and req.messages[-1].role == "assistant":
        return req.messages[-1]
    msg = Message(role="assistant")
    req.messages.append(msg)
    return msg


def _content_parts(content: Any) -> list[Any]:
    if isinstance(content, str):
        return [TextPart(content)] if content else []
    if not isinstance(content, list):
        return []

    parts: list[Any] = []
    for raw in content:
        if isinstance(raw, str):
            if raw:
                parts.append(TextPart(raw))
            continue
        if not isinstance(raw, dict):
            continue
        kind = raw.get("type")
        if kind in ("input_text", "output_text", "text", "summary_text", "refusal"):
            text = raw.get("text") if kind != "refusal" else raw.get("refusal")
            if isinstance(text, str) and text:
                parts.append(TextPart(text))
        elif kind == "input_image":
            image = _image_part(raw.get("image_url"))
            if image is not None:
                parts.append(image)
    return parts


def _image_part(raw: Any) -> Optional[ImagePart]:
    url = raw if isinstance(raw, str) else raw.get("url") if isinstance(raw, dict) else None
    if not isinstance(url, str) or not url:
        return None
    if url.startswith("data:"):
        header, _, data = url.partition(",")
        media_type = header[len("data:") :].split(";", 1)[0] or "image/png"
        return ImagePart(media_type=media_type, data=data, is_url=False)
    return ImagePart(media_type="image/*", data=url, is_url=True)


def _summary_text(item: dict[str, Any]) -> str:
    chunks: list[str] = []
    for entry in item.get("summary") or []:
        if isinstance(entry, str):
            chunks.append(entry)
        elif isinstance(entry, dict) and isinstance(entry.get("text"), str):
            chunks.append(entry["text"])
    if not chunks:
        # Newer builds carry the visible text under `content` instead.
        for entry in item.get("content") or []:
            if isinstance(entry, dict) and isinstance(entry.get("text"), str):
                chunks.append(entry["text"])
    return "\n".join(c for c in chunks if c)


def _tool_def(raw: Any) -> Optional[ToolDef]:
    """Responses puts name/description/parameters at the top level; Chat nests
    them under ``function``. Accept both, ignore tool types we cannot execute."""
    if not isinstance(raw, dict):
        return None
    nested = raw.get("function")
    spec = nested if isinstance(nested, dict) else raw
    kind = raw.get("type")
    if kind not in (None, "function") and not isinstance(nested, dict):
        return None
    name = spec.get("name")
    if not isinstance(name, str) or not name:
        return None
    params = spec.get("parameters")
    return ToolDef(
        name=name,
        description=str(spec.get("description") or ""),
        parameters=params if isinstance(params, dict) else {"type": "object", "properties": {}},
    )


def _stringify(value: Any) -> str:
    """Tool output arrives as a string, a content-part list, or a bare object."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        chunks = [
            entry if isinstance(entry, str) else entry.get("text", "") if isinstance(entry, dict) else ""
            for entry in value
        ]
        joined = "".join(c for c in chunks if isinstance(c, str))
        if joined:
            return joined
    try:
        return json.dumps(value, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(value)


def _as_int(value: Any) -> Optional[int]:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_float(value: Any) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


# --------------------------------------------------------------------------
# response object construction
# --------------------------------------------------------------------------


def _emits_reasoning(ctx: DialectContext) -> bool:
    return ctx.preset.reasoning == ReasoningPolicy.RESPONSES_ITEM


def _response_id(raw: str, ctx: DialectContext) -> str:
    rid = (raw or "").strip() or ctx.request_id or ctx.new_id("resp")
    return rid if rid.startswith("resp_") else f"resp_{rid}"


def _item_id(prefix: str, response_id: str, output_index: int) -> str:
    """Item ids are derived, not random.

    The streaming path announces an item long before :func:`egress` rebuilds the
    final response object; deriving both from the response id and the output
    index is what keeps the id in ``output_item.added`` equal to the id in
    ``response.completed``.
    """
    base = response_id[len("resp_") :] if response_id.startswith("resp_") else response_id
    return f"{prefix}_{base}_{output_index}"


def _reasoning_item(
    item_id: str, text: str, signature: Optional[str], status: str
) -> dict[str, Any]:
    return {
        "type": "reasoning",
        "id": item_id,
        "summary": [{"type": "summary_text", "text": text}] if text else [],
        "encrypted_content": signature,
        "status": status,
    }


def _message_item(item_id: str, text: Optional[str], status: str) -> dict[str, Any]:
    content = [_output_text_part(text)] if text is not None else []
    return {
        "type": "message",
        "id": item_id,
        "role": "assistant",
        "status": status,
        "content": content,
    }


def _output_text_part(text: str) -> dict[str, Any]:
    return {"type": "output_text", "text": text, "annotations": []}


def _function_call_item(
    item_id: str, call_id: str, name: str, arguments: str, status: str
) -> dict[str, Any]:
    return {
        "type": "function_call",
        "id": item_id,
        "call_id": call_id,
        "name": name,
        "arguments": arguments,
        "status": status,
    }


def _incomplete_details(stop_reason: StopReason) -> Optional[dict[str, Any]]:
    if stop_reason == "length":
        return {"reason": "max_output_tokens"}
    if stop_reason == "content_filter":
        return {"reason": "content_filter"}
    return None


def _usage(usage: Usage) -> dict[str, Any]:
    return {
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "total_tokens": usage.input_tokens + usage.output_tokens,
        "input_tokens_details": {"cached_tokens": usage.cached_input_tokens},
        "output_tokens_details": {"reasoning_tokens": usage.reasoning_tokens},
    }


def _envelope(
    response_id: str,
    model: str,
    created: int,
    *,
    status: str,
    output: list[dict[str, Any]],
    output_text: str,
    usage: Optional[dict[str, Any]],
    incomplete_details: Optional[dict[str, Any]],
) -> dict[str, Any]:
    """The response object itself — shared by the skeleton and the final body so
    the streaming and non-streaming shapes cannot drift apart."""
    return {
        "id": response_id,
        "object": "response",
        "created_at": created,
        "status": status,
        "model": model,
        "output": output,
        "output_text": output_text,
        "usage": usage,
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
        "instructions": None,
        "metadata": {},
        "incomplete_details": incomplete_details,
        "error": None,
    }


def _output_items(
    response: CanonicalResponse, ctx: DialectContext, response_id: str
) -> list[dict[str, Any]]:
    """Reasoning first, then the message, then one item per tool call.

    The streaming path emits items in exactly this order, so the output indices
    (and therefore the derived item ids) line up between the two paths.
    """
    items: list[dict[str, Any]] = []
    index = 0

    if _emits_reasoning(ctx):
        for part in response.reasoning_parts():
            if not part.text and not part.signature:
                continue
            items.append(
                _reasoning_item(
                    _item_id("rs", response_id, index),
                    "" if part.redacted else part.text,
                    part.signature,
                    "completed",
                )
            )
            index += 1

    text = response.text()
    if text:
        items.append(_message_item(_item_id("msg", response_id, index), text, "completed"))
        index += 1

    for call in response.tool_calls():
        items.append(
            _function_call_item(
                _item_id("fc", response_id, index),
                call.id,
                call.name,
                call.arguments,  # raw, verbatim
                "completed",
            )
        )
        index += 1

    return items


def _output_text(items: list[dict[str, Any]]) -> str:
    """``output_text`` is the message items and nothing else — no reasoning,
    no arguments. Deriving it from the item list keeps the two in agreement."""
    return "".join(
        part.get("text", "")
        for item in items
        if item.get("type") == "message"
        for part in item.get("content") or []
    )


def _terminal(
    response: CanonicalResponse,
    ctx: DialectContext,
    response_id: str,
    items: list[dict[str, Any]],
) -> dict[str, Any]:
    """The finished response object, given an already-decided item list.

    The two callers disagree about the items on purpose: :func:`egress` derives
    them from the response parts, the streaming path hands in the ones it
    actually announced. Everything else — id, model, usage, status — has to come
    out identical either way, so it is computed here once.
    """
    incomplete = _incomplete_details(response.stop_reason)
    return _envelope(
        response_id,
        ctx.client_model or response.model or "k3",
        ctx.created,
        status="incomplete" if incomplete else "completed",
        output=items,
        output_text=_output_text(items),
        usage=_usage(response.usage),
        incomplete_details=incomplete,
    )


# --------------------------------------------------------------------------
# egress
# --------------------------------------------------------------------------


def egress(response: CanonicalResponse, ctx: DialectContext) -> dict[str, Any]:
    """Raise a canonical response into a Responses ``response`` object."""
    response_id = _response_id(response.id, ctx)
    return _terminal(response, ctx, response_id, _output_items(response, ctx, response_id))


# --------------------------------------------------------------------------
# egress_stream
# --------------------------------------------------------------------------


async def egress_stream(
    events: AsyncIterator[StreamEvent], ctx: DialectContext
) -> AsyncIterator[bytes]:
    """Translate the canonical event stream into Responses named-event SSE."""
    seq = 0
    response_id = ""
    model = ctx.client_model or "k3"
    started = False

    #: Every item we finished, in the order we announced it. The terminal
    #: object is built from this list rather than re-derived from the response
    #: parts, because only this list describes what the client actually saw: a
    #: turn that goes text -> tool call -> text streams three items where the
    #: parts describe two, and re-deriving would renumber every id.
    items: list[dict[str, Any]] = []

    # Exactly one item is open at a time, and it closes before the next one is
    # announced — Codex reads `output_item.done` as "item N is final", so an
    # item that finished after a later one started would be a lie. `open_buf`
    # holds only the open item's deltas, which is what its `.done` must carry.
    open_kind: Optional[str] = None
    open_id = ""
    open_index = 0
    open_buf: list[str] = []
    open_tool: Optional[int] = None  # canonical index of the open tool call

    # canonical tool index -> {"item_id", "output_index", "call_id", "name",
    # "closed"}; kept after the close so a late end event is a no-op.
    tools: dict[int, dict[str, Any]] = {}

    def pack(name: str, **fields: Any) -> bytes:
        nonlocal seq
        payload: dict[str, Any] = {"type": name, "sequence_number": seq}
        payload.update(fields)
        seq += 1
        return sse(payload, event=name)

    def start_frames() -> list[bytes]:
        nonlocal started
        started = True
        skeleton = _envelope(
            response_id,
            model,
            ctx.created,
            status="in_progress",
            output=[],
            output_text="",
            usage=None,
            incomplete_details=None,
        )
        return [
            pack("response.created", response=skeleton),
            pack("response.in_progress", response=skeleton),
        ]

    def begin(prefix: str, kind: str) -> list[bytes]:
        """Retire the open item and take the next index for a new one.

        The index is ``len(items)`` rather than a running counter: with
        close-before-open the two are the same number, and deriving it makes the
        "no gaps, no interleaving" invariant impossible to break by accident.
        """
        nonlocal open_kind, open_id, open_index
        out = close_open()
        open_kind = kind
        open_index = len(items)
        open_id = _item_id(prefix, response_id, open_index)
        open_buf.clear()  # this item accumulates only its own deltas
        return out

    def finish(item: dict[str, Any]) -> None:
        nonlocal open_kind, open_tool
        open_kind = None
        open_tool = None
        open_buf.clear()
        items.append(item)

    def open_reasoning() -> list[bytes]:
        out = begin("rs", "reasoning")
        out.append(
            pack(
                "response.output_item.added",
                output_index=open_index,
                item=_reasoning_item(open_id, "", None, "in_progress"),
            )
        )
        return out

    def close_reasoning(signature: Optional[str]) -> list[bytes]:
        # The signature describes the segment that just ended, so it rides on
        # this item's encrypted_content — never on a later one's.
        full = "".join(open_buf)
        item = _reasoning_item(open_id, full, signature, "completed")
        finish(item)
        return [
            pack(
                "response.reasoning_summary_text.done",
                item_id=open_id,
                output_index=open_index,
                summary_index=0,
                text=full,
            ),
            pack("response.output_item.done", output_index=open_index, item=item),
        ]

    def open_text() -> list[bytes]:
        out = begin("msg", "message")
        out.extend(
            [
                pack(
                    "response.output_item.added",
                    output_index=open_index,
                    item=_message_item(open_id, None, "in_progress"),
                ),
                pack(
                    "response.content_part.added",
                    item_id=open_id,
                    output_index=open_index,
                    content_index=0,
                    part=_output_text_part(""),
                ),
            ]
        )
        return out

    def close_text() -> list[bytes]:
        full = "".join(open_buf)
        item = _message_item(open_id, full, "completed")
        finish(item)
        return [
            pack(
                "response.output_text.done",
                item_id=open_id,
                output_index=open_index,
                content_index=0,
                text=full,
            ),
            pack(
                "response.content_part.done",
                item_id=open_id,
                output_index=open_index,
                content_index=0,
                part=_output_text_part(full),
            ),
            pack("response.output_item.done", output_index=open_index, item=item),
        ]

    def open_tool_call(index: int, call_id: str, name: str) -> list[bytes]:
        nonlocal open_tool
        out = begin("fc", "function_call")
        open_tool = index
        tools[index] = {
            "item_id": open_id,
            "output_index": open_index,
            "call_id": call_id,
            "name": name,
            "closed": False,
        }
        out.append(
            pack(
                "response.output_item.added",
                output_index=open_index,
                item=_function_call_item(open_id, call_id, name, "", "in_progress"),
            )
        )
        return out

    def close_tool_call(arguments: str) -> list[bytes]:
        entry = tools[open_tool]
        entry["closed"] = True
        item = _function_call_item(
            entry["item_id"],
            entry["call_id"],
            entry["name"],
            arguments,  # raw, verbatim
            "completed",
        )
        finish(item)
        return [
            pack(
                "response.function_call_arguments.done",
                item_id=entry["item_id"],
                output_index=entry["output_index"],
                arguments=arguments,
            ),
            pack(
                "response.output_item.done",
                output_index=entry["output_index"],
                item=item,
            ),
        ]

    def close_open() -> list[bytes]:
        """Close whichever item is open, if any. A tool closes on the arguments
        seen so far — the engine sends a call's fragments contiguously, so by
        the time anything else opens they are all in."""
        if open_kind == "reasoning":
            return close_reasoning(None)
        if open_kind == "message":
            return close_text()
        if open_kind == "function_call":
            return close_tool_call("".join(open_buf))
        return []

    async for ev in events:
        if isinstance(ev, StreamStart):
            response_id = _response_id(ev.id, ctx)
            model = ctx.client_model or ev.model or model
            for frame in start_frames():
                yield frame
            continue

        # A malformed upstream could skip StreamStart; the client still needs
        # the envelope before any item event.
        if not started:
            response_id = _response_id(response_id, ctx)
            for frame in start_frames():
                yield frame

        if isinstance(ev, ReasoningDelta):
            if not _emits_reasoning(ctx):
                continue  # STRIP: the ledger still restores it upstream
            if open_kind != "reasoning":
                for frame in open_reasoning():
                    yield frame
            open_buf.append(ev.text)
            yield pack(
                "response.reasoning_summary_text.delta",
                item_id=open_id,
                output_index=open_index,
                summary_index=0,
                delta=ev.text,
            )

        elif isinstance(ev, ReasoningEnd):
            if not _emits_reasoning(ctx):
                continue
            if open_kind != "reasoning":
                if not ev.text and not ev.signature:
                    continue
                for frame in open_reasoning():
                    yield frame
            if ev.text and not open_buf:
                open_buf.append(ev.text)  # segment arrived whole, without deltas
            for frame in close_reasoning(ev.signature):
                yield frame

        elif isinstance(ev, TextDelta):
            if open_kind != "message":
                for frame in open_text():
                    yield frame
            open_buf.append(ev.text)
            yield pack(
                "response.output_text.delta",
                item_id=open_id,
                output_index=open_index,
                content_index=0,
                delta=ev.text,
            )

        elif isinstance(ev, TextEnd):
            # TextEnd carries the whole turn's text, which after a split is more
            # than this item's; the deltas we already streamed are the truth.
            if open_kind != "message":
                continue
            if ev.text and not open_buf:
                open_buf.append(ev.text)
            for frame in close_text():
                yield frame

        elif isinstance(ev, ToolCallStart):
            for frame in open_tool_call(ev.index, ev.id, ev.name):
                yield frame

        elif isinstance(ev, ToolCallArgsDelta):
            entry = tools.get(ev.index)
            if entry is None or entry["closed"]:
                continue  # no open item to attach the fragment to
            open_buf.append(ev.text)
            yield pack(
                "response.function_call_arguments.delta",
                item_id=entry["item_id"],
                output_index=entry["output_index"],
                delta=ev.text,
            )

        elif isinstance(ev, ToolCallEnd):
            entry = tools.get(ev.index)
            if entry is not None and entry["closed"]:
                continue  # already flushed when the next item was announced
            if entry is None:
                for frame in open_tool_call(ev.index, ev.id, ev.name):
                    yield frame
            arguments = ev.arguments if ev.arguments is not None else "".join(open_buf)
            for frame in close_tool_call(arguments):
                yield frame

        elif isinstance(ev, StreamEnd):
            for frame in close_open():
                yield frame
            # The terminal object lists the items we actually announced, in the
            # order we announced them. Re-deriving it from the response parts
            # would number the items from the parts instead and could hand the
            # client ids it never saw. Everything else matches egress() exactly.
            final = _terminal(ev.response, ctx, response_id, items)
            name = (
                "response.incomplete"
                if ev.stop_reason == "length"
                else "response.completed"
            )
            yield pack(name, response=final)
            # No [DONE]: the Responses API terminates on the named event.


__all__ = ["ingress", "egress", "egress_stream"]
