"""Engine chunks in, canonical events out.

This is the single place where K3's output becomes something a dialect can
render. Streaming and non-streaming share it: a non-streaming engine response is
turned into a short synthetic chunk list and run through the same loop, so the
two paths cannot drift.

Ordering guarantee for dialects: within a segment you get
``ReasoningDelta* ReasoningEnd`` before any ``TextDelta``, and every
``ToolCallStart`` is followed by its ``ToolCallArgsDelta*`` and ``ToolCallEnd``.
Reasoning may open again after text (interleaved thinking); dialects that can
only represent one thinking block should concatenate.
"""

from __future__ import annotations

from typing import Any, AsyncIterator, Iterable, Optional, Protocol

from .ir import (
    CanonicalResponse,
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
    Usage,
)
from .reasoning import (
    ReasoningLedger,
    encode_signature,
    strip_inline_think,
    upstream_assistant_from_response,
)
from .toolcalls import ParsedReasoning, ParsedText, ParsedToolCall, ToolCallParser, get_parser


class EngineLike(Protocol):  # pragma: no cover - structural typing only
    async def chat(self, payload: dict[str, Any]) -> dict[str, Any]: ...

    def chat_stream(self, payload: dict[str, Any]) -> AsyncIterator[dict[str, Any]]: ...


_FINISH_MAP: dict[str, StopReason] = {
    "stop": "stop",
    "length": "length",
    "max_tokens": "length",
    "tool_calls": "tool_calls",
    "function_call": "tool_calls",
    "content_filter": "content_filter",
}


class _InlineThinkSplitter:
    """Pull ``<think>…</think>`` out of a content stream into reasoning.

    Only used when the engine is configured to carry reasoning inline. Holds
    back just enough text that a tag split across chunks never leaks.
    """

    TOKENS = ("<think>", "</think>")

    def __init__(self) -> None:
        self.buf = ""
        self.inside = False

    def feed(self, text: str) -> list[tuple[str, str]]:
        out: list[tuple[str, str]] = []
        self.buf += text
        while True:
            token = "</think>" if self.inside else "<think>"
            idx = self.buf.find(token)
            if idx == -1:
                keep = _hold(self.buf, self.TOKENS)
                emit, self.buf = self.buf[: len(self.buf) - keep], self.buf[len(self.buf) - keep :]
                if emit:
                    out.append(("reasoning" if self.inside else "text", emit))
                return out
            head = self.buf[:idx]
            if head:
                out.append(("reasoning" if self.inside else "text", head))
            self.buf = self.buf[idx + len(token) :]
            self.inside = not self.inside

    def finish(self) -> list[tuple[str, str]]:
        if not self.buf:
            return []
        out = [("reasoning" if self.inside else "text", self.buf)]
        self.buf = ""
        return out


def _hold(buf: str, tokens: Iterable[str]) -> int:
    keep = 0
    for tok in tokens:
        limit = min(len(buf), len(tok) - 1)
        for k in range(limit, 0, -1):
            if buf.endswith(tok[:k]):
                keep = max(keep, k)
                break
    return keep


def response_to_chunks(resp: dict[str, Any]) -> list[dict[str, Any]]:
    """Turn a non-streaming engine response into synthetic stream chunks."""
    choices = resp.get("choices") or [{}]
    choice = choices[0] if choices else {}
    msg = choice.get("message") or {}
    chunks: list[dict[str, Any]] = []

    reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""
    if reasoning:
        chunks.append({"choices": [{"index": 0, "delta": {"reasoning_content": reasoning}}]})
    content = msg.get("content")
    if isinstance(content, list):  # some engines return part lists
        content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
    if content:
        chunks.append({"choices": [{"index": 0, "delta": {"content": content}}]})
    tool_calls = msg.get("tool_calls") or []
    if tool_calls:
        chunks.append(
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": i,
                                    "id": tc.get("id"),
                                    "type": "function",
                                    "function": {
                                        "name": (tc.get("function") or {}).get("name"),
                                        "arguments": (tc.get("function") or {}).get("arguments", ""),
                                    },
                                }
                                for i, tc in enumerate(tool_calls)
                            ]
                        },
                    }
                ]
            }
        )
    final: dict[str, Any] = {
        "id": resp.get("id"),
        "model": resp.get("model"),
        "choices": [{"index": 0, "delta": {}, "finish_reason": choice.get("finish_reason") or "stop"}],
    }
    if resp.get("usage"):
        final["usage"] = resp["usage"]
    chunks.append(final)
    # Identity has to ride on every chunk, not just the last: `run` establishes
    # the stream id from the first chunk it sees, and a non-streaming response
    # whose id only appeared at the end would silently degrade to "k3".
    for chunk in chunks:
        chunk.setdefault("id", resp.get("id"))
        chunk.setdefault("model", resp.get("model"))
    return chunks


def _usage_from(raw: Optional[dict[str, Any]]) -> Optional[Usage]:
    if not raw:
        return None
    details = raw.get("completion_tokens_details") or {}
    prompt_details = raw.get("prompt_tokens_details") or {}
    return Usage(
        input_tokens=int(raw.get("prompt_tokens") or raw.get("input_tokens") or 0),
        output_tokens=int(raw.get("completion_tokens") or raw.get("output_tokens") or 0),
        cached_input_tokens=int(prompt_details.get("cached_tokens") or 0),
        reasoning_tokens=int(details.get("reasoning_tokens") or 0),
    )


class _ToolAccumulator:
    """Native ``tool_calls`` deltas arrive fragmented and out of order."""

    def __init__(self) -> None:
        self.by_index: dict[int, dict[str, Any]] = {}
        self.order: list[int] = []

    def update(self, raw: dict[str, Any]) -> int:
        index = int(raw.get("index") or 0)
        entry = self.by_index.get(index)
        if entry is None:
            entry = {"id": "", "name": "", "arguments": "", "saw_args": False}
            self.by_index[index] = entry
            self.order.append(index)
        if raw.get("id"):
            entry["id"] = raw["id"]
        fn = raw.get("function") or {}
        fragment = fn.get("name") or ""
        # A name may be split across fragments, and some engines repeat the full
        # name on every delta. Fragments always precede the arguments, so once
        # arguments have started a name is a repeat, not a continuation.
        if fragment and not entry["saw_args"]:
            entry["name"] += fragment
        args = fn.get("arguments") or ""
        if args:
            entry["saw_args"] = True
            entry["arguments"] += args
        return index


async def run(
    engine: EngineLike,
    payload: dict[str, Any],
    *,
    tool_parser: str = "kimi_k3",
    ledger: Optional[ReasoningLedger] = None,
    reasoning_field: str = "reasoning_content",
    stream: bool = True,
    model_label: Optional[str] = None,
) -> AsyncIterator[StreamEvent]:
    """Drive the engine and yield canonical events."""
    ledger = ledger or ReasoningLedger()
    ledger_id = ledger.new_id()
    parser: ToolCallParser = get_parser(tool_parser)
    splitter = _InlineThinkSplitter() if reasoning_field == "inline" else None
    accumulator = _ToolAccumulator()

    reasoning_buf: list[str] = []
    segment_buf: list[str] = []
    text_buf: list[str] = []
    tool_parts: list[ToolCallPart] = []
    parts: list[Any] = []
    reasoning_open = False
    stream_id = ""
    model = model_label or payload.get("model", "k3")
    stop_reason: StopReason = "stop"
    usage = Usage()
    started = False
    tool_index = 0

    def close_reasoning() -> list[StreamEvent]:
        nonlocal reasoning_open
        if not reasoning_open:
            return []
        reasoning_open = False
        text = "".join(segment_buf)
        segment_buf.clear()
        signature = encode_signature(text, ledger_id)
        parts.append(ReasoningPart(text=text, signature=signature))
        return [ReasoningEnd(text=text, signature=signature)]

    def emit_parsed(events: list[Any]) -> list[StreamEvent]:
        nonlocal tool_index
        out: list[StreamEvent] = []
        for ev in events:
            if isinstance(ev, ParsedReasoning):
                out.extend(handle_reasoning(ev.text))
            elif isinstance(ev, ParsedText):
                if not ev.text:
                    continue
                out.extend(close_reasoning())
                text_buf.append(ev.text)
                out.append(TextDelta(ev.text))
            elif isinstance(ev, ParsedToolCall):
                out.extend(close_reasoning())
                idx = tool_index
                tool_index += 1
                out.append(ToolCallStart(index=idx, id=ev.id, name=ev.name))
                if ev.arguments:
                    out.append(ToolCallArgsDelta(index=idx, text=ev.arguments))
                out.append(ToolCallEnd(index=idx, id=ev.id, name=ev.name, arguments=ev.arguments))
                tool_parts.append(ToolCallPart(id=ev.id, name=ev.name, arguments=ev.arguments))
        return out

    def handle_content(text: str) -> list[StreamEvent]:
        out: list[StreamEvent] = []
        if splitter is None:
            return emit_parsed(list(parser.feed(text)))
        for kind, piece in splitter.feed(text):
            if kind == "reasoning":
                out.extend(handle_reasoning(piece))
            else:
                out.extend(emit_parsed(list(parser.feed(piece))))
        return out

    def handle_reasoning(text: str) -> list[StreamEvent]:
        nonlocal reasoning_open
        if not text:
            return []
        reasoning_open = True
        reasoning_buf.append(text)
        segment_buf.append(text)
        return [ReasoningDelta(text)]

    if stream:
        source = engine.chat_stream(payload)
    else:
        resp = await engine.chat(payload)
        source = _aiter(response_to_chunks(resp))

    native_started: set[int] = set()
    #: index -> the call id the client was told; frozen once announced
    announced: dict[int, str] = {}
    #: index -> how many argument characters have already been streamed out
    sent_args: dict[int, int] = {}

    async for chunk in source:
        # Engines are inconsistent about which chunks carry identity, so take it
        # from any chunk that has it rather than only the first.
        if chunk.get("id"):
            stream_id = str(chunk["id"])
        if chunk.get("model") and not model_label:
            model = str(chunk["model"])

        if not started:
            started = True
            stream_id = stream_id or "k3"
            yield StreamStart(id=stream_id, model=model)

        chunk_usage = _usage_from(chunk.get("usage"))
        if chunk_usage:
            usage = usage.merge(chunk_usage)

        for choice in chunk.get("choices") or []:
            delta = choice.get("delta") or choice.get("message") or {}

            reasoning_text = delta.get("reasoning_content") or delta.get("reasoning") or ""
            if reasoning_text:
                for ev in handle_reasoning(reasoning_text):
                    yield ev

            content = delta.get("content")
            if isinstance(content, list):
                content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
            if content:
                for ev in handle_content(content):
                    yield ev

            for raw_call in delta.get("tool_calls") or []:
                for ev in close_reasoning():
                    yield ev
                index = accumulator.update(raw_call)
                entry = accumulator.by_index[index]
                if index not in native_started:
                    # Hold the start until the call's identity has settled. The
                    # name is only final once arguments begin, and the client
                    # sees `function.name` exactly once — in this frame. An id
                    # or name announced early and revised later leaves the
                    # client's tool_result pointing at a call K3 never made.
                    if entry["name"] and entry["saw_args"]:
                        native_started.add(index)
                        announced[index] = entry["id"] or f"call_{index}"
                        yield ToolCallStart(
                            index=index, id=announced[index], name=entry["name"]
                        )
                else:
                    announced.setdefault(index, entry["id"] or f"call_{index}")
                if index in native_started:
                    # Fragments that arrived before the start still have to
                    # reach the client: dialects rebuild arguments from the
                    # deltas alone, so anything skipped here is lost JSON.
                    pending = entry["arguments"][sent_args.get(index, 0) :]
                    if pending:
                        sent_args[index] = len(entry["arguments"])
                        yield ToolCallArgsDelta(index=index, text=pending)

            finish = choice.get("finish_reason")
            if finish:
                stop_reason = _FINISH_MAP.get(finish, "stop")

    if not started:
        yield StreamStart(id="k3", model=model)

    if splitter is not None:
        for kind, piece in splitter.finish():
            if kind == "reasoning":
                for ev in handle_reasoning(piece):
                    yield ev
            else:
                for ev in emit_parsed(list(parser.feed(piece))):
                    yield ev

    for ev in emit_parsed(list(parser.finish())):
        yield ev
    for ev in close_reasoning():
        yield ev

    # Native tool calls only close once the engine stops sending fragments.
    for index in accumulator.order:
        entry = accumulator.by_index[index]
        if index not in native_started:
            native_started.add(index)
            announced[index] = entry["id"] or f"call_{index}"
            yield ToolCallStart(index=index, id=announced[index], name=entry["name"])
        # The id the client was told is the one that has to reach the ledger,
        # even if a later fragment carried a different one.
        call_id = announced[index]
        pending = entry["arguments"][sent_args.get(index, 0) :]
        if pending:
            sent_args[index] = len(entry["arguments"])
            yield ToolCallArgsDelta(index=index, text=pending)
        yield ToolCallEnd(
            index=index, id=call_id, name=entry["name"], arguments=entry["arguments"]
        )
        tool_parts.append(
            ToolCallPart(id=call_id, name=entry["name"], arguments=entry["arguments"])
        )

    text = "".join(text_buf)
    if text:
        parts.append(TextPart(text))
        yield TextEnd(text)
    parts.extend(tool_parts)

    if tool_parts and stop_reason == "stop":
        # A text-format parser found calls the engine reported as a plain stop.
        stop_reason = "tool_calls"

    reasoning_text = "".join(reasoning_buf)
    upstream_message = upstream_assistant_from_response(
        text=text,
        reasoning=reasoning_text,
        tool_calls=tool_parts,
        reasoning_field=reasoning_field,
    )
    ledger.reserve(ledger_id, reasoning_text)
    ledger.complete(ledger_id, upstream_message)

    if not usage.output_tokens:
        usage.output_tokens = max(1, (len(text) + len(reasoning_text)) // 4)
    if not usage.reasoning_tokens and reasoning_text:
        usage.reasoning_tokens = max(1, len(reasoning_text) // 4)

    response = CanonicalResponse(
        id=stream_id or "k3",
        model=model,
        parts=parts,
        stop_reason=stop_reason,
        usage=usage,
        upstream_message=upstream_message,
    )
    yield StreamEnd(stop_reason=stop_reason, usage=usage, response=response)


async def _aiter(items: Iterable[dict[str, Any]]) -> AsyncIterator[dict[str, Any]]:
    for item in items:
        yield item


async def collect_response(events: AsyncIterator[StreamEvent]) -> CanonicalResponse:
    """Drain an event stream down to the response it describes."""
    final: Optional[CanonicalResponse] = None
    async for ev in events:
        if isinstance(ev, StreamEnd):
            final = ev.response
    if final is None:
        raise RuntimeError("engine stream ended without a final event")
    return final


__all__ = ["run", "collect_response", "response_to_chunks", "EngineLike"]
