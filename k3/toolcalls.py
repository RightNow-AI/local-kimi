"""Tool-call parsers.

The engine either emits native OpenAI ``tool_calls`` deltas (because it was
started with its own ``--tool-call-parser``), or it emits the model's raw text
and we parse the calls out ourselves. A preset picks which.

All parsers are incremental: ``feed()`` may be called with an arbitrary split of
the token stream, including splits that land in the middle of a control token.
Text that could still turn out to be the start of a control token is held back
until we know, so a tag never leaks to the client.

Parsers emit *complete* tool calls. Streaming partial argument JSON to a client
buys nothing — every client buffers it until the call closes anyway — and it
makes the parsers far harder to get right.
"""

from __future__ import annotations

import ast
import json
import re
import uuid
from dataclasses import dataclass
from typing import Callable, Iterable, Optional, Union


@dataclass(slots=True)
class ParsedText:
    text: str


@dataclass(slots=True)
class ParsedToolCall:
    name: str
    arguments: str
    id: str


ParseEvent = Union[ParsedText, ParsedToolCall]


def new_call_id(prefix: str = "call") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:24]}"


def _hold_back(buf: str, tokens: Iterable[str]) -> int:
    """How many trailing chars of ``buf`` might be a partial control token."""
    keep = 0
    for tok in tokens:
        limit = min(len(buf), len(tok) - 1)
        for k in range(limit, 0, -1):
            if buf.endswith(tok[:k]):
                if k > keep:
                    keep = k
                break
    return keep


def _coerce_arguments(value: object) -> str:
    """Normalise a parsed ``arguments`` value into a JSON object string."""
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return "{}"
        try:
            json.loads(stripped)
        except json.JSONDecodeError:
            return json.dumps({"value": value}, ensure_ascii=False)
        return stripped
    if value is None:
        return "{}"
    try:
        return json.dumps(value, ensure_ascii=False)
    except (TypeError, ValueError):
        return "{}"


class ToolCallParser:
    """Base class: passthrough. The engine already did the parsing."""

    name = "passthrough"

    def feed(self, text: str) -> list[ParseEvent]:
        return [ParsedText(text)] if text else []

    def finish(self) -> list[ParseEvent]:
        return []

    def reset(self) -> None:  # pragma: no cover - trivial
        pass


class KimiToolParser(ToolCallParser):
    """Kimi-family control tokens — the K3 native format.

    ``<|tool_calls_section_begin|>``
    ``<|tool_call_begin|>functions.NAME:IDX<|tool_call_argument_begin|>{...}<|tool_call_end|>``
    ``<|tool_calls_section_end|>``

    The section wrapper is optional; some templates emit bare calls.
    """

    name = "kimi"

    SECTION_BEGIN = "<|tool_calls_section_begin|>"
    SECTION_END = "<|tool_calls_section_end|>"
    CALL_BEGIN = "<|tool_call_begin|>"
    ARG_BEGIN = "<|tool_call_argument_begin|>"
    CALL_END = "<|tool_call_end|>"

    TOKENS = (SECTION_BEGIN, SECTION_END, CALL_BEGIN, ARG_BEGIN, CALL_END)
    _ID_RE = re.compile(r"^(?:functions?\.)?(?P<name>[^:\s]+?)(?::(?P<idx>\d+))?$")

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._buf = ""
        self._state = "out"  # out | section | call_id | call_args
        self._in_section = False
        self._pending_name = ""
        self._pending_index = 0
        #: Has the current run of text inside a section produced any
        #: non-whitespace yet? A whitespace-only run is formatting and is
        #: dropped; once a run is known to carry prose, every byte of it —
        #: including its whitespace — is output and must be emitted verbatim.
        self._section_text = False

    def feed(self, text: str) -> list[ParseEvent]:
        if not text:
            return []
        self._buf += text
        return self._drain(final=False)

    def finish(self) -> list[ParseEvent]:
        events = self._drain(final=True)
        if self._buf:
            # Unterminated construct: surface it as text rather than eat output.
            if self._state in ("call_id", "call_args"):
                events.append(ParsedText(self._reconstruct_partial()))
            else:
                events.append(ParsedText(self._buf))
            self._buf = ""
        self._state = "out"
        self._section_text = False
        return [e for e in events if not (isinstance(e, ParsedText) and not e.text)]

    def _reconstruct_partial(self) -> str:
        if self._state == "call_id":
            return self.CALL_BEGIN + self._buf
        return self.CALL_BEGIN + self._pending_name + self.ARG_BEGIN + self._buf

    def _drain(self, final: bool) -> list[ParseEvent]:
        out: list[ParseEvent] = []
        while True:
            if self._state == "out":
                sec = self._buf.find(self.SECTION_BEGIN)
                call = self._buf.find(self.CALL_BEGIN)
                idx, tok, nxt = _first_of(
                    (sec, self.SECTION_BEGIN, "section"),
                    (call, self.CALL_BEGIN, "call_id"),
                )
                if idx == -1:
                    keep = 0 if final else _hold_back(self._buf, self.TOKENS)
                    emit, self._buf = self._buf[: len(self._buf) - keep], self._buf[len(self._buf) - keep :]
                    if emit:
                        out.append(ParsedText(emit))
                    return out
                if idx:
                    out.append(ParsedText(self._buf[:idx]))
                self._buf = self._buf[idx + len(tok) :]
                self._in_section = nxt == "section"
                self._state = nxt
                self._section_text = False

            elif self._state == "section":
                call = self._buf.find(self.CALL_BEGIN)
                end = self._buf.find(self.SECTION_END)
                idx, tok, nxt = _first_of(
                    (call, self.CALL_BEGIN, "call_id"),
                    (end, self.SECTION_END, "out"),
                )
                if idx == -1:
                    keep = 0 if final else _hold_back(self._buf, self.TOKENS)
                    emit = self._buf[: len(self._buf) - keep]
                    tail = self._buf[len(self._buf) - keep :]
                    if self._section_text or emit.strip():
                        # Prose inside a section is model output, not framing —
                        # surface it rather than eat it. A max_tokens cut right
                        # after the section token lands here.
                        if emit:
                            out.append(ParsedText(emit))
                            self._section_text = True
                        self._buf = tail
                    elif final:
                        # Whitespace between calls is formatting; drop it.
                        self._buf = tail
                    # Otherwise hold the leading whitespace rather than deciding
                    # now: a later chunk may turn this run into prose, and
                    # whether we dropped it must not depend on the chunk split.
                    return out
                run = self._buf[:idx]
                if run and (self._section_text or run.strip()):
                    out.append(ParsedText(run))
                self._buf = self._buf[idx + len(tok) :]
                if nxt == "out":
                    self._in_section = False
                self._state = nxt
                self._section_text = False

            elif self._state == "call_id":
                idx = self._buf.find(self.ARG_BEGIN)
                if idx == -1:
                    return out
                self._pending_name = self._buf[:idx].strip()
                self._buf = self._buf[idx + len(self.ARG_BEGIN) :]
                self._state = "call_args"

            elif self._state == "call_args":
                idx = self._buf.find(self.CALL_END)
                if idx == -1:
                    return out
                args = self._buf[:idx].strip()
                self._buf = self._buf[idx + len(self.CALL_END) :]
                name, call_id = self._split_id(self._pending_name)
                out.append(ParsedToolCall(name=name, arguments=_coerce_arguments(args), id=call_id))
                self._pending_name = ""
                self._state = "section" if self._in_section else "out"
                self._section_text = False

    def _split_id(self, raw: str) -> tuple[str, str]:
        m = self._ID_RE.match(raw)
        if not m:
            return (raw or "unknown"), new_call_id()
        name = m.group("name") or "unknown"
        idx = m.group("idx")
        # Reuse the model's own call index so retries stay stable within a turn.
        suffix = f"{self._pending_index}" if idx is None else idx
        self._pending_index += 1
        return name, f"call_{name}_{suffix}_{uuid.uuid4().hex[:8]}"


def _first_of(*candidates: tuple[int, str, str]) -> tuple[int, str, str]:
    best = (-1, "", "")
    for idx, tok, nxt in candidates:
        if idx == -1:
            continue
        if best[0] == -1 or idx < best[0]:
            best = (idx, tok, nxt)
    return best


class HermesToolParser(ToolCallParser):
    """``<tool_call>{"name": ..., "arguments": {...}}</tool_call>``."""

    name = "hermes"

    OPEN = "<tool_call>"
    CLOSE = "</tool_call>"
    TOKENS = (OPEN, CLOSE)

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._buf = ""
        self._inside = False

    def feed(self, text: str) -> list[ParseEvent]:
        if not text:
            return []
        self._buf += text
        return self._drain(final=False)

    def finish(self) -> list[ParseEvent]:
        events = self._drain(final=True)
        if self._buf:
            events.append(ParsedText((self.OPEN + self._buf) if self._inside else self._buf))
            self._buf = ""
            self._inside = False
        return [e for e in events if not (isinstance(e, ParsedText) and not e.text)]

    def _drain(self, final: bool) -> list[ParseEvent]:
        out: list[ParseEvent] = []
        while True:
            if not self._inside:
                idx = self._buf.find(self.OPEN)
                if idx == -1:
                    keep = 0 if final else _hold_back(self._buf, self.TOKENS)
                    emit, self._buf = self._buf[: len(self._buf) - keep], self._buf[len(self._buf) - keep :]
                    if emit:
                        out.append(ParsedText(emit))
                    return out
                if idx:
                    out.append(ParsedText(self._buf[:idx]))
                self._buf = self._buf[idx + len(self.OPEN) :]
                self._inside = True
            else:
                idx = self._buf.find(self.CLOSE)
                if idx == -1:
                    return out
                body = self._buf[:idx].strip()
                self._buf = self._buf[idx + len(self.CLOSE) :]
                self._inside = False
                call = _parse_name_arguments_json(body)
                out.append(call if call else ParsedText(self.OPEN + body + self.CLOSE))


def _parse_name_arguments_json(body: str) -> Optional[ParsedToolCall]:
    try:
        obj = json.loads(body)
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict):
        return None
    name = obj.get("name") or obj.get("function")
    if not isinstance(name, str) or not name:
        return None
    args = obj.get("arguments")
    if args is None:
        args = obj.get("parameters")
    return ParsedToolCall(name=name, arguments=_coerce_arguments(args), id=new_call_id())


class JsonToolParser(ToolCallParser):
    """Bare JSON object (optionally in a ```json fence) as the whole reply.

    There is no way to know a leading ``{`` will turn out to be a tool call, so
    this parser buffers a response that starts like JSON and decides at the end.
    Anything else streams through untouched.
    """

    name = "json"

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._buf = ""
        self._buffering: Optional[bool] = None

    def feed(self, text: str) -> list[ParseEvent]:
        if not text:
            return []
        if self._buffering is False:
            return [ParsedText(text)]
        self._buf += text
        if self._buffering is None:
            probe = self._buf.lstrip()
            if not probe:
                return []
            if probe.startswith("{") or probe.startswith("["):
                self._buffering = True
                return []
            if probe.startswith("```"):
                self._buffering = True
                return []
            # A fence is three characters, so a one-character chunk of it is
            # undecided rather than a rejection. Latching here would stream a
            # fenced tool call out as plain text.
            if len(probe) < 3 and "```".startswith(probe):
                return []
            self._buffering = False
            emit, self._buf = self._buf, ""
            return [ParsedText(emit)]
        return []

    def finish(self) -> list[ParseEvent]:
        if not self._buf:
            self.reset()
            return []
        body = _strip_code_fence(self._buf.strip())
        events: list[ParseEvent] = []
        parsed = _parse_name_arguments_json(body)
        if parsed:
            events.append(parsed)
        else:
            multi = _parse_json_array_of_calls(body)
            if multi:
                events.extend(multi)
            else:
                events.append(ParsedText(self._buf))
        self.reset()
        return events


def _parse_json_array_of_calls(body: str) -> list[ParsedToolCall]:
    try:
        obj = json.loads(body)
    except json.JSONDecodeError:
        return []
    if not isinstance(obj, list):
        return []
    calls: list[ParsedToolCall] = []
    for item in obj:
        if not isinstance(item, dict):
            return []
        parsed = _parse_name_arguments_json(json.dumps(item))
        if not parsed:
            return []
        calls.append(parsed)
    return calls


_FENCE_RE = re.compile(r"^```[a-zA-Z0-9_-]*\s*\n?(.*?)\n?```$", re.DOTALL)


def _strip_code_fence(text: str) -> str:
    m = _FENCE_RE.match(text.strip())
    return m.group(1).strip() if m else text


class PythonicToolParser(ToolCallParser):
    """``[get_weather(city="Beijing"), other(x=1)]`` as the whole reply."""

    name = "pythonic"

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._buf = ""
        self._buffering: Optional[bool] = None

    def feed(self, text: str) -> list[ParseEvent]:
        if not text:
            return []
        if self._buffering is False:
            return [ParsedText(text)]
        self._buf += text
        if self._buffering is None:
            probe = self._buf.lstrip()
            if not probe:
                return []
            if probe.startswith("["):
                self._buffering = True
                return []
            self._buffering = False
            emit, self._buf = self._buf, ""
            return [ParsedText(emit)]
        return []

    def finish(self) -> list[ParseEvent]:
        if not self._buf:
            self.reset()
            return []
        calls = _parse_pythonic(self._buf.strip())
        events: list[ParseEvent] = list(calls) if calls else [ParsedText(self._buf)]
        self.reset()
        return events


def _parse_pythonic(body: str) -> list[ParsedToolCall]:
    try:
        tree = ast.parse(body.strip(), mode="eval")
    except SyntaxError:
        return []
    node = tree.body
    items = node.elts if isinstance(node, ast.List) else [node]
    calls: list[ParsedToolCall] = []
    for item in items:
        if not isinstance(item, ast.Call) or not isinstance(item.func, ast.Name):
            return []
        args: dict[str, object] = {}
        for kw in item.keywords:
            if kw.arg is None:
                return []
            try:
                args[kw.arg] = ast.literal_eval(kw.value)
            except (ValueError, SyntaxError):
                return []
        calls.append(
            ParsedToolCall(
                name=item.func.id,
                arguments=json.dumps(args, ensure_ascii=False),
                id=new_call_id(),
            )
        )
    return calls


_REGISTRY: dict[str, Callable[[], ToolCallParser]] = {
    "passthrough": ToolCallParser,
    "none": ToolCallParser,
    "kimi": KimiToolParser,
    "kimi_k2": KimiToolParser,
    "hermes": HermesToolParser,
    "json": JsonToolParser,
    "pythonic": PythonicToolParser,
}


def get_parser(name: str) -> ToolCallParser:
    try:
        return _REGISTRY[name]()
    except KeyError:
        raise ValueError(
            f"unknown tool-call parser {name!r}; known: {', '.join(sorted(_REGISTRY))}"
        ) from None


def parser_names() -> list[str]:
    return sorted(_REGISTRY)


def parse_all(parser: ToolCallParser, text: str) -> list[ParseEvent]:
    """Convenience for non-streaming paths and tests."""
    events = list(parser.feed(text))
    events.extend(parser.finish())
    return events


__all__ = [
    "ParsedText",
    "ParsedToolCall",
    "ParseEvent",
    "ToolCallParser",
    "KimiToolParser",
    "HermesToolParser",
    "JsonToolParser",
    "PythonicToolParser",
    "get_parser",
    "parser_names",
    "parse_all",
    "new_call_id",
]
