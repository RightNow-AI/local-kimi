"""Canonical intermediate representation.

Every client dialect lowers into this IR on the way in and is raised back out
of it on the way out. The engine adapter is the only thing that talks to K3.

    client body --ingress--> CanonicalRequest --> upstream payload
    upstream chunks --> StreamEvent* --egress--> client body/SSE

Two invariants matter more than anything else in this file:

1. ``ToolCallPart.arguments`` is the *raw* JSON string the model produced. We
   never round-trip it through ``json.loads``/``json.dumps`` on the way back to
   the engine, because whitespace and key order changes shift the token stream.
2. ``ReasoningPart.text`` is the *raw* ``reasoning_content`` K3 emitted, and
   ``ReasoningPart.signature`` carries whatever we need to rebuild it exactly.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Iterable, Literal, Optional, Union

Role = Literal["system", "user", "assistant", "tool"]

StopReason = Literal[
    "stop",
    "length",
    "tool_calls",
    "content_filter",
    "error",
]


# --------------------------------------------------------------------------
# content parts
# --------------------------------------------------------------------------


@dataclass(slots=True)
class TextPart:
    text: str


@dataclass(slots=True)
class ReasoningPart:
    """A block of K3 ``reasoning_content``.

    ``signature`` is the opaque token we hand to the client so the exact bytes
    can be recovered on the next turn even if the client reshapes or drops the
    visible text. ``redacted`` marks a block whose text the client is not
    allowed to see (we still restore it upstream from the signature).
    """

    text: str
    signature: Optional[str] = None
    redacted: bool = False


@dataclass(slots=True)
class ToolCallPart:
    id: str
    name: str
    arguments: str  # raw JSON text, verbatim from the model

    def arguments_obj(self) -> dict[str, Any]:
        raw = (self.arguments or "").strip()
        if not raw:
            return {}
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {"value": parsed}


@dataclass(slots=True)
class ToolResultPart:
    tool_call_id: str
    content: str
    is_error: bool = False
    name: Optional[str] = None


@dataclass(slots=True)
class ImagePart:
    media_type: str
    data: str  # base64 payload or a URL
    is_url: bool = False


Part = Union[TextPart, ReasoningPart, ToolCallPart, ToolResultPart, ImagePart]


@dataclass(slots=True)
class Message:
    role: Role
    parts: list[Part] = field(default_factory=list)
    name: Optional[str] = None
    #: When we recovered a byte-exact upstream assistant message from the
    #: reasoning ledger, it lands here and is sent to K3 verbatim.
    verbatim_upstream: Optional[dict[str, Any]] = None

    def text(self) -> str:
        return "".join(p.text for p in self.parts if isinstance(p, TextPart))

    def reasoning(self) -> str:
        return "".join(p.text for p in self.parts if isinstance(p, ReasoningPart))

    def tool_calls(self) -> list[ToolCallPart]:
        return [p for p in self.parts if isinstance(p, ToolCallPart)]

    def tool_results(self) -> list[ToolResultPart]:
        return [p for p in self.parts if isinstance(p, ToolResultPart)]


# --------------------------------------------------------------------------
# request / response
# --------------------------------------------------------------------------


@dataclass(slots=True)
class ToolDef:
    name: str
    description: str = ""
    parameters: dict[str, Any] = field(default_factory=lambda: {"type": "object", "properties": {}})


@dataclass(slots=True)
class CanonicalRequest:
    model: str
    messages: list[Message] = field(default_factory=list)
    system: list[str] = field(default_factory=list)
    tools: list[ToolDef] = field(default_factory=list)
    tool_choice: Any = None
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    stop: list[str] = field(default_factory=list)
    stream: bool = False
    reasoning_effort: Optional[str] = None
    thinking_budget: Optional[int] = None
    thinking_enabled: Optional[bool] = None
    parallel_tool_calls: Optional[bool] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    #: dialect-specific leftovers we pass straight through to the engine
    passthrough: dict[str, Any] = field(default_factory=dict)
    #: original client body, kept for recording and debugging
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0
    reasoning_tokens: int = 0

    def merge(self, other: "Usage") -> "Usage":
        return Usage(
            input_tokens=other.input_tokens or self.input_tokens,
            output_tokens=other.output_tokens or self.output_tokens,
            cached_input_tokens=other.cached_input_tokens or self.cached_input_tokens,
            reasoning_tokens=other.reasoning_tokens or self.reasoning_tokens,
        )


@dataclass(slots=True)
class CanonicalResponse:
    id: str
    model: str
    parts: list[Part] = field(default_factory=list)
    stop_reason: StopReason = "stop"
    usage: Usage = field(default_factory=Usage)
    #: the exact assistant message as K3 would want it echoed back
    upstream_message: dict[str, Any] = field(default_factory=dict)

    def text(self) -> str:
        return "".join(p.text for p in self.parts if isinstance(p, TextPart))

    def reasoning_parts(self) -> list[ReasoningPart]:
        return [p for p in self.parts if isinstance(p, ReasoningPart)]

    def tool_calls(self) -> list[ToolCallPart]:
        return [p for p in self.parts if isinstance(p, ToolCallPart)]


# --------------------------------------------------------------------------
# stream events
# --------------------------------------------------------------------------


@dataclass(slots=True)
class StreamStart:
    id: str
    model: str


@dataclass(slots=True)
class ReasoningDelta:
    text: str


@dataclass(slots=True)
class ReasoningEnd:
    text: str
    signature: Optional[str] = None


@dataclass(slots=True)
class TextDelta:
    text: str


@dataclass(slots=True)
class TextEnd:
    text: str


@dataclass(slots=True)
class ToolCallStart:
    index: int
    id: str
    name: str


@dataclass(slots=True)
class ToolCallArgsDelta:
    index: int
    text: str


@dataclass(slots=True)
class ToolCallEnd:
    index: int
    id: str
    name: str
    arguments: str


@dataclass(slots=True)
class StreamEnd:
    stop_reason: StopReason
    usage: Usage
    response: CanonicalResponse


StreamEvent = Union[
    StreamStart,
    ReasoningDelta,
    ReasoningEnd,
    TextDelta,
    TextEnd,
    ToolCallStart,
    ToolCallArgsDelta,
    ToolCallEnd,
    StreamEnd,
]


def collect(events: Iterable[StreamEvent]) -> CanonicalResponse:
    """Fold a finished event list into the response it describes."""
    for ev in reversed(list(events)):
        if isinstance(ev, StreamEnd):
            return ev.response
    raise ValueError("event stream ended without StreamEnd")
