"""Shared contract every dialect module implements.

    ingress(body, ctx)        -> CanonicalRequest
    egress(response, ctx)     -> dict            (non-streaming client body)
    egress_stream(events, ctx)-> AsyncIterator[bytes]  (raw SSE)

``egress_stream`` receives the *canonical* event stream from
:mod:`k3.pipeline` and is responsible for translating it into whatever SSE
shape its client expects, including that client's own framing quirks.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Optional, Protocol

from ..ir import CanonicalRequest, CanonicalResponse, StreamEvent
from ..presets import Preset
from ..reasoning import ReasoningLedger


@dataclass(slots=True)
class DialectContext:
    """Everything a dialect needs that is not in the request body."""

    preset: Preset
    ledger: ReasoningLedger
    #: model id to report back to the client (usually what it asked for)
    client_model: str = "k3"
    #: model id actually sent to the engine
    upstream_model: str = "k3"
    request_id: str = ""
    created: int = field(default_factory=lambda: int(time.time()))
    headers: dict[str, str] = field(default_factory=dict)
    #: What the client asked for via ``stream_options.include_usage``. ``None``
    #: means it said nothing, in which case the preset default applies.
    stream_usage: Optional[bool] = None
    #: Whether the client enabled extended thinking. ``False`` means it asked
    #: for none, and the Messages API never returns thinking blocks in that
    #: case — nor accepts them back on the next turn.
    thinking_enabled: Optional[bool] = None
    #: Estimated prompt size, used where a client wants input tokens up front
    #: (Anthropic reports them in ``message_start``, before the engine has told
    #: us anything). Backfilled from real usage as soon as we have it.
    input_tokens_hint: int = 0

    def new_id(self, prefix: str) -> str:
        return f"{prefix}_{uuid.uuid4().hex[:24]}"


class Dialect(Protocol):  # pragma: no cover - structural typing only
    def ingress(self, body: dict[str, Any], ctx: DialectContext) -> CanonicalRequest: ...

    def egress(self, response: CanonicalResponse, ctx: DialectContext) -> dict[str, Any]: ...

    def egress_stream(
        self, events: AsyncIterator[StreamEvent], ctx: DialectContext
    ) -> AsyncIterator[bytes]: ...


def sse(data: Any, event: Optional[str] = None) -> bytes:
    """Encode one Server-Sent Event.

    Anthropic sends a named ``event:`` line; OpenAI sends bare ``data:``. Both
    terminate with a blank line.
    """
    if isinstance(data, (dict, list)):
        payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    else:
        payload = str(data)
    prefix = f"event: {event}\n" if event else ""
    return f"{prefix}data: {payload}\n\n".encode("utf-8")


def estimate_tokens(text: str) -> int:
    """Cheap token estimate for count_tokens and usage backfill.

    Deliberately crude: it is only ever used when the engine did not report
    usage, and every caller treats it as an estimate.
    """
    if not text:
        return 0
    return max(1, (len(text) + 3) // 4)


__all__ = ["DialectContext", "Dialect", "sse", "estimate_tokens"]
