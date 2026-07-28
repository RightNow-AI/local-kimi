"""Reasoning translation, in both directions.

K3 emits ``reasoning_content`` and expects the complete assistant message back,
verbatim, on the next turn. Every client wants that in a different shape:

    Claude Code   ``thinking`` content blocks with a ``signature``
    Codex         ``reasoning`` output items with ``encrypted_content``
    OpenAI chat   nothing at all — the field is dropped on the floor
    Kimi Code     ``reasoning_content`` passed straight through

So we need a vehicle that survives each of those shapes and lets us rebuild the
exact bytes on the way back in. We use three, in order of preference:

1. **Self-contained signature.** ``k3r1.<b64url(zlib(json))>`` carries the
   reasoning text itself plus a ledger id. Survives process restarts, survives
   the client dropping the visible thinking text, needs no shared state. This
   is the vehicle for ``signature`` (Anthropic) and ``encrypted_content``
   (Responses).
2. **Ledger lookup.** The signature's ledger id resolves to the *complete*
   upstream assistant message — including raw tool-call argument strings, which
   the client cannot round-trip losslessly because it parses them into objects.
   Strictly better than (1) when it hits; (1) is the fallback when it misses.
3. **Fingerprint recovery.** For clients that strip reasoning entirely, we hash
   the visible parts of the assistant turn (text + tool names + tool args) and
   look the reasoning up by that. This is what keeps K3 from degrading across
   agent loops on a plain OpenAI client.

If all three miss we degrade to whatever visible text we have, which is exactly
the pre-k3 behaviour — never worse.
"""

from __future__ import annotations

import base64
import hashlib
import json
import threading
import time
import uuid
import zlib
from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from .ir import Message, ReasoningPart, TextPart, ToolCallPart

SIG_SELF_CONTAINED = "k3r1."
SIG_LEDGER_ONLY = "k3l1."

#: Above this many bytes we stop inlining reasoning into the signature and fall
#: back to a ledger reference. Clients do carry multi-kilobyte signatures fine
#: (real Anthropic ones are ~1-2 KiB), but there is no reason to ship a 400 KiB
#: header-sized blob through an agent loop.
MAX_INLINE_SIGNATURE_BYTES = 64 * 1024

#: Hard ceiling on the *decompressed* size of a signature we will materialise.
#: The server is open by default and ``decode_signature`` runs on fully
#: attacker-controlled bytes for every assistant message of every request, so an
#: uncapped ``zlib.decompress`` is a decompression bomb: a 340 KiB payload
#: expands to 256 MiB. 8 MiB is roughly two million tokens of reasoning — orders
#: of magnitude past anything a real model turn can produce (K3's own context is
#: a small fraction of that), so this can only ever reject an attack.
MAX_DECOMPRESSED_SIGNATURE_BYTES = 8 * 1024 * 1024

#: Base64 expands 3 bytes to 4 chars, so this is the longest payload our own
#: encoder can emit. Checked before decoding, so a hostile signature cannot make
#: us allocate its decoded form either.
_MAX_SIGNATURE_B64_CHARS = 4 * ((MAX_INLINE_SIGNATURE_BYTES + 2) // 3)


class ReasoningPolicy(str, Enum):
    """How a preset surfaces K3 reasoning to its client."""

    #: Anthropic ``thinking`` content blocks (Claude Code).
    THINKING_BLOCKS = "thinking_blocks"
    #: Responses API ``reasoning`` items (Codex).
    RESPONSES_ITEM = "responses_item"
    #: OpenAI chat ``delta.reasoning_content`` passthrough (Kimi Code, Cline).
    REASONING_CONTENT = "reasoning_content"
    #: Inline ``<think>…</think>`` inside the visible content.
    INLINE_TAGS = "inline_tags"
    #: Drop it. Ledger fingerprinting still restores it upstream.
    STRIP = "strip"


# --------------------------------------------------------------------------
# signature codec
# --------------------------------------------------------------------------


def encode_signature(reasoning: str, ledger_id: str) -> str:
    """Pack reasoning + ledger id into an opaque, client-safe token."""
    payload = json.dumps({"r": reasoning, "i": ledger_id}, ensure_ascii=False, separators=(",", ":"))
    blob = zlib.compress(payload.encode("utf-8"), 6)
    if len(blob) > MAX_INLINE_SIGNATURE_BYTES:
        return SIG_LEDGER_ONLY + ledger_id
    return SIG_SELF_CONTAINED + base64.urlsafe_b64encode(blob).decode("ascii")


def decode_signature(signature: Optional[str]) -> tuple[Optional[str], Optional[str]]:
    """Return ``(reasoning_text, ledger_id)`` for a signature we minted.

    Unknown or malformed signatures (including genuine Anthropic ones, if a
    client ever replays a transcript recorded against the real API) return
    ``(None, None)`` rather than raising — a bad signature must never take down
    a request.
    """
    if not signature:
        return None, None
    if signature.startswith(SIG_LEDGER_ONLY):
        return None, signature[len(SIG_LEDGER_ONLY) :] or None
    if not signature.startswith(SIG_SELF_CONTAINED):
        return None, None
    raw = signature[len(SIG_SELF_CONTAINED) :]
    # ``encode_signature`` falls back to a ledger-only signature above
    # MAX_INLINE_SIGNATURE_BYTES, so a longer payload cannot be one of ours.
    if len(raw) > _MAX_SIGNATURE_B64_CHARS:
        return None, None
    try:
        pad = "=" * (-len(raw) % 4)
        blob = base64.urlsafe_b64decode(raw + pad)
        if len(blob) > MAX_INLINE_SIGNATURE_BYTES:
            return None, None
        # Bounded, incremental inflate rather than a single unbounded call.
        inflater = zlib.decompressobj()
        data = inflater.decompress(blob, MAX_DECOMPRESSED_SIGNATURE_BYTES)
        if inflater.unconsumed_tail or not inflater.eof:
            # ``unconsumed_tail``: the blob wanted to expand past the cap, i.e.
            # a bomb rather than our payload. ``not eof``: the stream never
            # reached its end marker, so it is truncated — ``zlib.decompress``
            # raised on that, and we must stay just as strict.
            return None, None
        payload = json.loads(data.decode("utf-8"))
    except Exception:
        return None, None
    if not isinstance(payload, dict):
        return None, None
    reasoning = payload.get("r")
    ledger_id = payload.get("i")
    return (
        reasoning if isinstance(reasoning, str) else None,
        ledger_id if isinstance(ledger_id, str) else None,
    )


# --------------------------------------------------------------------------
# fingerprinting
# --------------------------------------------------------------------------


def fingerprint_parts(text: str, tool_calls: list[tuple[str, str]]) -> str:
    """Hash the parts of an assistant turn that every client round-trips.

    ``tool_calls`` is ``[(name, arguments_json)]``. Arguments are normalised
    through ``json.loads`` first, because a client that parses and re-serialises
    them changes whitespace — the whole reason we need this fallback.
    """
    h = hashlib.sha256()
    h.update(text.strip().encode("utf-8"))
    for name, args in tool_calls:
        h.update(b"\x00")
        h.update(name.encode("utf-8"))
        h.update(b"\x00")
        h.update(_normalise_args(args).encode("utf-8"))
    return h.hexdigest()


def _normalise_args(args: str) -> str:
    raw = (args or "").strip()
    if not raw:
        return ""
    try:
        return json.dumps(json.loads(raw), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    except json.JSONDecodeError:
        return raw


def fingerprint_message(msg: Message) -> str:
    return fingerprint_parts(
        msg.text(),
        [(tc.name, tc.arguments) for tc in msg.tool_calls()],
    )


# --------------------------------------------------------------------------
# ledger
# --------------------------------------------------------------------------


@dataclass
class LedgerEntry:
    id: str
    reasoning: str
    created_at: float
    #: the exact assistant message dict to send back to K3
    upstream_message: dict[str, Any] = field(default_factory=dict)
    fingerprint: Optional[str] = None


class ReasoningLedger:
    """Bounded, TTL'd store of assistant turns keyed by id and fingerprint.

    Thread-safe: uvicorn may run the app across threads, and the recorder reads
    it out of band.
    """

    def __init__(self, max_entries: int = 4096, ttl_seconds: float = 6 * 3600) -> None:
        self.max_entries = max_entries
        self.ttl_seconds = ttl_seconds
        self._by_id: "OrderedDict[str, LedgerEntry]" = OrderedDict()
        self._by_fingerprint: dict[str, str] = {}
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0
        self.fingerprint_hits = 0

    # -- writing ----------------------------------------------------------

    def new_id(self) -> str:
        return uuid.uuid4().hex

    def reserve(self, ledger_id: str, reasoning: str) -> LedgerEntry:
        """Record reasoning as soon as it finishes, before the turn completes."""
        entry = LedgerEntry(id=ledger_id, reasoning=reasoning, created_at=time.time())
        with self._lock:
            self._by_id[ledger_id] = entry
            self._by_id.move_to_end(ledger_id)
            self._evict_locked()
        return entry

    def complete(self, ledger_id: str, upstream_message: dict[str, Any]) -> None:
        """Attach the finished assistant message once the turn is done."""
        fp = _fingerprint_upstream(upstream_message)
        with self._lock:
            entry = self._by_id.get(ledger_id)
            if entry is None:
                entry = LedgerEntry(
                    id=ledger_id,
                    reasoning=str(upstream_message.get("reasoning_content") or ""),
                    created_at=time.time(),
                )
                self._by_id[ledger_id] = entry
            entry.upstream_message = upstream_message

            # The index maps a fingerprint to exactly one entry, and collisions
            # are routine: two turns with identical visible output (a repeated
            # ``ls``, a repeated tool-only turn) fingerprint the same. Hand the
            # mapping to the newest entry, but first clear it off the previous
            # owner — ``_drop_locked``/``_evict_locked`` pop whatever
            # ``entry.fingerprint`` says, so leaving it set there would later
            # delete a mapping that now points at a different, still-live entry.
            previous_id = self._by_fingerprint.get(fp)
            if previous_id is not None and previous_id != ledger_id:
                previous = self._by_id.get(previous_id)
                if previous is not None and previous.fingerprint == fp:
                    previous.fingerprint = None
            # Symmetrically, a re-completed entry no longer owns its old index
            # slot; retire it so nothing dangles.
            if entry.fingerprint and entry.fingerprint != fp:
                if self._by_fingerprint.get(entry.fingerprint) == ledger_id:
                    del self._by_fingerprint[entry.fingerprint]

            entry.fingerprint = fp
            self._by_fingerprint[fp] = ledger_id
            self._by_id.move_to_end(ledger_id)
            self._evict_locked()

    # -- reading ----------------------------------------------------------

    def get(self, ledger_id: Optional[str]) -> Optional[LedgerEntry]:
        if not ledger_id:
            return None
        with self._lock:
            entry = self._by_id.get(ledger_id)
            if entry is None:
                self.misses += 1
                return None
            if self._expired(entry):
                self._drop_locked(ledger_id)
                self.misses += 1
                return None
            self._by_id.move_to_end(ledger_id)
            self.hits += 1
            return entry

    def find_by_fingerprint(self, fp: str) -> Optional[LedgerEntry]:
        with self._lock:
            ledger_id = self._by_fingerprint.get(fp)
            if not ledger_id:
                return None
            entry = self._by_id.get(ledger_id)
            if entry is None or self._expired(entry):
                self._by_fingerprint.pop(fp, None)
                return None
            self._by_id.move_to_end(ledger_id)
            self.fingerprint_hits += 1
            return entry

    # -- housekeeping -----------------------------------------------------

    def stats(self) -> dict[str, int]:
        with self._lock:
            return {
                "entries": len(self._by_id),
                "hits": self.hits,
                "misses": self.misses,
                "fingerprint_hits": self.fingerprint_hits,
            }

    def clear(self) -> None:
        with self._lock:
            self._by_id.clear()
            self._by_fingerprint.clear()

    def _expired(self, entry: LedgerEntry) -> bool:
        return self.ttl_seconds > 0 and (time.time() - entry.created_at) > self.ttl_seconds

    def _drop_locked(self, ledger_id: str) -> None:
        entry = self._by_id.pop(ledger_id, None)
        if entry and entry.fingerprint:
            self._by_fingerprint.pop(entry.fingerprint, None)

    def _evict_locked(self) -> None:
        while len(self._by_id) > self.max_entries:
            oldest, entry = self._by_id.popitem(last=False)
            if entry.fingerprint:
                self._by_fingerprint.pop(entry.fingerprint, None)


def _fingerprint_upstream(msg: dict[str, Any]) -> str:
    content = msg.get("content") or ""
    if not isinstance(content, str):
        content = json.dumps(content, sort_keys=True)
    calls: list[tuple[str, str]] = []
    for tc in msg.get("tool_calls") or []:
        fn = tc.get("function") or {}
        calls.append((str(fn.get("name") or ""), str(fn.get("arguments") or "")))
    return fingerprint_parts(content, calls)


# --------------------------------------------------------------------------
# restoration
# --------------------------------------------------------------------------


@dataclass
class Restored:
    """What we managed to recover for one assistant turn on the way back in."""

    reasoning: str = ""
    upstream_message: Optional[dict[str, Any]] = None
    source: str = "none"  # ledger | signature | echo | fingerprint | none

    @property
    def exact(self) -> bool:
        return self.source == "ledger"


def restore_assistant(msg: Message, ledger: ReasoningLedger) -> Restored:
    """Recover K3's original assistant message from a client's echo of it.

    Tries, in order: ledger by signature id, self-contained signature, visible
    thinking text, ledger by fingerprint.
    """
    reasoning_parts = [p for p in msg.parts if isinstance(p, ReasoningPart)]

    # A ledger hit carries the whole turn — including raw tool-call argument
    # strings the client cannot round-trip — so it wins outright.
    inline_pieces: list[str] = []
    ledger_reasoning = ""
    for part in reasoning_parts:
        inline, ledger_id = decode_signature(part.signature)
        entry = ledger.get(ledger_id)
        if entry is not None and entry.upstream_message:
            return Restored(
                reasoning=entry.reasoning,
                upstream_message=dict(entry.upstream_message),
                source="ledger",
            )
        if inline:
            inline_pieces.append(inline)
        elif entry is not None and entry.reasoning:
            ledger_reasoning = entry.reasoning

    # Interleaved thinking arrives as several blocks; each signature carries its
    # own segment, so the turn's reasoning is their concatenation.
    if inline_pieces:
        return Restored(reasoning="".join(inline_pieces), source="signature")
    if ledger_reasoning:
        return Restored(reasoning=ledger_reasoning, source="ledger")

    echoed = "".join(p.text for p in reasoning_parts if p.text and not p.redacted)
    if echoed:
        return Restored(reasoning=echoed, source="echo")

    fp = fingerprint_message(msg)
    entry = ledger.find_by_fingerprint(fp)
    if entry is not None:
        if entry.upstream_message:
            return Restored(
                reasoning=entry.reasoning,
                upstream_message=dict(entry.upstream_message),
                source="fingerprint",
            )
        return Restored(reasoning=entry.reasoning, source="fingerprint")

    return Restored()


def build_upstream_assistant(
    msg: Message,
    restored: Restored,
    reasoning_field: str = "reasoning_content",
) -> dict[str, Any]:
    """Assemble the assistant message K3 should see for this turn.

    A ledger hit wins outright — those are the original bytes. Otherwise we
    rebuild from the client's echo and splice the recovered reasoning back in.
    """
    if restored.upstream_message:
        out = dict(restored.upstream_message)
        # ``none`` and ``inline`` are sentinels, not field names — moving the
        # reasoning to a key literally called "none" would leak it upstream.
        reasoning = out.pop("reasoning_content", None)
        if reasoning:
            if reasoning_field == "inline":
                out["content"] = f"<think>{reasoning}</think>{out.get('content') or ''}"
            elif reasoning_field != "none":
                out[reasoning_field] = reasoning
        return out

    out: dict[str, Any] = {"role": "assistant"}
    text = msg.text()
    tool_calls = msg.tool_calls()

    if restored.reasoning and reasoning_field == "inline":
        out["content"] = f"<think>{restored.reasoning}</think>{text}"
    else:
        out["content"] = text or None
        if restored.reasoning and reasoning_field not in ("none", "inline"):
            out[reasoning_field] = restored.reasoning

    if tool_calls:
        out["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.name, "arguments": tc.arguments},
            }
            for tc in tool_calls
        ]
        if out.get("content") is None:
            out["content"] = ""
    elif out.get("content") is None:
        out["content"] = ""

    return out


def upstream_assistant_from_response(
    text: str,
    reasoning: str,
    tool_calls: list[ToolCallPart],
    reasoning_field: str = "reasoning_content",
) -> dict[str, Any]:
    """The assistant message we will want echoed back on the next turn."""
    out: dict[str, Any] = {"role": "assistant", "content": text}
    if reasoning and reasoning_field not in ("none", "inline"):
        out[reasoning_field] = reasoning
    elif reasoning and reasoning_field == "inline":
        out["content"] = f"<think>{reasoning}</think>{text}"
    if tool_calls:
        out["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.name, "arguments": tc.arguments},
            }
            for tc in tool_calls
        ]
    return out


def strip_inline_think(text: str) -> tuple[str, str]:
    """Split ``<think>…</think>`` prefix out of content into (reasoning, rest)."""
    if "<think>" not in text:
        return "", text
    start = text.index("<think>")
    end = text.find("</think>", start)
    if end == -1:
        return text[start + len("<think>") :], text[:start]
    reasoning = text[start + len("<think>") : end]
    rest = text[:start] + text[end + len("</think>") :]
    return reasoning, rest


__all__ = [
    "ReasoningPolicy",
    "ReasoningLedger",
    "LedgerEntry",
    "Restored",
    "encode_signature",
    "decode_signature",
    "fingerprint_parts",
    "fingerprint_message",
    "restore_assistant",
    "build_upstream_assistant",
    "upstream_assistant_from_response",
    "strip_inline_think",
    "MAX_INLINE_SIGNATURE_BYTES",
    "MAX_DECOMPRESSED_SIGNATURE_BYTES",
]
