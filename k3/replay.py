"""Replay recorded traffic against the current code.

A cassette pins the *client-facing bytes* for a request. Replaying re-runs it
with the engine stubbed out by its own recording, so any change in a preset,
dialect, or parser shows up as a diff instead of a GitHub issue.

Some fields are legitimately non-deterministic — message ids, timestamps, the
random ledger id inside a reasoning signature, tool-call ids. Those are
normalised rather than ignored: a signature is decoded and replaced by a hash of
*the reasoning it carries*, so the comparison still fails if the reasoning
payload changes. That is the invariant worth pinning.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any, Optional

import httpx

from .detect import detect
from .reasoning import decode_signature
from .record import Cassette, ReplayEngine, parse_sse
from .server import ServerConfig, create_app
from .upstream import UpstreamConfig

#: Header names we never replay (they describe the recording, not the request).
DROP_HEADERS = {"host", "content-length", "authorization", "x-api-key", "accept-encoding"}

_ID_FIELDS = {"id", "message_id", "item_id", "response_id"}
_TIME_FIELDS = {"created", "created_at", "recorded_at"}
_SIG_FIELDS = {"signature", "encrypted_content", "data"}
_TOOL_ID_FIELDS = {"tool_use_id", "call_id", "tool_call_id"}

_ID_PREFIXES = ("msg_", "chatcmpl-", "chatcmpl_", "resp_", "rs_", "fc_", "call_", "toolu_", "cmpl-")
_ID_RE = re.compile(
    r"\b(?:"
    + "|".join(re.escape(prefix) for prefix in _ID_PREFIXES)
    + r")[A-Za-z0-9_-]{6,}\b"
)


def _norm_id(value: str, ids: dict[str, str]) -> str:
    for prefix in _ID_PREFIXES:
        if value.startswith(prefix):
            if value not in ids:
                ids[value] = f"<id:{len(ids) + 1}>"
            return ids[value]
    return value


def _norm_signature(value: str) -> str:
    """Replace a signature with a stable digest of the reasoning it carries."""
    reasoning, _ledger_id = decode_signature(value)
    if reasoning is not None:
        digest = hashlib.sha256(reasoning.encode("utf-8")).hexdigest()[:16]
        return f"<sig:{digest}>"
    if value.startswith("k3l1."):
        return "<sig:ledger-only>"
    return value


def normalize(obj: Any) -> Any:
    """Strip non-determinism while preserving identity relationships and meaning."""
    ids: dict[str, str] = {}

    def walk(value: Any) -> Any:
        if isinstance(value, dict):
            out: dict[str, Any] = {}
            for key, item in value.items():
                if key in _TIME_FIELDS and isinstance(item, (int, float, str)):
                    out[key] = 0
                elif key in _SIG_FIELDS and isinstance(item, str) and item:
                    out[key] = _norm_signature(item)
                elif key in _ID_FIELDS and isinstance(item, str):
                    out[key] = _norm_id(item, ids)
                elif key in _TOOL_ID_FIELDS and isinstance(item, str):
                    out[key] = _norm_id(item, ids)
                else:
                    out[key] = walk(item)
            return out
        if isinstance(value, list):
            return [walk(item) for item in value]
        if isinstance(value, str):
            return _ID_RE.sub(lambda match: _norm_id(match.group(0), ids), value)
        return value

    return walk(obj)


def diff_json(expected: Any, actual: Any, path: str = "$", limit: int = 25) -> list[str]:
    """Readable, path-anchored differences; capped so output stays usable."""
    diffs: list[str] = []

    def walk(exp: Any, act: Any, where: str) -> None:
        if len(diffs) >= limit:
            return
        if type(exp) is not type(act) and not (
            isinstance(exp, (int, float)) and isinstance(act, (int, float))
        ):
            diffs.append(f"{where}: type {type(exp).__name__} != {type(act).__name__}")
            return
        if isinstance(exp, dict):
            # sorted(): the cap means only the first `limit` diffs are shown, so
            # set iteration order would make truncated output nondeterministic.
            for key in sorted(exp.keys() | act.keys(), key=str):
                if len(diffs) >= limit:
                    return
                if key not in exp:
                    diffs.append(f"{where}.{key}: unexpected {act[key]!r}")
                elif key not in act:
                    diffs.append(f"{where}.{key}: missing (expected {exp[key]!r})")
                else:
                    walk(exp[key], act[key], f"{where}.{key}")
        elif isinstance(exp, list):
            if len(exp) != len(act):
                diffs.append(f"{where}: length {len(exp)} != {len(act)}")
            for i, (a, b) in enumerate(zip(exp, act)):
                if len(diffs) >= limit:
                    return
                walk(a, b, f"{where}[{i}]")
        elif exp != act:
            diffs.append(f"{where}: {exp!r} != {act!r}")

    walk(expected, actual, path)
    return diffs


def normalize_upstream(payload: Any, drop_reasoning: bool = False) -> Any:
    """Normalise the engine payload before comparing it.

    Tool-call argument *whitespace* is allowed to differ: when the ledger is
    cold (as it always is on replay) arguments are rebuilt from the client's
    parsed copy, and that is acceptable degradation. Reasoning content is
    compared exactly, because losing it is the failure this suite exists to
    catch.

    ``drop_reasoning`` is set only for presets whose client has no reasoning
    vehicle at all (policy ``strip``). For those, restoration runs entirely off
    the server-side fingerprint index, which a cold replay cannot reproduce by
    construction — so comparing it would assert something false. Every other
    preset carries a signature through the client and *must* restore reasoning
    even on a cold server; that is the assertion worth having, and it stays.
    """
    if not isinstance(payload, dict):
        return payload
    out = dict(payload)
    messages = []
    for msg in out.get("messages") or []:
        if not isinstance(msg, dict):
            messages.append(msg)
            continue
        msg = dict(msg)
        if drop_reasoning:
            msg.pop("reasoning_content", None)
        calls = []
        for call in msg.get("tool_calls") or []:
            call = dict(call)
            fn = dict(call.get("function") or {})
            raw = fn.get("arguments")
            if isinstance(raw, str):
                try:
                    fn["arguments"] = _canonical_json(raw)
                except ValueError:
                    pass
            call["function"] = fn
            call.pop("id", None)  # ids are regenerated per run
            calls.append(call)
        if calls:
            msg["tool_calls"] = calls
        msg.pop("tool_call_id", None)
        messages.append(msg)
    if messages:
        out["messages"] = messages
    return out


def _is_ledger_only_preset(name: str) -> bool:
    from . import presets as presets_mod
    from .reasoning import ReasoningPolicy

    try:
        return presets_mod.get(name).reasoning is ReasoningPolicy.STRIP
    except ValueError:
        return False


def _canonical_json(raw: str) -> str:
    import json as _json

    return _json.dumps(_json.loads(raw), sort_keys=True, separators=(",", ":"), ensure_ascii=False)


@dataclass
class ReplayResult:
    cassette: Cassette
    ok: bool
    diffs: list[str] = field(default_factory=list)
    expected_preset: str = ""
    actual_preset: str = ""
    status: int = 0

    @property
    def name(self) -> str:
        return self.cassette.name or f"{self.cassette.preset}{self.cassette.path}"


def _replay_headers(cassette: Cassette) -> dict[str, str]:
    return {
        k: v
        for k, v in (cassette.request_headers or {}).items()
        if k.lower() not in DROP_HEADERS and v != "<redacted>"
    }


async def replay_cassette(
    cassette: Cassette, *, check_detection: bool = True, check_upstream: bool = True
) -> ReplayResult:
    """Re-run one cassette against the current code."""
    headers = _replay_headers(cassette)

    actual_preset = ""
    if check_detection:
        detection = detect(cassette.path, headers, cassette.request_body)
        actual_preset = detection.preset.name

    cfg = ServerConfig(
        upstream=UpstreamConfig(model="k3"),
        # Detection is part of what we are testing, so nothing is forced here.
        forced_client=None,
    )
    engine = ReplayEngine(cassette)
    app = create_app(cfg, engine=engine)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://k3.test") as client:
        response = await client.post(cassette.path, json=cassette.request_body, headers=headers)

    diffs: list[str] = []
    if check_detection and actual_preset and cassette.preset and actual_preset != cassette.preset:
        diffs.append(f"detection: recorded {cassette.preset!r}, now {actual_preset!r}")

    if response.status_code != cassette.client_status:
        diffs.append(f"status: {cassette.client_status} != {response.status_code}")

    if check_upstream and cassette.upstream_request is not None:
        drop = _is_ledger_only_preset(cassette.preset)
        expected_up = normalize(normalize_upstream(cassette.upstream_request, drop))
        actual_up = normalize(normalize_upstream(engine.seen_payload, drop))
        diffs.extend(diff_json(expected_up, actual_up, "$.upstream"))

    if cassette.streamed:
        expected = normalize(parse_sse("".join(cassette.client_sse)))
        actual = normalize(parse_sse(response.text))
        diffs.extend(diff_json(expected, actual, "$.sse"))
    else:
        expected = normalize(cassette.client_body)
        try:
            actual = normalize(response.json())
        except ValueError:
            actual = response.text
        diffs.extend(diff_json(expected, actual, "$.body"))

    return ReplayResult(
        cassette=cassette,
        ok=not diffs,
        diffs=diffs,
        expected_preset=cassette.preset,
        actual_preset=actual_preset,
        status=response.status_code,
    )


async def replay_all(
    cassettes: list[Cassette], *, check_detection: bool = True, check_upstream: bool = True
) -> list[ReplayResult]:
    results = []
    for cassette in cassettes:
        results.append(
            await replay_cassette(
                cassette, check_detection=check_detection, check_upstream=check_upstream
            )
        )
    return results


__all__ = [
    "replay_cassette",
    "replay_all",
    "ReplayResult",
    "normalize",
    "normalize_upstream",
    "diff_json",
]
