"""Client auto-detection.

You can tell who is calling from the request itself, so ``--client`` is an
override rather than a requirement. A POST to ``/v1/messages`` carrying
``anthropic-version`` is Claude Code. ``/v1/responses`` is Codex. Everything
else on ``/v1/chat/completions`` is generic OpenAI unless a user-agent or a
vendor header says otherwise.

Scoring is deliberately blunt and explainable: a *strong* signal (user-agent or
a vendor header) makes a preset eligible, weak signals only break ties, and a
route always has a fallback so an unrecognised client still gets served. Every
decision carries a human-readable reason, which is what you actually want at
3am when a client update silently changes its user-agent.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

from . import presets as presets_mod
from .presets import FALLBACK_BY_PATH, Preset

UA_POINTS = 40
HEADER_POINTS = 25
BODY_KEY_POINTS = 2
FALLBACK_SCORE = 1


@dataclass(slots=True)
class Detection:
    preset: Preset
    score: int
    reasons: list[str] = field(default_factory=list)
    forced: bool = False
    fallback: bool = False

    @property
    def reason(self) -> str:
        if self.forced:
            return f"--client {self.preset.name}"
        if not self.reasons:
            return f"route fallback for this path"
        return ", ".join(self.reasons)


def _normalise_headers(headers: Mapping[str, str]) -> dict[str, str]:
    return {k.lower(): v for k, v in headers.items()}


def score_preset(
    preset: Preset,
    path: str,
    headers: Mapping[str, str],
    body: Optional[Mapping[str, Any]] = None,
) -> tuple[int, list[str], bool]:
    """Return ``(score, reasons, is_fallback)``; score < 0 means not eligible."""
    if path not in preset.detect.paths:
        return -1, [], False

    reasons: list[str] = []
    score = 0
    strong = False

    ua = headers.get("user-agent", "")
    for pattern in preset.detect.user_agents:
        if ua and re.search(pattern, ua):
            score += UA_POINTS
            strong = True
            reasons.append(f"user-agent ~ /{pattern}/")
            break

    for name, pattern in preset.detect.headers.items():
        value = headers.get(name)
        if value is not None and re.search(pattern, value):
            score += HEADER_POINTS
            strong = True
            reasons.append(f"header {name}")

    if strong:
        score += preset.detect.weight
        if body:
            hits = [k for k in preset.detect.body_keys if k in body]
            if hits:
                score += BODY_KEY_POINTS * len(hits)
                reasons.append(f"body keys {'+'.join(hits)}")
        return score, reasons, False

    if FALLBACK_BY_PATH.get(path) == preset.name:
        return FALLBACK_SCORE, [], True

    return -1, [], False


def detect(
    path: str,
    headers: Mapping[str, str],
    body: Optional[Mapping[str, Any]] = None,
    forced: Optional[str] = None,
) -> Detection:
    """Pick the preset for one request. ``forced`` is the ``--client`` override."""
    if forced:
        return Detection(preset=presets_mod.get(forced), score=10_000, forced=True)

    hdrs = _normalise_headers(headers)
    best: Optional[Detection] = None
    for preset in presets_mod.all_presets():
        score, reasons, is_fallback = score_preset(preset, path, hdrs, body)
        if score < 0:
            continue
        if best is None or score > best.score:
            best = Detection(preset=preset, score=score, reasons=reasons, fallback=is_fallback)

    if best is not None:
        return best

    # Route is served but no preset claims it: fall back by dialect family so a
    # request is never rejected purely because detection came up empty.
    fallback_name = FALLBACK_BY_PATH.get(path, "openai")
    return Detection(
        preset=presets_mod.get(fallback_name), score=0, fallback=True, reasons=[]
    )


__all__ = ["Detection", "detect", "score_preset"]
