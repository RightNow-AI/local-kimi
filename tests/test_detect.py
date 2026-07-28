"""Client auto-detection tests.

``--client`` is an override, not a requirement, so detection has to be right on
its own. Two properties matter more than the individual cases:

* a route always resolves to *something* usable (never an exception, never a
  preset from the wrong dialect family), and
* a preset with a higher ``weight`` must not win a route without a strong
  signal of its own — otherwise a plain ``curl`` to ``/v1/chat/completions``
  starts getting served Kimi Code's reasoning passthrough.
"""

from __future__ import annotations

import re
from typing import Any, Mapping, Optional

import pytest

from k3 import presets as presets_mod
from k3.detect import Detection, detect, score_preset

CLAUDE_HEADERS = {
    "anthropic-version": "2023-06-01",
    "user-agent": "claude-cli/1.0.60 (external, cli)",
    "content-type": "application/json",
}


def named(
    path: str,
    headers: Optional[Mapping[str, str]] = None,
    body: Optional[Mapping[str, Any]] = None,
    forced: Optional[str] = None,
) -> Detection:
    return detect(path, dict(headers or {}), body, forced)


# --------------------------------------------------------------------------
# 1-4. the two non-chat routes
# --------------------------------------------------------------------------


def test_claude_code_is_detected_from_route_plus_headers() -> None:
    d = named("/v1/messages", CLAUDE_HEADERS, {"system": "be nice", "max_tokens": 1024})

    assert d.preset.name == "claude-code"
    assert d.forced is False
    assert d.fallback is False
    assert "user-agent" in d.reason
    assert "anthropic-version" in d.reason


def test_messages_route_falls_back_to_claude_code_without_headers() -> None:
    d = named("/v1/messages", {})

    assert d.preset.name == "claude-code"
    assert d.forced is False
    assert d.fallback is True
    assert "fallback" in d.reason


def test_codex_is_detected_from_its_user_agent() -> None:
    d = named("/v1/responses", {"user-agent": "codex_cli_rs/0.20.0"})

    assert d.preset.name == "codex"
    assert d.fallback is False
    assert "user-agent" in d.reason


def test_responses_route_falls_back_to_codex_without_headers() -> None:
    d = named("/v1/responses", {})

    assert d.preset.name == "codex"
    assert d.fallback is True


# --------------------------------------------------------------------------
# 5-9. the crowded chat route
# --------------------------------------------------------------------------


def test_plain_chat_completions_client_is_generic_openai() -> None:
    """The regression that matters: weight must not beat a missing signal.

    ``kimi-code``, ``cline`` and ``opencode`` all carry weight 20 on this exact
    path. None of them may win it without a user-agent or vendor header, or a
    plain OpenAI SDK call gets served somebody else's dialect.
    """
    d = named(
        "/v1/chat/completions",
        {"user-agent": "OpenAI/Python 1.51.0", "content-type": "application/json"},
        {"model": "k3", "messages": []},
    )

    assert d.preset.name == "openai"
    assert d.preset.name not in ("kimi-code", "cline", "opencode")
    assert d.forced is False


@pytest.mark.parametrize("ua", ["python-requests/2.32.3", "curl/8.4.0", "", "Go-http-client/2.0"])
def test_unremarkable_user_agents_all_land_on_openai(ua: str) -> None:
    assert named("/v1/chat/completions", {"user-agent": ua}).preset.name == "openai"


def test_cline_is_detected_from_its_referer_header() -> None:
    d = named("/v1/chat/completions", {"http-referer": "https://cline.bot"})

    assert d.preset.name == "cline"
    assert d.fallback is False
    assert "http-referer" in d.reason


def test_kimi_code_is_detected_from_its_user_agent() -> None:
    d = named("/v1/chat/completions", {"user-agent": "kimi-cli/0.4.1"})

    assert d.preset.name == "kimi-code"
    assert d.fallback is False


def test_opencode_is_detected_from_its_user_agent() -> None:
    assert named("/v1/chat/completions", {"user-agent": "opencode/0.3.11"}).preset.name == "opencode"


def test_aider_is_detected_through_litellm() -> None:
    d = named("/v1/chat/completions", {"user-agent": "Aider/0.60.0 litellm/1.48.0"})

    assert d.preset.name == "aider"
    assert d.fallback is False


# --------------------------------------------------------------------------
# 10-13. the override, header casing, and the failure modes
# --------------------------------------------------------------------------


def test_forced_client_overrides_every_other_signal() -> None:
    d = named("/v1/messages", CLAUDE_HEADERS, forced="codex")

    assert d.preset.name == "codex"
    assert d.forced is True
    assert d.fallback is False
    assert "--client" in d.reason


def test_header_names_are_matched_case_insensitively() -> None:
    d = named(
        "/v1/messages",
        {"Anthropic-Version": "2023-06-01", "USER-AGENT": "claude-cli/1.0.60"},
    )

    assert d.preset.name == "claude-code"
    assert d.fallback is False
    assert "anthropic-version" in d.reason
    assert "user-agent" in d.reason


def test_unknown_path_returns_a_usable_fallback_instead_of_raising() -> None:
    d = named("/v1/embeddings", {"user-agent": "whatever/1.0"})

    assert d.preset is not None
    assert d.preset.name in presets_mod.names()
    assert d.fallback is True
    assert d.forced is False


def test_forced_unknown_preset_raises() -> None:
    with pytest.raises(ValueError):
        named("/v1/chat/completions", {}, forced="nope")


# --------------------------------------------------------------------------
# 14. /v1/models is shape-sensitive
# --------------------------------------------------------------------------


def test_models_route_resolves_per_dialect() -> None:
    """The two dialects return different JSON, so this must not be sticky."""
    assert named("/v1/models", CLAUDE_HEADERS).preset.name == "claude-code"
    assert named("/v1/models", {}).preset.name == "openai"

    claude = presets_mod.get("claude-code")
    openai = presets_mod.get("openai")
    assert claude.dialect != openai.dialect


# --------------------------------------------------------------------------
# 15-16. properties over the whole preset table
# --------------------------------------------------------------------------

#: A regex is not a literal, so each preset gets a hand-built user-agent that
#: its own first pattern matches. Presets absent from this map are skipped.
UA_SAMPLES = {
    "claude-code": "claude-cli/1.0.60",       # r"claude-cli/"
    "codex": "codex_cli_rs/0.20.0",           # r"codex[_-]cli"
    "kimi-code": "kimi-cli/0.4.1",            # r"kimi[_-]?(code|cli)"
    "cline": "cline/3.2.0",                   # r"(?i)cline"
    "opencode": "opencode/0.3.11",            # r"(?i)opencode"
    "aider": "Aider/0.60.0",                  # r"(?i)aider"
}


@pytest.mark.parametrize(
    "preset", presets_mod.all_presets(), ids=[p.name for p in presets_mod.all_presets()]
)
def test_every_preset_detects_its_own_user_agent(preset) -> None:
    if not preset.detect.user_agents:
        pytest.skip(f"{preset.name} has no user-agent rules")
    sample = UA_SAMPLES.get(preset.name)
    if sample is None:
        # No hand-built sample for this pattern; add one when the preset lands.
        pytest.skip(f"no literal user-agent sample for {preset.name}")

    pattern = preset.detect.user_agents[0]
    assert re.search(pattern, sample), f"sample {sample!r} does not match /{pattern}/"

    paths = [p for p in preset.detect.paths if p != "/v1/models"]
    assert paths, f"{preset.name} only detects on /v1/models"

    d = named(paths[0], {"user-agent": sample})
    assert d.preset.name == preset.name, (
        f"{sample!r} on {paths[0]} resolved to {d.preset.name} ({d.reason})"
    )
    assert d.fallback is False


def test_presets_validate_clean() -> None:
    assert presets_mod.validate() == []


# --------------------------------------------------------------------------
# score_preset directly
# --------------------------------------------------------------------------


def test_score_preset_rejects_a_path_it_does_not_serve() -> None:
    codex = presets_mod.get("codex")
    score, reasons, is_fallback = score_preset(codex, "/v1/chat/completions", {})

    assert score < 0
    assert reasons == []
    assert is_fallback is False


def test_score_preset_marks_the_route_fallback_without_signals() -> None:
    claude = presets_mod.get("claude-code")
    score, reasons, is_fallback = score_preset(claude, "/v1/messages", {})

    assert is_fallback is True
    assert score > 0
    assert reasons == []


def test_a_strong_signal_outscores_the_route_fallback() -> None:
    claude = presets_mod.get("claude-code")
    strong, _, _ = score_preset(claude, "/v1/messages", {"anthropic-version": "2023-06-01"})
    weak, _, _ = score_preset(claude, "/v1/messages", {})

    assert strong > weak


def test_body_keys_only_count_once_a_strong_signal_exists() -> None:
    kimi = presets_mod.get("kimi-code")
    body = {"model": "k3", "messages": []}
    without, _, _ = score_preset(kimi, "/v1/chat/completions", {}, body)
    with_ua, reasons, _ = score_preset(
        kimi, "/v1/chat/completions", {"user-agent": "kimi-cli/0.4.1"}, body
    )

    assert without < 0
    assert with_ua > 0
    assert any("user-agent" in r for r in reasons)
