"""The conformance suite.

Every cassette in ``tests/cassettes`` is replayed against the current code with
the engine stubbed out by its own recording. A change to any preset, dialect,
parser, or the reasoning translation shows up here as a diff.

When one of these fails, read the diff before "fixing" the cassette. A cassette
is a statement about what a real client accepted.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from k3 import presets as presets_mod
from k3.record import load_cassettes
from k3.replay import replay_cassette

CASSETTE_DIR = Path(__file__).parent / "cassettes"
CASSETTES = load_cassettes(CASSETTE_DIR)


def _ids(cassette) -> str:
    return cassette.name or f"{cassette.preset}-{cassette.path}"


def test_cassettes_exist() -> None:
    assert CASSETTES, f"no cassettes in {CASSETTE_DIR}; run scripts/build_fixtures.py"


@pytest.mark.parametrize("cassette", CASSETTES, ids=[_ids(c) for c in CASSETTES])
async def test_cassette_replays_identically(cassette) -> None:
    result = await replay_cassette(cassette)
    assert result.ok, "\n  ".join(["client-facing output changed:", *result.diffs[:15]])


@pytest.mark.parametrize("cassette", CASSETTES, ids=[_ids(c) for c in CASSETTES])
def test_cassette_has_no_credentials(cassette) -> None:
    """Cassettes are meant to be committed, so this must hold for every one."""
    for name, value in (cassette.request_headers or {}).items():
        if name in ("authorization", "x-api-key", "cookie", "proxy-authorization"):
            assert value == "<redacted>", f"{name} leaked into {cassette.name}"


def test_every_preset_is_covered() -> None:
    """A preset with no cassette is a preset nothing is pinning."""
    covered = {c.preset for c in CASSETTES}
    missing = {p.name for p in presets_mod.all_presets()} - covered
    assert not missing, f"presets with no cassette: {sorted(missing)}"


def test_stable_presets_have_recorded_traffic() -> None:
    """``stable`` is a claim about real traffic, not synthesized bodies."""
    recorded = {c.preset for c in CASSETTES if c.source == "recorded"}
    for preset in presets_mod.all_presets():
        if preset.status == "stable":
            assert preset.name in recorded, (
                f"{preset.name} is marked stable but has no recorded cassette"
            )


def test_recorded_cassettes_exercise_reasoning_restoration() -> None:
    """At least one real capture must send reasoning back to the engine.

    This is the whole point of the project: if no cassette proves reasoning
    survived a client round trip, nothing here is testing the hard part.
    """
    proven = []
    for cassette in CASSETTES:
        if cassette.source != "recorded":
            continue
        payload = cassette.upstream_request or {}
        for msg in payload.get("messages") or []:
            if isinstance(msg, dict) and msg.get("reasoning_content"):
                proven.append(cassette.name)
                break
    assert proven, "no recorded cassette carries reasoning_content back upstream"
