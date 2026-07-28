"""The command line.

``k3`` is a CLI first: the paste block, the preset list, and the two diagnostic
commands (``detect``, ``doctor``) are what somebody runs before they trust the
proxy with a real agent loop. These tests drive the Typer app in-process.

``serve`` is deliberately never invoked — it ends in ``uvicorn.run`` and would
block forever. Everything *around* it is tested instead, including the preset
validation that runs before uvicorn is ever imported.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

import k3
from k3 import cli
from k3 import presets as presets_mod
from k3.cli import app
from k3.record import load_cassettes

runner = CliRunner()

CASSETTE_DIR = Path(__file__).parent / "cassettes"


@pytest.fixture(autouse=True)
def _wide_console(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stop rich from wrapping assertions into two lines.

    ``k3.cli.console`` is built at import time but reads ``COLUMNS`` off
    ``os.environ`` on every render, so setting it here is enough.
    """
    monkeypatch.setenv("COLUMNS", "300")
    monkeypatch.setenv("LINES", "80")


def flat(text: str) -> str:
    """Collapse rich's wrapping so substring assertions survive any width."""
    return " ".join(text.split())


# --------------------------------------------------------------------------
# presets
# --------------------------------------------------------------------------


def test_presets_lists_every_preset() -> None:
    result = runner.invoke(app, ["presets"])
    assert result.exit_code == 0, result.output
    for name in presets_mod.names():
        assert name in result.output, f"{name} missing from `k3 presets`"


def test_presets_verbose_adds_the_notes() -> None:
    plain = runner.invoke(app, ["presets"])
    verbose = runner.invoke(app, ["presets", "--verbose"])
    assert verbose.exit_code == 0, verbose.output

    body = flat(verbose.output)
    documented = [p for p in presets_mod.all_presets() if p.notes]
    assert documented, "no preset carries notes; this test would prove nothing"
    for preset in documented:
        assert flat(preset.notes) in body, f"notes for {preset.name} missing from -v output"
        assert flat(preset.notes) not in flat(plain.output), (
            f"notes for {preset.name} already printed without --verbose"
        )
    # --verbose is strictly additive.
    for name in presets_mod.names():
        assert name in verbose.output


# --------------------------------------------------------------------------
# version
# --------------------------------------------------------------------------


def test_version_prints_the_package_version() -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0, result.output
    assert result.output.strip() == k3.__version__


# --------------------------------------------------------------------------
# detect
# --------------------------------------------------------------------------


def test_detect_identifies_claude_code_from_its_headers() -> None:
    result = runner.invoke(
        app,
        [
            "detect",
            "--path",
            "/v1/messages",
            "-H",
            "user-agent: claude-cli/1.0.60",
            "-H",
            "anthropic-version: 2023-06-01",
        ],
    )
    assert result.exit_code == 0, result.output
    body = flat(result.output)
    assert "claude-code" in body
    # The reason is the point of the command: it must say *why*, not just who.
    assert "user-agent" in body
    assert "anthropic_messages" in body


def test_detect_rejects_a_header_without_a_colon() -> None:
    result = runner.invoke(app, ["detect", "-H", "user-agent claude-cli/1.0.60"])
    assert result.exit_code == 2, result.output
    body = flat(result.output)
    assert "bad header" in body
    assert "Name: value" in body


# --------------------------------------------------------------------------
# doctor
# --------------------------------------------------------------------------


def test_doctor_mock_reports_presets_and_a_reachable_engine() -> None:
    result = runner.invoke(app, ["doctor", "--mock"])
    assert result.exit_code == 0, result.output
    body = flat(result.output)
    assert f"{len(presets_mod.all_presets())} presets validate" in body
    assert "engine reachable at" in body
    assert "unreachable" not in body


def test_doctor_fails_when_the_engine_is_dead() -> None:
    """Port 9 (discard) refuses immediately, so this stays fast."""
    result = runner.invoke(app, ["doctor", "--upstream", "http://127.0.0.1:9/v1"])
    assert result.exit_code == 1, result.output
    body = flat(result.output)
    assert "engine unreachable at" in body
    assert "http://127.0.0.1:9/v1" in body
    # Presets are structural and still validate even with no engine.
    assert "presets validate" in body


# --------------------------------------------------------------------------
# replay
# --------------------------------------------------------------------------


def test_replay_on_an_empty_directory_fails_loudly(tmp_path: Path) -> None:
    result = runner.invoke(app, ["replay", str(tmp_path)])
    assert result.exit_code == 1, result.output
    assert "no cassettes" in flat(result.output)


def test_replay_of_the_committed_cassettes_all_match() -> None:
    expected = load_cassettes(CASSETTE_DIR)
    assert expected, f"no cassettes in {CASSETTE_DIR}"

    result = runner.invoke(app, ["replay", str(CASSETTE_DIR)])
    assert result.exit_code == 0, result.output
    body = flat(result.output)
    assert f"{len(expected)}/{len(expected)} cassettes match" in body
    assert "FAIL" not in body
    for cassette in expected:
        assert cassette.name in body, f"{cassette.name} not reported by `k3 replay`"


# --------------------------------------------------------------------------
# help / serve validation
# --------------------------------------------------------------------------


def test_help_lists_every_command() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0, result.output
    body = flat(result.output)
    for command in ("serve", "presets", "detect", "doctor", "replay", "version"):
        assert command in body, f"`{command}` missing from `k3 --help`"


def test_serve_rejects_an_unknown_preset_before_binding_a_port() -> None:
    """The check sits above ``import uvicorn``, so this never starts a server."""
    result = runner.invoke(app, ["serve", "--client", "nope"])
    assert result.exit_code == 2, result.output
    body = flat(result.output)
    assert "unknown preset" in body
    assert "nope" in body
    # It should tell you what you *could* have said.
    for name in presets_mod.names():
        assert name in body


# --------------------------------------------------------------------------
# shell detection and the paste block
# --------------------------------------------------------------------------


def test_detect_shell_returns_a_shell_we_can_render() -> None:
    assert cli.detect_shell() in {"posix", "powershell", "cmd"}


@pytest.mark.parametrize(
    "shell, expected",
    [
        ("posix", "export ANTHROPIC_BASE_URL=http://localhost:8080"),
        ("powershell", '$env:ANTHROPIC_BASE_URL = "http://localhost:8080"'),
        ("cmd", "set ANTHROPIC_BASE_URL=http://localhost:8080"),
    ],
)
def test_render_env_uses_the_right_syntax_per_shell(shell: str, expected: str) -> None:
    lines = presets_mod.get("claude-code").setup.render_env(
        "http://localhost:8080", "local", "k3", shell
    )
    assert expected in lines, lines
    # Every declared variable is rendered, once, in that shell's syntax.
    keys = [key for key, _ in presets_mod.get("claude-code").setup.env]
    assert len(lines) == len(keys)
    for key, line in zip(keys, lines):
        assert key in line


def test_render_env_defaults_to_posix() -> None:
    setup = presets_mod.get("claude-code").setup
    assert setup.render_env("http://localhost:8080", "local", "k3") == setup.render_env(
        "http://localhost:8080", "local", "k3", "posix"
    )
