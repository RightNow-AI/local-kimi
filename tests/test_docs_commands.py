"""Keep public k3 commands and preset names tied to the implementation.

This test is intentionally static. It reads the Typer declarations and preset
constructors from source, so a stale documentation command fails even if an
installed copy of k3 happens to accept different flags.
"""

from __future__ import annotations

import ast
import re
import shlex
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CLI_SOURCE = ROOT / "k3" / "cli.py"
PRESETS_SOURCE = ROOT / "k3" / "presets.py"
DOCS = [ROOT / "README.md", *sorted((ROOT / "docs").glob("*.md"))]
PRESET_LABEL = re.compile(r"^Preset: `([^`]+)`$", re.MULTILINE)


def _command_flags() -> dict[str, set[str]]:
    tree = ast.parse(CLI_SOURCE.read_text(encoding="utf-8"))
    commands: dict[str, set[str]] = {}

    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue

        command_name: str | None = None
        for decorator in node.decorator_list:
            if not isinstance(decorator, ast.Call):
                continue
            func = decorator.func
            if not (
                isinstance(func, ast.Attribute)
                and func.attr == "command"
                and isinstance(func.value, ast.Name)
                and func.value.id == "app"
            ):
                continue
            command_name = node.name.replace("_", "-")
            if decorator.args:
                explicit = decorator.args[0]
                if isinstance(explicit, ast.Constant) and isinstance(explicit.value, str):
                    command_name = explicit.value
            break

        if command_name is None:
            continue

        flags: set[str] = set()
        positional = node.args.args
        defaults = [None] * (len(positional) - len(node.args.defaults)) + list(node.args.defaults)
        for argument, default in zip(positional, defaults):
            if not isinstance(default, ast.Call):
                continue
            option = default.func
            if not (isinstance(option, ast.Attribute) and option.attr == "Option"):
                continue

            explicit_flags = {
                item.value
                for item in default.args
                if isinstance(item, ast.Constant)
                and isinstance(item.value, str)
                and item.value.startswith("-")
            }
            flags.update(explicit_flags)
            if not any(flag.startswith("--") for flag in explicit_flags):
                flags.add("--" + argument.arg.replace("_", "-"))

        commands[command_name] = flags

    return commands


def _preset_names() -> set[str]:
    tree = ast.parse(PRESETS_SOURCE.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not (isinstance(node.func, ast.Name) and node.func.id == "Preset"):
            continue
        for keyword in node.keywords:
            if (
                keyword.arg == "name"
                and isinstance(keyword.value, ast.Constant)
                and isinstance(keyword.value.value, str)
            ):
                names.add(keyword.value.value)
    return names


def _documented_k3_invocations() -> list[tuple[Path, int, list[str]]]:
    """Collect k3 invocations from fenced code blocks only.

    Prose is not shell. An earlier version ran shlex.split over every line of
    every document, so an ordinary apostrophe in a sentence raised "No closing
    quotation" and failed the suite on a paragraph rather than on a command.
    Only fenced blocks are commands, and a line inside one that still will not
    parse is skipped rather than crashing the collector, because this test
    exists to catch documented flags that do not exist, not to lint prose.
    """
    invocations: list[tuple[Path, int, list[str]]] = []
    for path in DOCS:
        in_fence = False
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            stripped = line.strip()
            if stripped.startswith("```"):
                in_fence = not in_fence
                continue
            if not in_fence:
                continue
            try:
                tokens = shlex.split(stripped)
            except ValueError:
                continue
            if "k3" not in tokens:
                continue
            index = tokens.index("k3")
            if index + 1 >= len(tokens) or tokens[index + 1].startswith("-"):
                continue
            invocations.append((path, line_number, tokens[index + 1 :]))
    return invocations


def test_documented_k3_flags_exist_in_cli() -> None:
    commands = _command_flags()
    invocations = _documented_k3_invocations()
    assert invocations, "no k3 command lines found in README.md or docs/*.md"

    for path, line_number, tokens in invocations:
        command, *arguments = tokens
        assert command in commands, f"{path}:{line_number}: unknown k3 command {command!r}"
        for token in arguments:
            if not token.startswith("--"):
                continue
            flag = token.split("=", 1)[0]
            assert flag in commands[command], (
                f"{path}:{line_number}: {flag!r} is not declared for `k3 {command}`"
            )


def test_documented_preset_labels_exist() -> None:
    known = _preset_names()
    documented: list[tuple[Path, str]] = []
    for path in DOCS:
        text = path.read_text(encoding="utf-8")
        documented.extend((path, name) for name in PRESET_LABEL.findall(text))

    # No assertion that labels exist. The point of this test is to catch a
    # preset named in the docs that does not exist in the code, so documenting
    # zero presets is not a failure. Requiring at least one would couple the
    # docs to a label syntax rather than to the truth of what they claim.
    for path, name in documented:
        assert name in known, f"{path}: unknown k3 preset {name!r}"
