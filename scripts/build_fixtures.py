"""Build the conformance fixture set in ``tests/cassettes``.

Two kinds of cassette land here:

``recorded``   real traffic captured off the wire from the actual client, via
               ``k3 serve --record``. These are the ones that prove a preset.
``synthetic``  request bodies written to match a client's documented wire
               format, replayed against the mock engine. These pin behaviour
               but do not prove it — they get replaced by real recordings as
               those are captured.

Run: ``uv run python scripts/build_fixtures.py [--raw DIR]``

``--raw`` points at a directory of freshly recorded cassettes to curate into the
fixture set (deduplicated, renamed, gzipped). Without it, only the synthetic
cassettes are rebuilt.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Optional

import httpx

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from k3.record import Cassette, load_cassettes  # noqa: E402
from k3.server import ServerConfig, create_app  # noqa: E402
from k3.upstream import MockUpstream, UpstreamConfig  # noqa: E402

OUT = ROOT / "tests" / "cassettes"

WEATHER_TOOL_OPENAI = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Look up the current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string", "description": "City name"}},
            "required": ["city"],
        },
    },
}

WEATHER_TOOL_RESPONSES = {
    "type": "function",
    "name": "get_weather",
    "description": "Look up the current weather for a city.",
    "parameters": {
        "type": "object",
        "properties": {"city": {"type": "string", "description": "City name"}},
        "required": ["city"],
    },
}


class Session:
    """One server instance, so multi-turn captures share a ledger like real life."""

    def __init__(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="k3-fixtures-"))
        cfg = ServerConfig(
            mock=True,
            tool_parser="kimi_k3",
            record_dir=str(self.tmp),
            upstream=UpstreamConfig(model="k3"),
        )
        # Pin fixture generation to the production K3 parser/emission pair.
        self.app = create_app(
            cfg, engine=MockUpstream(cfg.upstream, tool_parser=cfg.tool_parser or "kimi_k3")
        )

    async def post(self, path: str, body: dict[str, Any], headers: dict[str, str]) -> httpx.Response:
        transport = httpx.ASGITransport(app=self.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://k3.test") as client:
            return await client.post(path, json=body, headers=headers)

    def take(self, name: str) -> Cassette:
        """The most recently written cassette, relabelled for the fixture set."""
        files = sorted(self.tmp.glob("*.json"), key=lambda p: p.stat().st_mtime)
        cassette = Cassette.load(files[-1])
        cassette.name = name
        cassette.source = "synthetic"
        return cassette

    def cleanup(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)


def write(cassette: Cassette) -> Path:
    target = cassette.save(OUT, compress=True)
    size = target.stat().st_size
    print(f"  {cassette.source:9s} {cassette.name:38s} {size / 1024:7.1f} KiB")
    return target


# --------------------------------------------------------------------------
# synthetic captures
# --------------------------------------------------------------------------


async def build_openai() -> list[Cassette]:
    out: list[Cassette] = []
    ua = {"user-agent": "OpenAI/Python 1.51.0", "content-type": "application/json"}
    session = Session()
    try:
        body = {
            "model": "k3",
            "messages": [
                {"role": "system", "content": "You are a terse assistant."},
                {"role": "user", "content": "What is the weather in Beijing?"},
            ],
            "tools": [WEATHER_TOOL_OPENAI],
            "tool_choice": "auto",
            "max_tokens": 1024,
        }
        resp = await session.post("/v1/chat/completions", body, ua)
        out.append(session.take("openai-tools-json"))

        # Turn two: the client echoes the assistant message back with reasoning
        # stripped, so restoration has to go through the fingerprint path.
        assistant = resp.json()["choices"][0]["message"]
        call = assistant["tool_calls"][0]
        body2 = dict(body)
        body2["messages"] = [
            *body["messages"],
            assistant,
            {"role": "tool", "tool_call_id": call["id"], "content": "22C, sunny"},
        ]
        await session.post("/v1/chat/completions", body2, ua)
        out.append(session.take("openai-turn2-fingerprint-json"))

        body3 = {
            "model": "k3",
            "messages": [{"role": "user", "content": "Say hello."}],
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        await session.post("/v1/chat/completions", body3, ua)
        out.append(session.take("openai-plain-stream"))
    finally:
        session.cleanup()
    return out


async def build_codex() -> list[Cassette]:
    out: list[Cassette] = []
    ua = {"user-agent": "codex_cli_rs/0.20.0", "content-type": "application/json"}
    session = Session()
    try:
        body = {
            "model": "k3",
            "instructions": "You are a coding agent.",
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "What is the weather in Beijing?"}],
                }
            ],
            "tools": [WEATHER_TOOL_RESPONSES],
            "tool_choice": "auto",
            "reasoning": {"effort": "medium", "summary": "auto"},
            "include": ["reasoning.encrypted_content"],
            "store": False,
            "max_output_tokens": 2048,
        }
        resp = await session.post("/v1/responses", body, ua)
        out.append(session.take("codex-tools-json"))

        # Turn two echoes the reasoning item back; encrypted_content is the
        # vehicle that has to carry K3's exact bytes.
        items = resp.json()["output"]
        call = next(i for i in items if i["type"] == "function_call")
        body2 = dict(body)
        body2["input"] = [
            *body["input"],
            *items,
            {"type": "function_call_output", "call_id": call["call_id"], "output": "22C, sunny"},
        ]
        body2["stream"] = True
        await session.post("/v1/responses", body2, ua)
        out.append(session.take("codex-turn2-reasoning-stream"))
    finally:
        session.cleanup()
    return out


async def build_others() -> list[Cassette]:
    out: list[Cassette] = []
    session = Session()
    try:
        await session.post(
            "/v1/chat/completions",
            {
                "model": "kimi-k2-thinking",
                "messages": [{"role": "user", "content": "Explain reasoning round trips."}],
                "stream": True,
            },
            {"user-agent": "kimi-cli/0.4.1"},
        )
        out.append(session.take("kimi-code-reasoning-stream"))

        await session.post(
            "/v1/chat/completions",
            {
                "model": "k3",
                "messages": [{"role": "user", "content": "Refactor this function."}],
                "temperature": 0,
            },
            {"user-agent": "Aider/0.60.0 litellm/1.48.0"},
        )
        out.append(session.take("aider-inline-think-json"))

        await session.post(
            "/v1/chat/completions",
            {
                "model": "k3",
                "messages": [{"role": "user", "content": "Read the file."}],
                "stream": True,
            },
            {"http-referer": "https://cline.bot", "x-title": "Cline"},
        )
        out.append(session.take("cline-reasoning-stream"))

        await session.post(
            "/v1/chat/completions",
            {
                "model": "k3",
                "messages": [{"role": "user", "content": "List the files in src/."}],
                "tools": [WEATHER_TOOL_OPENAI],
                "stream": True,
            },
            {"user-agent": "opencode/0.3.11"},
        )
        out.append(session.take("opencode-tools-stream"))
    finally:
        session.cleanup()
    return out


# --------------------------------------------------------------------------
# curating real recordings
# --------------------------------------------------------------------------


def curate_raw(raw_dir: Path) -> list[Cassette]:
    """Pick the cassettes worth keeping out of a raw recording session.

    Claude Code fires a lot of near-identical background requests; keeping all
    of them would bloat the suite without testing anything new. We keep the
    turns that exercise distinct paths.
    """
    cassettes = load_cassettes(raw_dir)
    if not cassettes:
        print(f"  no cassettes in {raw_dir}")
        return []

    kept: list[Cassette] = []
    seen_plain = False
    for cassette in cassettes:
        body = cassette.request_body or {}

        if cassette.preset == "codex":
            items = body.get("input") or []
            has_output = any(
                isinstance(i, dict) and i.get("type") == "function_call_output" for i in items
            )
            cassette.name = (
                "codex-turn2-tool-output-stream" if has_output else "codex-turn1-tools-stream"
            )
            cassette.source = "recorded"
            kept.append(cassette)
            continue

        messages = body.get("messages") or []
        has_prior_thinking = any(
            isinstance(m.get("content"), list)
            and any(b.get("type") == "thinking" for b in m["content"] if isinstance(b, dict))
            for m in messages
            if isinstance(m, dict)
        )
        has_tool_result = any(
            isinstance(m.get("content"), list)
            and any(b.get("type") == "tool_result" for b in m["content"] if isinstance(b, dict))
            for m in messages
            if isinstance(m, dict)
        )

        if has_prior_thinking and has_tool_result:
            cassette.name = "claude-code-turn2-thinking-restore-stream"
        elif cassette.streamed and body.get("tools"):
            cassette.name = "claude-code-turn1-tools-stream"
        elif not seen_plain and not cassette.streamed:
            seen_plain = True
            cassette.name = "claude-code-background-json"
        else:
            continue
        cassette.source = "recorded"
        kept.append(cassette)

    return kept


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, help="Directory of freshly recorded cassettes.")
    parser.add_argument(
        "--keep-synthetic-only",
        action="store_true",
        help="Rebuild synthetic cassettes without touching recorded ones.",
    )
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)

    if not args.keep_synthetic_only:
        for path in OUT.glob("*.json.gz"):
            data = Cassette.load(path)
            if data.source == "synthetic":
                path.unlink()

    # claude-code, openai and codex are covered by real captures now
    # (scripts/capture_openai.py and `k3 serve --record` sessions), so only the
    # clients we have no binary for are synthesized here.
    print("synthetic cassettes:")
    for cassette in await build_others():
        write(cassette)

    if args.raw:
        print("recorded cassettes:")
        for cassette in curate_raw(args.raw):
            write(cassette)

    total = sum(p.stat().st_size for p in OUT.glob("*.json.gz"))
    print(f"\n{len(list(OUT.glob('*.json.gz')))} cassettes, {total / 1024:.0f} KiB total")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
