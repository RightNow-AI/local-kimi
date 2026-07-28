"""Capture real OpenAI SDK traffic into the fixture set.

The official ``openai`` Python client is the reference implementation of the
Chat Completions dialect, so letting *it* build the requests is meaningfully
different from hand-writing JSON bodies: it sets its own headers, its own
serialisation, and its own streaming expectations.

The transport is in-process (``httpx.ASGITransport``) rather than a socket —
the bytes are the SDK's, the round trip just doesn't touch the loopback
interface. Cassettes from here are marked ``recorded``.

Run: ``uv run python scripts/capture_openai.py``
"""

from __future__ import annotations

import asyncio
import shutil
import sys
import tempfile
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from k3.record import Cassette  # noqa: E402
from k3.server import ServerConfig, create_app  # noqa: E402
from k3.upstream import MockUpstream, UpstreamConfig  # noqa: E402

OUT = ROOT / "tests" / "cassettes"

TOOL = {
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


async def main() -> int:
    try:
        from openai import AsyncOpenAI
    except ImportError:
        print("openai SDK not installed; run: uv sync --group dev")
        return 1

    tmp = Path(tempfile.mkdtemp(prefix="k3-openai-"))
    cfg = ServerConfig(mock=True, record_dir=str(tmp), upstream=UpstreamConfig(model="k3"))
    app = create_app(cfg, engine=MockUpstream(cfg.upstream))

    http_client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://k3.test"
    )
    client = AsyncOpenAI(api_key="local", base_url="http://k3.test/v1", http_client=http_client)

    written: list[tuple[str, str]] = []

    def take(name: str) -> None:
        files = sorted(tmp.glob("*.json"), key=lambda p: p.stat().st_mtime)
        cassette = Cassette.load(files[-1])
        cassette.name = name
        cassette.source = "recorded"
        target = cassette.save(OUT, compress=True)
        written.append((name, f"{target.stat().st_size / 1024:.1f} KiB"))

    try:
        messages = [
            {"role": "system", "content": "You are a terse assistant."},
            {"role": "user", "content": "What is the weather in Beijing?"},
        ]
        first = await client.chat.completions.create(
            model="k3", messages=messages, tools=[TOOL], tool_choice="auto", max_tokens=1024
        )
        take("openai-sdk-tools-json")

        call = first.choices[0].message.tool_calls[0]
        await client.chat.completions.create(
            model="k3",
            messages=[
                *messages,
                first.choices[0].message.model_dump(exclude_none=True),
                {"role": "tool", "tool_call_id": call.id, "content": "22C, sunny"},
            ],
            tools=[TOOL],
            max_tokens=1024,
        )
        take("openai-sdk-turn2-json")

        stream = await client.chat.completions.create(
            model="k3",
            messages=[{"role": "user", "content": "Say hello."}],
            stream=True,
            stream_options={"include_usage": True},
        )
        async for _ in stream:
            pass
        take("openai-sdk-stream")
    finally:
        await http_client.aclose()
        shutil.rmtree(tmp, ignore_errors=True)

    print("recorded from the openai SDK:")
    for name, size in written:
        print(f"  {name:34s} {size}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
