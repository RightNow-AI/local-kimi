"""Record real client traffic, replay it as a conformance suite.

Claude Code, Codex and the rest all ship updates, and a preset that worked last
month can break silently. Without recorded traffic you find out from GitHub
issues. So: ``k3 serve --record DIR`` tees every request through to disk —
what the client sent, what we sent the engine, what the engine streamed back,
and the exact bytes we returned — and ``k3 replay DIR`` re-runs those cassettes
against the current code with the engine stubbed out by its own recording.

Anything that changes the client-facing bytes shows up as a diff. That is the
whole trick, and it is why this module exists before the sixth preset rather
than after it.
"""

from __future__ import annotations

import gzip
import json
import os
import re
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncIterator, Iterable, Optional

CASSETTE_VERSION = 1

#: Headers we never write to disk — cassettes get committed to repos.
REDACTED_HEADERS = {
    "authorization",
    "x-api-key",
    "proxy-authorization",
    "cookie",
    "set-cookie",
    "openai-organization",
}
REDACTED = "<redacted>"


def _safe_headers(headers: dict[str, str]) -> dict[str, str]:
    return {
        k.lower(): (REDACTED if k.lower() in REDACTED_HEADERS else v)
        for k, v in headers.items()
    }


def _slug(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "-", text).strip("-").lower() or "root"


@dataclass
class Cassette:
    preset: str
    path: str
    detected_via: str
    request_headers: dict[str, str] = field(default_factory=dict)
    request_body: Any = None
    upstream_request: Any = None
    upstream_chunks: list[dict[str, Any]] = field(default_factory=list)
    upstream_response: Optional[dict[str, Any]] = None
    client_status: int = 200
    client_body: Any = None
    client_sse: list[str] = field(default_factory=list)
    streamed: bool = False
    recorded_at: float = field(default_factory=time.time)
    name: str = ""
    #: ``recorded`` means it came off the wire from the real client.
    #: ``synthetic`` means the request body was written by hand to match the
    #: client's documented format. Both pin behaviour; only one proves it.
    source: str = "recorded"

    def to_dict(self) -> dict[str, Any]:
        return {
            "k3_cassette": CASSETTE_VERSION,
            "name": self.name,
            "recorded_at": self.recorded_at,
            "source": self.source,
            "preset": self.preset,
            "detected_via": self.detected_via,
            "streamed": self.streamed,
            "request": {
                "path": self.path,
                "headers": self.request_headers,
                "body": self.request_body,
            },
            "upstream": {
                "request": self.upstream_request,
                "chunks": self.upstream_chunks,
                "response": self.upstream_response,
            },
            "client": {
                "status": self.client_status,
                "body": self.client_body,
                "sse": self.client_sse,
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Cassette":
        request = data.get("request") or {}
        upstream = data.get("upstream") or {}
        client = data.get("client") or {}
        return cls(
            preset=data.get("preset", ""),
            path=request.get("path", ""),
            detected_via=data.get("detected_via", ""),
            request_headers=request.get("headers") or {},
            request_body=request.get("body"),
            upstream_request=upstream.get("request"),
            upstream_chunks=upstream.get("chunks") or [],
            upstream_response=upstream.get("response"),
            client_status=client.get("status", 200),
            client_body=client.get("body"),
            client_sse=client.get("sse") or [],
            streamed=bool(data.get("streamed")),
            recorded_at=data.get("recorded_at", 0.0),
            name=data.get("name", ""),
            source=data.get("source", "recorded"),
        )

    @classmethod
    def load(cls, path: os.PathLike[str] | str) -> "Cassette":
        path = Path(path)
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt", encoding="utf-8") as fh:  # type: ignore[operator]
            return cls.from_dict(json.load(fh))

    def save(self, directory: os.PathLike[str] | str, compress: bool = False) -> Path:
        """Write the cassette out.

        Real client traffic is large — Claude Code's system prompt alone runs to
        hundreds of kilobytes — so committed fixtures are gzipped. Ad-hoc
        recordings default to plain JSON because you want to read those.
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        if not self.name:
            stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime(self.recorded_at))
            kind = "stream" if self.streamed else "json"
            self.name = f"{stamp}-{self.preset}-{_slug(self.path)}-{kind}-{uuid.uuid4().hex[:6]}"
        target = directory / (f"{self.name}.json.gz" if compress else f"{self.name}.json")
        opener = gzip.open if compress else open
        with opener(target, "wt", encoding="utf-8") as fh:  # type: ignore[operator]
            json.dump(self.to_dict(), fh, indent=2, ensure_ascii=False)
            fh.write("\n")
        return target


class Recorder:
    """Owns the cassette directory. Cheap to construct, safe to disable."""

    def __init__(
        self, directory: Optional[os.PathLike[str] | str], compress: bool = False
    ) -> None:
        self.directory = Path(directory) if directory else None
        self.compress = compress
        self.written: list[Path] = []

    @property
    def enabled(self) -> bool:
        return self.directory is not None

    def start(
        self,
        *,
        path: str,
        headers: dict[str, str],
        body: Any,
        preset: str,
        detected_via: str,
    ) -> Optional[Cassette]:
        if not self.enabled:
            return None
        return Cassette(
            preset=preset,
            path=path,
            detected_via=detected_via,
            request_headers=_safe_headers(headers),
            request_body=body,
        )

    def finish(self, cassette: Optional[Cassette]) -> Optional[Path]:
        if cassette is None or self.directory is None:
            return None
        target = cassette.save(self.directory, compress=self.compress)
        self.written.append(target)
        return target


class RecordingEngine:
    """Tees engine traffic into a cassette without changing pipeline behaviour."""

    def __init__(self, inner: Any, cassette: Optional[Cassette]) -> None:
        self.inner = inner
        self.cassette = cassette

    async def chat(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self.cassette is not None:
            self.cassette.upstream_request = payload
        result = await self.inner.chat(payload)
        if self.cassette is not None:
            self.cassette.upstream_response = result
        return result

    async def chat_stream(self, payload: dict[str, Any]) -> AsyncIterator[dict[str, Any]]:
        if self.cassette is not None:
            self.cassette.upstream_request = payload
        async for chunk in self.inner.chat_stream(payload):
            if self.cassette is not None:
                self.cassette.upstream_chunks.append(chunk)
            yield chunk

    async def models(self) -> Optional[dict[str, Any]]:
        return await self.inner.models()

    async def health(self) -> tuple[bool, str]:
        return await self.inner.health()

    async def aclose(self) -> None:
        await self.inner.aclose()


class ReplayEngine:
    """Serves a cassette's recorded engine traffic back to the pipeline."""

    def __init__(self, cassette: Cassette) -> None:
        self.cassette = cassette
        self.seen_payload: Optional[dict[str, Any]] = None

    async def chat(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.seen_payload = payload
        if self.cassette.upstream_response is not None:
            return self.cassette.upstream_response
        # Recorded as a stream but replayed non-streaming: fold the chunks.
        return _fold_chunks(self.cassette.upstream_chunks)

    async def chat_stream(self, payload: dict[str, Any]) -> AsyncIterator[dict[str, Any]]:
        self.seen_payload = payload
        chunks: Iterable[dict[str, Any]] = self.cassette.upstream_chunks
        if not chunks and self.cassette.upstream_response is not None:
            from .pipeline import response_to_chunks

            chunks = response_to_chunks(self.cassette.upstream_response)
        for chunk in chunks:
            yield chunk

    async def models(self) -> Optional[dict[str, Any]]:
        return {"object": "list", "data": []}

    async def health(self) -> tuple[bool, str]:
        return True, "replay"

    async def aclose(self) -> None:
        return None


def _fold_chunks(chunks: list[dict[str, Any]]) -> dict[str, Any]:
    content: list[str] = []
    reasoning: list[str] = []
    finish = "stop"
    usage: Optional[dict[str, Any]] = None
    model = "k3"
    ident = "chatcmpl-replay"
    for chunk in chunks:
        model = chunk.get("model") or model
        ident = chunk.get("id") or ident
        if chunk.get("usage"):
            usage = chunk["usage"]
        for choice in chunk.get("choices") or []:
            delta = choice.get("delta") or {}
            if delta.get("content"):
                content.append(delta["content"])
            if delta.get("reasoning_content"):
                reasoning.append(delta["reasoning_content"])
            if choice.get("finish_reason"):
                finish = choice["finish_reason"]
    out: dict[str, Any] = {
        "id": ident,
        "object": "chat.completion",
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "".join(content),
                    "reasoning_content": "".join(reasoning),
                },
                "finish_reason": finish,
            }
        ],
    }
    if usage:
        out["usage"] = usage
    return out


def parse_sse(raw: str | bytes) -> list[dict[str, Any]]:
    """Split an SSE body into ``{"event": name, "data": parsed}`` records."""
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", "replace")
    records: list[dict[str, Any]] = []
    for block in raw.split("\n\n"):
        block = block.strip("\n")
        if not block.strip():
            continue
        event: Optional[str] = None
        data_lines: list[str] = []
        for line in block.split("\n"):
            if line.startswith("event:"):
                event = line[6:].strip()
            elif line.startswith("data:"):
                data_lines.append(line[5:].strip())
        if not data_lines:
            continue
        payload = "\n".join(data_lines)
        if payload == "[DONE]":
            records.append({"event": event, "data": "[DONE]"})
            continue
        try:
            records.append({"event": event, "data": json.loads(payload)})
        except json.JSONDecodeError:
            records.append({"event": event, "data": payload})
    return records


def load_cassettes(directory: os.PathLike[str] | str) -> list[Cassette]:
    directory = Path(directory)
    if not directory.exists():
        return []
    paths = sorted([*directory.glob("*.json"), *directory.glob("*.json.gz")])
    return [Cassette.load(p) for p in paths]


__all__ = [
    "Cassette",
    "Recorder",
    "RecordingEngine",
    "ReplayEngine",
    "parse_sse",
    "load_cassettes",
    "CASSETTE_VERSION",
]
