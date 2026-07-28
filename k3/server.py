"""The HTTP surface.

Every dialect route is registered unconditionally so auto-detection has
something to detect; ``--client`` pins the preset rather than narrowing the
server. Requests flow:

    route -> detect -> dialect.ingress -> build_payload -> pipeline.run
          -> dialect.egress / dialect.egress_stream

Errors are rendered in the shape the *detected client* expects, because a
client that cannot parse your error message reports it as a hang.
"""

from __future__ import annotations

import hmac
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Optional

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from . import pipeline, presets as presets_mod
from .detect import detect
from .dialects import get_dialect
from .dialects.base import DialectContext, estimate_tokens, sse
from .ir import StreamEvent
from .presets import Preset
from .record import Cassette, Recorder, RecordingEngine
from .reasoning import ReasoningLedger
from .upstream import MockUpstream, Upstream, UpstreamConfig, UpstreamError, build_payload, resolve_model

log = logging.getLogger("k3")


@dataclass
class ServerConfig:
    host: str = "127.0.0.1"
    port: int = 8080
    #: ``--client`` override; ``None`` means auto-detect per request.
    forced_client: Optional[str] = None
    #: Override the detected preset's parser. ``None`` keeps the preset choice.
    tool_parser: Optional[str] = None
    #: Token incoming requests must present. ``None`` means open (local use).
    auth_token: Optional[str] = None
    record_dir: Optional[str] = None
    #: gzip cassettes on write; real client traffic is large.
    record_compress: bool = False
    #: Serve only the routes the forced client actually uses.
    strict_routes: bool = False
    mock: bool = False
    #: Origins allowed to call the server from a browser. Empty disables CORS.
    cors_origins: list[str] = field(default_factory=list)
    upstream: UpstreamConfig = field(default_factory=UpstreamConfig)
    ledger_entries: int = 4096
    ledger_ttl: float = 6 * 3600
    public_url: Optional[str] = None

    @property
    def base_url(self) -> str:
        if self.public_url:
            return self.public_url.rstrip("/")
        host = "localhost" if self.host in ("0.0.0.0", "127.0.0.1", "::") else self.host
        return f"http://{host}:{self.port}"

    def preset(self) -> Optional[Preset]:
        return presets_mod.get(self.forced_client) if self.forced_client else None


# --------------------------------------------------------------------------
# error rendering
# --------------------------------------------------------------------------


def _anthropic_error(status: int, message: str, kind: str) -> dict[str, Any]:
    return {"type": "error", "error": {"type": kind, "message": message}}


def _openai_error(status: int, message: str, kind: str) -> dict[str, Any]:
    return {"error": {"message": message, "type": kind, "param": None, "code": kind}}


def error_body(dialect: str, status: int, message: str, kind: str = "api_error") -> dict[str, Any]:
    if dialect == "anthropic_messages":
        return _anthropic_error(status, message, kind)
    return _openai_error(status, message, kind)


def error_event(dialect: str, message: str, kind: str = "api_error") -> bytes:
    """Mid-stream failure, in the shape the client's SSE parser expects."""
    if dialect == "anthropic_messages":
        return sse({"type": "error", "error": {"type": kind, "message": message}}, "error")
    if dialect == "openai_responses":
        # The Responses parser dispatches on the event name; an unnamed frame is
        # invisible to it and the stream reads as a hang rather than a failure.
        return sse(
            {"type": "error", "code": kind, "message": message, "param": None}, "error"
        )
    return sse({"error": {"message": message, "type": kind, "code": kind}})


def error_frames(dialect: str, message: str, kind: str = "api_error") -> list[bytes]:
    """Everything a client needs to treat the stream as finished, not stalled."""
    frames = [error_event(dialect, message, kind)]
    if dialect == "openai_chat":
        # Chat clients block until [DONE]; without it the error never surfaces.
        frames.append(sse("[DONE]"))
    return frames


def _status_for(exc: Exception) -> tuple[int, str, str]:
    if isinstance(exc, UpstreamError):
        kind = "connection_error" if exc.kind == "connection" else "upstream_error"
        return (exc.status if exc.status >= 400 else 502), str(exc), kind
    return 500, f"{type(exc).__name__}: {exc}", "internal_error"


# --------------------------------------------------------------------------
# app
# --------------------------------------------------------------------------


def create_app(cfg: Optional[ServerConfig] = None, engine: Any = None) -> FastAPI:
    cfg = cfg or ServerConfig()
    app = FastAPI(title="k3", version=__import__("k3").__version__, docs_url=None, redoc_url=None)
    if cfg.cors_origins:
        # Off unless asked for. A wildcard here lets any page the user visits
        # drive their local engine and read the replies back.
        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(cfg.cors_origins),
            allow_methods=["*"],
            allow_headers=["*"],
        )

    if engine is None:
        engine = (
            MockUpstream(cfg.upstream, tool_parser=cfg.tool_parser or "kimi_k3")
            if cfg.mock
            else Upstream(cfg.upstream)
        )

    app.state.config = cfg
    app.state.engine = engine
    app.state.ledger = ReasoningLedger(max_entries=cfg.ledger_entries, ttl_seconds=cfg.ledger_ttl)
    app.state.recorder = Recorder(cfg.record_dir, compress=cfg.record_compress)
    app.state.counters = {"requests": 0, "streamed": 0, "errors": 0}
    app.state.by_preset = {}

    @app.on_event("shutdown")
    async def _shutdown() -> None:  # pragma: no cover - lifecycle
        close = getattr(app.state.engine, "aclose", None)
        if close is not None:
            await close()

    # -- helpers ---------------------------------------------------------

    def _authorized(request: Request) -> bool:
        if not cfg.auth_token:
            return True
        header = request.headers.get("authorization", "")
        token = header[7:] if header.lower().startswith("bearer ") else ""
        # compare_digest: token checks that short-circuit leak length and prefix.
        return any(
            hmac.compare_digest(cfg.auth_token, candidate)
            for candidate in (token, request.headers.get("x-api-key", ""))
            if candidate
        )

    def _route_allowed(path: str) -> bool:
        if not cfg.strict_routes:
            return True
        preset = cfg.preset()
        return preset is None or path in preset.routes

    async def _read_body(request: Request) -> dict[str, Any]:
        raw = await request.body()
        if not raw:
            return {}
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON body: {exc}") from exc
        if parsed is None:
            return {}
        if not isinstance(parsed, dict):
            # Every route here takes a JSON object. Letting an array through
            # turns a client typo into an AttributeError deep in a handler.
            raise ValueError(
                f"request body must be a JSON object, got {type(parsed).__name__}"
            )
        return parsed

    async def handle(request: Request, path: str) -> Any:
        started = time.perf_counter()
        headers = dict(request.headers)

        if not _route_allowed(path):
            preset = cfg.preset()
            return JSONResponse(
                status_code=404,
                content=error_body(
                    preset.dialect if preset else "openai_chat",
                    404,
                    f"{path} is not served under --client {cfg.forced_client} with --strict-routes",
                    "not_found",
                ),
            )

        try:
            body = await _read_body(request)
        except ValueError as exc:
            # The body is unusable, but the path and headers still identify the
            # client — and a client that cannot parse your error reports a hang.
            dialect_name = detect(path, headers, None, cfg.forced_client).preset.dialect
            return JSONResponse(
                status_code=400,
                content=error_body(dialect_name, 400, str(exc), "invalid_request_error"),
            )

        detection = detect(path, headers, body, cfg.forced_client)
        preset = detection.preset
        dialect_name = preset.dialect
        dialect = get_dialect(dialect_name)

        app.state.counters["requests"] += 1
        app.state.by_preset[preset.name] = app.state.by_preset.get(preset.name, 0) + 1

        cassette: Optional[Cassette] = app.state.recorder.start(
            path=path,
            headers=headers,
            body=body,
            preset=preset.name,
            detected_via=detection.reason,
        )

        ctx = DialectContext(
            preset=preset,
            ledger=app.state.ledger,
            client_model=str(body.get("model") or cfg.upstream.model),
            headers={k.lower(): v for k, v in headers.items()},
            stream_usage=_requested_stream_usage(body),
        )

        try:
            req = dialect.ingress(body, ctx)
        except Exception as exc:  # a malformed body must not 500 silently
            app.state.counters["errors"] += 1
            log.exception("ingress failed for %s under preset %s", path, preset.name)
            return JSONResponse(
                status_code=400,
                content=error_body(dialect_name, 400, f"could not parse request: {exc}", "invalid_request_error"),
            )

        ctx.thinking_enabled = req.thinking_enabled
        upstream_model = resolve_model(req.model, preset, cfg.upstream)
        ctx.upstream_model = upstream_model
        payload = build_payload(req, preset, app.state.ledger, cfg.upstream, upstream_model)
        ctx.input_tokens_hint = _estimate_prompt_tokens(payload)

        engine_for_request = RecordingEngine(app.state.engine, cassette) if cassette else app.state.engine

        log.info(
            "%s -> %s (%s) model=%s->%s stream=%s tools=%d",
            path,
            preset.name,
            detection.reason,
            req.model,
            upstream_model,
            req.stream,
            len(req.tools),
        )

        def make_events(stream: bool) -> AsyncIterator[StreamEvent]:
            return pipeline.run(
                engine_for_request,
                payload,
                tool_parser=cfg.tool_parser or preset.tool_parser,
                ledger=app.state.ledger,
                reasoning_field=cfg.upstream.reasoning_field,
                stream=stream,
                model_label=ctx.client_model,
            )

        if not req.stream:
            try:
                response = await pipeline.collect_response(make_events(stream=False))
            except Exception as exc:
                app.state.counters["errors"] += 1
                status, message, kind = _status_for(exc)
                log.error("engine call failed: %s", message)
                if cassette:
                    cassette.client_status = status
                    cassette.client_body = error_body(dialect_name, status, message, kind)
                    app.state.recorder.finish(cassette)
                return JSONResponse(
                    status_code=status, content=error_body(dialect_name, status, message, kind)
                )

            out = dialect.egress(response, ctx)
            if cassette:
                cassette.client_body = out
                app.state.recorder.finish(cassette)
            log.info("  done in %.0fms (%d out tokens)", (time.perf_counter() - started) * 1000, response.usage.output_tokens)
            return JSONResponse(content=out)

        # Streaming: pull the first event before committing to a 200 so an
        # engine that is down produces a real status code, not a dead stream.
        app.state.counters["streamed"] += 1
        events = make_events(stream=True)
        try:
            first = await events.__anext__()
        except StopAsyncIteration:
            first = None
        except Exception as exc:
            app.state.counters["errors"] += 1
            status, message, kind = _status_for(exc)
            log.error("engine stream failed to start: %s", message)
            if cassette:
                cassette.client_status = status
                cassette.client_body = error_body(dialect_name, status, message, kind)
                app.state.recorder.finish(cassette)
            return JSONResponse(
                status_code=status, content=error_body(dialect_name, status, message, kind)
            )

        async def canonical() -> AsyncIterator[StreamEvent]:
            if first is not None:
                yield first
            async for event in events:
                yield event

        async def body_iter() -> AsyncIterator[bytes]:
            if cassette:
                cassette.streamed = True
            try:
                async for chunk in dialect.egress_stream(canonical(), ctx):
                    if cassette:
                        cassette.client_sse.append(chunk.decode("utf-8", "replace"))
                    yield chunk
            except Exception as exc:  # noqa: BLE001 - surfaced to the client
                app.state.counters["errors"] += 1
                _, message, kind = _status_for(exc)
                log.error("stream failed mid-flight: %s", message)
                for chunk in error_frames(dialect_name, message, kind):
                    if cassette:
                        cassette.client_sse.append(chunk.decode("utf-8", "replace"))
                    yield chunk
            finally:
                if cassette:
                    app.state.recorder.finish(cassette)

        return StreamingResponse(
            body_iter(),
            media_type="text/event-stream",
            headers={
                "cache-control": "no-cache",
                "connection": "keep-alive",
                "x-k3-preset": preset.name,
            },
        )

    def _unauthorized(path: str) -> JSONResponse:
        preset = cfg.preset()
        dialect_name = preset.dialect if preset else ("anthropic_messages" if "messages" in path else "openai_chat")
        return JSONResponse(
            status_code=401,
            content=error_body(dialect_name, 401, "missing or invalid credentials", "authentication_error"),
        )

    # -- routes ----------------------------------------------------------

    @app.post("/v1/messages")
    async def messages(request: Request) -> Any:
        if not _authorized(request):
            return _unauthorized("/v1/messages")
        return await handle(request, "/v1/messages")

    @app.post("/v1/messages/count_tokens")
    async def messages_count_tokens(request: Request) -> Any:
        if not _authorized(request):
            return _unauthorized("/v1/messages/count_tokens")
        try:
            body = await _read_body(request)
        except ValueError as exc:
            return JSONResponse(
                status_code=400,
                content=error_body("anthropic_messages", 400, str(exc), "invalid_request_error"),
            )
        detection = detect("/v1/messages", dict(request.headers), body, cfg.forced_client)
        from .dialects import anthropic_messages

        ctx = DialectContext(
            preset=detection.preset,
            ledger=app.state.ledger,
            client_model=str(body.get("model") or cfg.upstream.model),
        )
        try:
            return JSONResponse(content=anthropic_messages.count_tokens(body, ctx))
        except Exception as exc:  # noqa: BLE001
            # Claude Code calls this before every request; a text/plain 500 here
            # is a parse error on the client and reads as a hang.
            log.exception("count_tokens failed")
            return JSONResponse(
                status_code=400,
                content=error_body(
                    "anthropic_messages", 400, f"could not parse request: {exc}",
                    "invalid_request_error",
                ),
            )

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request) -> Any:
        if not _authorized(request):
            return _unauthorized("/v1/chat/completions")
        return await handle(request, "/v1/chat/completions")

    @app.post("/v1/responses")
    async def responses(request: Request) -> Any:
        if not _authorized(request):
            return _unauthorized("/v1/responses")
        return await handle(request, "/v1/responses")

    @app.get("/v1/models")
    async def models(request: Request) -> Any:
        if not _authorized(request):
            return _unauthorized("/v1/models")
        detection = detect("/v1/models", dict(request.headers), None, cfg.forced_client)
        preset = detection.preset if "/v1/models" in detection.preset.routes else (cfg.preset() or presets_mod.get("openai"))
        ids = [cfg.upstream.model]
        if cfg.upstream.small_model:
            ids.append(cfg.upstream.small_model)
        ids.extend(m for m in preset.models.advertise if m not in ids)
        now = int(time.time())
        if preset.dialect == "anthropic_messages":
            return JSONResponse(
                {
                    "data": [
                        {
                            "type": "model",
                            "id": mid,
                            "display_name": mid,
                            "created_at": "1970-01-01T00:00:00Z",
                        }
                        for mid in ids
                    ],
                    "has_more": False,
                    "first_id": ids[0],
                    "last_id": ids[-1],
                }
            )
        return JSONResponse(
            {
                "object": "list",
                "data": [
                    {"id": mid, "object": "model", "created": now, "owned_by": "k3"} for mid in ids
                ],
            }
        )

    @app.get("/health")
    async def health(request: Request) -> Any:
        ok, detail = await app.state.engine.health()
        body: dict[str, Any] = {"status": "ok" if ok else "degraded"}
        # Stays reachable without credentials so container health checks work,
        # but the engine URL and config are not public.
        if _authorized(request):
            body.update(
                {
                    "engine": detail,
                    "upstream": cfg.upstream.base_url,
                    "mock": cfg.mock,
                    "client": cfg.forced_client or "auto",
                }
            )
        return JSONResponse(status_code=200 if ok else 503, content=body)

    @app.get("/k3/presets")
    async def list_presets(request: Request) -> Any:
        if not _authorized(request):
            return _unauthorized("/k3/presets")
        return JSONResponse(
            {
                "active": cfg.forced_client or "auto",
                "presets": [
                    {
                        "name": p.name,
                        "title": p.title,
                        "status": p.status,
                        "dialect": p.dialect,
                        "routes": p.routes,
                        "tool_parser": p.tool_parser,
                        "reasoning": p.reasoning.value,
                        "notes": p.notes,
                    }
                    for p in presets_mod.all_presets()
                ],
            }
        )

    @app.get("/k3/stats")
    async def stats(request: Request) -> Any:
        if not _authorized(request):
            return _unauthorized("/k3/stats")
        return JSONResponse(
            {
                "counters": app.state.counters,
                "by_preset": app.state.by_preset,
                "ledger": app.state.ledger.stats(),
                "recording": str(cfg.record_dir) if cfg.record_dir else None,
            }
        )

    return app


def _requested_stream_usage(body: dict[str, Any]) -> Optional[bool]:
    """``stream_options.include_usage``, or ``None`` when the client is silent.

    An explicit ``false`` has to be honoured: the extra ``choices: []`` chunk
    crashes any client that indexes ``choices[0]`` without a guard, and a client
    that opted out is exactly the one that will not have that guard.
    """
    options = body.get("stream_options")
    if not isinstance(options, dict) or "include_usage" not in options:
        return None
    return bool(options.get("include_usage"))


def _estimate_prompt_tokens(payload: dict[str, Any]) -> int:
    total = 0
    for msg in payload.get("messages") or []:
        content = msg.get("content")
        if isinstance(content, str):
            total += estimate_tokens(content)
        elif isinstance(content, list):
            total += estimate_tokens(json.dumps(content, ensure_ascii=False))
        for call in msg.get("tool_calls") or []:
            fn = call.get("function") or {}
            total += estimate_tokens(str(fn.get("name", "")) + str(fn.get("arguments", "")))
        total += 4
    for tool in payload.get("tools") or []:
        total += estimate_tokens(json.dumps(tool, ensure_ascii=False))
    return max(1, total)


__all__ = ["ServerConfig", "create_app", "error_body", "error_event"]
