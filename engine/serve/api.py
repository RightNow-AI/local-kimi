"""OpenAI-compatible HTTP surface for the local inference engine."""

from __future__ import annotations

import inspect
import json
import time
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Annotated, Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    ValidationError,
    field_validator,
)

from .contracts import ChatPrompt, ChatTokenizer, InferenceEngine, SamplingParams
from .runtime import (
    CompletionDelta,
    CompletionEnd,
    CompletionEvent,
    GenerationContractError,
    GenerationRuntime,
)
from .stub import ByteChatTokenizer, EchoEngine


@dataclass(frozen=True, slots=True)
class ServerConfig:
    model: str = "k3"
    default_max_tokens: int = 512
    serialize_engine: bool = True

    def __post_init__(self) -> None:
        if not self.model.strip():
            raise ValueError("model cannot be empty")
        if self.default_max_tokens < 1:
            raise ValueError("default_max_tokens must be at least 1")


class ChatCompletionInput(BaseModel):
    model_config = ConfigDict(extra="allow", strict=True)

    model: str
    messages: Annotated[list[dict[str, Any]], Field(min_length=1)]
    max_tokens: Annotated[StrictInt, Field(ge=1)] | None = None
    max_completion_tokens: Annotated[StrictInt, Field(ge=1)] | None = None
    stream: StrictBool = False
    temperature: Annotated[int | float, Field(ge=0)] = 0.0
    top_p: Annotated[int | float, Field(gt=0, le=1)] = 1.0
    tools: list[dict[str, Any]] | None = None
    tool_choice: Any = None
    reasoning_effort: str | None = None
    stream_options: dict[str, Any] | None = None

    @field_validator("model")
    @classmethod
    def _model_is_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("model cannot be empty")
        return value


class ClientInputError(ValueError):
    def __init__(self, message: str, param: str | None = None) -> None:
        super().__init__(message)
        self.param = param


def create_app(
    engine: InferenceEngine,
    tokenizer: ChatTokenizer,
    config: ServerConfig | None = None,
) -> FastAPI:
    """Create a server around one engine and one tokenizer implementation."""

    config = config or ServerConfig()
    runtime = GenerationRuntime(
        engine,
        tokenizer,
        serialize_engine=config.serialize_engine,
    )
    app = FastAPI(
        title="local-kimi engine",
        version="0.1.0",
        docs_url=None,
        redoc_url=None,
    )
    app.state.engine = engine
    app.state.tokenizer = tokenizer
    app.state.runtime = runtime
    app.state.config = config
    app.state.started_at = int(time.time())

    @app.get("/health")
    async def health() -> JSONResponse:
        ok = True
        detail = "ready"
        probe = getattr(engine, "health", None)
        if probe is not None:
            try:
                result = probe()
                if inspect.isawaitable(result):
                    result = await result
                if isinstance(result, tuple) and len(result) == 2:
                    ok, detail = bool(result[0]), str(result[1])
            except Exception as exc:
                ok, detail = False, f"{type(exc).__name__}: {exc}"
        return JSONResponse(
            status_code=200 if ok else 503,
            content={"status": "ok" if ok else "degraded", "engine": detail},
        )

    @app.get("/v1/models")
    async def models() -> JSONResponse:
        return JSONResponse(
            {
                "object": "list",
                "data": [
                    {
                        "id": config.model,
                        "object": "model",
                        "created": app.state.started_at,
                        "owned_by": "local-kimi",
                    }
                ],
            }
        )

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request) -> Any:
        try:
            body = await _read_json_object(request)
            chat = _validate_chat(body)
            prompt_ids = _encode_prompt(tokenizer, chat)
            params = _sampling_params(chat, config)
        except ClientInputError as exc:
            return _error_response(str(exc), param=exc.param)

        completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        created = int(time.time())
        events = runtime.run(prompt_ids, params)

        if not chat.stream:
            try:
                reasoning, content, end = await _collect(events)
            except Exception as exc:
                return _engine_error(exc)
            return JSONResponse(
                {
                    "id": completion_id,
                    "object": "chat.completion",
                    "created": created,
                    "model": chat.model,
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": content,
                                "reasoning_content": reasoning,
                            },
                            "finish_reason": end.finish_reason,
                            "logprobs": None,
                        }
                    ],
                    "usage": _usage(end),
                }
            )

        try:
            first = await events.__anext__()
        except StopAsyncIteration:
            await events.aclose()
            return _engine_error(GenerationContractError("engine returned no completion events"))
        except Exception as exc:
            await events.aclose()
            return _engine_error(exc)

        async def stream_body() -> AsyncIterator[bytes]:
            try:
                if await request.is_disconnected():
                    return
                yield _sse(
                    _chunk(
                        completion_id,
                        created,
                        chat.model,
                        {"role": "assistant", "content": ""},
                    )
                )

                async for event in _with_first(first, events):
                    if await request.is_disconnected():
                        return
                    if isinstance(event, CompletionDelta):
                        if event.text:
                            field = (
                                "content"
                                if event.channel == "content"
                                else "reasoning_content"
                            )
                            yield _sse(
                                _chunk(
                                    completion_id,
                                    created,
                                    chat.model,
                                    {field: event.text},
                                )
                            )
                        continue

                    yield _sse(
                        _chunk(
                            completion_id,
                            created,
                            chat.model,
                            {},
                            finish_reason=event.finish_reason,
                        )
                    )
                    yield _sse(
                        {
                            "id": completion_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": chat.model,
                            "choices": [],
                            "usage": _usage(event),
                        }
                    )
                yield b"data: [DONE]\n\n"
            finally:
                await events.aclose()

        return StreamingResponse(
            stream_body(),
            media_type="text/event-stream",
            headers={
                "cache-control": "no-cache",
                "connection": "keep-alive",
            },
        )

    return app


def create_stub_app(config: ServerConfig | None = None) -> FastAPI:
    """Create a runnable CPU-only app with no model weights."""

    tokenizer = ByteChatTokenizer()
    return create_app(EchoEngine(tokenizer), tokenizer, config)


async def _read_json_object(request: Request) -> dict[str, Any]:
    raw = await request.body()
    if not raw:
        raise ClientInputError("request body must be a JSON object")
    try:
        parsed = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ClientInputError(f"invalid JSON body: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ClientInputError(
            f"request body must be a JSON object, got {type(parsed).__name__}"
        )
    return parsed


def _validate_chat(body: dict[str, Any]) -> ChatCompletionInput:
    try:
        return ChatCompletionInput.model_validate(body)
    except ValidationError as exc:
        first = exc.errors(include_url=False)[0]
        location = ".".join(str(part) for part in first.get("loc", ())) or None
        message = str(first.get("msg") or "invalid request")
        if location:
            message = f"{location}: {message}"
        raise ClientInputError(message, param=location) from exc


def _encode_prompt(tokenizer: ChatTokenizer, chat: ChatCompletionInput) -> tuple[int, ...]:
    prompt = ChatPrompt(
        messages=tuple(dict(message) for message in chat.messages),
        tools=tuple(dict(tool) for tool in (chat.tools or [])),
        tool_choice=chat.tool_choice,
        reasoning_effort=chat.reasoning_effort,
    )
    try:
        return tuple(tokenizer.encode_prompt(prompt))
    except (TypeError, ValueError) as exc:
        raise ClientInputError(f"could not tokenize request: {exc}", param="messages") from exc


def _sampling_params(chat: ChatCompletionInput, config: ServerConfig) -> SamplingParams:
    max_tokens = chat.max_completion_tokens
    if max_tokens is None:
        max_tokens = chat.max_tokens
    if max_tokens is None:
        max_tokens = config.default_max_tokens
    return SamplingParams(
        max_tokens=max_tokens,
        temperature=float(chat.temperature),
        top_p=float(chat.top_p),
    )


async def _collect(
    events: AsyncIterator[CompletionEvent],
) -> tuple[str, str, CompletionEnd]:
    reasoning: list[str] = []
    content: list[str] = []
    end: CompletionEnd | None = None
    try:
        async for event in events:
            if isinstance(event, CompletionEnd):
                end = event
            elif event.channel == "reasoning":
                reasoning.append(event.text)
            else:
                content.append(event.text)
    finally:
        await events.aclose()
    if end is None:
        raise GenerationContractError("engine returned no final completion event")
    return "".join(reasoning), "".join(content), end


async def _with_first(
    first: CompletionEvent,
    events: AsyncIterator[CompletionEvent],
) -> AsyncIterator[CompletionEvent]:
    yield first
    async for event in events:
        yield event


def _chunk(
    completion_id: str,
    created: int,
    model: str,
    delta: dict[str, Any],
    *,
    finish_reason: str | None = None,
) -> dict[str, Any]:
    return {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": delta,
                "finish_reason": finish_reason,
            }
        ],
    }


def _usage(end: CompletionEnd) -> dict[str, int]:
    return {
        "prompt_tokens": end.prompt_tokens,
        "completion_tokens": end.completion_tokens,
        "total_tokens": end.total_tokens,
    }


def _sse(payload: dict[str, Any]) -> bytes:
    return f"data: {json.dumps(payload, ensure_ascii=False, separators=(',', ':'))}\n\n".encode(
        "utf-8"
    )


def _error_response(message: str, *, param: str | None = None) -> JSONResponse:
    return JSONResponse(
        status_code=400,
        content=_error_body(message, "invalid_request_error", param),
    )


def _engine_error(exc: Exception) -> JSONResponse:
    kind = "engine_contract_error" if isinstance(exc, GenerationContractError) else "engine_error"
    return JSONResponse(
        status_code=500,
        content=_error_body(f"{type(exc).__name__}: {exc}", kind, None),
    )


def _error_body(message: str, kind: str, param: str | None) -> dict[str, Any]:
    return {
        "error": {
            "message": message,
            "type": kind,
            "param": param,
            "code": kind,
        }
    }


__all__ = ["ChatCompletionInput", "ServerConfig", "create_app", "create_stub_app"]
