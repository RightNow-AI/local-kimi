"""Small OpenAI-compatible chat-completions client for endpoint evaluations."""

from __future__ import annotations

import time
from collections.abc import Mapping
from typing import Any


def chat_completions_url(base_url: str) -> str:
    normalized = base_url.rstrip("/")
    if normalized.endswith("/v1"):
        return f"{normalized}/chat/completions"
    return f"{normalized}/v1/chat/completions"


def _completed_response(payload: Any) -> dict[str, Any]:
    out = {
        "scoreable": False,
        "cleanTerminal": False,
        "content": "",
        "reasoning": "",
        "toolCalls": [],
        "finishReason": None,
        "malformed": False,
        "malformedDetail": None,
    }

    def refuse(detail: str) -> dict[str, Any]:
        out["malformed"] = True
        out["malformedDetail"] = detail
        return out

    if not isinstance(payload, Mapping):
        return refuse("completion payload is not an object")
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return refuse("completion choices is not a non-empty list")
    choice = choices[0]
    if not isinstance(choice, Mapping):
        return refuse("completion choice is not an object")
    message = choice.get("message")
    if not isinstance(message, Mapping):
        return refuse("completion message is not an object")

    content = message.get("content")
    if content is not None and not isinstance(content, str):
        return refuse("completion content is not text")
    reasoning = message.get("reasoning")
    if reasoning is None:
        reasoning = message.get("reasoning_content")
    if reasoning is not None and not isinstance(reasoning, str):
        return refuse("completion reasoning is not text")
    out["content"] = content or ""
    out["reasoning"] = reasoning or ""

    raw_tool_calls = message.get("tool_calls") or []
    if not isinstance(raw_tool_calls, list):
        return refuse("completion tool_calls is not a list")
    tool_calls = []
    for raw_call in raw_tool_calls:
        if not isinstance(raw_call, Mapping):
            return refuse("completion tool call is not an object")
        function = raw_call.get("function")
        if not isinstance(function, Mapping):
            return refuse("completion tool call function is not an object")
        tool_calls.append(
            {
                "id": raw_call.get("id"),
                "type": raw_call.get("type"),
                "name": function.get("name"),
                "arguments": function.get("arguments"),
            }
        )
    out["toolCalls"] = tool_calls

    finish_reason = choice.get("finish_reason")
    if finish_reason is not None and not isinstance(finish_reason, str):
        return refuse("finish_reason is not text")
    out["finishReason"] = finish_reason
    out["cleanTerminal"] = finish_reason in ("stop", "eos", "tool_calls")
    out["scoreable"] = out["cleanTerminal"]
    return out


class OpenAIEndpointClient:
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str | None,
        timeout_s: float,
    ) -> None:
        if not base_url or not base_url.strip():
            raise ValueError("base_url is required")
        if timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout_s = float(timeout_s)

    def complete(
        self,
        *,
        model_id: str,
        messages: list[dict[str, Any]],
        sampling: Mapping[str, Any],
        tools: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        import httpx

        body: dict[str, Any] = {
            "model": model_id,
            "messages": messages,
            "temperature": sampling["temperature"],
            "top_p": sampling["top_p"],
            "max_tokens": sampling["max_tokens"],
            "n": 1,
            "stream": False,
        }
        if tools is not None:
            body["tools"] = tools
            body["tool_choice"] = sampling.get("tool_choice", "auto")
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else None
        started = time.perf_counter()
        try:
            response = httpx.post(
                chat_completions_url(self.base_url),
                json=body,
                headers=headers,
                timeout=httpx.Timeout(self.timeout_s, connect=min(60.0, self.timeout_s)),
            )
        except Exception as exc:
            return {
                "scoreable": False,
                "cleanTerminal": False,
                "httpStatus": None,
                "transportError": f"{type(exc).__name__}: {str(exc)[:500]}",
                "request": body,
                "wallS": round(time.perf_counter() - started, 3),
            }
        if response.status_code != 200:
            return {
                "scoreable": False,
                "cleanTerminal": False,
                "httpStatus": response.status_code,
                "httpBodyPreview": response.text[:2000],
                "transportError": None,
                "request": body,
                "wallS": round(time.perf_counter() - started, 3),
            }
        try:
            payload = response.json()
        except Exception as exc:
            return {
                "scoreable": False,
                "cleanTerminal": False,
                "httpStatus": response.status_code,
                "transportError": None,
                "malformed": True,
                "malformedDetail": f"response body is not JSON: {type(exc).__name__}",
                "httpBodyPreview": response.text[:2000],
                "request": body,
                "wallS": round(time.perf_counter() - started, 3),
            }
        completed = _completed_response(payload)
        completed.update(
            {
                "httpStatus": response.status_code,
                "transportError": None,
                "request": body,
                "rawResponse": payload,
                "wallS": round(time.perf_counter() - started, 3),
            }
        )
        return completed


__all__ = ["OpenAIEndpointClient", "chat_completions_url"]
