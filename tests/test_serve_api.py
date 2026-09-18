import asyncio

import httpx
import pytest
from fastapi.testclient import TestClient

from engine.serve import ByteChatTokenizer, EchoEngine, ServerConfig, create_app


def _server(*, delay: float = 0.0, emit_eos: bool = True):
    tokenizer = ByteChatTokenizer()
    engine = EchoEngine(
        tokenizer,
        reasoning="reasoning bytes survive",
        delay=delay,
        emit_eos=emit_eos,
    )
    app = create_app(
        engine,
        tokenizer,
        ServerConfig(model="k3-test", default_max_tokens=256),
    )
    return app, engine


def test_health_and_models_advertise_the_served_engine():
    app, _ = _server()

    with TestClient(app) as client:
        health = client.get("/health")
        models = client.get("/v1/models")

    assert health.status_code == 200
    assert health.json() == {"status": "ok", "engine": "cpu echo engine"}
    assert models.status_code == 200
    assert models.json()["object"] == "list"
    assert models.json()["data"][0]["id"] == "k3-test"


def test_non_streaming_preserves_reasoning_and_reports_real_token_counts():
    app, engine = _server()
    request = {
        "model": "k3-test",
        "messages": [{"role": "user", "content": "snowman: \u2603"}],
        "max_tokens": 256,
    }

    with TestClient(app) as client:
        response = client.post("/v1/chat/completions", json=request)

    assert response.status_code == 200
    body = response.json()
    choice = body["choices"][0]
    assert choice["message"]["reasoning_content"] == "reasoning bytes survive"
    assert choice["message"]["content"] == "echo: snowman: \u2603"
    assert choice["finish_reason"] == "stop"

    usage = body["usage"]
    assert usage["prompt_tokens"] == len(engine.prompt_token_ids[0])
    assert usage["completion_tokens"] == len(engine.generated_token_ids[0])
    assert usage["total_tokens"] == (
        len(engine.prompt_token_ids[0]) + len(engine.generated_token_ids[0])
    )
    assert usage["prompt_tokens"] != max(1, len("snowman: \u2603") // 4)


def test_max_tokens_without_eos_finishes_with_length():
    app, engine = _server(emit_eos=False)

    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "k3-test",
                "messages": [{"role": "user", "content": "long answer"}],
                "max_tokens": 4,
            },
        )

    assert response.status_code == 200
    body = response.json()
    assert body["choices"][0]["finish_reason"] == "length"
    assert body["usage"]["completion_tokens"] == 4
    assert len(engine.generated_token_ids[0]) == 4


@pytest.mark.parametrize(
    "body",
    [
        ["not", "an", "object"],
        {"messages": [{"role": "user", "content": "hello"}]},
        {"model": "k3-test"},
        {
            "model": "k3-test",
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": "12",
        },
    ],
    ids=["non-dict", "missing-model", "missing-messages", "non-integer-max-tokens"],
)
def test_malformed_bodies_return_openai_json_400(body):
    app, _ = _server()

    with TestClient(app) as client:
        response = client.post("/v1/chat/completions", json=body)

    assert response.status_code == 400
    assert response.headers["content-type"].startswith("application/json")
    error = response.json()["error"]
    assert error["type"] == "invalid_request_error"
    assert isinstance(error["message"], str)
    assert error["message"]


@pytest.mark.asyncio
async def test_default_runtime_serializes_a_single_threaded_engine():
    app, engine = _server(delay=0.001)
    transport = httpx.ASGITransport(app=app)
    request = {
        "model": "k3-test",
        "messages": [{"role": "user", "content": "concurrent"}],
        "max_tokens": 256,
    }

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        first, second = await asyncio.gather(
            client.post("/v1/chat/completions", json=request),
            client.post("/v1/chat/completions", json=request),
        )

    assert first.status_code == 200
    assert second.status_code == 200
    assert engine.max_active_generations == 1
