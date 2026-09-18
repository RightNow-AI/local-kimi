import asyncio
import json

import pytest
from fastapi.testclient import TestClient

from engine.serve import ByteChatTokenizer, EchoEngine, ServerConfig, create_app


def _stream_server(*, delay: float = 0.0):
    tokenizer = ByteChatTokenizer()
    engine = EchoEngine(
        tokenizer,
        reasoning="separate reasoning",
        response_prefix="answer: ",
        delay=delay,
    )
    app = create_app(
        engine,
        tokenizer,
        ServerConfig(model="k3-stream", default_max_tokens=256),
    )
    return app, engine


def _data_frames(response) -> list[str]:
    return [
        line[len("data: ") :]
        for line in response.iter_lines()
        if line.startswith("data: ")
    ]


def test_stream_is_valid_chat_chunks_with_reasoning_usage_and_done():
    app, engine = _stream_server()

    with TestClient(app) as client:
        with client.stream(
            "POST",
            "/v1/chat/completions",
            json={
                "model": "k3-stream",
                "messages": [{"role": "user", "content": "hello"}],
                "max_tokens": 256,
                "stream": True,
                "stream_options": {"include_usage": True},
            },
        ) as response:
            frames = _data_frames(response)

    assert response.status_code == 200
    assert frames[-1] == "[DONE]"

    chunks = [json.loads(frame) for frame in frames[:-1]]
    assert chunks
    assert all(chunk["object"] == "chat.completion.chunk" for chunk in chunks)
    assert len({chunk["id"] for chunk in chunks}) == 1

    deltas = [choice["delta"] for chunk in chunks for choice in chunk["choices"]]
    reasoning = "".join(delta.get("reasoning_content", "") for delta in deltas)
    content = "".join(delta.get("content", "") for delta in deltas)
    assert reasoning == "separate reasoning"
    assert content == "answer: hello"
    assert "separate reasoning" not in content

    finish_chunks = [
        chunk
        for chunk in chunks
        if chunk["choices"] and chunk["choices"][0]["finish_reason"] is not None
    ]
    assert len(finish_chunks) == 1
    assert finish_chunks[0]["choices"][0]["finish_reason"] == "stop"

    usage_chunks = [chunk for chunk in chunks if not chunk["choices"]]
    assert len(usage_chunks) == 1
    usage = usage_chunks[0]["usage"]
    assert usage["prompt_tokens"] == len(engine.prompt_token_ids[0])
    assert usage["completion_tokens"] == len(engine.generated_token_ids[0])
    assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]


@pytest.mark.asyncio
async def test_disconnect_closes_generation_and_releases_the_engine():
    app, engine = _stream_server(delay=0.01)
    request_body = json.dumps(
        {
            "model": "k3-stream",
            "messages": [{"role": "user", "content": "disconnect me"}],
            "max_tokens": 256,
            "stream": True,
        }
    ).encode("utf-8")
    disconnect = asyncio.Event()
    request_sent = False
    sent: list[dict] = []

    async def receive():
        nonlocal request_sent
        if not request_sent:
            request_sent = True
            return {"type": "http.request", "body": request_body, "more_body": False}
        await disconnect.wait()
        return {"type": "http.disconnect"}

    async def send(message):
        sent.append(message)
        if message["type"] == "http.response.body" and message.get("more_body"):
            disconnect.set()

    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/v1/chat/completions",
        "raw_path": b"/v1/chat/completions",
        "query_string": b"",
        "root_path": "",
        "headers": [(b"content-type", b"application/json")],
        "client": ("127.0.0.1", 12345),
        "server": ("127.0.0.1", 80),
    }

    await asyncio.wait_for(app(scope, receive, send), timeout=2.0)

    assert any(message["type"] == "http.response.start" for message in sent)
    assert engine.active_generations == 0
    assert engine.cancelled_generations == 1
