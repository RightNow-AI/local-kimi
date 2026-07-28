"""The real Kimi K3 encoder is the oracle for prompt bytes."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, AsyncIterator

import httpx
import pytest

from k3 import pipeline, presets as presets_mod, server as server_mod
from k3.ir import (
    ReasoningDelta,
    ReasoningPart,
    StreamEnd,
    TextDelta,
    TextPart,
    ToolDef,
)
from k3.presets import TOOL_PARSER_DESCRIPTIONS, all_presets
from k3.server import ServerConfig, create_app
from k3.template import (
    render_kimi_k3_prompt,
    render_kimi_k3_tool_call,
    render_tool_prompt,
)
from k3.toolcalls import (
    KimiK3ToolParser,
    ParsedReasoning,
    ParsedText,
    ParsedToolCall,
    parse_all,
    parser_names,
)
from k3.upstream import MockUpstream, UpstreamConfig, _mock_generate


ROOT = Path(__file__).resolve().parents[1]


def load_oracle() -> ModuleType:
    path = ROOT / "reference" / "encoding_k3.py"
    spec = importlib.util.spec_from_file_location("test_encoding_k3_oracle", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ORACLE = load_oracle()

GOLDEN_MESSAGES = [
    {"role": "system", "content": "You are a terse assistant."},
    {"role": "user", "content": "What is the weather in Beijing?"},
    {
        "role": "assistant",
        "reasoning_content": "The user wants weather. I should call get_weather.",
        "content": "",
        "tool_calls": [
            {
                "id": "call_abc123",
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "arguments": '{"city":  "Beijing", "units":"c"}',
                },
            }
        ],
    },
    {"role": "tool", "tool_call_id": "call_abc123", "content": "22C, sunny"},
    {"role": "assistant", "reasoning_content": "Got it.", "content": "22C and sunny."},
]

GOLDEN_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Look up the weather.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }
]


def oracle_render(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
    **kwargs: Any,
) -> str:
    segments = ORACLE.build_chat_segments(messages, tools=tools, **kwargs)
    return "".join(segment.text for segment in segments)


def oracle_assistant_bytes(message: dict[str, Any]) -> str:
    """Render a completed assistant turn with Moonshot's encoder."""
    return oracle_render([message], thinking=True, add_generation_prompt=False)


def oracle_calls(message: dict[str, Any]) -> list[ParsedToolCall]:
    return [
        event
        for event in parse_all(KimiK3ToolParser(), oracle_assistant_bytes(message))
        if isinstance(event, ParsedToolCall)
    ]


def assert_call_contract(
    calls: list[ParsedToolCall], expected: list[tuple[str, dict[str, Any]]]
) -> None:
    assert [(call.name, json.loads(call.arguments)) for call in calls] == expected


ORACLE_MULTI_CALL_MESSAGE = {
    "role": "assistant",
    "reasoning_content": "Two calls are required.",
    "content": "",
    "tool_calls": [
        {
            "id": "zero",
            "type": "function",
            "function": {"name": "ping", "arguments": "{}"},
        },
        {
            "id": "escaped",
            "type": "function",
            "function": {
                "name": 'lookup&"quoted',
                "arguments": '{"key&\\"quoted":"A&B","count":2}',
            },
        },
    ],
}

ORACLE_EXPECTED_CALLS = [
    ("ping", {}),
    ('lookup&"quoted', {'key&"quoted': "A&B", "count": 2}),
]


class RawK3Engine:
    """Stream oracle bytes as engine content, including hostile chunk splits."""

    def __init__(self, raw: str) -> None:
        self.raw = raw

    async def chat_stream(self, payload: dict[str, Any]) -> AsyncIterator[dict[str, Any]]:
        for offset in range(0, len(self.raw), 7):
            yield {
                "id": "oracle-k3",
                "model": "k3",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": self.raw[offset : offset + 7]},
                        "finish_reason": None,
                    }
                ],
            }
        yield {
            "id": "oracle-k3",
            "model": "k3",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }

    async def chat(self, payload: dict[str, Any]) -> dict[str, Any]:
        return {
            "id": "oracle-k3",
            "model": "k3",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": self.raw},
                    "finish_reason": "stop",
                }
            ],
        }


def test_k3_renderer_matches_real_encoder_and_checked_in_golden_byte_for_byte():
    expected = oracle_render(
        GOLDEN_MESSAGES,
        GOLDEN_TOOLS,
        thinking=True,
        add_generation_prompt=True,
    )
    actual = render_kimi_k3_prompt(
        GOLDEN_MESSAGES,
        GOLDEN_TOOLS,
        thinking=True,
        add_generation_prompt=True,
    )

    assert actual.encode("utf-8") == expected.encode("utf-8")
    checked_in = (ROOT / "reference" / "GOLDEN-real-k3-prompt.txt").read_text(
        encoding="utf-8"
    )
    assert actual.encode("utf-8") == checked_in.encode("utf-8")


def test_k3_renderer_matches_oracle_for_all_argument_types_and_empty_think():
    messages = [
        {"role": "user", "content": "Inspect this."},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "opaque-a",
                    "type": "function",
                    "function": {
                        "name": "inspect",
                        "arguments": (
                            '{"string":"x","boolean":true,"null":null,'
                            '"number":2.5,"object":{"a":1},"array":[1,2]}'
                        ),
                    },
                }
            ],
        },
    ]

    assert render_kimi_k3_prompt(messages) == oracle_render(messages)


def test_k3_renderer_matches_oracle_when_tool_results_arrive_out_of_order():
    messages = [
        {"role": "user", "content": "Run both."},
        {
            "role": "assistant",
            "reasoning": "I need both results.",
            "content": "",
            "tool_calls": [
                {
                    "id": "opaque-a",
                    "type": "function",
                    "function": {"name": "alpha", "arguments": '{"x":1}'},
                },
                {
                    "id": "opaque-b",
                    "type": "function",
                    "function": {"name": "beta", "arguments": '{"y":2}'},
                },
            ],
        },
        {"role": "tool", "tool_call_id": "opaque-b", "content": "B"},
        {"role": "tool", "tool_call_id": "opaque-a", "content": "A"},
    ]

    assert render_kimi_k3_prompt(messages) == oracle_render(messages)


def test_kimi_k3_is_the_documented_production_default():
    assert "kimi_k3" in parser_names()
    assert TOOL_PARSER_DESCRIPTIONS["kimi_k3"] == "Kimi K3 XTML format"
    assert all(preset.tool_parser == "kimi_k3" for preset in all_presets())
    prompted = render_tool_prompt(
        [ToolDef(name="get_weather", parameters={"type": "object"})],
        style="kimi_k3",
    )
    assert prompted.startswith("# Tools\nHere are the available tools")
    assert '```json\n[{"function":' in prompted
    assert '"type":"function"}]\n```' in prompted
    assert "<|tool_calls_section_begin|>" not in prompted


def test_mock_upstream_defaults_to_parseable_k3_xtml_without_removing_k2():
    payload = {
        "messages": [{"role": "user", "content": "Weather in Beijing"}],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                },
            }
        ],
    }

    _, k3_content = _mock_generate(payload)
    k3_calls = [
        event
        for event in parse_all(KimiK3ToolParser(), k3_content)
        if isinstance(event, ParsedToolCall)
    ]
    _, k2_content = _mock_generate(payload, "kimi_k2")

    assert [call.name for call in k3_calls] == ["get_weather"]
    assert json.loads(k3_calls[0].arguments) == {"city": "Weather in Beijing"}
    assert "<|open|>tools<|sep|>" in k3_content
    assert "<|tool_calls_section_begin|>" not in k3_content
    assert "<|tool_calls_section_begin|>" in k2_content


async def test_server_configured_tool_parser_reaches_pipeline(monkeypatch: pytest.MonkeyPatch):
    observed: list[str] = []
    real_run = server_mod.pipeline.run

    def recording_run(*args: Any, **kwargs: Any) -> AsyncIterator[Any]:
        observed.append(kwargs["tool_parser"])
        return real_run(*args, **kwargs)

    # Make the preset disagree so this fails if ServerConfig is ignored.
    monkeypatch.setattr(presets_mod.OPENAI, "tool_parser", "kimi_k2")
    monkeypatch.setattr(server_mod.pipeline, "run", recording_run)
    cfg = ServerConfig(
        mock=True,
        forced_client="openai",
        tool_parser="kimi_k3",
        upstream=UpstreamConfig(model="k3"),
    )
    app = create_app(
        cfg,
        engine=MockUpstream(cfg.upstream, tool_parser=cfg.tool_parser or "kimi_k3"),
    )
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://k3.test") as client:
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": "k3",
                "messages": [{"role": "user", "content": "hello"}],
                "stream": False,
            },
        )

    assert response.status_code == 200, response.text
    assert observed == ["kimi_k3"]


@pytest.mark.parametrize("stream", [True, False])
async def test_oracle_k3_turn_splits_reasoning_and_visible_response_in_pipeline(
    stream: bool,
):
    reasoning = "PRIVATE reasoning only."
    visible = "VISIBLE answer only."
    raw = oracle_assistant_bytes(
        {
            "role": "assistant",
            "reasoning_content": reasoning,
            "content": visible,
        }
    )
    events = [
        event
        async for event in pipeline.run(
            RawK3Engine(raw),
            {"model": "k3", "messages": []},
            stream=stream,
        )
    ]

    parsed_reasoning = "".join(
        event.text for event in events if isinstance(event, ReasoningDelta)
    )
    parsed_content = "".join(event.text for event in events if isinstance(event, TextDelta))
    final = next(event for event in events if isinstance(event, StreamEnd))

    assert parsed_reasoning == reasoning
    assert parsed_content == visible
    assert reasoning not in parsed_content
    assert visible not in parsed_reasoning
    assert [part.text for part in final.response.parts if isinstance(part, ReasoningPart)] == [
        reasoning
    ]
    assert [part.text for part in final.response.parts if isinstance(part, TextPart)] == [visible]


def test_oracle_parser_asserts_full_call_list_names_and_arguments():
    raw = oracle_assistant_bytes(ORACLE_MULTI_CALL_MESSAGE)
    calls = oracle_calls(ORACLE_MULTI_CALL_MESSAGE)

    assert "&amp;" in raw and "&quot;" in raw
    assert [call.arguments for call in calls] == [
        "{}",
        json.dumps(ORACLE_EXPECTED_CALLS[1][1], ensure_ascii=False),
    ]
    assert_call_contract(calls, ORACLE_EXPECTED_CALLS)


@pytest.mark.parametrize(
    "mutation",
    [
        "first_call_only",
        "drop_zero_argument_call",
        "drop_all_calls",
        "names_only",
        "drop_last_argument",
        "stringify_typed_argument",
        "leave_name_html_escaped",
        "leave_argument_key_html_escaped",
        "reverse_call_order",
        "double_encode_arguments",
    ],
)
def test_oracle_call_contract_kills_known_semantic_mutations(mutation: str):
    baseline = [
        ParsedToolCall(name=name, arguments=json.dumps(arguments), id=f"call-{index}")
        for index, (name, arguments) in enumerate(ORACLE_EXPECTED_CALLS)
    ]

    if mutation == "first_call_only":
        mutant = baseline[:1]
    elif mutation == "drop_zero_argument_call":
        mutant = baseline[1:]
    elif mutation == "drop_all_calls":
        mutant = []
    elif mutation == "names_only":
        mutant = [
            ParsedToolCall(name=call.name, arguments="{}", id=call.id) for call in baseline
        ]
    elif mutation == "drop_last_argument":
        mutant = [
            baseline[0],
            ParsedToolCall(
                name=baseline[1].name,
                arguments='{"key&\\"quoted":"A&B"}',
                id=baseline[1].id,
            ),
        ]
    elif mutation == "stringify_typed_argument":
        mutant = [
            baseline[0],
            ParsedToolCall(
                name=baseline[1].name,
                arguments='{"key&\\"quoted":"A&B","count":"2"}',
                id=baseline[1].id,
            ),
        ]
    elif mutation == "leave_name_html_escaped":
        mutant = [
            baseline[0],
            ParsedToolCall(
                name="lookup&amp;&quot;quoted",
                arguments=baseline[1].arguments,
                id=baseline[1].id,
            ),
        ]
    elif mutation == "leave_argument_key_html_escaped":
        mutant = [
            baseline[0],
            ParsedToolCall(
                name=baseline[1].name,
                arguments='{"key&amp;&quot;quoted":"A&B","count":2}',
                id=baseline[1].id,
            ),
        ]
    elif mutation == "reverse_call_order":
        mutant = list(reversed(baseline))
    else:
        mutant = [
            baseline[0],
            ParsedToolCall(
                name=baseline[1].name,
                arguments=json.dumps(baseline[1].arguments),
                id=baseline[1].id,
            ),
        ]

    with pytest.raises(AssertionError):
        assert_call_contract(mutant, ORACLE_EXPECTED_CALLS)


@pytest.mark.parametrize("mutation", ["concatenate_into_content", "swap_channels"])
def test_oracle_channel_contract_kills_known_routing_mutations(mutation: str):
    reasoning = "private"
    visible = "public"
    if mutation == "concatenate_into_content":
        mutant: list[object] = [ParsedText(reasoning + visible)]
    else:
        mutant = [ParsedReasoning(visible), ParsedText(reasoning)]

    with pytest.raises(AssertionError):
        assert "".join(
            event.text for event in mutant if isinstance(event, ParsedReasoning)
        ) == reasoning
        assert "".join(event.text for event in mutant if isinstance(event, ParsedText)) == visible


def test_json_string_arguments_render_byte_for_byte_like_the_oracle():
    arguments = '{"city":"Beijing","active":true,"count":2}'
    message = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "wire-shape",
                "type": "function",
                "function": {"name": "inspect", "arguments": arguments},
            }
        ],
    }
    oracle_bytes = oracle_assistant_bytes(message)
    start = oracle_bytes.index("<|open|>call ")
    close = "<|close|>call<|sep|>"
    expected = oracle_bytes[start : oracle_bytes.index(close, start) + len(close)]

    assert render_kimi_k3_tool_call("inspect", arguments) == expected


def test_unparseable_json_string_arguments_fall_back_without_being_dropped():
    arguments = '{"city":'
    rendered = render_kimi_k3_tool_call("inspect", arguments)
    raw = "<|open|>tools<|sep|>" + rendered + "<|close|>tools<|sep|>"
    calls = [
        event
        for event in parse_all(KimiK3ToolParser(), raw)
        if isinstance(event, ParsedToolCall)
    ]

    assert len(calls) == 1
    assert calls[0].arguments == arguments
