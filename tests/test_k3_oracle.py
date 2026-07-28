"""The real Kimi K3 encoder is the oracle for prompt bytes."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

from k3.ir import ToolDef
from k3.presets import TOOL_PARSER_DESCRIPTIONS, all_presets
from k3.template import render_kimi_k3_prompt, render_tool_prompt
from k3.toolcalls import KimiK3ToolParser, ParsedToolCall, parse_all, parser_names
from k3.upstream import _mock_generate


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


def test_kimi_k3_is_selectable_and_documented_without_flipping_defaults():
    assert "kimi_k3" in parser_names()
    assert TOOL_PARSER_DESCRIPTIONS["kimi_k3"] == "Kimi K3 XTML format"
    assert all(preset.tool_parser != "kimi_k3" for preset in all_presets())
    prompted = render_tool_prompt(
        [ToolDef(name="get_weather", parameters={"type": "object"})],
        style="kimi_k3",
    )
    assert prompted.startswith("# Tools\nHere are the available tools")
    assert '```json\n[{"function":' in prompted
    assert '"type":"function"}]\n```' in prompted
    assert "<|tool_calls_section_begin|>" not in prompted


def test_mock_upstream_can_emit_parseable_k3_xtml_without_removing_k2():
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

    _, k3_content = _mock_generate(payload, "kimi_k3")
    k3_calls = [
        event
        for event in parse_all(KimiK3ToolParser(), k3_content)
        if isinstance(event, ParsedToolCall)
    ]
    _, k2_content = _mock_generate(payload, "kimi")

    assert [call.name for call in k3_calls] == ["get_weather"]
    assert "<|open|>tools<|sep|>" in k3_content
    assert "<|tool_calls_section_begin|>" not in k3_content
    assert "<|tool_calls_section_begin|>" in k2_content
