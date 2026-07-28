"""Unit tests for the Anthropic Messages dialect.

These exercise :func:`k3.dialects.anthropic_messages.ingress` /
:func:`~k3.dialects.anthropic_messages.egress` /
:func:`~k3.dialects.anthropic_messages.egress_stream` /
:func:`~k3.dialects.anthropic_messages.count_tokens` in isolation — no server,
no engine, no network. A :class:`~k3.dialects.base.DialectContext` is built by
hand and canonical events are fed through a tiny async generator.
"""

from __future__ import annotations

import json
from dataclasses import replace
from typing import Any, AsyncIterator, Iterable

from k3 import presets as presets_mod
from k3.dialects import anthropic_messages as ant
from k3.dialects.base import DialectContext
from k3.ir import (
    CanonicalResponse,
    ImagePart,
    Message,
    ReasoningDelta,
    ReasoningEnd,
    ReasoningPart,
    StreamEnd,
    StreamEvent,
    StreamStart,
    TextDelta,
    TextPart,
    ToolCallArgsDelta,
    ToolCallEnd,
    ToolCallPart,
    ToolCallStart,
    ToolResultPart,
    Usage,
)
from k3.reasoning import ReasoningLedger, ReasoningPolicy
from k3.record import parse_sse

# --------------------------------------------------------------------------
# harness
# --------------------------------------------------------------------------


def make_ctx(preset_name: str = "claude-code", **overrides: Any) -> DialectContext:
    preset = presets_mod.get(preset_name)
    if "reasoning" in overrides:
        preset = replace(preset, reasoning=overrides.pop("reasoning"))
    return DialectContext(
        preset=preset,
        ledger=ReasoningLedger(),
        client_model="k3",
        **overrides,
    )


async def _agen(events: Iterable[StreamEvent]) -> AsyncIterator[StreamEvent]:
    for event in events:
        yield event


async def run_stream(events: Iterable[StreamEvent], ctx: DialectContext) -> list[dict[str, Any]]:
    """Drive ``egress_stream`` and decode the SSE bytes it produced."""
    chunks: list[bytes] = []
    async for chunk in ant.egress_stream(_agen(events), ctx):
        assert isinstance(chunk, (bytes, bytearray))
        chunks.append(bytes(chunk))
    return parse_sse(b"".join(chunks))


def types_of(records: list[dict[str, Any]]) -> list[str]:
    return [r["data"]["type"] for r in records]


def response_with(*parts: Any, stop_reason: str = "stop") -> CanonicalResponse:
    return CanonicalResponse(
        id="msg_fixed",
        model="k3",
        parts=list(parts),
        stop_reason=stop_reason,  # type: ignore[arg-type]
        usage=Usage(input_tokens=11, output_tokens=7),
    )


def parts_of(msg: Message, kind: type) -> list[Any]:
    return [p for p in msg.parts if isinstance(p, kind)]


# --------------------------------------------------------------------------
# 1. system prompt, both spellings
# --------------------------------------------------------------------------


def test_system_as_plain_string_lands_in_req_system() -> None:
    req = ant.ingress({"system": "you are helpful"}, make_ctx())
    assert req.system == ["you are helpful"]


def test_system_as_text_blocks_lands_in_req_system() -> None:
    body = {
        "system": [
            {"type": "text", "text": "first"},
            {"type": "text", "text": "second"},
        ]
    }
    req = ant.ingress(body, make_ctx())
    assert req.system == ["first", "second"]


# --------------------------------------------------------------------------
# 2/3. tool results split out of the user turn, and come first
# --------------------------------------------------------------------------


def test_tool_result_plus_text_splits_into_tool_then_user() -> None:
    body = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": ant.TOOL_ID_PREFIX + "call_7",
                        "content": "42",
                    },
                    {"type": "text", "text": "and now explain it"},
                ],
            }
        ]
    }
    req = ant.ingress(body, make_ctx())

    assert [m.role for m in req.messages] == ["tool", "user"]

    results = req.messages[0].tool_results()
    assert len(results) == 1
    assert isinstance(results[0], ToolResultPart)
    assert results[0].tool_call_id == "call_7"
    assert results[0].content == "42"
    assert results[0].is_error is False

    assert req.messages[1].text() == "and now explain it"


def test_tool_result_content_list_is_flattened_and_is_error_survives() -> None:
    body = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "call_9",
                        "is_error": True,
                        "content": [
                            {"type": "text", "text": "line one"},
                            {"type": "text", "text": "line two"},
                        ],
                    }
                ],
            }
        ]
    }
    req = ant.ingress(body, make_ctx())

    assert [m.role for m in req.messages] == ["tool"]
    result = req.messages[0].tool_results()[0]
    assert isinstance(result.content, str)
    assert result.content == "line one\nline two"
    assert result.is_error is True


# --------------------------------------------------------------------------
# 4/5. assistant reasoning + tool calls
# --------------------------------------------------------------------------


def test_thinking_and_tool_use_become_reasoning_and_tool_call_parts() -> None:
    body = {
        "messages": [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "thinking",
                        "thinking": "weigh the options",
                        "signature": "k3r1.abc",
                    },
                    {
                        "type": "tool_use",
                        "id": ant.TOOL_ID_PREFIX + "call_42",
                        "name": "do_thing",
                        "input": {"a": 1},
                    },
                ],
            }
        ]
    }
    req = ant.ingress(body, make_ctx())

    assert len(req.messages) == 1
    msg = req.messages[0]
    assert msg.role == "assistant"

    reasoning = parts_of(msg, ReasoningPart)
    assert len(reasoning) == 1
    assert reasoning[0].text == "weigh the options"
    assert reasoning[0].signature == "k3r1.abc"
    assert reasoning[0].redacted is False

    calls = msg.tool_calls()
    assert len(calls) == 1
    assert calls[0].id == "call_42"  # un-prefixed by from_anthropic_tool_id
    assert not calls[0].id.startswith(ant.TOOL_ID_PREFIX)
    assert calls[0].name == "do_thing"
    assert json.loads(calls[0].arguments) == {"a": 1}


def test_unprefixed_tool_use_id_passes_through_untouched() -> None:
    body = {
        "messages": [
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "toolu_real_01", "name": "f", "input": {}}
                ],
            }
        ]
    }
    req = ant.ingress(body, make_ctx())
    assert req.messages[0].tool_calls()[0].id == "toolu_real_01"


def test_redacted_thinking_becomes_redacted_reasoning_part() -> None:
    body = {
        "messages": [
            {
                "role": "assistant",
                "content": [{"type": "redacted_thinking", "data": "opaque-blob"}],
            }
        ]
    }
    req = ant.ingress(body, make_ctx())

    reasoning = parts_of(req.messages[0], ReasoningPart)
    assert len(reasoning) == 1
    assert reasoning[0].redacted is True
    assert reasoning[0].signature == "opaque-blob"
    assert reasoning[0].text == ""


# --------------------------------------------------------------------------
# 6. images
# --------------------------------------------------------------------------


def test_base64_image_block_becomes_inline_image_part() -> None:
    body = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": "QUJD",
                        },
                    }
                ],
            }
        ]
    }
    req = ant.ingress(body, make_ctx())

    images = parts_of(req.messages[0], ImagePart)
    assert len(images) == 1
    assert images[0].is_url is False
    assert images[0].media_type == "image/jpeg"
    assert images[0].data == "QUJD"


def test_url_image_block_becomes_url_image_part() -> None:
    body = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {"type": "url", "url": "https://example.test/a.png"},
                    }
                ],
            }
        ]
    }
    req = ant.ingress(body, make_ctx())

    images = parts_of(req.messages[0], ImagePart)
    assert len(images) == 1
    assert images[0].is_url is True
    assert images[0].data == "https://example.test/a.png"


# --------------------------------------------------------------------------
# 7. tools and tool_choice
# --------------------------------------------------------------------------


def test_input_schema_maps_to_tool_parameters() -> None:
    schema = {"type": "object", "properties": {"q": {"type": "string"}}, "required": ["q"]}
    req = ant.ingress(
        {"tools": [{"name": "search", "description": "find things", "input_schema": schema}]},
        make_ctx(),
    )
    assert len(req.tools) == 1
    assert req.tools[0].name == "search"
    assert req.tools[0].description == "find things"
    assert req.tools[0].parameters == schema


def test_tool_choice_auto_maps_to_auto() -> None:
    assert ant.ingress({"tool_choice": {"type": "auto"}}, make_ctx()).tool_choice == "auto"


def test_tool_choice_any_maps_to_required() -> None:
    assert ant.ingress({"tool_choice": {"type": "any"}}, make_ctx()).tool_choice == "required"


def test_tool_choice_none_maps_to_none_string() -> None:
    assert ant.ingress({"tool_choice": {"type": "none"}}, make_ctx()).tool_choice == "none"


def test_tool_choice_named_tool_maps_to_openai_function_shape() -> None:
    req = ant.ingress({"tool_choice": {"type": "tool", "name": "x"}}, make_ctx())
    assert req.tool_choice == {"type": "function", "function": {"name": "x"}}


# --------------------------------------------------------------------------
# 8. thinking budget -> reasoning effort, at the documented boundaries
# --------------------------------------------------------------------------


def _effort_for(budget: int) -> str | None:
    body = {"thinking": {"type": "enabled", "budget_tokens": budget}}
    return ant.ingress(body, make_ctx()).reasoning_effort


def test_thinking_budget_boundaries_map_to_effort_levels() -> None:
    # The boundaries themselves, per ``_effort_from_budget``: <=4096 low,
    # <=16384 medium, above that high.
    assert _effort_for(4096) == "low"
    assert _effort_for(16384) == "medium"
    assert _effort_for(32000) == "high"


def test_thinking_budget_just_past_each_boundary_moves_up_a_level() -> None:
    assert _effort_for(4097) == "medium"
    assert _effort_for(16385) == "high"


def test_thinking_budget_is_recorded_and_thinking_enabled_is_flagged() -> None:
    req = ant.ingress({"thinking": {"type": "enabled", "budget_tokens": 4096}}, make_ctx())
    assert req.thinking_budget == 4096
    assert req.thinking_enabled is True


# --------------------------------------------------------------------------
# 9/10. stop sequences and the empty body
# --------------------------------------------------------------------------


def test_stop_sequences_as_string_becomes_a_list() -> None:
    req = ant.ingress({"stop_sequences": "STOP"}, make_ctx())
    assert req.stop == ["STOP"]


def test_stop_sequences_as_list_stays_a_list() -> None:
    req = ant.ingress({"stop_sequences": ["A", "B"]}, make_ctx())
    assert req.stop == ["A", "B"]


def test_empty_body_yields_a_usable_request() -> None:
    req = ant.ingress({}, make_ctx())
    assert req.model == "k3"
    assert req.messages == []
    assert req.system == []
    assert req.tools == []
    assert req.stop == []
    assert req.stream is False
    # preset defaults still fill in
    assert req.max_tokens == presets_mod.get("claude-code").defaults.max_tokens


# --------------------------------------------------------------------------
# 11. reasoning egress under each policy
# --------------------------------------------------------------------------


def test_reasoning_becomes_thinking_block_under_claude_code() -> None:
    response = response_with(
        ReasoningPart(text="deliberating", signature="k3r1.sig"),
        TextPart("done"),
    )
    body = ant.egress(response, make_ctx())

    thinking = [b for b in body["content"] if b["type"] == "thinking"]
    assert len(thinking) == 1
    assert thinking[0]["thinking"] == "deliberating"
    assert thinking[0]["signature"] == "k3r1.sig"
    # and it comes before the visible text
    assert body["content"][0]["type"] == "thinking"
    assert body["content"][1] == {"type": "text", "text": "done"}


def test_strip_policy_emits_no_thinking_block() -> None:
    response = response_with(
        ReasoningPart(text="deliberating", signature="k3r1.sig"),
        TextPart("done"),
    )
    body = ant.egress(response, make_ctx(reasoning=ReasoningPolicy.STRIP))

    assert [b["type"] for b in body["content"]] == ["text"]
    assert body["content"][0]["text"] == "done"
    assert "deliberating" not in json.dumps(body)


def test_inline_tags_policy_wraps_reasoning_in_think_tags() -> None:
    response = response_with(
        ReasoningPart(text="deliberating", signature="k3r1.sig"),
        TextPart("done"),
    )
    body = ant.egress(response, make_ctx(reasoning=ReasoningPolicy.INLINE_TAGS))

    assert [b["type"] for b in body["content"]] == ["text"]
    assert body["content"][0]["text"] == "<think>deliberating</think>done"


def test_redacted_reasoning_becomes_redacted_thinking_block() -> None:
    response = response_with(ReasoningPart(text="", signature="blob", redacted=True))
    body = ant.egress(response, make_ctx())
    assert body["content"][0] == {"type": "redacted_thinking", "data": "blob"}


# --------------------------------------------------------------------------
# 12/13/14. tool_use blocks, stop reasons, empty responses
# --------------------------------------------------------------------------


def test_tool_use_input_is_the_parsed_arguments_object_and_id_is_prefixed() -> None:
    raw_args = '{"zeta":  1, "alpha":"x  y"}'
    response = response_with(
        ToolCallPart(id="call_1", name="do_thing", arguments=raw_args),
        stop_reason="tool_calls",
    )
    body = ant.egress(response, make_ctx())

    blocks = [b for b in body["content"] if b["type"] == "tool_use"]
    assert len(blocks) == 1
    assert blocks[0]["input"] == {"zeta": 1, "alpha": "x  y"}
    assert isinstance(blocks[0]["input"], dict)
    assert blocks[0]["name"] == "do_thing"
    assert blocks[0]["id"] == ant.TOOL_ID_PREFIX + "call_1"
    assert ant.from_anthropic_tool_id(blocks[0]["id"]) == "call_1"


def test_stop_reason_mapping() -> None:
    ctx = make_ctx()
    assert ant.egress(response_with(TextPart("a"), stop_reason="stop"), ctx)["stop_reason"] == "end_turn"
    assert ant.egress(response_with(TextPart("a"), stop_reason="length"), ctx)["stop_reason"] == "max_tokens"
    assert (
        ant.egress(response_with(TextPart("a"), stop_reason="tool_calls"), ctx)["stop_reason"]
        == "tool_use"
    )


def test_response_with_no_parts_still_has_non_empty_content() -> None:
    body = ant.egress(response_with(), make_ctx())
    assert isinstance(body["content"], list)
    assert body["content"] != []
    assert body["content"] == [{"type": "text", "text": ""}]
    assert body["role"] == "assistant"
    assert body["type"] == "message"
    assert body["id"].startswith("msg_")


# --------------------------------------------------------------------------
# 15-19. streaming
# --------------------------------------------------------------------------


def _stream_end(stop_reason: str = "stop", **kwargs: Any) -> StreamEnd:
    response = kwargs.pop("response", None) or response_with(stop_reason=stop_reason)
    return StreamEnd(
        stop_reason=stop_reason,  # type: ignore[arg-type]
        usage=Usage(input_tokens=5, output_tokens=3),
        response=response,
    )


async def test_stream_envelope_has_one_start_one_delta_one_stop() -> None:
    events = [
        StreamStart(id="msg_x", model="k3"),
        TextDelta("hello"),
        _stream_end("tool_calls"),
    ]
    records = await run_stream(events, make_ctx())
    kinds = types_of(records)

    assert kinds[0] == "message_start"
    assert kinds[-1] == "message_stop"
    assert kinds.count("message_start") == 1
    assert kinds.count("message_stop") == 1
    assert kinds.count("message_delta") == 1

    delta = next(r["data"] for r in records if r["data"]["type"] == "message_delta")
    assert delta["delta"]["stop_reason"] == "tool_use"

    # every record carries the matching named event line
    for record in records:
        assert record["event"] == record["data"]["type"]


async def test_signature_delta_lands_inside_the_thinking_block() -> None:
    events = [
        StreamStart(id="msg_x", model="k3"),
        ReasoningDelta("weigh "),
        ReasoningDelta("options"),
        ReasoningEnd(text="weigh options", signature="k3r1.sig"),
        TextDelta("answer"),
        _stream_end("stop"),
    ]
    records = await run_stream(events, make_ctx())
    kinds = types_of(records)

    start_idx = next(
        i
        for i, r in enumerate(records)
        if r["data"]["type"] == "content_block_start"
        and r["data"]["content_block"]["type"] == "thinking"
    )
    sig_idx = next(
        i
        for i, r in enumerate(records)
        if r["data"]["type"] == "content_block_delta"
        and r["data"]["delta"]["type"] == "signature_delta"
    )
    stop_idx = next(
        i for i, k in enumerate(kinds) if k == "content_block_stop" and i > start_idx
    )

    assert start_idx < sig_idx < stop_idx
    assert records[sig_idx]["data"]["delta"]["signature"] == "k3r1.sig"
    assert records[sig_idx]["data"]["index"] == records[start_idx]["data"]["index"]

    thinking = "".join(
        r["data"]["delta"]["thinking"]
        for r in records
        if r["data"]["type"] == "content_block_delta"
        and r["data"]["delta"]["type"] == "thinking_delta"
    )
    assert thinking == "weigh options"


async def test_interleaved_reasoning_produces_nested_non_overlapping_blocks() -> None:
    events = [
        StreamStart(id="msg_x", model="k3"),
        ReasoningDelta("r1"),
        ReasoningEnd(text="r1", signature="sig1"),
        TextDelta("t1"),
        ReasoningDelta("r2"),
        ReasoningEnd(text="r2", signature="sig2"),
        TextDelta("t2"),
        _stream_end("stop"),
    ]
    records = await run_stream(events, make_ctx())

    open_index: int | None = None
    seen: list[tuple[int, str]] = []
    highest = -1
    for record in records:
        data = record["data"]
        if data["type"] == "content_block_start":
            assert open_index is None, "a block was opened while another was still open"
            index = data["index"]
            assert index > highest, "content block indices must increase monotonically"
            highest = index
            open_index = index
            seen.append((index, data["content_block"]["type"]))
        elif data["type"] == "content_block_delta":
            assert open_index is not None, "delta outside any content block"
            assert data["index"] == open_index
        elif data["type"] == "content_block_stop":
            assert open_index is not None, "content_block_stop without a start"
            assert data["index"] == open_index
            open_index = None

    assert open_index is None, "a content block was never closed"
    assert [kind for _, kind in seen] == ["thinking", "text", "thinking", "text"]
    assert [index for index, _ in seen] == [0, 1, 2, 3]

    starts = sum(1 for r in records if r["data"]["type"] == "content_block_start")
    stops = sum(1 for r in records if r["data"]["type"] == "content_block_stop")
    assert starts == stops == 4

    sigs = [
        r["data"]["delta"]["signature"]
        for r in records
        if r["data"]["type"] == "content_block_delta"
        and r["data"]["delta"]["type"] == "signature_delta"
    ]
    assert sigs == ["sig1", "sig2"]


async def test_empty_event_stream_still_produces_a_well_formed_envelope() -> None:
    records = await run_stream([], make_ctx())
    kinds = types_of(records)

    assert kinds[0] == "message_start"
    assert kinds[-1] == "message_stop"
    assert kinds.count("message_start") == 1
    assert kinds.count("message_delta") == 1
    assert kinds.count("message_stop") == 1
    assert "content_block_start" not in kinds
    assert "content_block_stop" not in kinds

    start = records[0]["data"]["message"]
    assert start["role"] == "assistant"
    assert start["type"] == "message"
    assert start["content"] == []
    assert start["id"].startswith("msg_")
    assert records[-2]["data"]["delta"]["stop_reason"] == "end_turn"


async def test_tool_call_arg_deltas_concatenate_to_the_original_arguments() -> None:
    raw_args = '{"zeta":  1, "alpha":"x  y"}'
    pieces = [raw_args[:6], raw_args[6:15], raw_args[15:]]
    assert "".join(pieces) == raw_args

    events: list[StreamEvent] = [
        StreamStart(id="msg_x", model="k3"),
        ToolCallStart(index=0, id="call_1", name="do_thing"),
        *[ToolCallArgsDelta(index=0, text=piece) for piece in pieces],
        ToolCallEnd(index=0, id="call_1", name="do_thing", arguments=raw_args),
        _stream_end("tool_calls"),
    ]
    records = await run_stream(events, make_ctx())

    block_start = next(
        r["data"]
        for r in records
        if r["data"]["type"] == "content_block_start"
        and r["data"]["content_block"]["type"] == "tool_use"
    )
    assert block_start["content_block"]["id"] == ant.TOOL_ID_PREFIX + "call_1"
    assert block_start["content_block"]["name"] == "do_thing"

    partials = [
        r["data"]["delta"]["partial_json"]
        for r in records
        if r["data"]["type"] == "content_block_delta"
        and r["data"]["delta"]["type"] == "input_json_delta"
    ]
    assert len(partials) == len(pieces)
    assert "".join(partials) == raw_args


# --------------------------------------------------------------------------
# 20. count_tokens
# --------------------------------------------------------------------------


def test_count_tokens_returns_a_positive_int_and_grows_with_the_prompt() -> None:
    small = {"messages": [{"role": "user", "content": "hi"}]}
    large = {
        "system": "you are a careful assistant with a long standing brief",
        "messages": [
            {"role": "user", "content": "hi " * 500},
            {"role": "assistant", "content": [{"type": "text", "text": "sure " * 200}]},
        ],
        "tools": [
            {
                "name": "search",
                "description": "find things on the internet",
                "input_schema": {"type": "object", "properties": {"q": {"type": "string"}}},
            }
        ],
    }

    ctx = make_ctx()
    small_result = ant.count_tokens(small, ctx)
    large_result = ant.count_tokens(large, ctx)

    assert set(small_result) == {"input_tokens"}
    assert isinstance(small_result["input_tokens"], int)
    assert not isinstance(small_result["input_tokens"], bool)
    assert small_result["input_tokens"] > 0
    assert large_result["input_tokens"] > small_result["input_tokens"]


def test_count_tokens_on_empty_body_is_still_positive() -> None:
    assert ant.count_tokens({}, make_ctx())["input_tokens"] > 0
