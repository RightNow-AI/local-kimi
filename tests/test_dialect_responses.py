"""Unit tests for the OpenAI Responses dialect (the Codex preset).

``ingress`` / ``egress`` / ``egress_stream`` are exercised in isolation: no
server, no engine, no network. The streaming assertions are the load-bearing
ones — Codex is strict about ``sequence_number`` monotonicity, about
``output_item.added``/``done`` pairing, and about the terminal response object
describing exactly the items that were announced along the way.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, AsyncIterator, Iterable

from k3 import presets as presets_mod
from k3.dialects import openai_responses as resp
from k3.dialects.base import DialectContext
from k3.ir import (
    CanonicalResponse,
    Message,
    ReasoningDelta,
    ReasoningEnd,
    ReasoningPart,
    StreamEnd,
    StreamEvent,
    StreamStart,
    TextDelta,
    TextEnd,
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

#: Non-canonical whitespace on purpose: the raw argument string has to survive
#: byte-identical, because re-serialising it shifts the upstream token stream.
RAW_ARGS = '{"a":  1, "b":"x"}'


# --------------------------------------------------------------------------
# harness
# --------------------------------------------------------------------------


def make_ctx(**overrides: Any) -> DialectContext:
    preset = presets_mod.get("codex")
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
    chunks: list[bytes] = []
    async for chunk in resp.egress_stream(_agen(events), ctx):
        assert isinstance(chunk, (bytes, bytearray))
        chunks.append(bytes(chunk))
    return parse_sse(b"".join(chunks))


def types_of(records: list[dict[str, Any]]) -> list[str]:
    return [r["data"]["type"] for r in records]


def parts_of(msg: Message, kind: type) -> list[Any]:
    return [p for p in msg.parts if isinstance(p, kind)]


def canonical_response(stop_reason: str = "stop") -> CanonicalResponse:
    return CanonicalResponse(
        id="strm1",
        model="k3",
        parts=[
            ReasoningPart(text="weigh options", signature="k3r1.sig"),
            TextPart("Hello world"),
            ToolCallPart(id="call_1", name="do_thing", arguments=RAW_ARGS),
        ],
        stop_reason=stop_reason,  # type: ignore[arg-type]
        usage=Usage(input_tokens=12, output_tokens=8),
    )


def full_event_stream(stop_reason: str = "tool_calls") -> list[StreamEvent]:
    return [
        StreamStart(id="strm1", model="k3"),
        ReasoningDelta("weigh "),
        ReasoningDelta("options"),
        ReasoningEnd(text="weigh options", signature="k3r1.sig"),
        TextDelta("Hello "),
        TextDelta("world"),
        TextEnd("Hello world"),
        ToolCallStart(index=0, id="call_1", name="do_thing"),
        ToolCallArgsDelta(index=0, text=RAW_ARGS[:8]),
        ToolCallArgsDelta(index=0, text=RAW_ARGS[8:]),
        ToolCallEnd(index=0, id="call_1", name="do_thing", arguments=RAW_ARGS),
        StreamEnd(
            stop_reason=stop_reason,  # type: ignore[arg-type]
            usage=Usage(input_tokens=12, output_tokens=8),
            response=canonical_response(stop_reason),
        ),
    ]


# --------------------------------------------------------------------------
# 1. instructions + bare-string input
# --------------------------------------------------------------------------


def test_instructions_become_system_and_string_input_becomes_one_user_message() -> None:
    req = resp.ingress(
        {"instructions": "be terse", "input": "what is 2+2?"},
        make_ctx(),
    )
    assert req.system == ["be terse"]
    assert len(req.messages) == 1
    assert req.messages[0].role == "user"
    assert req.messages[0].text() == "what is 2+2?"


# --------------------------------------------------------------------------
# 2/3. sibling items merge into one assistant turn
# --------------------------------------------------------------------------


def test_function_call_reasoning_and_message_items_merge_into_one_assistant_message() -> None:
    body = {
        "input": [
            {
                "type": "function_call",
                "id": "fc_1",
                "call_id": "call_1",
                "name": "do_thing",
                "arguments": RAW_ARGS,
            },
            {
                "type": "reasoning",
                "id": "rs_1",
                "summary": [{"type": "summary_text", "text": "weigh options"}],
                "encrypted_content": "k3r1.sig",
            },
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "Hello world"}],
            },
        ]
    }
    req = resp.ingress(body, make_ctx())

    # restore_assistant() only works if all three land on ONE message.
    assert len(req.messages) == 1, [m.role for m in req.messages]
    msg = req.messages[0]
    assert msg.role == "assistant"

    reasoning = parts_of(msg, ReasoningPart)
    assert len(reasoning) == 1
    assert reasoning[0].text == "weigh options"
    assert reasoning[0].signature == "k3r1.sig"

    assert msg.text() == "Hello world"

    calls = msg.tool_calls()
    assert len(calls) == 1
    assert calls[0].id == "call_1"  # call_id wins over the item handle
    assert calls[0].name == "do_thing"


def test_function_call_arguments_survive_byte_identical() -> None:
    body = {
        "input": [
            {
                "type": "function_call",
                "call_id": "call_1",
                "name": "do_thing",
                "arguments": RAW_ARGS,
            }
        ]
    }
    req = resp.ingress(body, make_ctx())
    args = req.messages[0].tool_calls()[0].arguments
    assert args == RAW_ARGS
    assert args == '{"a":  1, "b":"x"}'
    assert "  1" in args  # the double space is intact


# --------------------------------------------------------------------------
# 4. function_call_output closes the assistant turn
# --------------------------------------------------------------------------


def test_function_call_output_becomes_a_tool_message_and_closes_the_turn() -> None:
    body = {
        "input": [
            {
                "type": "reasoning",
                "summary": [{"type": "summary_text", "text": "first thought"}],
                "encrypted_content": "sig-a",
            },
            {
                "type": "function_call",
                "call_id": "call_1",
                "name": "do_thing",
                "arguments": RAW_ARGS,
            },
            {"type": "function_call_output", "call_id": "call_1", "output": "42"},
            {
                "type": "reasoning",
                "summary": [{"type": "summary_text", "text": "second thought"}],
                "encrypted_content": "sig-b",
            },
        ]
    }
    req = resp.ingress(body, make_ctx())

    assert [m.role for m in req.messages] == ["assistant", "tool", "assistant"]

    results = req.messages[1].tool_results()
    assert len(results) == 1
    assert isinstance(results[0], ToolResultPart)
    assert results[0].tool_call_id == "call_1"
    assert results[0].content == "42"

    first = parts_of(req.messages[0], ReasoningPart)
    second = parts_of(req.messages[2], ReasoningPart)
    assert [p.text for p in first] == ["first thought"]
    assert [p.signature for p in first] == ["sig-a"]
    assert [p.text for p in second] == ["second thought"]
    assert [p.signature for p in second] == ["sig-b"]


# --------------------------------------------------------------------------
# 5/6/7. tools, sampling knobs, robustness
# --------------------------------------------------------------------------


def test_tools_parse_in_both_responses_and_chat_completions_shapes() -> None:
    schema = {"type": "object", "properties": {"q": {"type": "string"}}}
    body = {
        "tools": [
            {
                "type": "function",
                "name": "responses_shape",
                "description": "flat",
                "parameters": schema,
            },
            {
                "type": "function",
                "function": {
                    "name": "chat_shape",
                    "description": "nested",
                    "parameters": schema,
                },
            },
        ]
    }
    req = resp.ingress(body, make_ctx())

    assert [t.name for t in req.tools] == ["responses_shape", "chat_shape"]
    assert [t.description for t in req.tools] == ["flat", "nested"]
    assert all(t.parameters == schema for t in req.tools)


def test_max_output_tokens_and_reasoning_effort_map_onto_the_request() -> None:
    req = resp.ingress(
        {"max_output_tokens": 1234, "reasoning": {"effort": "high"}},
        make_ctx(),
    )
    assert req.max_tokens == 1234
    assert req.reasoning_effort == "high"


def test_defaults_fill_gaps_but_never_override_the_client() -> None:
    defaults = presets_mod.get("codex").defaults
    req = resp.ingress({}, make_ctx())
    assert req.max_tokens == defaults.max_tokens
    assert req.reasoning_effort == defaults.reasoning_effort


def test_unknown_item_types_and_empty_body_do_not_raise() -> None:
    body = {
        "input": [
            {"type": "web_search_call", "id": "ws_1", "status": "completed"},
            {"type": "local_shell_call", "id": "ls_1"},
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hi"}]},
        ]
    }
    req = resp.ingress(body, make_ctx())
    assert [m.role for m in req.messages] == ["user"]
    assert req.messages[0].text() == "hi"

    empty = resp.ingress({}, make_ctx())
    assert empty.messages == []
    assert empty.system == []
    assert empty.tools == []
    assert empty.model == "k3"


# --------------------------------------------------------------------------
# 8/9/10. egress output items
# --------------------------------------------------------------------------


def test_output_items_are_ordered_reasoning_message_function_call() -> None:
    body = resp.egress(canonical_response(), make_ctx())
    output = body["output"]

    assert [item["type"] for item in output] == ["reasoning", "message", "function_call"]
    assert output[0]["id"].startswith("rs_")
    assert output[1]["id"].startswith("msg_")
    assert output[2]["id"].startswith("fc_")

    assert output[2]["call_id"] == "call_1"
    assert output[2]["name"] == "do_thing"
    assert output[2]["arguments"] == RAW_ARGS

    assert body["id"].startswith("resp_")
    assert body["object"] == "response"
    assert body["status"] == "completed"


def test_encrypted_content_carries_the_reasoning_signature() -> None:
    body = resp.egress(canonical_response(), make_ctx())
    reasoning = next(i for i in body["output"] if i["type"] == "reasoning")
    assert reasoning["encrypted_content"] == "k3r1.sig"
    assert reasoning["summary"] == [{"type": "summary_text", "text": "weigh options"}]


def test_strip_policy_emits_no_reasoning_item() -> None:
    body = resp.egress(canonical_response(), make_ctx(reasoning=ReasoningPolicy.STRIP))
    assert [item["type"] for item in body["output"]] == ["message", "function_call"]
    assert "weigh options" not in body["output_text"]
    assert all("encrypted_content" not in item for item in body["output"])


# --------------------------------------------------------------------------
# 11/12. terminal status and output_text
# --------------------------------------------------------------------------


def test_length_stop_reason_marks_the_response_incomplete() -> None:
    body = resp.egress(canonical_response("length"), make_ctx())
    assert body["status"] == "incomplete"
    assert body["incomplete_details"] == {"reason": "max_output_tokens"}


def test_completed_response_has_no_incomplete_details() -> None:
    body = resp.egress(canonical_response("stop"), make_ctx())
    assert body["status"] == "completed"
    assert body["incomplete_details"] is None


def test_output_text_is_the_concatenated_text() -> None:
    response = CanonicalResponse(
        id="strm1",
        model="k3",
        parts=[TextPart("Hello "), ReasoningPart(text="ignored"), TextPart("world")],
    )
    body = resp.egress(response, make_ctx())
    assert body["output_text"] == "Hello world"

    message = next(i for i in body["output"] if i["type"] == "message")
    assert message["content"] == [
        {"type": "output_text", "text": "Hello world", "annotations": []}
    ]


# --------------------------------------------------------------------------
# 13-17. streaming
# --------------------------------------------------------------------------


async def test_sequence_numbers_start_at_zero_and_increase_by_one() -> None:
    records = await run_stream(full_event_stream(), make_ctx())
    assert records, "stream produced nothing"

    seqs = [r["data"]["sequence_number"] for r in records]
    assert seqs == list(range(len(records))), seqs


async def test_event_order_and_single_terminal_event_without_done_sentinel() -> None:
    records = await run_stream(full_event_stream(), make_ctx())
    kinds = types_of(records)

    assert kinds[0] == "response.created"
    assert kinds[1] == "response.in_progress"

    terminals = [k for k in kinds if k in ("response.completed", "response.incomplete")]
    assert terminals == ["response.completed"]
    assert kinds[-1] == "response.completed"

    # named-event SSE: the event line always matches the payload type
    for record in records:
        assert record["event"] == record["data"]["type"]

    assert all(record["data"] != "[DONE]" for record in records)
    assert "[DONE]" not in kinds


async def test_incomplete_terminal_event_when_the_engine_ran_out_of_room() -> None:
    records = await run_stream(full_event_stream("length"), make_ctx())
    kinds = types_of(records)
    assert kinds[-1] == "response.incomplete"
    assert kinds.count("response.incomplete") == 1
    assert "response.completed" not in kinds
    assert records[-1]["data"]["response"]["status"] == "incomplete"


async def test_every_output_item_added_has_a_matching_done_in_index_order() -> None:
    records = await run_stream(full_event_stream(), make_ctx())

    added = [r["data"] for r in records if r["data"]["type"] == "response.output_item.added"]
    done = [r["data"] for r in records if r["data"]["type"] == "response.output_item.done"]

    assert [d["output_index"] for d in added] == list(range(len(added)))
    assert len(added) == len(done)
    assert [d["output_index"] for d in done] == [d["output_index"] for d in added]
    assert [d["item"]["id"] for d in done] == [d["item"]["id"] for d in added]
    assert [d["item"]["type"] for d in added] == ["reasoning", "message", "function_call"]

    for entry in added:
        assert entry["item"]["status"] == "in_progress"
    for entry in done:
        assert entry["item"]["status"] == "completed"


async def test_function_call_argument_deltas_concatenate_to_the_done_arguments() -> None:
    records = await run_stream(full_event_stream(), make_ctx())

    deltas = [
        r["data"]["delta"]
        for r in records
        if r["data"]["type"] == "response.function_call_arguments.delta"
    ]
    done = next(
        r["data"]
        for r in records
        if r["data"]["type"] == "response.function_call_arguments.done"
    )

    assert len(deltas) == 2
    assert "".join(deltas) == done["arguments"]
    assert done["arguments"] == RAW_ARGS


async def test_terminal_response_lists_exactly_the_items_announced_while_streaming() -> None:
    records = await run_stream(full_event_stream(), make_ctx())

    announced = [
        r["data"]["item"]["id"]
        for r in records
        if r["data"]["type"] == "response.output_item.added"
    ]
    final = records[-1]["data"]["response"]
    assert [item["id"] for item in final["output"]] == announced
    assert [item["type"] for item in final["output"]] == [
        "reasoning",
        "message",
        "function_call",
    ]

    # and the response id itself is stable from the very first frame
    created_id = records[0]["data"]["response"]["id"]
    assert created_id.startswith("resp_")
    assert final["id"] == created_id
    for item_id in announced:
        assert item_id.split("_", 1)[1].startswith(created_id[len("resp_") :])


async def test_streaming_and_non_streaming_bodies_agree() -> None:
    ctx = make_ctx()
    records = await run_stream(full_event_stream(), ctx)
    streamed = records[-1]["data"]["response"]

    response_id = records[0]["data"]["response"]["id"]
    direct = resp.egress(replace(canonical_response(), id=response_id), ctx)

    assert streamed == direct
