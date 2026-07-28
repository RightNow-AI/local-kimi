"""Pipeline contract tests.

``k3.pipeline.run`` is the single place engine output becomes canonical events,
so the guarantees a dialect is allowed to rely on are pinned here:

* reasoning always closes before visible text opens,
* control tokens never reach a ``TextDelta``,
* the streaming and non-streaming paths cannot drift,
* the reasoning ledger holds byte-exact tool-call arguments afterwards.

Everything is driven through a local fake engine; nothing here touches a socket
or a clock.
"""

from __future__ import annotations

from typing import Any, AsyncIterator, Optional

import pytest

from k3 import pipeline
from k3.ir import (
    CanonicalResponse,
    ReasoningDelta,
    ReasoningEnd,
    ReasoningPart,
    StreamEnd,
    StreamStart,
    TextDelta,
    TextEnd,
    ToolCallArgsDelta,
    ToolCallEnd,
    ToolCallStart,
)
from k3.reasoning import ReasoningLedger, decode_signature

# --------------------------------------------------------------------------
# fake engine + chunk helpers
# --------------------------------------------------------------------------


class FakeEngine:
    """Records payloads and replays canned chunks / a canned response."""

    def __init__(self, chunks: list[dict[str, Any]], response: Optional[dict[str, Any]] = None) -> None:
        self.chunks, self.response, self.calls = chunks, response, []

    async def chat(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.calls.append(payload)
        return self.response

    async def chat_stream(self, payload: dict[str, Any]) -> AsyncIterator[dict[str, Any]]:
        self.calls.append(payload)
        for c in self.chunks:
            yield c


PAYLOAD: dict[str, Any] = {"model": "k3", "messages": [{"role": "user", "content": "weather?"}]}

USAGE = {
    "prompt_tokens": 120,
    "completion_tokens": 34,
    "total_tokens": 154,
    "completion_tokens_details": {"reasoning_tokens": 12},
    "prompt_tokens_details": {"cached_tokens": 64},
}


def chunk(delta: dict[str, Any], **extra: Any) -> dict[str, Any]:
    """An OpenAI chat-completion streaming chunk."""
    out: dict[str, Any] = {
        "id": "c1",
        "model": "k3",
        "choices": [
            {"index": 0, "delta": delta, "finish_reason": extra.pop("finish_reason", None)}
        ],
    }
    out.update(extra)
    return out


def kimi_section(name: str = "get_weather", idx: int = 0, args: str = '{"city":"Beijing"}') -> str:
    return (
        "<|tool_calls_section_begin|>"
        f"<|tool_call_begin|>functions.{name}:{idx}"
        f"<|tool_call_argument_begin|>{args}"
        "<|tool_call_end|>"
        "<|tool_calls_section_end|>"
    )


CONTROL_TOKENS = (
    "<|tool_calls_section_begin|>",
    "<|tool_calls_section_end|>",
    "<|tool_call_begin|>",
    "<|tool_call_argument_begin|>",
    "<|tool_call_end|>",
    "<|",
    "|>",
)


def split_every(text: str, size: int) -> list[str]:
    return [text[i : i + size] for i in range(0, len(text), size)]


async def collect(
    engine: FakeEngine,
    *,
    payload: Optional[dict[str, Any]] = None,
    tool_parser: str = "kimi",
    ledger: Optional[ReasoningLedger] = None,
    stream: bool = True,
    reasoning_field: str = "reasoning_content",
) -> list[Any]:
    return [
        ev
        async for ev in pipeline.run(
            engine,
            dict(payload or PAYLOAD),
            tool_parser=tool_parser,
            ledger=ledger if ledger is not None else ReasoningLedger(),
            reasoning_field=reasoning_field,
            stream=stream,
        )
    ]


def types_of(events: list[Any]) -> list[type]:
    return [type(ev) for ev in events]


def only(events: list[Any], kind: type) -> list[Any]:
    return [ev for ev in events if isinstance(ev, kind)]


def text_of(events: list[Any]) -> str:
    return "".join(ev.text for ev in only(events, TextDelta))


def reasoning_of(events: list[Any]) -> str:
    return "".join(ev.text for ev in only(events, ReasoningDelta))


def final(events: list[Any]) -> StreamEnd:
    ends = only(events, StreamEnd)
    assert len(ends) == 1, f"expected exactly one StreamEnd, got {len(ends)}"
    return ends[0]


def assert_no_control_tokens(events: list[Any]) -> None:
    for ev in only(events, TextDelta):
        for token in CONTROL_TOKENS:
            assert token not in ev.text, f"control token {token!r} leaked into TextDelta {ev.text!r}"


# --------------------------------------------------------------------------
# 1. ordering contract
# --------------------------------------------------------------------------


async def test_reasoning_closes_before_the_first_text_delta() -> None:
    engine = FakeEngine(
        [
            chunk({"reasoning_content": "think "}),
            chunk({"reasoning_content": "harder"}),
            chunk({"content": "Hello"}),
            chunk({"content": " world"}),
            chunk({}, finish_reason="stop"),
        ]
    )
    events = await collect(engine)

    assert types_of(events) == [
        StreamStart,
        ReasoningDelta,
        ReasoningDelta,
        ReasoningEnd,
        TextDelta,
        TextDelta,
        TextEnd,
        StreamEnd,
    ]

    # And, stated as the contract rather than as a literal sequence:
    first_text = next(i for i, ev in enumerate(events) if isinstance(ev, TextDelta))
    reasoning_positions = [
        i for i, ev in enumerate(events) if isinstance(ev, (ReasoningDelta, ReasoningEnd))
    ]
    assert reasoning_positions, "expected reasoning events"
    assert max(reasoning_positions) < first_text
    assert reasoning_of(events) == "think harder"
    assert text_of(events) == "Hello world"


# --------------------------------------------------------------------------
# 2. signature round-trip
# --------------------------------------------------------------------------


async def test_reasoning_end_signature_decodes_back_to_the_reasoning_text() -> None:
    engine = FakeEngine(
        [
            chunk({"reasoning_content": "step one. "}),
            chunk({"reasoning_content": "step two."}),
            chunk({"content": "done"}),
            chunk({}, finish_reason="stop"),
        ]
    )
    events = await collect(engine)

    ends = only(events, ReasoningEnd)
    assert len(ends) == 1
    reasoning_end = ends[0]
    assert reasoning_end.text == "step one. step two."

    decoded, ledger_id = decode_signature(reasoning_end.signature)
    assert decoded == "step one. step two."
    assert ledger_id  # opaque, never compared literally

    part = final(events).response.reasoning_parts()[0]
    assert isinstance(part, ReasoningPart)
    assert part.text == "step one. step two."
    assert part.signature == reasoning_end.signature


# --------------------------------------------------------------------------
# 3. text-format tool calls
# --------------------------------------------------------------------------


@pytest.mark.parametrize("size", [1, 3, 7, 13, 4096])
async def test_text_format_tool_call_survives_any_chunk_split(size: int) -> None:
    body = "Sure. " + kimi_section()
    engine = FakeEngine(
        [chunk({"content": piece}) for piece in split_every(body, size)]
        + [chunk({}, finish_reason="stop")]
    )
    events = await collect(engine)

    assert text_of(events) == "Sure. "
    assert_no_control_tokens(events)

    starts, args_deltas, ends = (
        only(events, ToolCallStart),
        only(events, ToolCallArgsDelta),
        only(events, ToolCallEnd),
    )
    assert len(starts) == 1 and len(ends) == 1
    assert starts[0].name == "get_weather"
    assert starts[0].index == 0 and ends[0].index == 0
    assert starts[0].id == ends[0].id
    assert starts[0].id  # generated, so never compared literally
    assert "".join(d.text for d in args_deltas) == '{"city":"Beijing"}'
    assert ends[0].arguments == '{"city":"Beijing"}'

    # ToolCallStart -> ToolCallArgsDelta* -> ToolCallEnd, in that order.
    tool_events = [
        ev for ev in events if isinstance(ev, (ToolCallStart, ToolCallArgsDelta, ToolCallEnd))
    ]
    assert types_of(tool_events) == [ToolCallStart, ToolCallArgsDelta, ToolCallEnd]

    calls = final(events).response.tool_calls()
    assert [(c.name, c.arguments) for c in calls] == [("get_weather", '{"city":"Beijing"}')]


async def test_tool_call_section_split_inside_every_control_token() -> None:
    """A one-character stream splits mid-token everywhere; nothing may leak."""
    body = "before " + kimi_section(name="search", args='{"q":"kimi"}') + " after"
    engine = FakeEngine(
        [chunk({"content": ch}) for ch in body] + [chunk({}, finish_reason="stop")]
    )
    events = await collect(engine)

    assert_no_control_tokens(events)
    assert text_of(events) == "before  after"
    ends = only(events, ToolCallEnd)
    assert [(e.name, e.arguments) for e in ends] == [("search", '{"q":"kimi"}')]


# --------------------------------------------------------------------------
# 4. stop_reason upgrade
# --------------------------------------------------------------------------


async def test_stop_reason_upgrades_to_tool_calls_when_the_parser_found_calls() -> None:
    engine = FakeEngine(
        [
            chunk({"content": kimi_section()}),
            chunk({}, finish_reason="stop"),
        ]
    )
    events = await collect(engine)

    end = final(events)
    assert end.stop_reason == "tool_calls"
    assert end.response.stop_reason == "tool_calls"


async def test_stop_reason_stays_stop_without_tool_calls() -> None:
    engine = FakeEngine([chunk({"content": "plain"}), chunk({}, finish_reason="stop")])
    assert final(await collect(engine)).stop_reason == "stop"


# --------------------------------------------------------------------------
# 5. native tool_calls deltas
# --------------------------------------------------------------------------


async def test_fragmented_native_tool_call_deltas_accumulate_into_one_call() -> None:
    engine = FakeEngine(
        [
            chunk(
                {
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "call_abc",
                            "type": "function",
                            "function": {"name": "get_weather", "arguments": '{"ci'},
                        }
                    ]
                }
            ),
            chunk({"tool_calls": [{"index": 0, "function": {"arguments": 'ty": "Bei'}}]}),
            chunk({"tool_calls": [{"index": 0, "function": {"arguments": 'jing"}'}}]}),
            chunk({}, finish_reason="tool_calls"),
        ]
    )
    events = await collect(engine)

    starts, args_deltas, ends = (
        only(events, ToolCallStart),
        only(events, ToolCallArgsDelta),
        only(events, ToolCallEnd),
    )
    assert len(starts) == 1
    assert starts[0].index == 0 and starts[0].id == "call_abc" and starts[0].name == "get_weather"

    assert len(ends) == 1, f"fragments must fold into ONE ToolCallEnd, got {len(ends)}"
    assert ends[0].index == 0
    assert ends[0].id == "call_abc"
    assert ends[0].name == "get_weather"
    assert ends[0].arguments == '{"city": "Beijing"}'
    assert "".join(d.text for d in args_deltas) == '{"city": "Beijing"}'

    calls = final(events).response.tool_calls()
    assert len(calls) == 1
    assert (calls[0].id, calls[0].name, calls[0].arguments) == (
        "call_abc",
        "get_weather",
        '{"city": "Beijing"}',
    )


# --------------------------------------------------------------------------
# 6. streaming vs non-streaming parity  (the anti-drift test)
# --------------------------------------------------------------------------


def _without_generated_ids(msg: dict[str, Any]) -> dict[str, Any]:
    """Tool-call ids are minted per parse, so they can never be compared."""
    out = dict(msg)
    calls = out.get("tool_calls")
    if calls:
        out["tool_calls"] = [{**tc, "id": f"<id:{i}>"} for i, tc in enumerate(calls)]
    return out


def _same_response(a: CanonicalResponse, b: CanonicalResponse) -> None:
    assert a.text() == b.text()
    assert [p.text for p in a.reasoning_parts()] == [p.text for p in b.reasoning_parts()]
    assert [(c.name, c.arguments) for c in a.tool_calls()] == [
        (c.name, c.arguments) for c in b.tool_calls()
    ]
    assert a.stop_reason == b.stop_reason
    assert _without_generated_ids(a.upstream_message) == _without_generated_ids(b.upstream_message)


async def test_streaming_and_non_streaming_agree_for_text_format_tool_calls() -> None:
    reasoning = "I should look the weather up."
    visible = "Sure, checking. "
    section = kimi_section(args='{"city": "Beijing", "unit": "c"}')

    streamed = FakeEngine(
        [chunk({"reasoning_content": reasoning}), chunk({"content": visible})]
        + [chunk({"content": p}) for p in split_every(section, 7)]
        + [chunk({}, finish_reason="stop", usage=USAGE)]
    )
    blocking = FakeEngine(
        [],
        response={
            "id": "c1",
            "model": "k3",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": visible + section,
                        "reasoning_content": reasoning,
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": USAGE,
        },
    )

    a = final(await collect(streamed)).response
    b = final(await collect(blocking, stream=False)).response

    _same_response(a, b)
    assert a.text() == visible
    assert [p.text for p in a.reasoning_parts()] == [reasoning]
    assert [(c.name, c.arguments) for c in a.tool_calls()] == [
        ("get_weather", '{"city": "Beijing", "unit": "c"}')
    ]
    assert a.stop_reason == "tool_calls"
    assert a.usage.input_tokens == b.usage.input_tokens
    assert a.usage.output_tokens == b.usage.output_tokens
    assert blocking.calls == [PAYLOAD]  # the non-streaming path really used chat()


async def test_streaming_and_non_streaming_agree_for_native_tool_calls() -> None:
    reasoning = "Tool time."
    visible = "One moment. "
    args = '{"city": "Beijing"}'

    streamed = FakeEngine(
        [
            chunk({"reasoning_content": reasoning}),
            chunk({"content": visible}),
            chunk(
                {
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "call_abc",
                            "type": "function",
                            "function": {"name": "get_weather", "arguments": '{"city": '},
                        }
                    ]
                }
            ),
            chunk({"tool_calls": [{"index": 0, "function": {"arguments": '"Beijing"}'}}]}),
            chunk({}, finish_reason="tool_calls", usage=USAGE),
        ]
    )
    blocking = FakeEngine(
        [],
        response={
            "id": "c1",
            "model": "k3",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": visible,
                        "reasoning_content": reasoning,
                        "tool_calls": [
                            {
                                "id": "call_abc",
                                "type": "function",
                                "function": {"name": "get_weather", "arguments": args},
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": USAGE,
        },
    )

    a = final(await collect(streamed)).response
    b = final(await collect(blocking, stream=False)).response

    _same_response(a, b)
    assert [(c.id, c.name, c.arguments) for c in a.tool_calls()] == [
        ("call_abc", "get_weather", args)
    ]
    assert a.stop_reason == "tool_calls"


async def test_streaming_and_non_streaming_agree_on_the_engine_id_and_model() -> None:
    """The completion id and model the engine reported survive either path.

    KNOWN FAILURE — a real drift between the two paths, left unfixed on purpose.
    ``response_to_chunks`` writes ``id``/``model`` onto the *final* synthetic
    chunk, but ``run`` only reads them from the *first* chunk it sees, so the
    non-streaming path silently drops both. ``CanonicalResponse.id`` collapses
    to the literal ``"k3"`` and ``model`` falls back to the request's model.
    """
    engine_id, engine_model = "chatcmpl-77", "engine-model"
    payload = {"model": "k3", "messages": []}

    streamed = FakeEngine(
        [
            {
                "id": engine_id,
                "model": engine_model,
                "choices": [{"index": 0, "delta": {"content": "hi"}, "finish_reason": None}],
            },
            {
                "id": engine_id,
                "model": engine_model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            },
        ]
    )
    blocking = FakeEngine(
        [],
        response={
            "id": engine_id,
            "model": engine_model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "hi"},
                    "finish_reason": "stop",
                }
            ],
        },
    )

    a = final(await collect(streamed, payload=payload)).response
    b = final(await collect(blocking, payload=payload, stream=False)).response

    assert a.id == engine_id and a.model == engine_model  # streaming is fine
    assert b.id == a.id, (
        "non-streaming dropped the engine completion id: "
        f"{b.id!r} != {a.id!r} (response_to_chunks puts it on the last chunk, "
        "run reads it from the first)"
    )
    assert b.model == a.model


# --------------------------------------------------------------------------
# 7. ledger registration
# --------------------------------------------------------------------------


async def test_ledger_holds_the_exact_upstream_assistant_message() -> None:
    reasoning = "The user wants Beijing weather."
    raw_args = '{"city": "Beijing", "unit": "c"}'
    ledger = ReasoningLedger()

    engine = FakeEngine(
        [
            chunk({"reasoning_content": reasoning}),
            chunk({"content": "Checking. "}),
            chunk({"content": kimi_section(args=raw_args)}),
            chunk({}, finish_reason="stop"),
        ]
    )
    events = await collect(engine, ledger=ledger)

    _, ledger_id = decode_signature(only(events, ReasoningEnd)[0].signature)
    entry = ledger.get(ledger_id)
    assert entry is not None, "the run must register its turn in the ledger"
    assert entry.reasoning == reasoning

    upstream = entry.upstream_message
    assert upstream["role"] == "assistant"
    assert upstream["reasoning_content"] == reasoning
    assert upstream["content"] == "Checking. "
    # The raw JSON string, verbatim: no json.loads/json.dumps round trip, so the
    # inner spacing the model produced is still there.
    assert upstream["tool_calls"][0]["function"]["arguments"] == raw_args
    assert upstream["tool_calls"][0]["function"]["name"] == "get_weather"
    assert final(events).response.upstream_message == upstream


# --------------------------------------------------------------------------
# 8. usage
# --------------------------------------------------------------------------


async def test_usage_from_the_final_chunk_lands_on_stream_end() -> None:
    engine = FakeEngine(
        [
            chunk({"reasoning_content": "hmm"}),
            chunk({"content": "hi"}),
            chunk({}, finish_reason="stop", usage=USAGE),
        ]
    )
    events = await collect(engine)

    usage = final(events).usage
    assert usage.input_tokens == 120
    assert usage.output_tokens == 34
    assert usage.reasoning_tokens == 12
    assert usage.cached_input_tokens == 64
    assert final(events).response.usage == usage


async def test_usage_is_estimated_when_the_engine_reports_none() -> None:
    engine = FakeEngine(
        [
            chunk({"reasoning_content": "a fairly long chain of thought here"}),
            chunk({"content": "a visible answer of some length"}),
            chunk({}, finish_reason="stop"),
        ]
    )
    usage = final(await collect(engine)).usage

    assert usage.output_tokens > 0
    assert usage.reasoning_tokens > 0
    assert usage.input_tokens == 0


# --------------------------------------------------------------------------
# 9. inline <think> splitting
# --------------------------------------------------------------------------

INLINE_BODY = "<think>abc</think>visible"


@pytest.mark.parametrize("at", list(range(1, len(INLINE_BODY))))
async def test_inline_think_is_extracted_whatever_the_split(at: int) -> None:
    engine = FakeEngine(
        [
            chunk({"content": INLINE_BODY[:at]}),
            chunk({"content": INLINE_BODY[at:]}),
            chunk({}, finish_reason="stop"),
        ]
    )
    events = await collect(engine, reasoning_field="inline")

    assert reasoning_of(events) == "abc"
    assert text_of(events) == "visible"
    for ev in only(events, TextDelta):
        assert "<think>" not in ev.text
        assert "</think>" not in ev.text
        assert "<" not in ev.text

    ends = only(events, ReasoningEnd)
    assert len(ends) == 1 and ends[0].text == "abc"
    first_text = next(i for i, ev in enumerate(events) if isinstance(ev, TextDelta))
    assert events.index(ends[0]) < first_text

    response = final(events).response
    assert response.text() == "visible"
    assert [p.text for p in response.reasoning_parts()] == ["abc"]


async def test_inline_think_split_inside_the_opening_tag() -> None:
    """The named regression: a chunk boundary landing inside ``<think>``."""
    engine = FakeEngine(
        [
            chunk({"content": "<th"}),
            chunk({"content": "ink>abc</think>visible"}),
            chunk({}, finish_reason="stop"),
        ]
    )
    events = await collect(engine, reasoning_field="inline")

    assert reasoning_of(events) == "abc"
    assert text_of(events) == "visible"
    assert all("<think>" not in ev.text for ev in only(events, TextDelta))


# --------------------------------------------------------------------------
# 10-12. degenerate streams and finish-reason mapping
# --------------------------------------------------------------------------


async def test_empty_stream_still_produces_a_valid_response() -> None:
    events = await collect(FakeEngine([]))

    assert types_of(events) == [StreamStart, StreamEnd]
    start, end = events
    assert isinstance(start, StreamStart)
    assert start.model == "k3"
    assert isinstance(end.response, CanonicalResponse)
    assert end.response.text() == ""
    assert end.response.tool_calls() == []
    assert end.stop_reason == "stop"
    assert end.response.upstream_message["role"] == "assistant"


async def test_collect_response_returns_the_stream_end_response() -> None:
    engine = FakeEngine(
        [
            chunk({"reasoning_content": "hmm"}),
            chunk({"content": "hi"}),
            chunk({}, finish_reason="stop"),
        ]
    )
    events = await collect(engine)

    async def replay() -> AsyncIterator[Any]:
        for ev in events:
            yield ev

    response = await pipeline.collect_response(replay())
    assert response is final(events).response


async def test_finish_reason_length_maps_to_stop_reason_length() -> None:
    engine = FakeEngine([chunk({"content": "truncated"}), chunk({}, finish_reason="length")])
    end = final(await collect(engine))

    assert end.stop_reason == "length"
    assert end.response.stop_reason == "length"
