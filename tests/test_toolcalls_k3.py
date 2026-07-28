"""Golden behavior for Kimi K3 XTML tool-call parsing."""

from __future__ import annotations

import json

from k3.toolcalls import (
    KimiK3ToolParser,
    ParsedReasoning,
    ParsedText,
    ParsedToolCall,
    parse_all,
)


OPEN = KimiK3ToolParser.OPEN
CLOSE = KimiK3ToolParser.CLOSE
SEP = KimiK3ToolParser.SEP
END = KimiK3ToolParser.END_OF_MSG
CONTROLS = (OPEN, CLOSE, SEP, END)


def open_tag(name: str, **attrs: object) -> str:
    rendered = "".join(f' {key}="{value}"' for key, value in attrs.items())
    return OPEN + name + rendered + SEP


def close_tag(name: str) -> str:
    return CLOSE + name + SEP


def argument(key: str, kind: str, body: str) -> str:
    return open_tag("argument", key=key, type=kind) + body + close_tag("argument")


def call(name: str, *children: str, index: int = 1) -> str:
    return (
        open_tag("call", tool=name, index=index)
        + "".join(children)
        + close_tag("call")
    )


def tools(*calls: str) -> str:
    return open_tag("tools") + "".join(calls) + close_tag("tools")


def normalise(events: list[object]) -> list[tuple[object, ...]]:
    out: list[tuple[object, ...]] = []
    for event in events:
        if isinstance(event, ParsedText):
            if out and out[-1][0] == "text":
                out[-1] = ("text", str(out[-1][1]) + event.text)
            else:
                out.append(("text", event.text))
        elif isinstance(event, ParsedToolCall):
            out.append(("call", event.name, event.arguments))
        else:
            raise AssertionError(f"unexpected event: {event!r}")
    return out


def text_events(events: list[object]) -> list[ParsedText]:
    return [event for event in events if isinstance(event, ParsedText)]


def tool_events(events: list[object]) -> list[ParsedToolCall]:
    return [event for event in events if isinstance(event, ParsedToolCall)]


def reasoning_events(events: list[object]) -> list[ParsedReasoning]:
    return [event for event in events if isinstance(event, ParsedReasoning)]


def test_k3_whole_string_rebuilds_typed_argument_elements():
    raw = tools(
        call(
            "inspect",
            argument("city", "string", "Beijing"),
            argument("active", "boolean", "true"),
            argument("count", "number", "3"),
            argument("missing", "null", "null"),
            argument("meta", "object", '{"unit": "c"}'),
            argument("days", "array", "[1, 2]"),
        )
    )

    events = parse_all(KimiK3ToolParser(), raw)

    calls = tool_events(events)
    assert len(calls) == 1
    assert calls[0].name == "inspect"
    assert json.loads(calls[0].arguments) == {
        "city": "Beijing",
        "active": True,
        "count": 3,
        "missing": None,
        "meta": {"unit": "c"},
        "days": [1, 2],
    }
    assert text_events(events) == []


def test_k3_think_and_response_elements_are_distinct_under_every_split():
    raw = (
        open_tag("message", role="assistant")
        + open_tag("think")
        + "private reasoning"
        + close_tag("think")
        + open_tag("response")
        + "visible answer"
        + close_tag("response")
        + close_tag("message")
        + END
    )

    for index in range(len(raw) + 1):
        parser = KimiK3ToolParser()
        events = parser.feed(raw[:index])
        events.extend(parser.feed(raw[index:]))
        events.extend(parser.finish())

        assert "".join(event.text for event in reasoning_events(events)) == "private reasoning"
        assert "".join(event.text for event in text_events(events)) == "visible answer"


def test_k3_json_block_body_is_byte_identical():
    body = '{\n  "city":  "Beijing",\n  "units":"c"\n}'
    raw = tools(
        call(
            "get_weather",
            open_tag("json", type="object") + body + close_tag("json"),
        )
    )

    parsed = tool_events(parse_all(KimiK3ToolParser(), raw))

    assert len(parsed) == 1
    assert parsed[0].arguments == body


def test_k3_chunk_split_invariance_at_every_index():
    raw = (
        "before "
        + tools(
            call("alpha", argument("x", "number", "1"), index=1),
            call("beta", argument("y", "string", "two"), index=2),
        )
        + " after"
    )
    expected = normalise(parse_all(KimiK3ToolParser(), raw))

    for index in range(len(raw) + 1):
        parser = KimiK3ToolParser()
        events = parser.feed(raw[:index])
        events.extend(parser.feed(raw[index:]))
        events.extend(parser.finish())
        assert normalise(events) == expected, f"split at {index} changed the parse"


def test_k3_control_tokens_never_appear_in_parsed_text():
    raw = (
        open_tag("message", role="assistant")
        + open_tag("response")
        + "before"
        + close_tag("response")
        + tools(call("ping", argument("ok", "boolean", "true")))
        + "after"
        + close_tag("message")
        + END
    )

    for index in range(len(raw) + 1):
        parser = KimiK3ToolParser()
        events = parser.feed(raw[:index])
        events.extend(parser.feed(raw[index:]))
        events.extend(parser.finish())
        for event in text_events(events):
            assert all(token not in event.text for token in CONTROLS)


def test_k3_unterminated_call_degrades_to_text_without_raising():
    raw = (
        open_tag("tools")
        + open_tag("call", tool="get_weather", index=1)
        + open_tag("argument", key="city", type="string")
        + "Bei"
    )

    events = parse_all(KimiK3ToolParser(), raw)

    assert tool_events(events) == []
    surfaced = "".join(event.text for event in text_events(events))
    assert "get_weather" in surfaced
    assert "Bei" in surfaced
    assert all(token not in surfaced for token in CONTROLS)


def test_k3_text_before_and_after_tools_is_preserved_in_order():
    raw = "before " + tools(call("ping", argument("value", "number", "1"))) + " after"

    assert normalise(parse_all(KimiK3ToolParser(), raw)) == [
        ("text", "before "),
        ("call", "ping", '{"value": 1}'),
        ("text", " after"),
    ]
