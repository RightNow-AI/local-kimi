"""Tests for :mod:`k3.toolcalls`.

The contract these tests pin down (see the module docstring of ``toolcalls``):

* every parser produces the same *logical* event stream no matter how the raw
  model output is chopped into chunks, and
* a control token never leaks into ``ParsedText`` — not even when a chunk
  boundary lands in the middle of one.

``ParsedToolCall.id`` embeds a random uuid, so every comparison here goes
through :func:`normalise`, which drops ids and coalesces adjacent text runs.
"""

from __future__ import annotations

import json

import pytest

from k3.toolcalls import (
    HermesToolParser,
    JsonToolParser,
    KimiToolParser,
    ParsedText,
    ParsedToolCall,
    PythonicToolParser,
    ToolCallParser,
    get_parser,
    parse_all,
    parser_names,
)

# --------------------------------------------------------------------------
# fixtures-as-constants
# --------------------------------------------------------------------------

SECTION_BEGIN = KimiToolParser.SECTION_BEGIN
SECTION_END = KimiToolParser.SECTION_END
CALL_BEGIN = KimiToolParser.CALL_BEGIN
ARG_BEGIN = KimiToolParser.ARG_BEGIN
CALL_END = KimiToolParser.CALL_END

HERMES_OPEN = HermesToolParser.OPEN
HERMES_CLOSE = HermesToolParser.CLOSE

#: Every token that must never reach the client as visible text.
CONTROL_TOKENS = (
    SECTION_BEGIN,
    SECTION_END,
    CALL_BEGIN,
    ARG_BEGIN,
    CALL_END,
    HERMES_OPEN,
    HERMES_CLOSE,
)


def kimi_call(call_id: str, args: str) -> str:
    return CALL_BEGIN + call_id + ARG_BEGIN + args + CALL_END


def kimi_section(*calls: str, joiner: str = "") -> str:
    return SECTION_BEGIN + joiner.join(calls) + SECTION_END


TEXTS: dict[str, str] = {
    # -- well formed ---------------------------------------------------
    "kimi_section_one": kimi_section(
        kimi_call("functions.get_weather:0", '{"city": "Beijing"}')
    ),
    "kimi_section_two": kimi_section(
        kimi_call("functions.alpha:0", '{"x": 1}'),
        kimi_call("functions.beta:1", '{"y": [2, 3], "z": "a b"}'),
        joiner="\n",
    ),
    "kimi_bare_call": kimi_call("functions.ping", "{}"),
    "kimi_text_around": (
        "before "
        + kimi_section(kimi_call("functions.f:0", '{"a": 1}'))
        + " after"
    ),
    "hermes_one": HERMES_OPEN
    + '{"name": "get_weather", "arguments": {"city": "Beijing"}}'
    + HERMES_CLOSE,
    "hermes_text_around": (
        "pre "
        + HERMES_OPEN
        + '{"name": "f", "arguments": {"a": 1}}'
        + HERMES_CLOSE
        + " post"
    ),
    "json_object": '{"name": "get_weather", "arguments": {"city": "Beijing"}}',
    "json_fenced": (
        '```json\n{"name": "get_weather", "arguments": {"city": "Beijing"}}\n```'
    ),
    "pythonic_one": '[get_weather(city="Beijing", days=3)]',
    #: A ``max_tokens`` cut landing right after the section token: everything
    #: the model wrote is inside an unterminated section.
    "kimi_section_truncated_after_begin": (
        "prefix"
        + SECTION_BEGIN
        + "the model then wrote a very long answer that never closes"
    ),
    #: Prose before and after a call, inside the section wrapper.
    "kimi_section_prose_between_calls": (
        SECTION_BEGIN
        + "I will now call the tool."
        + kimi_call("functions.f:0", "{}")
        + "SOME TRAILING TEXT"
        + SECTION_END
        + "after"
    ),
    #: Prose runs with their own leading/trailing whitespace, interleaved with
    #: two calls — the case where "drop whitespace" and "keep prose" collide.
    "kimi_section_prose_around_two_calls": (
        "lead "
        + SECTION_BEGIN
        + "  first I call alpha  "
        + kimi_call("functions.alpha:0", '{"x": 1}')
        + "\n  then beta:\n"
        + kimi_call("functions.beta:1", '{"y": 2}')
        + "  done  "
        + SECTION_END
        + " tail"
    ),
    # -- degenerate ----------------------------------------------------
    "kimi_unterminated_id": CALL_BEGIN + "foo",
    "kimi_unterminated_args": CALL_BEGIN + "functions.f:0" + ARG_BEGIN + '{"a": 1',
    "hermes_unterminated": HERMES_OPEN + "{bad json",
    "hermes_bad_json": HERMES_OPEN + "{bad json}" + HERMES_CLOSE,
    "json_not_a_call": '{"just": "an object"}',
    "json_plain_prose": "just some prose, no JSON here",
    "pythonic_not_calls": "[alpha, beta]",
    "pythonic_plain_prose": "hello there",
}

PARSER_NAMES = ("kimi", "hermes", "json", "pythonic", "passthrough")


def make(parser_name: str) -> ToolCallParser:
    return get_parser(parser_name)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def normalise(events: list) -> list[tuple]:
    """Reduce an event list to comparable tuples.

    * ``ParsedToolCall.id`` is dropped — it embeds a random uuid.
    * Adjacent ``ParsedText`` events are coalesced. A chunk boundary can split
      one run of text into several text events; that is a formatting artefact
      of the chunking, not a semantic difference. What must not change is the
      text *content* and its interleaving with tool calls.
    """
    out: list[tuple] = []
    for ev in events:
        if isinstance(ev, ParsedText):
            if out and out[-1][0] == "text":
                out[-1] = ("text", out[-1][1] + ev.text)
            else:
                out.append(("text", ev.text))
        elif isinstance(ev, ParsedToolCall):
            out.append(("call", ev.name, ev.arguments))
        else:  # pragma: no cover - guards against a new event type
            raise AssertionError(f"unexpected event type: {ev!r}")
    return out


def feed_chunks(parser_name: str, chunks) -> list:
    parser = make(parser_name)
    events: list = []
    for chunk in chunks:
        events.extend(parser.feed(chunk))
    events.extend(parser.finish())
    return events


def assert_split_invariant(parser_factory, text: str) -> None:
    """The event stream must not depend on how ``text`` was chunked.

    ``parser_factory`` must return a *fresh* parser on every call.
    """
    expected = normalise(parse_all(parser_factory(), text))

    def run(chunks) -> list[tuple]:
        parser = parser_factory()
        events: list = []
        for chunk in chunks:
            events.extend(parser.feed(chunk))
        events.extend(parser.finish())
        return normalise(events)

    # One character at a time — the worst case a real token stream can produce.
    assert run(list(text)) == expected, "char-by-char feed diverged from whole-string parse"

    # Every possible two-way split, including the two empty-chunk edges.
    for i in range(len(text) + 1):
        got = run([text[:i], text[i:]])
        assert got == expected, (
            f"split at index {i} diverged\n"
            f"  boundary: ...{text[max(0, i - 12):i]!r} | {text[i:i + 12]!r}..."
        )


def calls_of(events: list) -> list[ParsedToolCall]:
    return [e for e in events if isinstance(e, ParsedToolCall)]


def text_of(events: list) -> str:
    return "".join(e.text for e in events if isinstance(e, ParsedText))


# --------------------------------------------------------------------------
# 1. whole-string parse, one case per parser
# --------------------------------------------------------------------------


def test_kimi_section_with_one_call():
    events = parse_all(KimiToolParser(), TEXTS["kimi_section_one"])
    calls = calls_of(events)
    assert [c.name for c in calls] == ["get_weather"]
    assert json.loads(calls[0].arguments) == {"city": "Beijing"}
    assert calls[0].id
    assert text_of(events) == ""


def test_kimi_section_with_two_calls():
    events = parse_all(KimiToolParser(), TEXTS["kimi_section_two"])
    calls = calls_of(events)
    assert [c.name for c in calls] == ["alpha", "beta"]
    assert json.loads(calls[0].arguments) == {"x": 1}
    assert json.loads(calls[1].arguments) == {"y": [2, 3], "z": "a b"}
    assert calls[0].id != calls[1].id
    # Whitespace between calls inside a section is formatting, not content.
    assert text_of(events) == ""


def test_kimi_bare_call_without_section_wrapper():
    events = parse_all(KimiToolParser(), TEXTS["kimi_bare_call"])
    calls = calls_of(events)
    assert [c.name for c in calls] == ["ping"]
    assert json.loads(calls[0].arguments) == {}
    assert text_of(events) == ""


def test_hermes_tool_call():
    events = parse_all(HermesToolParser(), TEXTS["hermes_one"])
    calls = calls_of(events)
    assert [c.name for c in calls] == ["get_weather"]
    assert json.loads(calls[0].arguments) == {"city": "Beijing"}
    assert text_of(events) == ""


def test_json_object_reply():
    events = parse_all(JsonToolParser(), TEXTS["json_object"])
    calls = calls_of(events)
    assert [c.name for c in calls] == ["get_weather"]
    assert json.loads(calls[0].arguments) == {"city": "Beijing"}
    assert text_of(events) == ""


def test_json_fenced_object_reply():
    events = parse_all(JsonToolParser(), TEXTS["json_fenced"])
    calls = calls_of(events)
    assert [c.name for c in calls] == ["get_weather"]
    assert json.loads(calls[0].arguments) == {"city": "Beijing"}
    assert text_of(events) == ""


def test_json_array_of_calls():
    body = (
        '[{"name": "alpha", "arguments": {"x": 1}},'
        ' {"name": "beta", "arguments": {"y": 2}}]'
    )
    events = parse_all(JsonToolParser(), body)
    calls = calls_of(events)
    assert [c.name for c in calls] == ["alpha", "beta"]
    assert json.loads(calls[0].arguments) == {"x": 1}
    assert json.loads(calls[1].arguments) == {"y": 2}


def test_pythonic_reply():
    events = parse_all(PythonicToolParser(), "[fn(a=1)]")
    calls = calls_of(events)
    assert [c.name for c in calls] == ["fn"]
    assert json.loads(calls[0].arguments) == {"a": 1}
    assert text_of(events) == ""


def test_pythonic_reply_with_multiple_calls_and_types():
    events = parse_all(PythonicToolParser(), TEXTS["pythonic_one"])
    calls = calls_of(events)
    assert [c.name for c in calls] == ["get_weather"]
    assert json.loads(calls[0].arguments) == {"city": "Beijing", "days": 3}


# --------------------------------------------------------------------------
# 2. chunk-split invariance
# --------------------------------------------------------------------------

SPLIT_CASES = [
    ("kimi", "kimi_section_one"),
    ("kimi", "kimi_section_two"),
    ("kimi", "kimi_bare_call"),
    ("kimi", "kimi_text_around"),
    ("kimi", "kimi_section_truncated_after_begin"),
    ("kimi", "kimi_section_prose_between_calls"),
    ("kimi", "kimi_section_prose_around_two_calls"),
    ("kimi", "kimi_unterminated_id"),
    ("kimi", "kimi_unterminated_args"),
    ("kimi", "json_plain_prose"),
    ("hermes", "hermes_one"),
    ("hermes", "hermes_text_around"),
    ("hermes", "hermes_unterminated"),
    ("hermes", "hermes_bad_json"),
    ("hermes", "json_plain_prose"),
    ("json", "json_object"),
    ("json", "json_fenced"),
    ("json", "json_not_a_call"),
    ("json", "json_plain_prose"),
    ("pythonic", "pythonic_one"),
    ("pythonic", "pythonic_not_calls"),
    ("pythonic", "pythonic_plain_prose"),
    ("passthrough", "kimi_section_one"),
    ("passthrough", "json_plain_prose"),
]


@pytest.mark.parametrize(
    "parser_name,text_key",
    SPLIT_CASES,
    ids=[f"{p}-{t}" for p, t in SPLIT_CASES],
)
def test_split_invariance(parser_name, text_key):
    assert_split_invariant(lambda: make(parser_name), TEXTS[text_key])


def test_split_invariance_helper_detects_divergence():
    """The helper must actually be able to fail — guard against a no-op test."""

    class Leaky(ToolCallParser):
        """Emits every chunk verbatim, so chunking changes the event stream."""

        def feed(self, text):
            return [ParsedText(text)] if text else []

    # Leaky coalesces to the same text, so it *is* invariant under normalise;
    # this parser instead swallows a chunk boundary, which is a real divergence.
    class Dropper(ToolCallParser):
        def __init__(self):
            self.seen = 0

        def feed(self, text):
            self.seen += 1
            if self.seen == 2:
                return []
            return [ParsedText(text)] if text else []

    assert_split_invariant(Leaky, "hello world")
    with pytest.raises(AssertionError):
        assert_split_invariant(Dropper, "hello world")


# --------------------------------------------------------------------------
# 3. no tag leakage
# --------------------------------------------------------------------------

LEAK_CASES = [
    ("kimi", "kimi_section_one"),
    ("kimi", "kimi_section_two"),
    ("kimi", "kimi_bare_call"),
    ("kimi", "kimi_text_around"),
    ("kimi", "kimi_section_truncated_after_begin"),
    ("kimi", "kimi_section_prose_between_calls"),
    ("kimi", "kimi_section_prose_around_two_calls"),
    ("hermes", "hermes_one"),
    ("hermes", "hermes_text_around"),
    ("json", "json_object"),
    ("json", "json_fenced"),
    ("pythonic", "pythonic_one"),
]


@pytest.mark.parametrize(
    "parser_name,text_key",
    LEAK_CASES,
    ids=[f"{p}-{t}" for p, t in LEAK_CASES],
)
def test_no_control_token_leaks_into_text(parser_name, text_key):
    text = TEXTS[text_key]

    def check(events):
        visible = text_of(events)
        for token in CONTROL_TOKENS:
            assert token not in visible, f"{token!r} leaked into visible text {visible!r}"

    check(parse_all(make(parser_name), text))
    check(feed_chunks(parser_name, list(text)))
    for i in range(len(text) + 1):
        check(feed_chunks(parser_name, [text[:i], text[i:]]))


# --------------------------------------------------------------------------
# 4. text around calls is preserved, in order
# --------------------------------------------------------------------------


def test_text_around_kimi_section_is_preserved_in_order():
    events = parse_all(KimiToolParser(), TEXTS["kimi_text_around"])
    assert normalise(events) == [
        ("text", "before "),
        ("call", "f", '{"a": 1}'),
        ("text", " after"),
    ]


def test_text_around_hermes_call_is_preserved_in_order():
    events = parse_all(HermesToolParser(), TEXTS["hermes_text_around"])
    assert normalise(events) == [
        ("text", "pre "),
        ("call", "f", '{"a": 1}'),
        ("text", " post"),
    ]


def test_text_around_kimi_section_survives_char_by_char_feed():
    text = TEXTS["kimi_text_around"]
    assert normalise(feed_chunks("kimi", list(text))) == [
        ("text", "before "),
        ("call", "f", '{"a": 1}'),
        ("text", " after"),
    ]


# --------------------------------------------------------------------------
# 5. malformed input degrades to text, never swallowed
# --------------------------------------------------------------------------


def test_kimi_unterminated_call_id_comes_back_as_text():
    text = TEXTS["kimi_unterminated_id"]
    events = parse_all(KimiToolParser(), text)
    assert calls_of(events) == []
    assert text_of(events) == text


def test_kimi_unterminated_arguments_come_back_as_text():
    text = TEXTS["kimi_unterminated_args"]
    events = parse_all(KimiToolParser(), text)
    assert calls_of(events) == []
    assert text_of(events) == text


def test_kimi_unterminated_section_does_not_raise_or_invent_a_call():
    # Inside a section, whitespace between calls is formatting and is dropped
    # on purpose. What matters is that an unclosed section neither raises nor
    # fabricates a tool call.
    parser = KimiToolParser()
    parser.feed(SECTION_BEGIN + "\n")
    assert calls_of(parser.finish()) == []


def test_kimi_section_truncated_after_begin_keeps_the_prose():
    """A ``max_tokens`` cut just after the section token must not eat the reply.

    Regression: the ``section`` state used to discard its buffer unconditionally,
    so everything the model wrote after an unterminated section token vanished.
    """
    text = TEXTS["kimi_section_truncated_after_begin"]
    events = parse_all(KimiToolParser(), text)

    assert calls_of(events) == []
    visible = text_of(events)
    assert visible == "prefixthe model then wrote a very long answer that never closes"
    for token in CONTROL_TOKENS:
        assert token not in visible


def test_kimi_prose_between_calls_inside_a_section_is_not_swallowed():
    """Prose the model writes between calls inside a section is output."""
    text = TEXTS["kimi_section_prose_between_calls"]
    events = parse_all(KimiToolParser(), text)

    assert normalise(events) == [
        ("text", "I will now call the tool."),
        ("call", "f", "{}"),
        ("text", "SOME TRAILING TEXTafter"),
    ]
    for token in CONTROL_TOKENS:
        assert token not in text_of(events)


def test_kimi_section_prose_keeps_its_own_whitespace_but_drops_formatting():
    """A prose run survives verbatim; a whitespace-only run stays formatting."""
    events = parse_all(KimiToolParser(), TEXTS["kimi_section_prose_around_two_calls"])
    assert normalise(events) == [
        ("text", "lead   first I call alpha  "),
        ("call", "alpha", '{"x": 1}'),
        ("text", "\n  then beta:\n"),
        ("call", "beta", '{"y": 2}'),
        ("text", "  done   tail"),
    ]

    # The other side of the contract: a section whose only inter-call text is
    # whitespace still yields no visible text at all.
    assert text_of(parse_all(KimiToolParser(), TEXTS["kimi_section_two"])) == ""


@pytest.mark.parametrize(
    "text_key",
    [
        "kimi_section_truncated_after_begin",
        "kimi_section_prose_between_calls",
        "kimi_section_prose_around_two_calls",
    ],
)
def test_kimi_section_prose_survives_char_by_char_feed(text_key):
    """Whether prose is kept must not depend on where the chunk boundary fell."""
    text = TEXTS[text_key]
    assert normalise(feed_chunks("kimi", list(text))) == normalise(
        parse_all(KimiToolParser(), text)
    )


def test_hermes_unterminated_call_comes_back_as_text():
    text = TEXTS["hermes_unterminated"]
    events = parse_all(HermesToolParser(), text)
    assert calls_of(events) == []
    assert text_of(events) == text


def test_hermes_bad_json_comes_back_as_text():
    text = TEXTS["hermes_bad_json"]
    events = parse_all(HermesToolParser(), text)
    assert calls_of(events) == []
    assert text_of(events) == text


def test_json_parser_non_call_object_comes_back_as_text():
    text = TEXTS["json_not_a_call"]
    events = parse_all(JsonToolParser(), text)
    assert calls_of(events) == []
    assert text_of(events) == text


def test_json_parser_prose_streams_through():
    text = TEXTS["json_plain_prose"]
    events = parse_all(JsonToolParser(), text)
    assert calls_of(events) == []
    assert text_of(events) == text


def test_pythonic_parser_non_call_list_comes_back_as_text():
    text = TEXTS["pythonic_not_calls"]
    events = parse_all(PythonicToolParser(), text)
    assert calls_of(events) == []
    assert text_of(events) == text


@pytest.mark.parametrize(
    "parser_name,text_key",
    [
        ("kimi", "kimi_unterminated_id"),
        ("kimi", "kimi_unterminated_args"),
        ("hermes", "hermes_unterminated"),
        ("hermes", "hermes_bad_json"),
        ("json", "json_not_a_call"),
        ("pythonic", "pythonic_not_calls"),
    ],
)
def test_malformed_input_never_raises_under_any_chunking(parser_name, text_key):
    text = TEXTS[text_key]
    # char-by-char, then every two-way split — none may raise
    feed_chunks(parser_name, list(text))
    for i in range(len(text) + 1):
        feed_chunks(parser_name, [text[:i], text[i:]])


# --------------------------------------------------------------------------
# 6. passthrough parser
# --------------------------------------------------------------------------


def test_passthrough_returns_text_unchanged():
    text = "hello, world"
    assert parse_all(ToolCallParser(), text) == [ParsedText(text)]


def test_passthrough_emits_no_tool_calls_even_for_control_tokens():
    text = TEXTS["kimi_section_one"]
    events = parse_all(ToolCallParser(), text)
    assert calls_of(events) == []
    assert text_of(events) == text


def test_passthrough_ignores_empty_feed_and_finish():
    parser = ToolCallParser()
    assert parser.feed("") == []
    assert parser.finish() == []
    parser.reset()
    assert parser.feed("x") == [ParsedText("x")]


# --------------------------------------------------------------------------
# 7. registry
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name,expected",
    [
        ("passthrough", ToolCallParser),
        ("none", ToolCallParser),
        ("kimi", KimiToolParser),
        ("kimi_k2", KimiToolParser),
        ("hermes", HermesToolParser),
        ("json", JsonToolParser),
        ("pythonic", PythonicToolParser),
    ],
)
def test_get_parser_returns_the_registered_class(name, expected):
    parser = get_parser(name)
    assert type(parser) is expected
    # each call must hand back a fresh, independent parser
    assert get_parser(name) is not parser


def test_get_parser_unknown_name_raises_value_error():
    with pytest.raises(ValueError) as excinfo:
        get_parser("nope")
    message = str(excinfo.value)
    assert "nope" in message
    assert "kimi" in message


def test_parser_names_matches_registry():
    names = parser_names()
    assert names == sorted(names)
    assert set(names) == {
        "passthrough",
        "none",
        "kimi",
        "kimi_k2",
        "hermes",
        "json",
        "pythonic",
    }
