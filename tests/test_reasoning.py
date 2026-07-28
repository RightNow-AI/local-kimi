"""Tests for :mod:`k3.reasoning`.

The load-bearing property is byte-exactness: when a client echoes an assistant
turn back to us, K3 must see *the original bytes* again — including the raw
tool-call argument string, whitespace and key order intact, which no client can
round-trip on its own because it parses the arguments into an object.

Recovery has four tiers, and every one of them is pinned here:
``ledger`` > ``signature`` > ``echo`` > ``fingerprint`` > ``none``.
"""

from __future__ import annotations

import base64
import hashlib
import json
import time
import tracemalloc
import zlib

import pytest

from k3.ir import Message, ReasoningPart, TextPart, ToolCallPart
from k3.reasoning import (
    MAX_DECOMPRESSED_SIGNATURE_BYTES,
    MAX_INLINE_SIGNATURE_BYTES,
    SIG_LEDGER_ONLY,
    SIG_SELF_CONTAINED,
    ReasoningLedger,
    Restored,
    build_upstream_assistant,
    decode_signature,
    encode_signature,
    fingerprint_message,
    fingerprint_parts,
    restore_assistant,
    strip_inline_think,
    upstream_assistant_from_response,
)

# --------------------------------------------------------------------------
# shared scenario
# --------------------------------------------------------------------------

#: Deliberately kept inside cp1252 so a failure diff prints on a Windows
#: console. Full unicode coverage lives in the codec tests below.
REASONING = "first I weigh the options\nthen I pick a tool — carefully"
TEXT = "Here is the answer."

#: Non-canonical whitespace *and* non-alphabetical key order. A client that
#: does ``json.dumps(json.loads(...))`` destroys both; the ledger must not.
RAW_ARGS = '{"zeta":  1, "alpha":"x  y", "nested": {"b": [1,2] , "a":3}}'
TOOL_ID = "call_abc123"
TOOL_NAME = "do_thing"


def client_reserialised_args() -> str:
    """What a client hands back after parsing and re-serialising the args."""
    return json.dumps(json.loads(RAW_ARGS))


def make_upstream() -> dict:
    """The assistant message K3 emitted and wants echoed back verbatim."""
    return upstream_assistant_from_response(
        TEXT,
        REASONING,
        [ToolCallPart(id=TOOL_ID, name=TOOL_NAME, arguments=RAW_ARGS)],
    )


def make_client_echo(signature=None, with_reasoning=True) -> Message:
    """The ``ir.Message`` a client sends back on the next turn."""
    parts = []
    if with_reasoning:
        parts.append(ReasoningPart(text=REASONING, signature=signature))
    parts.append(TextPart(TEXT))
    parts.append(
        ToolCallPart(
            id=TOOL_ID,
            name=TOOL_NAME,
            arguments=client_reserialised_args(),
        )
    )
    return Message(role="assistant", parts=parts)


def populated_ledger(**kwargs) -> tuple[ReasoningLedger, str, dict]:
    ledger = ReasoningLedger(**kwargs)
    ledger_id = ledger.new_id()
    upstream = make_upstream()
    ledger.reserve(ledger_id, REASONING)
    ledger.complete(ledger_id, upstream)
    return ledger, ledger_id, upstream


def high_entropy_text(n: int) -> str:
    """Deterministic, effectively incompressible text of length ``n``."""
    chunks: list[str] = []
    total = 0
    seed = b"k3-deterministic-seed"
    while total < n:
        seed = hashlib.sha256(seed).digest()
        piece = base64.b64encode(seed).decode("ascii")
        chunks.append(piece)
        total += len(piece)
    return "".join(chunks)[:n]


def realistic_reasoning(n: int) -> str:
    """``n`` chars of plausible reasoning prose — compressible like the real thing.

    ``high_entropy_text`` is the worst case on purpose; this is the *normal*
    case, and it must stay inline-able no matter how big a real turn gets.
    """
    parts: list[str] = []
    total = 0
    i = 0
    while total < n:
        line = (
            f"step {i}: I should check whether the file at src/module_{i % 37}.py "
            f"still imports helper_{i % 11}, then re-run the tests — carefully.\n"
        )
        parts.append(line)
        total += len(line)
        i += 1
    return "".join(parts)[:n]


def zlib_bomb_signature(decompressed_bytes: int) -> str:
    """A well-formed ``k3r1.`` signature that inflates to ``decompressed_bytes``."""
    blob = zlib.compress(b"\x00" * decompressed_bytes, 9)
    return SIG_SELF_CONTAINED + base64.urlsafe_b64encode(blob).decode("ascii")


def measure(fn):
    """Run ``fn``, returning ``(result, seconds, peak_bytes_allocated)``."""
    tracemalloc.start()
    try:
        started = time.perf_counter()
        result = fn()
        elapsed = time.perf_counter() - started
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return result, elapsed, peak


# --------------------------------------------------------------------------
# 1. signature codec
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "reasoning",
    [
        "",
        "plain ascii",
        "with\nnewlines\r\nand\ttabs",
        "unicode: héllo wörld 世界 — ✓ 🤔",
        "trailing whitespace   ",
        '{"looks": "like json"}',
        "k3r1.not-really-a-signature",
    ],
    ids=[
        "empty",
        "ascii",
        "newlines",
        "unicode",
        "trailing-ws",
        "json-lookalike",
        "prefix-lookalike",
    ],
)
def test_encode_decode_signature_round_trip(reasoning):
    ledger_id = "ledger-id-42"
    signature = encode_signature(reasoning, ledger_id)
    assert signature.startswith(SIG_SELF_CONTAINED)
    inline, recovered_id = decode_signature(signature)
    assert recovered_id == ledger_id
    if reasoning:
        assert inline == reasoning
    else:
        # An empty reasoning string round-trips as an empty string.
        assert inline == ""


def test_signature_is_url_and_header_safe():
    signature = encode_signature("some\nreasoning — with unicode 世界", "lid")
    body = signature[len(SIG_SELF_CONTAINED) :]
    assert body.isascii()
    assert all(ch.isalnum() or ch in "-_=" for ch in body), body


def test_large_reasoning_falls_back_to_ledger_only_signature():
    big = high_entropy_text(200 * 1024)
    assert len(big) == 200 * 1024

    signature = encode_signature(big, "ledger-id-42")

    assert signature.startswith(SIG_LEDGER_ONLY)
    assert not signature.startswith(SIG_SELF_CONTAINED)
    assert signature == SIG_LEDGER_ONLY + "ledger-id-42"
    # Nothing near the inline budget got shipped.
    assert len(signature) < MAX_INLINE_SIGNATURE_BYTES

    inline, recovered_id = decode_signature(signature)
    assert inline is None
    assert recovered_id == "ledger-id-42"


def test_small_reasoning_stays_inline():
    signature = encode_signature("short", "lid")
    assert signature.startswith(SIG_SELF_CONTAINED)
    assert decode_signature(signature) == ("short", "lid")


@pytest.mark.parametrize(
    "signature",
    [
        None,
        "",
        "garbage",
        "k3r1",
        "k3r1.",
        "k3r1.!!!!",
        "k3r1.////",
        "k3l1.",
        "sig_abc123==",  # what a genuine Anthropic signature looks like
    ],
    ids=[
        "none",
        "empty",
        "garbage",
        "prefix-only",
        "empty-body",
        "not-base64",
        "not-zlib",
        "empty-ledger-id",
        "foreign-signature",
    ],
)
def test_decode_signature_returns_none_pair_for_bad_input(signature):
    assert decode_signature(signature) == (None, None)


@pytest.mark.parametrize("cut", [1, 3, 7, 13])
def test_decode_signature_survives_a_truncated_blob(cut):
    signature = encode_signature("some reasoning that is long enough to truncate", "lid")
    truncated = signature[:-cut]
    # Must not raise, and must not hand back a half-decoded payload.
    assert decode_signature(truncated) in {(None, None), ("", None)}
    assert decode_signature(truncated)[0] in (None, "")


def test_decode_signature_rejects_an_oversized_payload_without_decoding_it():
    """A zlib bomb: attacker-controlled bytes, reached on every request.

    Regression: ``decode_signature`` used to call ``zlib.decompress`` with no
    size cap, so a 340 KiB signature inflated to 256 MiB (~585 MB peak RSS) —
    a 1 MB request body reached ~1.7 GB.
    """
    signature = zlib_bomb_signature(256 * 1024 * 1024)
    # Small enough to fit many times over into one request body.
    assert len(signature) < 512 * 1024

    result, elapsed, peak = measure(lambda: decode_signature(signature))

    assert result == (None, None)
    assert peak < 4 * 1024 * 1024, f"inflated {peak} bytes decoding a bomb"
    assert elapsed < 1.0, f"took {elapsed:.3f}s to reject a bomb"


def test_decode_signature_caps_the_inflate_not_just_the_payload_size():
    """The cheap length gate is not the only defence — the inflate is bounded.

    This blob is comfortably under ``MAX_INLINE_SIGNATURE_BYTES``, so it passes
    the pre-decode size check and must be stopped by the decompression cap.
    """
    inflated = 56 * 1024 * 1024
    signature = zlib_bomb_signature(inflated)
    blob = base64.urlsafe_b64decode(signature[len(SIG_SELF_CONTAINED) :])
    assert len(blob) < MAX_INLINE_SIGNATURE_BYTES

    result, _, peak = measure(lambda: decode_signature(signature))

    assert result == (None, None)
    # zlib grows its output buffer geometrically, so peak sits near 2x the cap —
    # what matters is that it is nowhere near the 56 MiB this blob wanted.
    assert peak < 3 * MAX_DECOMPRESSED_SIGNATURE_BYTES, f"inflated {peak} bytes"
    assert peak < inflated // 2


def test_large_but_legitimate_reasoning_still_round_trips_exactly():
    """The caps must not be able to reject a real reasoning trace."""
    big = realistic_reasoning(200 * 1024)
    assert len(big) == 200 * 1024

    signature = encode_signature(big, "ledger-id-42")
    assert signature.startswith(SIG_SELF_CONTAINED), "200 KB of prose should stay inline"

    inline, ledger_id = decode_signature(signature)
    assert inline == big
    assert ledger_id == "ledger-id-42"


def test_ledger_only_signature_bypasses_the_size_caps():
    """``k3l1.`` carries no compressed payload, so the caps must not touch it."""
    ledger_id = "z" * (MAX_INLINE_SIGNATURE_BYTES * 2)
    assert decode_signature(SIG_LEDGER_ONLY + ledger_id) == (None, ledger_id)


def test_decode_signature_survives_a_corrupted_blob():
    signature = encode_signature("some reasoning", "lid")
    corrupted = signature[: len(SIG_SELF_CONTAINED) + 4] + "ZZZZ" + signature[len(SIG_SELF_CONTAINED) + 8 :]
    assert decode_signature(corrupted) == (None, None)


# --------------------------------------------------------------------------
# 2. byte-exact round trip through a client echo  (the crown jewel)
# --------------------------------------------------------------------------


def test_ledger_hit_round_trips_the_upstream_message_byte_for_byte():
    ledger, ledger_id, upstream = populated_ledger()
    signature = encode_signature(REASONING, ledger_id)

    # Sanity: the client really did mangle the argument bytes.
    assert client_reserialised_args() != RAW_ARGS

    echo = make_client_echo(signature=signature)
    restored = restore_assistant(echo, ledger)

    assert restored.source == "ledger"
    assert restored.exact is True
    assert restored.reasoning == REASONING

    built = build_upstream_assistant(echo, restored)

    assert built == upstream
    assert json.dumps(built, sort_keys=True, ensure_ascii=False) == json.dumps(
        upstream, sort_keys=True, ensure_ascii=False
    )
    # The whole point: the raw argument string, whitespace and key order intact.
    assert built["tool_calls"][0]["function"]["arguments"] == RAW_ARGS
    assert built["reasoning_content"] == REASONING
    assert built["content"] == TEXT


def test_ledger_hit_returns_a_copy_not_the_stored_dict():
    ledger, ledger_id, upstream = populated_ledger()
    echo = make_client_echo(signature=encode_signature(REASONING, ledger_id))

    built = build_upstream_assistant(echo, restore_assistant(echo, ledger))
    built["content"] = "mutated"

    again = build_upstream_assistant(echo, restore_assistant(echo, ledger))
    assert again["content"] == TEXT


# --------------------------------------------------------------------------
# 3. ledger miss falls back to the self-contained signature
# --------------------------------------------------------------------------


def test_ledger_miss_falls_back_to_the_self_contained_signature():
    _, ledger_id, _ = populated_ledger()
    signature = encode_signature(REASONING, ledger_id)

    fresh = ReasoningLedger()  # process restarted; nothing in memory
    echo = make_client_echo(signature=signature)

    restored = restore_assistant(echo, fresh)

    assert restored.source == "signature"
    assert restored.exact is False
    assert restored.reasoning == REASONING
    assert restored.upstream_message is None

    built = build_upstream_assistant(echo, restored)
    assert built["reasoning_content"] == REASONING
    assert built["content"] == TEXT
    # Without the ledger we can only echo what the client gave us back.
    assert built["tool_calls"][0]["function"]["arguments"] == client_reserialised_args()


def test_signature_fallback_works_when_the_client_dropped_the_visible_text():
    _, ledger_id, _ = populated_ledger()
    signature = encode_signature(REASONING, ledger_id)
    msg = Message(
        role="assistant",
        parts=[ReasoningPart(text="", signature=signature), TextPart(TEXT)],
    )
    restored = restore_assistant(msg, ReasoningLedger())
    assert restored.source == "signature"
    assert restored.reasoning == REASONING


# --------------------------------------------------------------------------
# 4. echo fallback
# --------------------------------------------------------------------------


def test_reasoning_part_without_signature_falls_back_to_echo():
    echo = make_client_echo(signature=None)
    restored = restore_assistant(echo, ReasoningLedger())

    assert restored.source == "echo"
    assert restored.reasoning == REASONING
    assert restored.upstream_message is None

    built = build_upstream_assistant(echo, restored)
    assert built["reasoning_content"] == REASONING


def test_redacted_reasoning_is_not_echoed():
    msg = Message(
        role="assistant",
        parts=[ReasoningPart(text="secret", redacted=True), TextPart(TEXT)],
    )
    restored = restore_assistant(msg, ReasoningLedger())
    assert restored.source == "none"
    assert restored.reasoning == ""


# --------------------------------------------------------------------------
# 5. fingerprint recovery
# --------------------------------------------------------------------------


def test_fingerprint_recovers_reasoning_the_client_stripped_entirely():
    ledger, _, upstream = populated_ledger()

    # The client kept only what it understands: text + tool call. No reasoning,
    # no signature — and it re-serialised the arguments on the way through.
    stripped = make_client_echo(with_reasoning=False)
    assert not [p for p in stripped.parts if isinstance(p, ReasoningPart)]
    assert stripped.tool_calls()[0].arguments != RAW_ARGS

    restored = restore_assistant(stripped, ledger)

    assert restored.source == "fingerprint"
    assert restored.reasoning == REASONING
    # A fingerprint hit still carries the original upstream message.
    built = build_upstream_assistant(stripped, restored)
    assert built == upstream
    assert built["tool_calls"][0]["function"]["arguments"] == RAW_ARGS


def test_fingerprint_is_stable_across_argument_whitespace_and_key_order():
    a = fingerprint_parts(TEXT, [(TOOL_NAME, RAW_ARGS)])
    b = fingerprint_parts(TEXT, [(TOOL_NAME, client_reserialised_args())])
    c = fingerprint_parts(
        TEXT,
        [(TOOL_NAME, json.dumps(json.loads(RAW_ARGS), sort_keys=True, indent=2))],
    )
    assert a == b == c

    # ...but not across a real change in content.
    assert a != fingerprint_parts(TEXT, [(TOOL_NAME, '{"zeta": 2}')])
    assert a != fingerprint_parts(TEXT + "!", [(TOOL_NAME, RAW_ARGS)])
    assert a != fingerprint_parts(TEXT, [("other_tool", RAW_ARGS)])


def test_fingerprint_message_matches_fingerprint_parts():
    msg = make_client_echo(with_reasoning=False)
    assert fingerprint_message(msg) == fingerprint_parts(
        TEXT, [(TOOL_NAME, client_reserialised_args())]
    )


def test_fingerprint_miss_on_a_different_turn():
    ledger, _, _ = populated_ledger()
    other = Message(role="assistant", parts=[TextPart("a completely different reply")])
    assert restore_assistant(other, ledger).source == "none"


# --------------------------------------------------------------------------
# 6. nothing recoverable
# --------------------------------------------------------------------------


def test_nothing_recoverable_degrades_cleanly():
    stripped = make_client_echo(with_reasoning=False)
    restored = restore_assistant(stripped, ReasoningLedger())

    assert restored.source == "none"
    assert restored.reasoning == ""
    assert restored.exact is False
    assert restored.upstream_message is None

    built = build_upstream_assistant(stripped, restored)

    assert built["role"] == "assistant"
    assert built["content"] == TEXT
    assert "reasoning_content" not in built
    assert built["tool_calls"] == [
        {
            "id": TOOL_ID,
            "type": "function",
            "function": {"name": TOOL_NAME, "arguments": client_reserialised_args()},
        }
    ]


def test_empty_assistant_message_still_builds():
    msg = Message(role="assistant", parts=[])
    built = build_upstream_assistant(msg, restore_assistant(msg, ReasoningLedger()))
    assert built == {"role": "assistant", "content": ""}


def test_tool_call_only_message_gets_empty_string_content():
    msg = Message(
        role="assistant",
        parts=[ToolCallPart(id=TOOL_ID, name=TOOL_NAME, arguments="{}")],
    )
    built = build_upstream_assistant(msg, Restored())
    assert built["content"] == ""
    assert built["tool_calls"][0]["function"]["name"] == TOOL_NAME


# --------------------------------------------------------------------------
# 7. ledger housekeeping: LRU eviction and TTL
# --------------------------------------------------------------------------


def test_ledger_evicts_the_oldest_entry_and_cleans_its_fingerprint_index():
    ledger = ReasoningLedger(max_entries=2)

    fingerprints = []
    for i in range(3):
        content = f"turn {i}"
        ledger.reserve(f"id{i}", f"reason {i}")
        ledger.complete(f"id{i}", {"role": "assistant", "content": content})
        fingerprints.append(fingerprint_parts(content, []))

    before = ledger.stats()
    assert before == {"entries": 2, "hits": 0, "misses": 0, "fingerprint_hits": 0}

    # The oldest entry is gone...
    assert ledger.get("id0") is None
    # ...and so is its fingerprint index entry, not just the id.
    assert ledger.find_by_fingerprint(fingerprints[0]) is None

    # The two newest survive, by id and by fingerprint.
    assert ledger.get("id1") is not None
    entry2 = ledger.get("id2")
    assert entry2 is not None
    assert entry2.reasoning == "reason 2"
    assert ledger.find_by_fingerprint(fingerprints[2]) is entry2

    after = ledger.stats()
    assert after == {"entries": 2, "hits": 2, "misses": 1, "fingerprint_hits": 1}


def test_shared_fingerprint_survives_eviction_of_the_older_entry():
    """Two turns with identical visible output — routine in an agent loop.

    Regression: ``complete`` reassigned the index without clearing the previous
    owner's ``fingerprint``, so evicting that older entry popped an index slot
    that had since been handed to a different, still-live entry.
    """
    ledger = ReasoningLedger(max_entries=2)
    repeated = {"role": "assistant", "content": "ls"}
    fp = fingerprint_parts("ls", [])

    ledger.reserve("older", "reason older")
    ledger.complete("older", dict(repeated))
    ledger.reserve("newer", "reason newer")
    ledger.complete("newer", dict(repeated))

    # The newest turn owns the mapping...
    owner = ledger.find_by_fingerprint(fp)
    assert owner is not None and owner.id == "newer"

    # ...and evicting the older one must not take that mapping with it.
    ledger.reserve("third", "reason third")
    ledger.complete("third", {"role": "assistant", "content": "something else"})
    assert ledger.get("older") is None

    still = ledger.find_by_fingerprint(fp)
    assert still is not None, "eviction removed a fingerprint owned by a live entry"
    assert still.id == "newer"
    assert still.reasoning == "reason newer"
    assert ledger.stats()["entries"] == 2


def test_fingerprint_recovery_still_works_after_a_colliding_turn_is_evicted():
    """End to end: the ``strip`` policy's only restoration vehicle must survive."""
    ledger = ReasoningLedger(max_entries=2)
    upstream = make_upstream()

    # An earlier turn with byte-identical visible output, then the real one.
    ledger.reserve("older", "stale reasoning")
    ledger.complete("older", dict(upstream))
    ledger.reserve("newer", REASONING)
    ledger.complete("newer", upstream)

    # Enough traffic to evict the older turn.
    ledger.reserve("filler", "filler")
    ledger.complete("filler", {"role": "assistant", "content": "unrelated"})
    assert ledger.get("older") is None

    # A client on the ``strip`` policy hands back text + tool call and nothing else.
    stripped = make_client_echo(with_reasoning=False)
    restored = restore_assistant(stripped, ledger)

    assert restored.source == "fingerprint"
    assert restored.reasoning == REASONING
    built = build_upstream_assistant(stripped, restored)
    assert built["tool_calls"][0]["function"]["arguments"] == RAW_ARGS


def test_recompleting_an_entry_retires_its_old_fingerprint():
    """A re-completed turn must not leave a stale index slot pointing at it."""
    ledger = ReasoningLedger()
    ledger.reserve("x", REASONING)
    ledger.complete("x", {"role": "assistant", "content": "first"})
    ledger.complete("x", {"role": "assistant", "content": "second"})

    assert ledger.find_by_fingerprint(fingerprint_parts("first", [])) is None
    entry = ledger.find_by_fingerprint(fingerprint_parts("second", []))
    assert entry is not None and entry.id == "x"


def test_ledger_expires_entries_after_the_ttl():
    ledger = ReasoningLedger(ttl_seconds=0.01)
    ledger.reserve("x", REASONING)
    ledger.complete("x", {"role": "assistant", "content": "hi"})
    assert ledger.stats()["entries"] == 1

    time.sleep(0.05)

    assert ledger.get("x") is None
    assert ledger.find_by_fingerprint(fingerprint_parts("hi", [])) is None
    assert ledger.stats() == {
        "entries": 0,
        "hits": 0,
        "misses": 1,
        "fingerprint_hits": 0,
    }


def test_expired_ledger_entry_falls_back_to_the_signature():
    ledger = ReasoningLedger(ttl_seconds=0.01)
    ledger_id = ledger.new_id()
    ledger.reserve(ledger_id, REASONING)
    ledger.complete(ledger_id, make_upstream())
    signature = encode_signature(REASONING, ledger_id)

    time.sleep(0.05)

    echo = make_client_echo(signature=signature)
    restored = restore_assistant(echo, ledger)
    assert restored.source == "signature"
    assert restored.reasoning == REASONING


def test_ledger_get_and_clear_bookkeeping():
    ledger, ledger_id, _ = populated_ledger()
    assert ledger.stats()["entries"] == 1

    assert ledger.get(ledger_id) is not None
    assert ledger.stats()["hits"] == 1

    assert ledger.get("no-such-id") is None
    assert ledger.stats()["misses"] == 1

    # A falsy id is not a lookup at all and must not be counted.
    assert ledger.get(None) is None
    assert ledger.get("") is None
    assert ledger.stats()["misses"] == 1

    ledger.clear()
    assert ledger.stats()["entries"] == 0
    assert ledger.get(ledger_id) is None


def test_complete_without_reserve_recovers_reasoning_from_the_message():
    ledger = ReasoningLedger()
    ledger.complete("orphan", {"role": "assistant", "content": TEXT, "reasoning_content": REASONING})
    entry = ledger.get("orphan")
    assert entry is not None
    assert entry.reasoning == REASONING


# --------------------------------------------------------------------------
# 8. strip_inline_think
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text,expected",
    [
        ("<think>abc</think>hello", ("abc", "hello")),
        ("pre<think>abc</think>post", ("abc", "prepost")),
        ("<think></think>hello", ("", "hello")),
        ("<think>multi\nline</think>tail", ("multi\nline", "tail")),
        ("no tags at all", ("", "no tags at all")),
        ("", ("", "")),
        ("<think>unterminated", ("unterminated", "")),
        ("visible<think>unterminated", ("unterminated", "visible")),
    ],
    ids=[
        "leading",
        "embedded",
        "empty-think",
        "multiline",
        "no-tags",
        "empty-string",
        "unterminated",
        "unterminated-with-prefix",
    ],
)
def test_strip_inline_think(text, expected):
    assert strip_inline_think(text) == expected


def test_strip_inline_think_round_trips_with_the_inline_builder():
    msg = Message(role="assistant", parts=[TextPart(TEXT)])
    built = build_upstream_assistant(
        msg, Restored(reasoning=REASONING, source="signature"), reasoning_field="inline"
    )
    assert strip_inline_think(built["content"]) == (REASONING, TEXT)


# --------------------------------------------------------------------------
# 9. build_upstream_assistant reasoning_field handling
# --------------------------------------------------------------------------


def test_build_upstream_default_field_carries_reasoning_content():
    msg = Message(role="assistant", parts=[TextPart(TEXT)])
    built = build_upstream_assistant(msg, Restored(reasoning=REASONING, source="signature"))
    assert built == {"role": "assistant", "content": TEXT, "reasoning_content": REASONING}


def test_build_upstream_with_reasoning_field_none_omits_reasoning():
    msg = Message(role="assistant", parts=[TextPart(TEXT)])
    built = build_upstream_assistant(
        msg, Restored(reasoning=REASONING, source="signature"), reasoning_field="none"
    )
    assert built == {"role": "assistant", "content": TEXT}
    assert REASONING not in json.dumps(built, ensure_ascii=False)


def test_build_upstream_with_reasoning_field_inline_wraps_in_think_tags():
    msg = Message(role="assistant", parts=[TextPart(TEXT)])
    built = build_upstream_assistant(
        msg, Restored(reasoning=REASONING, source="signature"), reasoning_field="inline"
    )
    assert built == {
        "role": "assistant",
        "content": f"<think>{REASONING}</think>{TEXT}",
    }
    assert "reasoning_content" not in built
    assert "inline" not in built


def test_build_upstream_with_a_custom_reasoning_field_name():
    msg = Message(role="assistant", parts=[TextPart(TEXT)])
    built = build_upstream_assistant(
        msg, Restored(reasoning=REASONING, source="signature"), reasoning_field="reasoning"
    )
    assert built == {"role": "assistant", "content": TEXT, "reasoning": REASONING}


def test_build_upstream_renames_reasoning_content_on_a_ledger_hit():
    ledger, ledger_id, _ = populated_ledger()
    echo = make_client_echo(signature=encode_signature(REASONING, ledger_id))
    restored = restore_assistant(echo, ledger)
    assert restored.source == "ledger"

    built = build_upstream_assistant(echo, restored, reasoning_field="reasoning")
    assert built["reasoning"] == REASONING
    assert "reasoning_content" not in built
    # The exact bytes still survive the rename.
    assert built["tool_calls"][0]["function"]["arguments"] == RAW_ARGS


# NOTE: the two tests below are expected to FAIL against the current source.
# ``build_upstream_assistant`` treats the "none"/"inline" *sentinels* as if
# they were upstream field names on the ledger-hit path, so the reasoning ends
# up under a key literally called "none" / "inline". They assert the correct
# behaviour, matching both the non-ledger path above and
# ``upstream_assistant_from_response``.


def test_build_upstream_field_none_omits_reasoning_on_a_ledger_hit():
    ledger, ledger_id, _ = populated_ledger()
    echo = make_client_echo(signature=encode_signature(REASONING, ledger_id))
    restored = restore_assistant(echo, ledger)
    assert restored.source == "ledger"

    built = build_upstream_assistant(echo, restored, reasoning_field="none")

    assert "reasoning_content" not in built
    assert "none" not in built, "reasoning leaked under a key literally named 'none'"
    assert REASONING not in json.dumps(built, ensure_ascii=False)
    # Everything else is still byte-exact.
    assert built["content"] == TEXT
    assert built["tool_calls"][0]["function"]["arguments"] == RAW_ARGS


def test_build_upstream_field_inline_wraps_in_think_tags_on_a_ledger_hit():
    ledger, ledger_id, _ = populated_ledger()
    echo = make_client_echo(signature=encode_signature(REASONING, ledger_id))
    restored = restore_assistant(echo, ledger)
    assert restored.source == "ledger"

    built = build_upstream_assistant(echo, restored, reasoning_field="inline")

    assert "inline" not in built, "reasoning leaked under a key literally named 'inline'"
    assert "reasoning_content" not in built
    assert built["content"] == f"<think>{REASONING}</think>{TEXT}"
    assert built["tool_calls"][0]["function"]["arguments"] == RAW_ARGS


# --------------------------------------------------------------------------
# upstream_assistant_from_response
# --------------------------------------------------------------------------


def test_upstream_assistant_from_response_keeps_raw_arguments():
    out = make_upstream()
    assert out["role"] == "assistant"
    assert out["content"] == TEXT
    assert out["reasoning_content"] == REASONING
    assert out["tool_calls"][0]["function"]["arguments"] == RAW_ARGS
    assert out["tool_calls"][0]["id"] == TOOL_ID
    assert out["tool_calls"][0]["type"] == "function"


@pytest.mark.parametrize("field", ["none", "inline"])
def test_upstream_assistant_from_response_sentinels_are_not_field_names(field):
    out = upstream_assistant_from_response(TEXT, REASONING, [], reasoning_field=field)
    assert field not in out
    if field == "none":
        assert out == {"role": "assistant", "content": TEXT}
    else:
        assert out == {"role": "assistant", "content": f"<think>{REASONING}</think>{TEXT}"}


def test_upstream_assistant_from_response_without_reasoning():
    out = upstream_assistant_from_response(TEXT, "", [])
    assert out == {"role": "assistant", "content": TEXT}
