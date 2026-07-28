import pytest
import torch

import engine.serve.klinear_engine as klinear_adapter
from engine.serve.contracts import (
    ChatPrompt,
    DecodedFragment,
    SamplingParams,
    TokenEvent,
    UsageEvent,
)
from engine.serve.klinear_engine import KimiChatTokenizer, KLinearEngine
from engine.serve.runtime import CompletionEnd, GenerationRuntime
from engine.serve.stub import ByteChatTokenizer

VERIFIED_CONTROL_TOKEN_IDS = {
    "[EOS]": 163585,
    "[EOT]": 163593,
    "<|im_end|>": 163586,
    "<|im_user|>": 163587,
    "<|im_assistant|>": 163588,
    "<|im_system|>": 163594,
    "<|im_middle|>": 163601,
    "<|tool_calls_section_begin|>": 163595,
    "<|tool_calls_section_end|>": 163596,
    "<|tool_call_begin|>": 163597,
    "<|tool_call_argument_begin|>": 163598,
    "<|tool_call_end|>": 163599,
}


class _SyntheticTiktokenModel:
    def __init__(self, token_bytes: dict[int, bytes]) -> None:
        self.token_bytes = token_bytes

    def decode_single_token_bytes(self, token_id: int) -> bytes:
        return self.token_bytes[token_id]


class _SyntheticMoonshotTokenizer:
    eos_token_id = VERIFIED_CONTROL_TOKEN_IDS["[EOS]"]
    unk_token_id = 163838

    def __init__(
        self,
        *,
        special_tokens: dict[str, int] | None = None,
        rendered_prompt: list[int] | None = None,
    ) -> None:
        self.special_tokens = dict(
            VERIFIED_CONTROL_TOKEN_IDS if special_tokens is None else special_tokens
        )
        self.model = _SyntheticTiktokenModel(
            {
                10: b"visible",
                11: b" response",
            }
        )
        self.rendered_prompt = rendered_prompt or [7001, 7002, 7003, 7004]
        self.template_calls: list[tuple[list[dict], dict]] = []

    def convert_tokens_to_ids(self, token: str) -> int:
        return self.special_tokens.get(token, self.unk_token_id)

    def apply_chat_template(self, messages, **kwargs):
        self.template_calls.append((messages, kwargs))
        return list(self.rendered_prompt)


def _scripted_engine(
    monkeypatch,
    token_ids: list[int],
    *,
    eos_token_ids: set[int],
):
    produced: list[int] = []
    closed: list[bool] = []

    def scripted_tokens(model, prompt, max_new_tokens, **kwargs):
        del model, kwargs
        try:
            for token_id in token_ids[:max_new_tokens]:
                produced.append(token_id)
                yield torch.tensor(
                    [token_id],
                    dtype=torch.long,
                    device=prompt.device,
                )
        finally:
            closed.append(True)

    monkeypatch.setattr(klinear_adapter, "generate_tokens", scripted_tokens)
    tiny_model = torch.nn.Linear(1, 1, bias=False)
    engine = KLinearEngine(
        tiny_model,
        eos_token_ids,
        device="cpu",
    )
    return engine, produced, closed


def test_adapter_resolves_verified_kimi_linear_control_tokens():
    tokenizer = KimiChatTokenizer(_SyntheticMoonshotTokenizer())

    assert tokenizer.control_token_ids == VERIFIED_CONTROL_TOKEN_IDS
    assert tokenizer.eos_token_ids == {163585, 163586, 163593}


def test_adapter_names_a_missing_required_control_token():
    special_tokens = dict(VERIFIED_CONTROL_TOKEN_IDS)
    del special_tokens["<|im_middle|>"]

    with pytest.raises(
        ValueError,
        match=r"missing required special token <\|im_middle\|>",
    ):
        KimiChatTokenizer(_SyntheticMoonshotTokenizer(special_tokens=special_tokens))


def test_rendered_user_turn_matches_the_authoritative_template_sequence():
    rendered = [
        163587,
        2482,
        163601,
        163586,
        163588,
        69702,
        163601,
    ]
    synthetic = _SyntheticMoonshotTokenizer(rendered_prompt=rendered)
    tokenizer = KimiChatTokenizer(synthetic)

    prompt_ids = tokenizer.encode_prompt(
        ChatPrompt(
            messages=({"role": "user", "content": ""},),
            reasoning_effort="high",
        )
    )

    assert prompt_ids == rendered
    messages, template_kwargs = synthetic.template_calls[0]
    assert messages == [{"role": "user", "content": ""}]
    assert template_kwargs == {
        "tokenize": True,
        "add_generation_prompt": True,
    }


def test_decoder_emits_content_only_for_template_without_reasoning_channel():
    tokenizer = KimiChatTokenizer(_SyntheticMoonshotTokenizer())
    decoder = tokenizer.new_decoder()

    fragments = decoder.push(10) + decoder.push(11) + decoder.finish()

    assert fragments == [
        DecodedFragment("content", "visible"),
        DecodedFragment("content", " response"),
    ]


def test_tool_call_envelope_becomes_openai_tool_calls_with_id_and_arguments_intact():
    tokenizer = KimiChatTokenizer(_SyntheticMoonshotTokenizer())
    arguments = '{"city":"Amman","unit":"celsius"}'
    envelope = (
        "<|tool_calls_section_begin|>"
        "<|tool_call_begin|>functions.get_weather:7"
        f"<|tool_call_argument_begin|>{arguments}"
        "<|tool_call_end|>"
        "<|tool_calls_section_end|>"
    )

    message = tokenizer.parse_assistant_output(envelope)

    assert message == {
        "content": None,
        "tool_calls": [
            {
                "id": "functions.get_weather:7",
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "arguments": arguments,
                },
            }
        ],
    }
    assert "reasoning_content" not in message


@pytest.mark.asyncio
async def test_adapter_yields_each_token_before_producing_the_rest(monkeypatch):
    engine, produced, _ = _scripted_engine(
        monkeypatch,
        [11, 12, 13],
        eos_token_ids={99},
    )
    source = engine.generate([101, 202], SamplingParams(max_tokens=3))

    first = await source.__anext__()

    assert first == TokenEvent(11)
    assert produced == [11]
    await source.aclose()


@pytest.mark.asyncio
async def test_closing_adapter_mid_generation_stops_token_production(monkeypatch):
    engine, produced, closed = _scripted_engine(
        monkeypatch,
        [21, 22, 23, 24],
        eos_token_ids={99},
    )
    source = engine.generate([1, 2, 3], SamplingParams(max_tokens=4))

    assert await source.__anext__() == TokenEvent(21)
    assert await source.__anext__() == TokenEvent(22)
    await source.aclose()

    assert produced == [21, 22]
    assert closed == [True]
    assert engine.active_generations == 0
    assert engine.cancelled_generations == 1


@pytest.mark.asyncio
async def test_generation_stops_on_im_end_and_not_only_eos(monkeypatch):
    tokenizer = KimiChatTokenizer(_SyntheticMoonshotTokenizer())
    engine, produced, _ = _scripted_engine(
        monkeypatch,
        [31, tokenizer.im_end_token_id, 32],
        eos_token_ids=set(tokenizer.eos_token_ids),
    )

    events = [
        event
        async for event in engine.generate(
            [5000, 7, 42],
            SamplingParams(max_tokens=10),
        )
    ]

    assert produced == [31, 163586]
    assert [event.token_id for event in events if isinstance(event, TokenEvent)] == [
        31,
        163586,
    ]
    assert [event for event in events if isinstance(event, UsageEvent)] == [
        UsageEvent(prompt_tokens=3, completion_tokens=2)
    ]


@pytest.mark.asyncio
async def test_usage_counts_exact_prompt_and_generated_token_ids(monkeypatch):
    engine, _, _ = _scripted_engine(
        monkeypatch,
        [31, 32, 99, 33],
        eos_token_ids={99},
    )
    prompt_ids = [5000, 7, 7, 42, 163584]

    events = [
        event
        async for event in engine.generate(
            prompt_ids,
            SamplingParams(max_tokens=10),
        )
    ]

    assert [event.token_id for event in events if isinstance(event, TokenEvent)] == [
        31,
        32,
        99,
    ]
    assert [event for event in events if isinstance(event, UsageEvent)] == [
        UsageEvent(prompt_tokens=len(prompt_ids), completion_tokens=3)
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("generated_ids", "max_tokens", "expected_reason"),
    [
        ([ord("a"), ord("b"), ord("c")], 2, "length"),
        ([ord("a"), ByteChatTokenizer.EOS, ord("b")], 5, "stop"),
    ],
)
async def test_runtime_finish_reason_tracks_length_and_eos(
    monkeypatch,
    generated_ids,
    max_tokens,
    expected_reason,
):
    tokenizer = ByteChatTokenizer()
    engine, _, _ = _scripted_engine(
        monkeypatch,
        generated_ids,
        eos_token_ids=set(tokenizer.eos_token_ids),
    )
    runtime = GenerationRuntime(engine, tokenizer, serialize_engine=False)

    events = [
        event
        async for event in runtime.run(
            [10, 20, 30, 40],
            SamplingParams(max_tokens=max_tokens),
        )
    ]

    ends = [event for event in events if isinstance(event, CompletionEnd)]
    assert len(ends) == 1
    assert ends[0].finish_reason == expected_reason
    assert ends[0].prompt_tokens == 4
    assert ends[0].completion_tokens == 2
