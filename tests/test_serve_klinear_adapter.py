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


class _SyntheticTiktokenModel:
    def __init__(self, token_bytes: dict[int, bytes]) -> None:
        self.token_bytes = token_bytes

    def decode_single_token_bytes(self, token_id: int) -> bytes:
        return self.token_bytes[token_id]


class _SyntheticMoonshotTokenizer:
    special_tokens = {
        "<|open|>": 1000,
        "<|close|>": 1001,
        "<|sep|>": 1002,
        "<|end_of_msg|>": 1003,
    }
    eos_token_id = 1004
    unk_token_id = -1

    def __init__(self) -> None:
        self.model = _SyntheticTiktokenModel(
            {
                10: b"reason",
                11: b"think",
                12: b"response",
                13: b"visible",
            }
        )
        self.template_calls: list[tuple[list[dict], dict]] = []

    def apply_chat_template(self, messages, **kwargs):
        self.template_calls.append((messages, kwargs))
        return [7001, 7002, 7003, 7004]


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


def test_moonshot_tokenizer_uses_real_prompt_ids_and_splits_xtml_channels():
    synthetic = _SyntheticMoonshotTokenizer()
    tokenizer = KimiChatTokenizer(synthetic)
    prompt = ChatPrompt(
        messages=({"role": "user", "content": "hello"},),
        reasoning_effort="medium",
    )

    prompt_ids = tokenizer.encode_prompt(prompt)
    decoder = tokenizer.new_decoder()
    fragments = []
    for token_id in (
        10,
        1001,
        11,
        1002,
        1000,
        12,
        1002,
        13,
        1001,
        12,
        1002,
    ):
        fragments.extend(decoder.push(token_id))
    fragments.extend(decoder.finish())

    assert prompt_ids == [7001, 7002, 7003, 7004]
    _, template_kwargs = synthetic.template_calls[0]
    assert template_kwargs["tokenize"] is True
    assert template_kwargs["add_generation_prompt"] is True
    assert template_kwargs["thinking_effort"] == "high"
    assert fragments == [
        DecodedFragment("reasoning", "reason"),
        DecodedFragment("content", "visible"),
    ]


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
