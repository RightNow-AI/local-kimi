from __future__ import annotations

import torch
from torch import nn

from engine.klinear.state import KDALayerState, KLinearDecodeState, MLALayerState
from engine.speculative.draft import propose
from engine.speculative.reference import ordinary_greedy, speculative_greedy
from engine.speculative.state_checkpoint import DecodeCheckpoint
from engine.speculative.verify import align_verification_logits, verify_greedy


def _one_hot_logits(predictions: torch.Tensor, vocab_size: int) -> torch.Tensor:
    logits = torch.full((*predictions.shape, vocab_size), -100.0)
    return logits.scatter_(-1, predictions.unsqueeze(-1), 100.0)


def _state_tensors(state: KLinearDecodeState) -> dict[str, torch.Tensor]:
    tensors: dict[str, torch.Tensor] = {}
    for index, layer in enumerate(state.layer_states):
        if isinstance(layer, KDALayerState):
            tensors[f"{index}.q_conv"] = layer.q_conv
            tensors[f"{index}.k_conv"] = layer.k_conv
            tensors[f"{index}.v_conv"] = layer.v_conv
            tensors[f"{index}.recurrent"] = layer.recurrent
        elif isinstance(layer, MLALayerState):
            tensors[f"{index}.compressed_kv"] = layer.compressed_kv
            tensors[f"{index}.rotary_key"] = layer.rotary_key
            tensors[f"{index}.key_pass"] = layer.key_pass
            tensors[f"{index}.value"] = layer.value
            tensors[f"{index}.position"] = layer.position
    if state.attention_mask is not None:
        tensors["attention_mask"] = state.attention_mask
    if state.position is not None:
        tensors["position"] = state.position
    return tensors


def test_decode_checkpoint_restores_every_tensor_exactly() -> None:
    torch.manual_seed(20260729)
    kda = KDALayerState(
        q_conv=torch.randn(1, 8, 4),
        k_conv=torch.randn(1, 8, 4),
        v_conv=torch.randn(1, 8, 4),
        recurrent=torch.randn(1, 2, 4, 4),
        is_static=True,
    )
    mla = MLALayerState(
        compressed_kv=torch.randn(1, 8, 3),
        rotary_key=torch.randn(1, 8, 2),
        key_pass=torch.randn(1, 2, 8, 3),
        value=torch.randn(1, 2, 8, 4),
        position=torch.tensor(2, dtype=torch.long),
    )
    state = KLinearDecodeState(
        (kda, None, mla),
        tokens_seen=2,
        attention_mask=torch.tensor([[1, 1, 0, 0, 0, 0, 0, 0]]),
        position=torch.tensor(2, dtype=torch.long),
    )
    checkpoint = DecodeCheckpoint(state, max_speculative_tokens=3)
    checkpoint.snapshot(state)

    original = {name: tensor.clone() for name, tensor in _state_tensors(state).items()}
    addresses = {name: tensor.data_ptr() for name, tensor in _state_tensors(state).items()}

    kda.q_conv.add_(11)
    kda.k_conv.mul_(3)
    kda.v_conv.zero_()
    kda.recurrent.add_(torch.randn_like(kda.recurrent))
    mla.compressed_kv[:, 2:5].add_(7)
    mla.rotary_key[:, 2:5].zero_()
    mla.key_pass[:, :, 2:5].mul_(-2)
    mla.value[:, :, 2:5].add_(9)
    mla.position.add_(3)
    state.attention_mask[:, 2:5].fill_(1)
    state.position.add_(3)

    restored = checkpoint.restore(state.with_tokens_seen(5))

    assert restored.tokens_seen == 2
    for name, tensor in _state_tensors(restored).items():
        assert tensor.data_ptr() == addresses[name]
        assert torch.equal(tensor, original[name]), name


def test_prompt_lookup_uses_most_recent_match_and_can_return_fewer_than_k() -> None:
    assert propose([1, 2, 8, 1, 2, 7, 1, 2], k=3, ngram=2) == [7, 1, 2]
    assert propose([1, 2, 3, 1, 2], k=8, ngram=2) == [3, 1, 2]
    assert propose([1, 2, 3], k=4, ngram=2) == []


def test_greedy_verifier_accepts_prefix_and_replaces_first_mismatch() -> None:
    draft = torch.tensor([[3, 9, 0, 0]])
    predictions = torch.tensor([[3, 4, 5, 1]])

    result = verify_greedy(draft, _one_hot_logits(predictions, vocab_size=10))

    assert torch.equal(result.model_ids, predictions)
    assert torch.equal(result.accepted_prefix_length, torch.tensor([1]))
    assert torch.equal(result.emitted_count, torch.tensor([2]))
    assert torch.equal(result.emitted_ids, torch.tensor([[3, 4, 0, 0]]))
    assert torch.equal(result.emitted_mask, torch.tensor([[True, True, False, False]]))


def test_alignment_uses_pre_round_logits_for_the_first_draft_token() -> None:
    next_logits = _one_hot_logits(torch.tensor([2]), vocab_size=7)
    forward_logits = _one_hot_logits(torch.tensor([[3, 4, 5]]), vocab_size=7)
    draft = torch.tensor([[2, 3, 4]])

    aligned = align_verification_logits(next_logits, forward_logits)
    result = verify_greedy(draft, aligned)

    assert torch.equal(result.model_ids, draft)
    assert torch.equal(result.accepted_prefix_length, torch.tensor([3]))
    assert torch.equal(result.emitted_ids, draft)
    assert torch.equal(result.emitted_count, torch.tensor([3]))


class _TinyPositionedCycle(nn.Module):
    """Follow a cycle except at positions that force a draft rejection."""

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        predictions = (input_ids + 1).remainder(4)
        positions = torch.arange(input_ids.shape[1]).view(1, -1)
        predictions = torch.where(
            positions.remainder(9) == 8,
            torch.full_like(predictions, 5),
            predictions,
        )
        return _one_hot_logits(predictions, vocab_size=6)


def test_reference_speculative_decode_matches_32_token_ordinary_greedy() -> None:
    model = _TinyPositionedCycle().eval()
    prompt = torch.tensor([[0, 1, 2, 3, 0, 1, 2, 3]], dtype=torch.long)

    ordinary = ordinary_greedy(model, prompt, max_new_tokens=32)
    speculative = speculative_greedy(
        model,
        prompt,
        max_new_tokens=32,
        k=4,
        ngram=2,
    )

    assert propose(prompt[0].tolist(), k=4, ngram=2) == [0, 1, 2, 3]
    first_round_logits = model(torch.cat((prompt, torch.tensor([[0, 1, 2, 3]])), dim=1))
    first_round_predictions = first_round_logits[:, 7:11].argmax(dim=-1)
    assert torch.equal(first_round_predictions[:, :2], torch.tensor([[0, 5]]))
    assert ordinary.shape == (1, 32)
    assert torch.equal(speculative, ordinary)
