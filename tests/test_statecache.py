from __future__ import annotations

from unittest.mock import patch

import torch

from engine.klinear.config import KLinearConfig
from engine.klinear.generate import decode, prefill
from engine.klinear.model import KLinearModel
from engine.klinear.state import KDALayerState, KLinearDecodeState, MLALayerState
from engine.statecache.key import prefix_key
from engine.statecache.session import warm_prefill
from engine.statecache.store import StateCache


def _tiny_config() -> KLinearConfig:
    return KLinearConfig(
        vocab_size=17,
        hidden_size=4,
        head_dim=2,
        intermediate_size=8,
        num_hidden_layers=3,
        num_attention_heads=2,
        num_key_value_heads=2,
        kv_lora_rank=2,
        qk_nope_head_dim=1,
        qk_rope_head_dim=1,
        v_head_dim=2,
        kda_num_heads=2,
        kda_head_dim=2,
        short_conv_kernel_size=2,
        num_experts=4,
        num_experts_per_token=2,
        num_shared_experts=1,
        moe_intermediate_size=3,
        first_k_dense_replace=1,
        moe_renormalize=True,
        routed_scaling_factor=2.446,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
        full_attention_layers=(3,),
        kda_layers=(1, 2),
    )


def _model() -> KLinearModel:
    torch.manual_seed(20260729)
    return KLinearModel(_tiny_config(), dtype=torch.float32).eval()


def _greedy_from_prefill(
    model: KLinearModel,
    prompt: torch.Tensor,
    token_count: int,
) -> torch.Tensor:
    output = prefill(model, prompt)
    state = output.state.reserve_decode_capacity(token_count)
    next_token = output.logits[:, -1].argmax(dim=-1)
    generated = []
    for _ in range(token_count):
        generated.append(next_token)
        output = decode(model, next_token.unsqueeze(1), state)
        state = output.state
        next_token = output.logits[:, -1].argmax(dim=-1)
    return torch.stack(generated, dim=1)


def _greedy_from_state(
    model: KLinearModel,
    state: KLinearDecodeState,
    first_token: torch.Tensor,
    token_count: int,
) -> torch.Tensor:
    next_token = first_token
    generated = []
    for _ in range(token_count):
        generated.append(next_token)
        output = decode(model, next_token.unsqueeze(1), state)
        state = output.state
        next_token = output.logits[:, -1].argmax(dim=-1)
    return torch.stack(generated, dim=1)


def _cached_greedy_request(
    model: KLinearModel,
    prompt: torch.Tensor,
    token_count: int,
    cache: StateCache,
    fingerprint: dict[str, str],
) -> torch.Tensor:
    state, _ = warm_prefill(
        model,
        prompt[:, :-1],
        cache,
        decode_capacity=token_count + 1,
        model_fingerprint=fingerprint,
    )
    output = prefill(model, prompt[:, -1:], state=state)
    first_token = output.logits[:, -1].argmax(dim=-1)
    return _greedy_from_state(model, output.state, first_token, token_count)


def _storage_pointers(state: KLinearDecodeState) -> tuple[int, ...]:
    pointers = []
    for layer in state.layer_states:
        if isinstance(layer, KDALayerState):
            pointers.extend(
                (
                    layer.q_conv.data_ptr(),
                    layer.k_conv.data_ptr(),
                    layer.v_conv.data_ptr(),
                    layer.recurrent.data_ptr(),
                )
            )
        elif isinstance(layer, MLALayerState):
            pointers.extend(
                (
                    layer.compressed_kv.data_ptr(),
                    layer.rotary_key.data_ptr(),
                    layer.key_pass.data_ptr(),
                    layer.value.data_ptr(),
                    layer.position.data_ptr(),
                )
            )
    pointers.append(state.position.data_ptr())
    return tuple(pointers)


def _compressed_snapshot_bytes(state: KLinearDecodeState) -> int:
    total = state.position.element_size()
    for layer in state.layer_states:
        if isinstance(layer, KDALayerState):
            total += sum(
                tensor.numel() * tensor.element_size()
                for tensor in (
                    layer.q_conv,
                    layer.k_conv,
                    layer.v_conv,
                    layer.recurrent,
                )
            )
        elif isinstance(layer, MLALayerState):
            count = state.tokens_seen
            total += (
                layer.compressed_kv[:, :count].numel()
                * layer.compressed_kv.element_size()
            )
            total += (
                layer.rotary_key[:, :count].numel()
                * layer.rotary_key.element_size()
            )
            total += layer.position.element_size()
    if state.attention_mask is not None:
        total += (
            state.attention_mask[:, : state.tokens_seen].numel()
            * state.attention_mask.element_size()
        )
    return total


def _assert_live_state_equal(
    actual: KLinearDecodeState,
    expected: KLinearDecodeState,
) -> None:
    assert actual.tokens_seen == expected.tokens_seen
    for current, reference in zip(
        actual.layer_states,
        expected.layer_states,
        strict=True,
    ):
        if isinstance(reference, KDALayerState):
            assert isinstance(current, KDALayerState)
            torch.testing.assert_close(current.q_conv, reference.q_conv)
            torch.testing.assert_close(current.k_conv, reference.k_conv)
            torch.testing.assert_close(current.v_conv, reference.v_conv)
            torch.testing.assert_close(current.recurrent, reference.recurrent)
        else:
            assert isinstance(reference, MLALayerState)
            assert isinstance(current, MLALayerState)
            count = expected.tokens_seen
            torch.testing.assert_close(
                current.compressed_kv[:, :count], reference.compressed_kv
            )
            torch.testing.assert_close(
                current.rotary_key[:, :count], reference.rotary_key
            )
            torch.testing.assert_close(
                current.key_pass[:, :, :count], reference.key_pass
            )
            torch.testing.assert_close(
                current.value[:, :, :count], reference.value
            )


def test_snapshot_mutate_restore_repeats_identical_generated_ids() -> None:
    model = _model()
    prompt = torch.tensor([[1, 4, 7]], dtype=torch.long)
    generated_count = 4
    expected = _greedy_from_prefill(model, prompt, generated_count)

    output = prefill(model, prompt)
    first_token = output.logits[:, -1].argmax(dim=-1)
    state = output.state.reserve_decode_capacity(generated_count)
    key = prefix_key(prompt, {"model": "tiny", "weights": "seed-20260729"})
    cache = StateCache(byte_budget=16 * 1024 * 1024)
    assert cache.save(key, state, state.tokens_seen)
    assert cache.entry_bytes(key) == _compressed_snapshot_bytes(state)
    pointers = _storage_pointers(state)

    mutation = decode(model, first_token.unsqueeze(1), state)
    mutation_token = mutation.logits[:, -1].argmax(dim=-1)
    mutation = decode(model, mutation_token.unsqueeze(1), mutation.state)
    state = mutation.state
    assert state.tokens_seen == prompt.shape[1] + 2
    with torch.inference_mode():
        for layer in state.layer_states:
            if isinstance(layer, MLALayerState):
                layer.key_pass[:, :, : prompt.shape[1]].zero_()
                layer.value[:, :, : prompt.shape[1]].zero_()

    assert cache.load(key, state, model=model)
    assert state.tokens_seen == prompt.shape[1]
    assert _storage_pointers(state) == pointers
    restored = _greedy_from_state(model, state, first_token, generated_count)

    assert torch.equal(restored, expected)


def test_two_fresh_restores_after_decode_match_uncached_generation() -> None:
    model = _model()
    prompt = torch.tensor([[3, 1, 4, 1, 5, 9, 2]], dtype=torch.long)
    generated_count = 8
    expected = _greedy_from_prefill(model, prompt, generated_count)
    cache = StateCache(byte_budget=16 * 1024 * 1024)
    fingerprint = {"model": "tiny", "weights": "seed-20260729"}

    results = [
        _cached_greedy_request(
            model,
            prompt,
            generated_count,
            cache,
            fingerprint,
        )
        for _ in range(3)
    ]

    assert all(torch.equal(result, expected) for result in results)


def test_different_model_fingerprints_cannot_share_a_snapshot() -> None:
    model = _model()
    prompt = torch.tensor([[1, 3, 6]], dtype=torch.long)
    fingerprint_a = {
        "checkpoint_directory_name": "checkpoint-a",
        "resident_weight_bytes": 100,
        "kernel_variants": {"dense": "reference"},
    }
    fingerprint_b = {
        "checkpoint_directory_name": "checkpoint-b",
        "resident_weight_bytes": 100,
        "kernel_variants": {"dense": "reference"},
    }
    key_a = prefix_key(prompt, fingerprint_a)
    key_b = prefix_key(prompt, fingerprint_b)
    assert key_a != key_b

    state = prefill(model, prompt).state.reserve_decode_capacity(1)
    cache = StateCache(byte_budget=16 * 1024 * 1024)
    assert cache.save(key_a, state, state.tokens_seen)
    target = cache.allocate_state(
        key_a,
        model=model,
        device="cpu",
        additional_tokens=1,
    )

    assert not cache.load(key_b, target, model=model)
    assert cache.load(key_a, target, model=model)


def test_lru_eviction_respects_the_byte_budget() -> None:
    model = _model()
    prompt = torch.tensor([[1, 5, 9]], dtype=torch.long)
    state = prefill(model, prompt).state.reserve_decode_capacity(1)

    probe = StateCache(byte_budget=16 * 1024 * 1024)
    assert probe.save("probe", state, state.tokens_seen)
    snapshot_bytes = probe.entry_bytes("probe")
    assert snapshot_bytes is not None

    cache = StateCache(byte_budget=2 * snapshot_bytes)
    assert cache.save("a", state, state.tokens_seen)
    assert cache.save("b", state, state.tokens_seen)
    target = cache.allocate_state(
        "a",
        model=model,
        device="cpu",
        additional_tokens=1,
    )
    assert cache.load("a", target, model=model)
    assert cache.save("c", state, state.tokens_seen)

    assert "a" in cache
    assert "b" not in cache
    assert "c" in cache
    assert cache.host_bytes <= cache.byte_budget


def test_partial_extension_restores_prefix_and_prefills_only_the_suffix() -> None:
    model = _model()
    fingerprint = {"model": "tiny", "weights": "seed-20260729"}
    prefix = torch.tensor([[1, 4, 7]], dtype=torch.long)
    extended = torch.tensor([[1, 4, 7, 6, 3]], dtype=torch.long)
    cache = StateCache(byte_budget=16 * 1024 * 1024)

    _, hit = warm_prefill(
        model,
        prefix,
        cache,
        decode_capacity=1,
        model_fingerprint=fingerprint,
    )
    assert not hit

    from engine.statecache import session

    with patch.object(session, "prefill", wraps=prefill) as exact_prefill:
        exact_state, hit = warm_prefill(
            model,
            prefix,
            cache,
            decode_capacity=1,
            model_fingerprint=fingerprint,
        )
    assert hit
    exact_prefill.assert_not_called()
    assert exact_state.tokens_seen == prefix.shape[1]

    with patch.object(session, "prefill", wraps=prefill) as tracked_prefill:
        restored, hit = warm_prefill(
            model,
            extended,
            cache,
            decode_capacity=1,
            model_fingerprint=fingerprint,
        )

    assert hit
    assert [call.args[1].shape[1] for call in tracked_prefill.call_args_list] == [1, 1]

    expected = prefill(model, extended).state
    _assert_live_state_equal(restored, expected)

    continuation = torch.tensor([[8]], dtype=torch.long)
    restored_output = decode(model, continuation, restored)
    expected_output = decode(model, continuation, expected)
    torch.testing.assert_close(restored_output.logits, expected_output.logits)
