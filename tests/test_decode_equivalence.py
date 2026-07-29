from __future__ import annotations

import torch

from engine.klinear.attention import MLAAttention
from engine.klinear.config import KLinearConfig
from engine.klinear.generate import decode, generate, prefill
from engine.klinear.model import KLinearModel
from engine.klinear.moe import shape_stable_expert_indices
from engine.klinear.state import KDALayerState, KLinearDecodeState, MLALayerState


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


def _growing_greedy_ids(
    model: KLinearModel,
    prompt: torch.Tensor,
    max_new_tokens: int,
) -> torch.Tensor:
    output = prefill(model, prompt)
    generated = []
    for _ in range(max_new_tokens):
        token = output.logits[:, -1].argmax(dim=-1)
        generated.append(token)
        output = decode(model, token.unsqueeze(1), output.state)
    return torch.stack(generated, dim=1)


def test_greedy_decode_tokens_match_the_growing_reference_exactly() -> None:
    torch.manual_seed(20260729)
    model = KLinearModel(_tiny_config(), dtype=torch.float32).eval()
    prompt = torch.tensor([[1, 4, 7]], dtype=torch.long)

    expected = _growing_greedy_ids(model, prompt, max_new_tokens=4)
    actual = generate(model, prompt, max_new_tokens=4, temperature=0.0)

    assert torch.equal(actual.generated_ids, expected)
    assert torch.equal(actual.token_ids, torch.cat((prompt, expected), dim=1))


def test_shape_stable_expert_gather_selects_the_naive_experts() -> None:
    expert_indices = torch.tensor([[4, 1, 7], [2, 6, 0]], dtype=torch.long)
    shared_expert_id = 8
    stable = shape_stable_expert_indices(expert_indices, shared_expert_id)
    expert_bank = torch.arange(9 * 5).view(9, 5)

    naive = torch.stack(
        [
            torch.stack([expert_bank[int(index)] for index in row] + [expert_bank[8]])
            for row in expert_indices
        ]
    )
    gathered = expert_bank.index_select(0, stable.reshape(-1)).view(2, 4, 5)

    assert torch.equal(stable[:, :-1], expert_indices)
    assert torch.equal(stable[:, -1], torch.full((2,), shared_expert_id))
    assert torch.equal(gathered, naive)


def test_preallocated_mla_state_matches_growing_state_after_n_steps() -> None:
    torch.manual_seed(91)
    attention = MLAAttention(
        hidden_size=4,
        num_heads=2,
        num_key_value_heads=2,
        kv_lora_rank=2,
        qk_nope_head_dim=1,
        qk_rope_head_dim=1,
        v_head_dim=2,
        dtype=torch.float32,
    ).eval()
    prefix = torch.randn(1, 3, 4)
    decode_inputs = torch.randn(1, 4, 4)
    _, prefix_state = attention(prefix, return_state=True)

    growing = prefix_state
    growing_outputs = []
    for index in range(decode_inputs.shape[1]):
        output, growing = attention(
            decode_inputs[:, index : index + 1],
            state=growing,
            return_state=True,
        )
        growing_outputs.append(output)

    reserved = KLinearDecodeState(
        (prefix_state,),
        tokens_seen=prefix.shape[1],
    ).reserve_decode_capacity(decode_inputs.shape[1])
    static = reserved.layer_states[0]
    assert isinstance(static, MLALayerState)
    storage_addresses = (
        static.compressed_kv.data_ptr(),
        static.rotary_key.data_ptr(),
        static.key_pass.data_ptr(),
        static.value.data_ptr(),
    )
    static_outputs = []
    for index in range(decode_inputs.shape[1]):
        output, static = attention(
            decode_inputs[:, index : index + 1],
            state=static,
            return_state=True,
        )
        static_outputs.append(output)

    length = prefix.shape[1] + decode_inputs.shape[1]
    assert storage_addresses == (
        static.compressed_kv.data_ptr(),
        static.rotary_key.data_ptr(),
        static.key_pass.data_ptr(),
        static.value.data_ptr(),
    )
    torch.testing.assert_close(torch.cat(static_outputs, dim=1), torch.cat(growing_outputs, dim=1))
    torch.testing.assert_close(static.compressed_kv[:, :length], growing.compressed_kv)
    torch.testing.assert_close(static.rotary_key[:, :length], growing.rotary_key)
    torch.testing.assert_close(static.key_pass[:, :, :length], growing.key_pass)
    torch.testing.assert_close(static.value[:, :, :length], growing.value)


def test_preallocated_full_model_state_matches_growing_state() -> None:
    torch.manual_seed(808)
    model = KLinearModel(_tiny_config(), dtype=torch.float32).eval()
    prompt = torch.tensor([[1, 6, 4]], dtype=torch.long)
    decode_ids = torch.tensor([[3, 8, 5]], dtype=torch.long)

    growing_output = prefill(model, prompt)
    for index in range(decode_ids.shape[1]):
        growing_output = decode(
            model,
            decode_ids[:, index : index + 1],
            growing_output.state,
        )

    static_prefill = prefill(model, prompt)
    static_state = static_prefill.state.reserve_decode_capacity(decode_ids.shape[1])
    static_output = static_prefill
    for index in range(decode_ids.shape[1]):
        static_output = decode(
            model,
            decode_ids[:, index : index + 1],
            static_state,
        )
        static_state = static_output.state

    assert static_state.tokens_seen == growing_output.state.tokens_seen
    for static_layer, growing_layer in zip(
        static_state.layer_states,
        growing_output.state.layer_states,
        strict=True,
    ):
        if isinstance(static_layer, KDALayerState):
            assert isinstance(growing_layer, KDALayerState)
            torch.testing.assert_close(static_layer.q_conv, growing_layer.q_conv)
            torch.testing.assert_close(static_layer.k_conv, growing_layer.k_conv)
            torch.testing.assert_close(static_layer.v_conv, growing_layer.v_conv)
            torch.testing.assert_close(static_layer.recurrent, growing_layer.recurrent)
        else:
            assert isinstance(static_layer, MLALayerState)
            assert isinstance(growing_layer, MLALayerState)
            length = growing_layer.sequence_length
            torch.testing.assert_close(
                static_layer.compressed_kv[:, :length],
                growing_layer.compressed_kv,
            )
            torch.testing.assert_close(
                static_layer.rotary_key[:, :length], growing_layer.rotary_key
            )
            torch.testing.assert_close(
                static_layer.key_pass[:, :, :length], growing_layer.key_pass
            )
            torch.testing.assert_close(
                static_layer.value[:, :, :length], growing_layer.value
            )
