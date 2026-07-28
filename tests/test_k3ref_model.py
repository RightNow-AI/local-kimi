import copy

import torch

from engine.k3ref.config import K3LayerConfig
from engine.k3ref.layer import K3ReferenceLayer
from engine.k3ref.model import K3DenseReferenceLayer, K3Model
from engine.k3ref.state import KDALayerState, MLALayerState


def _tiny_config(*, full_attention_layers=()) -> K3LayerConfig:
    return K3LayerConfig(
        hidden_size=2,
        num_attention_heads=1,
        num_key_value_heads=1,
        q_lora_rank=2,
        kv_lora_rank=2,
        qk_nope_head_dim=1,
        qk_rope_head_dim=1,
        v_head_dim=2,
        kda_head_dim=2,
        kda_num_heads=1,
        short_conv_kernel_size=2,
        routed_expert_hidden_size=2,
        moe_intermediate_size=2,
        num_experts=1,
        num_experts_per_token=1,
        num_shared_experts=0,
        attn_res_block_size=None,
        full_attention_layers=tuple(full_attention_layers),
    )


def _deterministic_layer(config: K3LayerConfig, layer_idx: int, value: float):
    layer = K3ReferenceLayer(
        config,
        layer_idx,
        dtype=torch.float32,
    ).eval()
    with torch.no_grad():
        for parameter in layer.parameters():
            parameter.fill_(value)
        if layer.is_kda:
            layer.self_attn.A_log.zero_()
            layer.self_attn.dt_bias.zero_()
            layer.self_attn.q_proj.weight.copy_(
                value * torch.tensor([[2.0, 1.0], [-1.0, 3.0]])
            )
            layer.self_attn.k_proj.weight.copy_(
                value * torch.tensor([[1.0, -2.0], [3.0, 1.0]])
            )
            layer.self_attn.v_proj.weight.copy_(
                value * torch.tensor([[3.0, 1.0], [-2.0, 2.0]])
            )
            layer.self_attn.o_proj.weight.copy_(
                value * torch.tensor([[2.0, -1.0], [1.0, 3.0]])
            )
            layer.self_attn.g_proj.weight.copy_(
                value * torch.tensor([[1.0, 2.0], [-2.0, 1.0]])
            )
            layer.self_attn.o_norm.weight.fill_(1.0)
    return layer


def test_two_layer_stack_threads_hidden_state_and_differs_from_one_layer():
    config = _tiny_config()
    first = _deterministic_layer(config, 1, 0.15)
    second = _deterministic_layer(config, 2, 0.25)
    one_layer = K3Model(
        config,
        [1],
        layers=[copy.deepcopy(first)],
    ).eval()
    two_layers = K3Model(config, [1, 2], layers=[first, second]).eval()
    hidden = torch.tensor([[[0.4, -0.2], [0.1, 0.3]]])

    one = one_layer(hidden_states=hidden)
    two = two_layers(hidden_states=hidden)

    assert torch.allclose(two.layer_hidden_states[0], one.hidden_states)
    assert not torch.allclose(two.hidden_states, one.hidden_states)


def test_kda_recurrent_state_is_carried_into_the_next_decode_token():
    config = _tiny_config()
    layer = _deterministic_layer(config, 1, 0.2)
    model = K3Model(config, [1], layers=[layer]).eval()

    first = model(hidden_states=torch.tensor([[[0.6, -0.1]]]))
    carried = model(
        hidden_states=torch.tensor([[[0.2, 0.5]]]),
        state=first.state,
    )
    reset = model(hidden_states=torch.tensor([[[0.2, 0.5]]]))

    first_cache = first.state.layer_states[0]
    carried_cache = carried.state.layer_states[0]
    assert isinstance(first_cache, KDALayerState)
    assert isinstance(carried_cache, KDALayerState)
    assert torch.count_nonzero(first_cache.recurrent) > 0
    assert not torch.allclose(carried.hidden_states, reset.hidden_states)
    assert not torch.allclose(carried_cache.recurrent, first_cache.recurrent)


def test_layer_indices_select_kda_and_mla_and_keep_distinct_cache_types():
    config = _tiny_config(full_attention_layers=(4,))
    kda = _deterministic_layer(config, 2, 0.1)
    mla = _deterministic_layer(config, 3, 0.1)
    model = K3Model(config, [2, 3], layers=[kda, mla]).eval()

    output = model(hidden_states=torch.tensor([[[0.3, -0.2], [0.4, 0.1]]]))

    assert model.layers[0].is_kda is True
    assert model.layers[1].is_kda is False
    assert isinstance(output.state.layer_states[0], KDALayerState)
    assert isinstance(output.state.layer_states[1], MLALayerState)
    mla_state = output.state.layer_states[1]
    assert mla_state.compressed_kv.shape == (1, 2, config.kv_lora_rank)
    assert mla_state.rotary_key.shape == (1, 2, config.qk_rope_head_dim)


def test_layer_zero_uses_the_dense_situ_mlp_path():
    config = _tiny_config()
    layer = K3DenseReferenceLayer(
        config,
        intermediate_size=3,
        dtype=torch.float32,
    ).eval()
    with torch.no_grad():
        for parameter in layer.parameters():
            parameter.fill_(0.1)
        layer.self_attn.A_log.zero_()
        layer.self_attn.dt_bias.zero_()
    model = K3Model(config, [0], layers=[layer]).eval()

    output = model(hidden_states=torch.tensor([[[0.2, -0.3]]]))

    assert output.hidden_states.shape == (1, 1, config.hidden_size)
    assert output.router_indices[0].shape == (1, 0)
