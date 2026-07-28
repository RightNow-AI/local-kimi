import torch

from engine.k3ref.config import K3LayerConfig
from engine.k3ref.embed import K3EmbeddingHead
from engine.k3ref.generate import generate, prefill, sample_logits
from engine.k3ref.layer import K3ReferenceLayer
from engine.k3ref.model import K3Model
from engine.k3ref.state import KDALayerState, MLALayerState


def _tiny_hybrid_model() -> K3Model:
    config = K3LayerConfig(
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
        full_attention_layers=(4,),
    )
    layers = [
        K3ReferenceLayer(config, 2, dtype=torch.float32).eval(),
        K3ReferenceLayer(config, 3, dtype=torch.float32).eval(),
    ]
    with torch.no_grad():
        for layer_index, layer in enumerate(layers, start=1):
            for parameter in layer.parameters():
                parameter.fill_(0.08 * layer_index)
            if layer.is_kda:
                value = 0.08 * layer_index
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
    embedding = torch.tensor(
        [
            [0.10, -0.20],
            [0.30, 0.05],
            [-0.15, 0.25],
            [0.20, 0.10],
            [-0.05, -0.30],
        ]
    )
    head = torch.tensor(
        [
            [0.20, -0.10],
            [-0.05, 0.30],
            [0.25, 0.15],
            [-0.20, 0.05],
            [0.10, 0.20],
        ]
    )
    embedding_head = K3EmbeddingHead(embedding, head)
    return K3Model(
        config,
        [2, 3],
        layers=layers,
        embedding_head=embedding_head,
    ).eval()


def test_greedy_sampling_is_deterministic_and_temperature_zero_is_greedy():
    logits = torch.tensor([[0.5, 2.0, 1.0], [3.0, -1.0, 0.0]])

    first = sample_logits(logits)
    second = sample_logits(logits)
    zero_temperature = sample_logits(logits, temperature=0.0, top_p=0.2)

    assert torch.equal(first, torch.tensor([1, 0]))
    assert torch.equal(second, first)
    assert torch.equal(zero_temperature, first)


def test_temperature_top_p_sampling_excludes_tokens_outside_the_nucleus():
    logits = torch.tensor([[10.0, 9.0, -10.0]])
    generator = torch.Generator().manual_seed(7)

    sampled = sample_logits(
        logits,
        temperature=1.0,
        top_p=0.6,
        generator=generator,
    )

    assert torch.equal(sampled, torch.tensor([0]))


def test_incremental_generation_matches_one_prefill_of_the_same_tokens():
    model = _tiny_hybrid_model()
    prompt = torch.tensor([[0, 1]])

    incremental = generate(model, prompt, max_new_tokens=3)
    single_prefill = prefill(model, incremental.token_ids)

    assert incremental.state.tokens_seen == incremental.token_ids.shape[1]
    assert single_prefill.state.tokens_seen == incremental.token_ids.shape[1]
    assert torch.allclose(
        incremental.final_logits[:, -1],
        single_prefill.logits[:, -1],
        atol=1e-5,
        rtol=1e-5,
    )
    incremental_kda, incremental_mla = incremental.state.layer_states
    prefill_kda, prefill_mla = single_prefill.state.layer_states
    assert isinstance(incremental_kda, KDALayerState)
    assert isinstance(prefill_kda, KDALayerState)
    assert isinstance(incremental_mla, MLALayerState)
    assert isinstance(prefill_mla, MLALayerState)
    assert torch.allclose(
        incremental_kda.recurrent,
        prefill_kda.recurrent,
        atol=1e-5,
        rtol=1e-5,
    )
    assert torch.allclose(
        incremental_mla.compressed_kv,
        prefill_mla.compressed_kv,
        atol=1e-5,
        rtol=1e-5,
    )
    assert torch.allclose(
        incremental_mla.rotary_key,
        prefill_mla.rotary_key,
        atol=1e-5,
        rtol=1e-5,
    )
