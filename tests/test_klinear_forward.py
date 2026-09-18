import torch
import torch.nn.functional as F

from engine.klinear.config import KLinearConfig
from engine.klinear.generate import generate
from engine.klinear.model import KLinearModel
from engine.klinear.moe import ExpertMLP
from engine.klinear.router import KLinearRouter
from engine.klinear.state import KDALayerState, MLALayerState


def _tiny_config() -> KLinearConfig:
    return KLinearConfig(
        vocab_size=13,
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


def test_router_renormalizes_selected_sigmoid_scores_then_applies_scaling_factor():
    router = KLinearRouter(
        hidden_size=2,
        num_experts=4,
        top_k=2,
        renormalize=True,
        routed_scaling_factor=2.446,
        use_grouped_topk=True,
        num_expert_group=1,
        topk_group=1,
        dtype=torch.float32,
    ).eval()
    with torch.no_grad():
        router.weight.copy_(
            torch.tensor(
                [
                    [3.0, 0.0],
                    [2.0, 0.0],
                    [1.0, 0.0],
                    [0.0, 0.0],
                ]
            )
        )
        router.e_score_correction_bias.zero_()

    indices, weights = router(torch.tensor([[[1.0, 0.0]]]))
    raw_scores = torch.sigmoid(torch.tensor([3.0, 2.0, 1.0, 0.0]))
    gathered = raw_scores[indices[0]]
    expected = gathered / gathered.sum() * 2.446

    assert set(indices[0].tolist()) == {0, 1}
    assert torch.allclose(weights[0], expected)
    assert torch.allclose(weights.sum(dim=-1), torch.tensor([2.446]))


def test_routed_expert_is_plain_swiglu_without_latent_projections():
    expert = ExpertMLP(2, 2, dtype=torch.float32).eval()
    with torch.no_grad():
        expert.w1.weight.copy_(torch.eye(2))
        expert.w2.weight.copy_(torch.eye(2))
        expert.w3.weight.copy_(torch.eye(2))
    hidden_states = torch.tensor([[1.0, -0.5]])

    output = expert(hidden_states)

    assert torch.allclose(output, F.silu(hidden_states) * hidden_states)


def test_tiny_model_runs_dense_kda_mla_moe_lm_head_and_generate():
    torch.manual_seed(20260728)
    config = _tiny_config()
    model = KLinearModel(config, dtype=torch.float32).eval()
    prompt = torch.tensor([[1, 4, 7]], dtype=torch.long)

    output = model(prompt)

    assert output.hidden_states.shape == (1, 3, config.hidden_size)
    assert output.logits.shape == (1, 3, config.vocab_size)
    assert len(output.layer_hidden_states) == 3
    assert isinstance(output.state.layer_states[0], KDALayerState)
    assert isinstance(output.state.layer_states[1], KDALayerState)
    assert isinstance(output.state.layer_states[2], MLALayerState)
    assert output.router_indices[0].shape == (3, 0)
    assert output.router_indices[1].shape == (3, 2)
    assert output.router_indices[2].shape == (3, 2)
    assert torch.allclose(
        output.router_weights[1].sum(dim=-1),
        torch.full((3,), config.routed_scaling_factor),
    )
    assert torch.allclose(
        output.router_weights[2].sum(dim=-1),
        torch.full((3,), config.routed_scaling_factor),
    )

    generated = generate(model, prompt[:, :2], max_new_tokens=2)

    assert generated.token_ids.dtype == torch.long
    assert generated.token_ids.shape == (1, 4)
    assert generated.generated_ids.shape == (1, 2)
    assert generated.state.tokens_seen == 4
    assert generated.final_logits.shape == (1, 1, config.vocab_size)
