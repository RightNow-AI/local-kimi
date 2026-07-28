import ast
from pathlib import Path
from types import SimpleNamespace

import torch

from engine.k3ref.config import K3LayerConfig
from engine.k3ref.layer import K3LayerOutput, K3ReferenceLayer
from engine.k3ref.norm import apply_attention_residual


def _moonshot_apply_attention_residual():
    source_path = Path(__file__).parents[1] / "reference" / "modeling_kimi_linear.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_apply_attn_res"
    )
    module = ast.Module(body=[function], type_ignores=[])
    namespace = {"torch": torch}
    exec(compile(ast.fix_missing_locations(module), str(source_path), "exec"), namespace)
    return namespace["_apply_attn_res"]


def test_learned_block_residual_mixer_matches_moonshot_source():
    torch.manual_seed(31)
    prefix = torch.randn(6, 5)
    residuals = torch.randn(6, 3, 5)
    projection_weight = torch.randn(1, 5)
    norm_weight = torch.randn(5)
    eps = 1e-5
    projection = SimpleNamespace(weight=projection_weight)
    norm = SimpleNamespace(weight=norm_weight, variance_epsilon=eps)

    expected = _moonshot_apply_attention_residual()(
        prefix, residuals, projection, norm
    )
    actual = apply_attention_residual(
        prefix, residuals, projection_weight, norm_weight, eps
    )

    assert torch.equal(actual, expected)


def test_complete_small_kda_moe_layer_runs_from_synthetic_weights():
    config = K3LayerConfig(
        hidden_size=4,
        num_attention_heads=2,
        num_key_value_heads=2,
        q_lora_rank=2,
        kv_lora_rank=2,
        qk_nope_head_dim=1,
        qk_rope_head_dim=1,
        v_head_dim=2,
        kda_head_dim=2,
        kda_num_heads=2,
        short_conv_kernel_size=2,
        routed_expert_hidden_size=3,
        moe_intermediate_size=2,
        num_experts=2,
        num_experts_per_token=1,
        num_shared_experts=1,
        attn_res_block_size=2,
        full_attention_layers=(),
    )
    layer = K3ReferenceLayer(config, layer_idx=2, dtype=torch.float32).eval()
    with torch.no_grad():
        layer.self_attn.A_log.zero_()
        layer.self_attn.dt_bias.zero_()
    hidden = torch.randn(2, 3, 4)

    result = layer(
        hidden,
        attention_mask=torch.ones(2, 3, dtype=torch.long),
        return_aux=True,
    )

    assert isinstance(result, K3LayerOutput)
    assert result.hidden_states.shape == hidden.shape
    assert result.hidden_states.dtype == hidden.dtype
    assert result.router_indices.shape == (6, 1)
    assert result.router_weights.shape == (6, 1)
    assert result.attention_state.recurrent.shape == (2, 2, 2, 2)
    assert result.block_residual.shape == (6, 1, 4)
