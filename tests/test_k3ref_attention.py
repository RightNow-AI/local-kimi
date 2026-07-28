import ast
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from engine.k3ref.attention import DepthwiseShortConv, KDAAttention, MLAAttention


def _moonshot_mla_types():
    source_path = Path(__file__).parents[1] / "reference" / "modeling_kimi_linear.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    wanted = {
        "KimiRMSNorm",
        "repeat_kv",
        "eager_attention_forward",
        "KimiMLAAttention",
    }
    nodes = [
        node
        for node in tree.body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in wanted
    ]
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__",
                names=[ast.alias(name="annotations")],
                level=0,
            ),
            *nodes,
        ],
        type_ignores=[],
    )
    namespace = {
        "ALL_ATTENTION_FUNCTIONS": {},
        "F": F,
        "nn": nn,
        "torch": torch,
    }
    exec(compile(ast.fix_missing_locations(module), str(source_path), "exec"), namespace)
    return namespace["KimiMLAAttention"]


class _MLAConfig:
    hidden_size = 8
    num_attention_heads = 2
    num_key_value_heads = 2
    attention_dropout = 0.0
    q_lora_rank = 4
    qk_rope_head_dim = 2
    kv_lora_rank = 3
    v_head_dim = 2
    qk_nope_head_dim = 2
    mla_use_nope = True
    mla_use_output_gate = True
    _attn_implementation = "eager"


def test_mla_matches_moonshot_eager_path_exactly():
    torch.manual_seed(23)
    reference = _moonshot_mla_types()(_MLAConfig(), layer_idx=3).eval()
    actual = MLAAttention(
        hidden_size=8,
        num_heads=2,
        num_key_value_heads=2,
        q_lora_rank=4,
        kv_lora_rank=3,
        qk_nope_head_dim=2,
        qk_rope_head_dim=2,
        v_head_dim=2,
        use_output_gate=True,
        rms_norm_eps=1e-6,
    ).eval()
    actual.load_state_dict(reference.state_dict())
    hidden = torch.randn(2, 4, 8)
    mask = torch.zeros(2, 1, 4, 4)
    mask.masked_fill_(
        torch.triu(torch.ones(4, 4, dtype=torch.bool), diagonal=1),
        torch.finfo(torch.float32).min,
    )

    expected = reference(hidden, attention_mask=mask)
    output = actual(hidden, attention_mask=mask)

    assert torch.equal(output, expected)


def test_short_convolution_is_causal_depthwise_and_uses_oldest_to_current_order():
    convolution = DepthwiseShortConv(hidden_size=1, kernel_size=3)
    with torch.no_grad():
        convolution.weight.copy_(torch.tensor([[[1.0, 10.0, 100.0]]]))
    values = torch.tensor([[[1.0], [2.0], [3.0]]])

    output, state = convolution(values)
    expected_pre_activation = torch.tensor([[[100.0], [210.0], [321.0]]])

    assert torch.allclose(output, F.silu(expected_pre_activation))
    assert torch.equal(state, torch.tensor([[[1.0, 2.0, 3.0]]]))


def test_kda_scalar_recurrence_matches_the_delta_rule_analytically():
    attention = KDAAttention(
        hidden_size=1,
        num_heads=1,
        head_dim=1,
        conv_size=1,
        gate_lower_bound=-5.0,
        rms_norm_eps=1e-5,
    ).eval()
    with torch.no_grad():
        for projection in (
            attention.q_proj,
            attention.k_proj,
            attention.v_proj,
            attention.o_proj,
        ):
            projection.weight.fill_(1.0)
        for convolution in (
            attention.q_conv1d,
            attention.k_conv1d,
            attention.v_conv1d,
        ):
            convolution.weight.fill_(1.0)
        attention.f_a_proj.weight.zero_()
        attention.f_b_proj.weight.zero_()
        attention.b_proj.weight.zero_()
        attention.g_proj.weight.zero_()
        attention.A_log.zero_()
        attention.dt_bias.zero_()
        attention.o_norm.weight.fill_(1.0)

    hidden = torch.tensor([[[1.0], [2.0]]])
    output = attention(hidden)

    values = F.silu(hidden.float()).flatten()
    decay = -2.5
    beta = 0.5
    recurrent = torch.tensor(0.0)
    expected = []
    for value in values:
        normalized = value / torch.sqrt(value.square() + 1e-6)
        recurrent = recurrent * torch.exp(torch.tensor(decay))
        recurrent = recurrent + beta * normalized * (value - normalized * recurrent)
        token_output = normalized * recurrent
        token_output = token_output / torch.sqrt(token_output.square() + 1e-5)
        expected.append(token_output * 0.5)
    expected = torch.stack(expected).view(1, 2, 1)

    assert torch.allclose(output, expected, atol=1e-6, rtol=1e-6)


def test_kda_a_log_is_per_head_dimension_and_shared_across_heads():
    attention = KDAAttention(
        hidden_size=6,
        num_heads=2,
        head_dim=3,
        gate_lower_bound=-5.0,
    )
    with torch.no_grad():
        # log(0) is -inf, which would make the first decay term degenerate and
        # disagrees with the expectation below, which is written for A = [1,2,4].
        attention.A_log.copy_(torch.tensor([1.0, 2.0, 4.0]).log())
        attention.dt_bias.zero_()
    raw_gate = torch.ones(1, 1, 2, 3)

    actual = attention._decay_gate(raw_gate)
    expected_per_dimension = -5.0 * torch.sigmoid(torch.tensor([1.0, 2.0, 4.0]))
    expected = expected_per_dimension.view(1, 1, 1, 3).expand(1, 1, 2, 3)

    assert torch.allclose(actual, expected, atol=1e-7, rtol=1e-7)


def test_moonshot_kda_call_pins_all_fused_recurrence_options():
    source = (
        Path(__file__).parents[1] / "reference" / "modeling_kimi_linear.py"
    ).read_text(encoding="utf-8")

    assert "use_qk_l2norm_in_kernel=True" in source
    assert "use_gate_in_kernel=True" in source
    assert "use_beta_sigmoid_in_kernel=True" in source
    assert "safe_gate=self.gate_lower_bound is not None" in source
    assert "lower_bound=self.gate_lower_bound" in source
    assert "transpose_state_layout=True" in source
