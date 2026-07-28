import copy
import ast
from pathlib import Path

import torch
from torch import nn

from engine.k3ref.config import K3LayerConfig
from engine.k3ref.moe import K3ExpertMLP, LatentMoE


def _situ(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    return torch.tanh(gate) * torch.sigmoid(gate) * up


def _moonshot_expert_types():
    source_path = Path(__file__).parents[1] / "reference" / "modeling_kimi_linear.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    wanted = {"SituAndMul", "_get_situ_activation_params", "KimiBlockSparseMLP"}
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
    namespace = {"ACT2FN": {}, "nn": nn, "torch": torch}
    exec(compile(ast.fix_missing_locations(module), str(source_path), "exec"), namespace)
    return namespace["KimiBlockSparseMLP"]


class _ExpertConfig:
    hidden_size = 3
    intermediate_size = 2
    hidden_act = "situ"
    activation_situ_beta = 4.0
    activation_situ_linear_beta = 25.0


def test_expert_situ_path_matches_moonshot_source_exactly():
    torch.manual_seed(5)
    reference = _moonshot_expert_types()(
        _ExpertConfig(), hidden_size=3, intermediate_size=2
    ).eval()
    actual = K3ExpertMLP(
        hidden_size=3,
        intermediate_size=2,
        situ_beta=4.0,
        situ_linear_beta=25.0,
    ).eval()
    actual.load_state_dict(reference.state_dict())
    hidden = torch.randn(2, 4, 3)

    assert torch.equal(actual(hidden), reference(hidden))


def test_two_expert_latent_moe_has_analytically_computable_output():
    moe = LatentMoE(
        hidden_size=2,
        latent_size=2,
        expert_intermediate_size=1,
        num_experts=2,
        top_k=2,
        num_shared_experts=1,
        rms_norm_eps=0.0,
        situ_beta=1.0,
        situ_linear_beta=None,
    ).eval()
    with torch.no_grad():
        moe.gate.weight.zero_()
        moe.gate.e_score_correction_bias.zero_()
        moe.routed_expert_down_proj.weight.copy_(torch.eye(2))
        moe.routed_expert_up_proj.weight.copy_(torch.eye(2))
        moe.routed_expert_norm.weight.fill_(1.0)

        first, second = moe.experts
        first.w1.weight.copy_(torch.tensor([[1.0, 0.0]]))
        first.w3.weight.copy_(torch.tensor([[0.0, 1.0]]))
        first.w2.weight.copy_(torch.tensor([[1.0], [0.0]]))
        second.w1.weight.copy_(torch.tensor([[0.0, 1.0]]))
        second.w3.weight.copy_(torch.tensor([[1.0, 0.0]]))
        second.w2.weight.copy_(torch.tensor([[0.0], [1.0]]))

        shared = moe.shared_experts
        shared.gate_proj.weight.copy_(torch.tensor([[1.0, 0.0]]))
        shared.up_proj.weight.copy_(torch.tensor([[0.0, 1.0]]))
        shared.down_proj.weight.copy_(torch.tensor([[1.0], [1.0]]))

    hidden = torch.tensor([[[1.0, 2.0]]])
    first_value = _situ(hidden[..., 0], hidden[..., 1])
    second_value = _situ(hidden[..., 1], hidden[..., 0])
    mixed = torch.stack((first_value * 0.5, second_value * 0.5), dim=-1)
    normalized = mixed / mixed.square().mean(dim=-1, keepdim=True).sqrt()
    shared_value = _situ(hidden[..., 0], hidden[..., 1]).unsqueeze(-1)
    expected = normalized + shared_value.expand_as(normalized)

    actual = moe(hidden)

    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)


def test_permuting_expert_order_and_router_rows_changes_nothing_observable():
    torch.manual_seed(17)
    original = LatentMoE(
        hidden_size=4,
        latent_size=3,
        expert_intermediate_size=2,
        num_experts=3,
        top_k=2,
        num_shared_experts=0,
        situ_beta=1.5,
        situ_linear_beta=3.0,
    ).eval()
    reordered = copy.deepcopy(original)
    permutation = torch.tensor([2, 0, 1])
    reordered.experts = nn.ModuleList(
        [copy.deepcopy(original.experts[index]) for index in permutation.tolist()]
    )
    with torch.no_grad():
        reordered.gate.weight.copy_(original.gate.weight[permutation])
        reordered.gate.e_score_correction_bias.copy_(
            original.gate.e_score_correction_bias[permutation]
        )
    hidden = torch.randn(2, 4, 4)

    expected = original(hidden)
    actual = reordered(hidden)

    assert torch.allclose(actual, expected, atol=2e-6, rtol=2e-6)


def test_real_checkpoint_shape_and_dtype_contracts_are_pinned_without_allocating():
    config_path = Path(__file__).parents[1] / "reference" / "config.json"
    config = K3LayerConfig.from_json(config_path)
    specs = config.real_tensor_specs()

    assert specs["routed_expert_down_proj.weight"] == ((3584, 7168), "BF16")
    assert specs["routed_expert_norm.weight"] == ((3584,), "BF16")
    assert specs["routed_expert_up_proj.weight"] == ((7168, 3584), "BF16")
    assert specs["experts.w1.weight_packed"] == ((3072, 1792), "U8")
    assert specs["experts.w1.weight_scale"] == ((3072, 112), "U8")
    assert specs["experts.w2.weight_packed"] == ((3584, 1536), "U8")
    assert specs["experts.w2.weight_scale"] == ((3584, 96), "U8")
    assert specs["shared_experts.gate_proj.weight"] == ((6144, 7168), "BF16")
    assert specs["self_attn.A_log"] == ((96,), "F32")
    assert specs["self_attn.dt_bias"] == ((12288,), "F32")
    assert specs["self_attn.k_conv1d.weight"] == ((12288, 1, 4), "F32")
    assert config.is_kda_layer(12)
    assert not config.is_kda_layer(11)

    moe = LatentMoE(
        hidden_size=7168,
        latent_size=3584,
        expert_intermediate_size=3072,
        num_experts=896,
        top_k=16,
        num_shared_experts=2,
        expert_provider=lambda *_: (),
        device="meta",
        dtype=torch.bfloat16,
    )
    assert moe.gate.weight.shape == (896, 7168)
    assert moe.gate.weight.dtype == torch.bfloat16
    assert moe.routed_expert_down_proj.weight.shape == (3584, 7168)
    assert moe.routed_expert_norm.weight.shape == (3584,)
    assert moe.routed_expert_up_proj.weight.shape == (7168, 3584)
    assert moe.shared_experts.gate_proj.weight.shape == (6144, 7168)
    assert moe.shared_experts.down_proj.weight.shape == (7168, 6144)
