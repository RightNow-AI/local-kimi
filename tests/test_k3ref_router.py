import ast
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from engine.k3ref.router import K3Router


def _moonshot_router_class():
    source_path = Path(__file__).parents[1] / "reference" / "modeling_kimi_linear.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    router = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "KimiMoEGate"
    )
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__",
                names=[ast.alias(name="annotations")],
                level=0,
            ),
            router,
        ],
        type_ignores=[],
    )
    namespace = {
        "F": F,
        "KimiLinearConfig": object,
        "math": math,
        "nn": nn,
        "torch": torch,
    }
    exec(compile(ast.fix_missing_locations(module), str(source_path), "exec"), namespace)
    return namespace["KimiMoEGate"]


class _Config:
    hidden_size = 7
    num_experts = 8
    num_experts_per_token = 3
    num_expert_group = 4
    topk_group = 2
    routed_scaling_factor = 1.25
    moe_router_activation_func = "sigmoid"
    moe_renormalize = True


def test_router_matches_moonshot_selection_bias_grouping_and_renormalization():
    torch.manual_seed(20260728)
    reference = _moonshot_router_class()(_Config()).eval()
    actual = K3Router(
        _Config.hidden_size,
        _Config.num_experts,
        _Config.num_experts_per_token,
        num_expert_group=_Config.num_expert_group,
        topk_group=_Config.topk_group,
        renormalize=_Config.moe_renormalize,
        routed_scaling_factor=_Config.routed_scaling_factor,
    ).eval()
    weight = torch.randn(_Config.num_experts, _Config.hidden_size)
    correction = torch.tensor([0.0, 0.8, -0.4, 0.1, 0.0, -0.2, 0.5, 0.3])
    with torch.no_grad():
        reference.weight.copy_(weight)
        reference.e_score_correction_bias.copy_(correction)
        actual.weight.copy_(weight)
        actual.e_score_correction_bias.copy_(correction)
    hidden_states = torch.randn(2, 5, _Config.hidden_size)

    expected_indices, expected_weights = reference(hidden_states)
    actual_indices, actual_weights = actual(hidden_states)

    assert torch.equal(actual_indices, expected_indices)
    assert torch.equal(actual_weights, expected_weights)
    assert torch.allclose(
        actual_weights.sum(dim=-1),
        torch.full((10,), _Config.routed_scaling_factor),
    )
