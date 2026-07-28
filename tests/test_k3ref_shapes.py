import json
from pathlib import Path

import torch

from engine.k3ref.config import K3LayerConfig
from engine.k3ref.layer import K3ReferenceLayer
from engine.k3ref.manifest import (
    BF16,
    F32,
    K3_EXPERT_CHECKPOINT_MANIFEST,
    K3_LAYER_TENSOR_MANIFEST,
    MXFP4_GROUP_SIZE,
    U8,
    runtime_parameter_manifest,
)


_TORCH_DTYPES = {
    BF16: torch.bfloat16,
    F32: torch.float32,
}


def test_every_constructed_kda_layer_parameter_matches_checkpoint_manifest():
    config_path = Path(__file__).parents[1] / "reference" / "config.json"
    config = K3LayerConfig.from_json(config_path)
    layer = K3ReferenceLayer(
        config,
        layer_idx=12,
        device="meta",
        dtype=torch.bfloat16,
    )
    expected = runtime_parameter_manifest(config.num_experts)
    actual = dict(layer.named_parameters())

    assert set(actual) == set(expected)
    for name, spec in expected.items():
        parameter = actual[name]
        assert tuple(parameter.shape) == spec.shape, name
        assert parameter.dtype == _TORCH_DTYPES[spec.dtype], name


def test_checkpoint_manifest_pins_kda_gate_axes_and_all_raw_expert_storage():
    assert K3_LAYER_TENSOR_MANIFEST["self_attn.b_proj.weight"].shape == (96, 7168)
    assert K3_LAYER_TENSOR_MANIFEST["self_attn.f_a_proj.weight"].shape == (
        128,
        7168,
    )
    assert K3_LAYER_TENSOR_MANIFEST["self_attn.f_b_proj.weight"].shape == (
        12288,
        128,
    )
    assert K3_LAYER_TENSOR_MANIFEST["self_attn.A_log"].shape == (128,)
    assert K3_LAYER_TENSOR_MANIFEST["self_attn.A_log"].dtype == F32
    assert K3_LAYER_TENSOR_MANIFEST["self_attn.dt_bias"].shape == (12288,)
    assert K3_LAYER_TENSOR_MANIFEST["self_attn.k_conv1d.weight"].shape == (
        12288,
        1,
        4,
    )

    assert MXFP4_GROUP_SIZE == 32
    assert K3_EXPERT_CHECKPOINT_MANIFEST[
        "block_sparse_moe.experts.{expert}.w1.weight_packed"
    ].shape == (3072, 1792)
    assert K3_EXPERT_CHECKPOINT_MANIFEST[
        "block_sparse_moe.experts.{expert}.w1.weight_scale"
    ].shape == (3072, 112)
    assert K3_EXPERT_CHECKPOINT_MANIFEST[
        "block_sparse_moe.experts.{expert}.w2.weight_packed"
    ].shape == (3584, 1536)
    assert K3_EXPERT_CHECKPOINT_MANIFEST[
        "block_sparse_moe.experts.{expert}.w2.weight_scale"
    ].shape == (3584, 96)
    assert K3_EXPERT_CHECKPOINT_MANIFEST[
        "block_sparse_moe.experts.{expert}.w3.weight_packed"
    ].shape == (3072, 1792)
    assert K3_EXPERT_CHECKPOINT_MANIFEST[
        "block_sparse_moe.experts.{expert}.w3.weight_scale"
    ].shape == (3072, 112)
    assert {spec.dtype for spec in K3_EXPERT_CHECKPOINT_MANIFEST.values()} == {U8}


def test_real_config_pins_situ_for_routed_and_shared_experts():
    config_path = Path(__file__).parents[1] / "reference" / "config.json"
    text_config = json.loads(config_path.read_text(encoding="utf-8"))["text_config"]

    assert text_config["hidden_act"] == "situ"
    assert text_config["activation_situ_beta"] == 4.0
    assert text_config["activation_situ_linear_beta"] == 25.0
