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


_EXPECTED_LAYER_12_NON_EXPERT = {
    "block_sparse_moe.gate.e_score_correction_bias": ((896,), F32),
    "block_sparse_moe.gate.weight": ((896, 7168), BF16),
    "block_sparse_moe.routed_expert_down_proj.weight": ((3584, 7168), BF16),
    "block_sparse_moe.routed_expert_norm.weight": ((3584,), BF16),
    "block_sparse_moe.routed_expert_up_proj.weight": ((7168, 3584), BF16),
    "block_sparse_moe.shared_experts.down_proj.weight": ((7168, 6144), BF16),
    "block_sparse_moe.shared_experts.gate_proj.weight": ((6144, 7168), BF16),
    "block_sparse_moe.shared_experts.up_proj.weight": ((6144, 7168), BF16),
    "input_layernorm.weight": ((7168,), BF16),
    "mlp_res_norm.weight": ((7168,), BF16),
    "mlp_res_proj.weight": ((1, 7168), BF16),
    "post_attention_layernorm.weight": ((7168,), BF16),
    "self_attention_res_norm.weight": ((7168,), BF16),
    "self_attention_res_proj.weight": ((1, 7168), BF16),
    "self_attn.A_log": ((128,), F32),
    "self_attn.b_proj.weight": ((96, 7168), BF16),
    "self_attn.dt_bias": ((12288,), F32),
    "self_attn.f_a_proj.weight": ((128, 7168), BF16),
    "self_attn.f_b_proj.weight": ((12288, 128), BF16),
    "self_attn.g_proj.weight": ((12288, 7168), BF16),
    "self_attn.k_conv1d.weight": ((12288, 1, 4), F32),
    "self_attn.k_proj.weight": ((12288, 7168), BF16),
    "self_attn.o_norm.weight": ((128,), F32),
    "self_attn.o_proj.weight": ((7168, 12288), BF16),
    "self_attn.q_conv1d.weight": ((12288, 1, 4), F32),
    "self_attn.q_proj.weight": ((12288, 7168), BF16),
    "self_attn.v_conv1d.weight": ((12288, 1, 4), F32),
    "self_attn.v_proj.weight": ((12288, 7168), BF16),
}


def test_layer_12_non_expert_manifest_exactly_matches_safetensors_header():
    actual = {
        name: (spec.shape, spec.dtype)
        for name, spec in K3_LAYER_TENSOR_MANIFEST.items()
    }

    assert actual == _EXPECTED_LAYER_12_NON_EXPERT


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


class _ManifestBackedStore:
    def __init__(self, layer_idx: int) -> None:
        self.prefix = f"layers.{layer_idx}."
        self.validated: set[str] = set()
        self.loaded: set[str] = set()

    def _relative_name(self, name: str) -> str:
        assert name.startswith(self.prefix)
        return name[len(self.prefix) :]

    def validate(self, name: str, spec) -> None:
        relative_name = self._relative_name(name)
        assert K3_LAYER_TENSOR_MANIFEST[relative_name] == spec
        self.validated.add(relative_name)

    def load(self, name: str, *, device, dtype=None) -> torch.Tensor:
        relative_name = self._relative_name(name)
        spec = K3_LAYER_TENSOR_MANIFEST[relative_name]
        self.loaded.add(relative_name)
        storage_dtype = _TORCH_DTYPES[spec.dtype]
        result_dtype = dtype if dtype is not None else storage_dtype
        return torch.empty(spec.shape, device=device, dtype=result_dtype)


def test_kda_loader_consumes_every_manifest_tensor_and_no_others():
    config_path = Path(__file__).parents[1] / "reference" / "config.json"
    config = K3LayerConfig.from_json(config_path)
    layer = K3ReferenceLayer(
        config,
        layer_idx=12,
        device="meta",
        dtype=torch.bfloat16,
    )
    store = _ManifestBackedStore(layer_idx=12)

    layer._load_raw_weights(store, device="meta", dtype=torch.bfloat16)

    expected = set(K3_LAYER_TENSOR_MANIFEST)
    assert store.validated == expected
    assert store.loaded == expected


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
