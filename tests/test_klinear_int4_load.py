import json

import pytest
import torch

from engine.klinear.quantized import W4A16Linear
from engine.klinear.weights import (
    CheckpointKind,
    SafetensorIndexStore,
    detect_checkpoint_kind,
)
from engine.quant.w4a16 import W4A16Tensor


def _tensor_bytes(tensor: torch.Tensor) -> bytes:
    return bytes(tensor.contiguous().view(torch.uint8).reshape(-1).tolist())


def _write_checkpoint(tmp_path, tensors: dict[str, torch.Tensor]) -> None:
    shard_name = "model-00001-of-00001.safetensors"
    header = {}
    payload = bytearray()
    for name, tensor in tensors.items():
        raw = _tensor_bytes(tensor)
        start = len(payload)
        payload.extend(raw)
        dtype = {
            torch.uint8: "U8",
            torch.bfloat16: "BF16",
        }[tensor.dtype]
        header[name] = {
            "dtype": dtype,
            "shape": list(tensor.shape),
            "data_offsets": [start, len(payload)],
        }
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    shard = len(header_bytes).to_bytes(8, "little") + header_bytes + payload
    (tmp_path / shard_name).write_bytes(shard)
    index = {
        "metadata": {
            "total_size": len(payload),
            "quantization": "W4A16",
            "group_size": 32,
            "scale_dtype": "BF16",
        },
        "weight_map": {name: shard_name for name in tensors},
    }
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(index), encoding="utf-8"
    )


def test_checkpoint_kind_detection_is_index_derived_and_fail_closed():
    bf16 = {
        "weight_map": {"sample.weight": "model.safetensors"},
    }
    w4a16 = {
        "metadata": {
            "quantization": "W4A16",
            "group_size": 32,
            "scale_dtype": "BF16",
        },
        "weight_map": {
            "sample.weight.w4a16_packed": "model.safetensors",
            "sample.weight.w4a16_scales": "model.safetensors",
            "sample.norm.weight": "model.safetensors",
        },
    }

    assert detect_checkpoint_kind(
        bf16, expected_bf16_names={"sample.weight"}
    ) is CheckpointKind.BF16
    assert detect_checkpoint_kind(w4a16) is CheckpointKind.W4A16
    with pytest.raises(ValueError, match="neither the BF16 contract"):
        detect_checkpoint_kind(
            {"weight_map": {"wrong.tensor": "model.safetensors"}},
            expected_bf16_names={"sample.weight"},
        )


def test_quantized_linear_reports_packed_plus_scale_bytes_not_bf16_bytes():
    encoded = W4A16Tensor(
        packed=torch.zeros(3, 16, dtype=torch.uint8),
        scales=torch.ones(3, 1, dtype=torch.bfloat16),
        original_shape=(3, 32),
    )
    linear = W4A16Linear.from_encoded(encoded)

    assert linear.resident_bytes == 3 * 16 + 3 * 1 * 2
    assert linear.resident_bytes == encoded.storage_bytes
    assert linear.resident_bytes != 3 * 32 * 2
    assert linear.weight is None


def test_mixed_checkpoint_keeps_plan_retained_linear_classes_in_bf16(tmp_path):
    quantized_name = "model.layers.0.mlp.gate_proj.weight"
    retained_names = (
        "model.layers.0.self_attn.f_a_proj.weight",
        "model.layers.1.block_sparse_moe.gate.weight",
        "model.layers.3.self_attn.kv_a_proj_with_mqa.weight",
    )
    tensors = {
        quantized_name + ".w4a16_packed": torch.zeros(2, 16, dtype=torch.uint8),
        quantized_name + ".w4a16_scales": torch.ones(
            2, 1, dtype=torch.bfloat16
        ),
    }
    tensors.update(
        {
            name: torch.ones(2, 32, dtype=torch.bfloat16)
            for name in retained_names
        }
    )
    _write_checkpoint(tmp_path, tensors)
    store = SafetensorIndexStore(tmp_path, validate_real_layout=False)
    factory = store.linear_factory()
    assert factory is not None
    assert isinstance(
        factory(
            quantized_name,
            2_304,
            9_216,
            device="meta",
            dtype=torch.bfloat16,
        ),
        W4A16Linear,
    )
    retained_dimensions = {
        retained_names[0]: (2_304, 128),
        retained_names[1]: (2_304, 256),
        retained_names[2]: (2_304, 576),
    }
    for name, (in_features, out_features) in retained_dimensions.items():
        module = factory(
            name,
            in_features,
            out_features,
            device="meta",
            dtype=torch.bfloat16,
        )
        assert isinstance(module, torch.nn.Linear)
        assert module.weight.dtype == torch.bfloat16

    quantized = store.load_linear_weight(
        quantized_name,
        (2, 32),
        device="cpu",
        dtype=torch.bfloat16,
    )
    assert isinstance(quantized, W4A16Tensor)
    for name in retained_names:
        retained = store.load_linear_weight(
            name,
            (2, 32),
            device="cpu",
            dtype=torch.bfloat16,
        )
        assert isinstance(retained, torch.Tensor)
        assert retained.dtype == torch.bfloat16
        assert retained.shape == (2, 32)


def test_mixed_checkpoint_missing_scale_tensor_is_rejected(tmp_path):
    packed_name = "model.layers.0.mlp.gate_proj.weight.w4a16_packed"
    index = {
        "metadata": {
            "quantization": "W4A16",
            "group_size": 32,
            "scale_dtype": "BF16",
        },
        "weight_map": {packed_name: "model.safetensors"},
    }
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(index), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="missing_scales"):
        SafetensorIndexStore(tmp_path, validate_real_layout=False)
