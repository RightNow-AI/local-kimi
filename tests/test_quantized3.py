from __future__ import annotations

import json

import pytest
import torch
import torch.nn.functional as F

from engine.klinear.quantized3 import W3A16Linear
from engine.klinear.weights import CheckpointKind, SafetensorIndexStore
from engine.quant.w3a16 import W3A16Tensor, dequantise, quantise


def _tensor_bytes(tensor: torch.Tensor) -> bytes:
    return bytes(tensor.contiguous().view(torch.uint8).reshape(-1).tolist())


def _write_checkpoint(
    directory,
    tensors: dict[str, torch.Tensor],
    *,
    quantization: str,
) -> None:
    directory.mkdir()
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
    (directory / shard_name).write_bytes(shard)
    index = {
        "metadata": {
            "total_size": len(payload),
            "quantization": quantization,
            "group_size": 32,
            "scale_dtype": "BF16",
        },
        "weight_map": {name: shard_name for name in tensors},
    }
    (directory / "model.safetensors.index.json").write_text(
        json.dumps(index), encoding="utf-8"
    )


def test_quantized3_linear_round_trip_reports_exact_resident_bytes() -> None:
    torch.manual_seed(20260729)
    out_features = 3
    in_features = 32
    group_size = 32
    encoded = quantise(
        torch.randn((out_features, in_features), dtype=torch.bfloat16),
        group_size=group_size,
    )

    linear = W3A16Linear.from_encoded(encoded)
    restored = linear.encoded

    assert torch.equal(restored.packed, encoded.packed)
    assert torch.equal(restored.scales, encoded.scales)
    assert restored.original_shape == encoded.original_shape
    assert restored.group_size == encoded.group_size
    packed_bytes = out_features * (in_features // 8 * 3)
    scale_bytes = out_features * (in_features // group_size) * 2
    assert linear.resident_bytes == packed_bytes + scale_bytes
    assert linear.resident_bytes == encoded.storage_bytes
    assert linear.weight is None


def test_quantized3_linear_forward_matches_dequantise_then_linear_on_cpu(
    monkeypatch,
) -> None:
    torch.manual_seed(20260730)
    weight = torch.randn((5, 32), dtype=torch.bfloat16)
    encoded = quantise(weight, group_size=32)
    linear = W3A16Linear.from_encoded(encoded)

    def reference(
        activations: torch.Tensor,
        packed: torch.Tensor,
        scales: torch.Tensor,
        *,
        group_size: int,
    ) -> torch.Tensor:
        reference_encoded = W3A16Tensor(
            packed=packed,
            scales=scales,
            original_shape=(packed.shape[0], packed.shape[1] // 3 * 8),
            group_size=group_size,
        )
        return F.linear(activations, dequantise(reference_encoded))

    monkeypatch.setattr(linear, "_dense_kernel", reference)
    activations = torch.randn((2, 4, 32), dtype=torch.bfloat16)
    expected = F.linear(activations, dequantise(encoded))

    actual = linear(activations)

    torch.testing.assert_close(actual, expected, atol=0.01, rtol=0.01)


def test_checkpoint_detector_distinguishes_w3a16_from_w4a16_content(
    tmp_path,
) -> None:
    w3_directory = tmp_path / "int3"
    _write_checkpoint(
        w3_directory,
        {
            "sample.weight.w3a16_packed": torch.zeros(2, 12, dtype=torch.uint8),
            "sample.weight.w3a16_scales": torch.ones(
                2, 1, dtype=torch.bfloat16
            ),
        },
        quantization="W3A16",
    )
    w4_directory = tmp_path / "int4"
    _write_checkpoint(
        w4_directory,
        {
            "sample.weight.w4a16_packed": torch.zeros(2, 16, dtype=torch.uint8),
            "sample.weight.w4a16_scales": torch.ones(
                2, 1, dtype=torch.bfloat16
            ),
        },
        quantization="W4A16",
    )

    w3_store = SafetensorIndexStore(w3_directory, validate_real_layout=False)
    w4_store = SafetensorIndexStore(w4_directory, validate_real_layout=False)
    loaded = w3_store.load_w3a16(
        "sample.weight",
        (2, 32),
        device="cpu",
    )

    assert w3_store.checkpoint_kind is CheckpointKind.W3A16
    assert w4_store.checkpoint_kind is CheckpointKind.W4A16
    assert w4_store.checkpoint_kind is not CheckpointKind.W3A16
    assert loaded.original_shape == (2, 32)
    assert loaded.packed.shape == (2, 12)
    assert loaded.scales.shape == (2, 1)


def test_checkpoint_detector_rejects_w4_width_under_w3_names(tmp_path) -> None:
    directory = tmp_path / "mislabeled"
    _write_checkpoint(
        directory,
        {
            "sample.weight.w3a16_packed": torch.zeros(2, 16, dtype=torch.uint8),
            "sample.weight.w3a16_scales": torch.ones(
                2, 1, dtype=torch.bfloat16
            ),
        },
        quantization="W3A16",
    )

    with pytest.raises(ValueError, match="3 bytes per 8 weights"):
        SafetensorIndexStore(directory, validate_real_layout=False)
