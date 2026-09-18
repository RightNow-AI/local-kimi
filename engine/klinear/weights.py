"""Low-level readers for BF16, W3A16, and W4A16 safetensors checkpoints."""

from __future__ import annotations

import json
from collections import OrderedDict, defaultdict
from collections.abc import Collection
from dataclasses import dataclass
from enum import Enum
from math import prod
from pathlib import Path
from typing import Any, BinaryIO, Mapping

import torch
from torch import nn

from engine.quant.klinear_plan import (
    TensorMetadata,
    build_klinear_quantization_plan,
)
from engine.quant.w3a16 import GROUP_SIZE as W3A16_GROUP_SIZE
from engine.quant.w3a16 import W3A16Tensor
from engine.quant.w4a16 import GROUP_SIZE, W4A16Tensor

from .manifest import (
    REAL_CHECKPOINT_MANIFEST,
    REAL_EXPERT_TEMPLATE_MANIFEST,
    TensorSpec,
    validate_real_checkpoint_layout,
)
from .quantized import LinearFactory, W4A16Linear
from .quantized3 import W3A16Linear

_PACKED_SUFFIX = ".w4a16_packed"
_SCALES_SUFFIX = ".w4a16_scales"
_W3A16_PACKED_SUFFIX = ".w3a16_packed"
_W3A16_SCALES_SUFFIX = ".w3a16_scales"

_DTYPES = {
    "BOOL": torch.bool,
    "U8": torch.uint8,
    "I8": torch.int8,
    "I16": torch.int16,
    "I32": torch.int32,
    "I64": torch.int64,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    "F32": torch.float32,
    "F64": torch.float64,
}

_DTYPE_BYTES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "I16": 2,
    "I32": 4,
    "I64": 8,
    "F16": 2,
    "BF16": 2,
    "F32": 4,
    "F64": 8,
}


class CheckpointKind(str, Enum):
    BF16 = "bf16"
    W3A16 = "w3a16"
    W4A16 = "w4a16"


def _build_real_w4a16_layout() -> tuple[frozenset[str], dict[str, TensorSpec]]:
    metadata = (
        TensorMetadata(name=name, shape=spec.shape, dtype=spec.dtype)
        for name, spec in REAL_CHECKPOINT_MANIFEST.items()
    )
    plan = build_klinear_quantization_plan(metadata)
    quantized_names = frozenset(
        decision.name for decision in plan.tensors if decision.quantize
    )
    layout: dict[str, TensorSpec] = {}
    for name, spec in REAL_CHECKPOINT_MANIFEST.items():
        if name not in quantized_names:
            layout[name] = spec
            continue
        rows = prod(spec.shape[:-1])
        reduction = spec.shape[-1]
        layout[name + _PACKED_SUFFIX] = TensorSpec((rows, reduction // 2), "U8")
        layout[name + _SCALES_SUFFIX] = TensorSpec(
            (rows, reduction // GROUP_SIZE), "BF16"
        )
    if len(quantized_names) != 20_150:
        raise AssertionError("Kimi-Linear W4A16 plan no longer selects 20,150 tensors")
    return quantized_names, layout


_REAL_W4A16_NAMES, _REAL_W4A16_MANIFEST = _build_real_w4a16_layout()


def _build_real_w3a16_layout() -> tuple[frozenset[str], dict[str, TensorSpec]]:
    metadata = (
        TensorMetadata(name=name, shape=spec.shape, dtype=spec.dtype)
        for name, spec in REAL_CHECKPOINT_MANIFEST.items()
    )
    plan = build_klinear_quantization_plan(metadata)
    quantized_names = frozenset(
        decision.name for decision in plan.tensors if decision.quantize
    )
    layout: dict[str, TensorSpec] = {}
    for name, spec in REAL_CHECKPOINT_MANIFEST.items():
        if name not in quantized_names:
            layout[name] = spec
            continue
        rows = prod(spec.shape[:-1])
        reduction = spec.shape[-1]
        layout[name + _W3A16_PACKED_SUFFIX] = TensorSpec(
            (rows, reduction // 8 * 3), "U8"
        )
        layout[name + _W3A16_SCALES_SUFFIX] = TensorSpec(
            (rows, reduction // W3A16_GROUP_SIZE), "BF16"
        )
    if len(quantized_names) != 20_150:
        raise AssertionError("Kimi-Linear W3A16 plan no longer selects 20,150 tensors")
    return quantized_names, layout


_REAL_W3A16_NAMES, _REAL_W3A16_MANIFEST = _build_real_w3a16_layout()


def _weight_map(index_payload: Mapping[str, Any]) -> dict[str, str]:
    weight_map = index_payload.get("weight_map")
    if not isinstance(weight_map, Mapping) or not weight_map:
        raise ValueError("safetensors index has no weight_map")
    if not all(
        isinstance(name, str)
        and name
        and isinstance(shard, str)
        and shard
        for name, shard in weight_map.items()
    ):
        raise ValueError("safetensors weight_map must contain non-empty string pairs")
    return dict(weight_map)


def _validate_w3a16_index_specs(
    packed_bases: Collection[str],
    tensor_specs: Mapping[str, TensorSpec],
    *,
    group_size: int,
) -> None:
    for base in packed_bases:
        packed_name = base + _W3A16_PACKED_SUFFIX
        scales_name = base + _W3A16_SCALES_SUFFIX
        try:
            packed = tensor_specs[packed_name]
            scales = tensor_specs[scales_name]
        except KeyError as error:
            raise ValueError(
                f"W3A16 tensor spec is missing for {error.args[0]}"
            ) from error
        if packed.dtype != "U8" or scales.dtype != "BF16":
            raise ValueError(
                f"W3A16 payload dtypes must be U8 and BF16 for {base}: "
                f"packed={packed.dtype}, scales={scales.dtype}"
            )
        if len(packed.shape) != 2 or len(scales.shape) != 2:
            raise ValueError(f"W3A16 payloads must be matrices for {base}")
        rows, packed_width = packed.shape
        scale_rows, scale_groups = scales.shape
        if rows <= 0 or scale_groups <= 0:
            raise ValueError(f"W3A16 payload dimensions must be positive for {base}")
        original_reduction = scale_groups * group_size
        expected_packed = (rows, original_reduction // 8 * 3)
        if scale_rows != rows or packed.shape != expected_packed:
            raise ValueError(
                f"W3A16 packed bytes must use 3 bytes per 8 weights for {base}: "
                f"packed={packed.shape}, scales={scales.shape}, "
                f"group_size={group_size}, expected_packed={expected_packed}"
            )
        original_elements = rows * original_reduction
        if rows * packed_width * 8 != original_elements * 3:
            raise ValueError(
                f"W3A16 packed byte ratio is invalid for {base}: "
                f"packed={packed.shape}, original_elements={original_elements}"
            )


def detect_checkpoint_kind(
    index_payload: Mapping[str, Any],
    *,
    expected_bf16_names: Collection[str] | None = None,
    tensor_specs: Mapping[str, TensorSpec] | None = None,
) -> CheckpointKind:
    """Classify an index from its names and reject unsupported layouts."""
    names = set(_weight_map(index_payload))
    metadata = index_payload.get("metadata")
    if metadata is None:
        metadata = {}
    if not isinstance(metadata, Mapping):
        raise ValueError("safetensors index metadata must be an object")
    quantization = metadata.get("quantization")
    if quantization not in (None, "W3A16", "W4A16"):
        raise ValueError(f"unsupported checkpoint quantization: {quantization}")

    w3a16_packed_bases = {
        name[: -len(_W3A16_PACKED_SUFFIX)]
        for name in names
        if name.endswith(_W3A16_PACKED_SUFFIX)
    }
    w3a16_scale_bases = {
        name[: -len(_W3A16_SCALES_SUFFIX)]
        for name in names
        if name.endswith(_W3A16_SCALES_SUFFIX)
    }

    packed_bases = {
        name[: -len(_PACKED_SUFFIX)]
        for name in names
        if name.endswith(_PACKED_SUFFIX)
    }
    scale_bases = {
        name[: -len(_SCALES_SUFFIX)]
        for name in names
        if name.endswith(_SCALES_SUFFIX)
    }
    if (w3a16_packed_bases or w3a16_scale_bases) and (
        packed_bases or scale_bases
    ):
        raise ValueError("checkpoint index mixes W3A16 and W4A16 payloads")
    if w3a16_packed_bases or w3a16_scale_bases:
        missing_scales = sorted(w3a16_packed_bases - w3a16_scale_bases)
        missing_packed = sorted(w3a16_scale_bases - w3a16_packed_bases)
        if missing_scales or missing_packed:
            raise ValueError(
                "W3A16 index has incomplete packed and scale pairs: "
                f"missing_scales={missing_scales}, missing_packed={missing_packed}"
            )
        duplicate_originals = sorted(w3a16_packed_bases & names)
        if duplicate_originals:
            raise ValueError(
                "W3A16 index contains packed and BF16 copies of the same tensor: "
                f"{duplicate_originals}"
            )
        if quantization not in (None, "W3A16"):
            raise ValueError(
                f"W3A16 tensor names disagree with quantization={quantization}"
            )
        group_size = metadata.get("group_size", W3A16_GROUP_SIZE)
        if (
            not isinstance(group_size, int)
            or isinstance(group_size, bool)
            or group_size <= 0
            or group_size % 8
        ):
            raise ValueError(
                "W3A16 index group size must be a positive multiple of 8, "
                f"got {group_size}"
            )
        scale_dtype = metadata.get("scale_dtype")
        if scale_dtype is not None and scale_dtype != "BF16":
            raise ValueError(
                f"W3A16 index scale dtype must be BF16, got {scale_dtype}"
            )
        if tensor_specs is not None:
            _validate_w3a16_index_specs(
                w3a16_packed_bases,
                tensor_specs,
                group_size=group_size,
            )
        return CheckpointKind.W3A16

    if packed_bases or scale_bases:
        missing_scales = sorted(packed_bases - scale_bases)
        missing_packed = sorted(scale_bases - packed_bases)
        if missing_scales or missing_packed:
            raise ValueError(
                "W4A16 index has incomplete packed and scale pairs: "
                f"missing_scales={missing_scales}, missing_packed={missing_packed}"
            )
        duplicate_originals = sorted(packed_bases & names)
        if duplicate_originals:
            raise ValueError(
                "W4A16 index contains packed and BF16 copies of the same tensor: "
                f"{duplicate_originals}"
            )
        if quantization not in (None, "W4A16"):
            raise ValueError(
                f"W4A16 tensor names disagree with quantization={quantization}"
            )
        group_size = metadata.get("group_size")
        if group_size is not None and group_size != GROUP_SIZE:
            raise ValueError(
                f"W4A16 index group size must be {GROUP_SIZE}, got {group_size}"
            )
        scale_dtype = metadata.get("scale_dtype")
        if scale_dtype is not None and scale_dtype != "BF16":
            raise ValueError(
                f"W4A16 index scale dtype must be BF16, got {scale_dtype}"
            )
        return CheckpointKind.W4A16

    if quantization == "W3A16":
        raise ValueError("W3A16 index declares quantization but has no packed tensors")
    if any(".w3a16_" in name for name in names):
        raise ValueError("checkpoint index contains unrecognized W3A16 tensor names")
    if quantization == "W4A16":
        raise ValueError("W4A16 index declares quantization but has no packed tensors")
    if any(".w4a16_" in name for name in names):
        raise ValueError("checkpoint index contains unrecognized W4A16 tensor names")
    expected = (
        set(REAL_CHECKPOINT_MANIFEST)
        if expected_bf16_names is None
        else set(expected_bf16_names)
    )
    if names == expected:
        return CheckpointKind.BF16
    raise ValueError(
        "checkpoint index is neither the BF16 contract nor a paired W3A16 or "
        "W4A16 layout"
    )


def validate_real_w4a16_layout(actual: Mapping[str, TensorSpec]) -> None:
    expected_names = set(_REAL_W4A16_MANIFEST)
    actual_names = set(actual)
    missing = sorted(expected_names - actual_names)
    unexpected = sorted(actual_names - expected_names)
    if missing or unexpected:
        raise ValueError(
            "checkpoint tensor names do not match Kimi-Linear W4A16: "
            f"missing={missing}, unexpected={unexpected}"
        )
    mismatches = [
        name
        for name, expected in _REAL_W4A16_MANIFEST.items()
        if actual[name] != expected
    ]
    if mismatches:
        details = [
            f"{name}: expected={_REAL_W4A16_MANIFEST[name]}, actual={actual[name]}"
            for name in mismatches
        ]
        raise ValueError("W4A16 checkpoint tensor specs disagree: " + "; ".join(details))


def validate_real_w3a16_layout(actual: Mapping[str, TensorSpec]) -> None:
    expected_names = set(_REAL_W3A16_MANIFEST)
    actual_names = set(actual)
    missing = sorted(expected_names - actual_names)
    unexpected = sorted(actual_names - expected_names)
    if missing or unexpected:
        raise ValueError(
            "checkpoint tensor names do not match Kimi-Linear W3A16: "
            f"missing={missing}, unexpected={unexpected}"
        )
    mismatches = [
        name
        for name, expected in _REAL_W3A16_MANIFEST.items()
        if actual[name] != expected
    ]
    if mismatches:
        details = [
            f"{name}: expected={_REAL_W3A16_MANIFEST[name]}, actual={actual[name]}"
            for name in mismatches
        ]
        raise ValueError(
            "W3A16 checkpoint tensor specs disagree: " + "; ".join(details)
        )


@dataclass(frozen=True)
class TensorLocation:
    path: Path
    data_start: int
    data_end: int
    spec: TensorSpec


class SafetensorIndexStore:
    """Read named tensors directly from Hugging Face sharded safetensors files."""

    def __init__(
        self,
        directory: str | Path,
        *,
        index_name: str = "model.safetensors.index.json",
        validate_real_layout: bool = True,
    ) -> None:
        self.directory = Path(directory)
        index_path = self.directory / index_name
        if not index_path.is_file():
            raise FileNotFoundError(f"safetensors index is missing: {index_path}")
        payload = json.loads(index_path.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError("safetensors index must contain an object")
        self._weight_map = _weight_map(payload)
        expected_names = None if validate_real_layout else set(self._weight_map)
        self.checkpoint_kind = detect_checkpoint_kind(
            payload, expected_bf16_names=expected_names
        )
        self._locations: dict[str, TensorLocation] = {}

        for shard_name in sorted(set(self._weight_map.values())):
            shard_path = self.directory / shard_name
            if not shard_path.is_file():
                raise FileNotFoundError(f"safetensors shard is missing: {shard_path}")
            with shard_path.open("rb") as handle:
                length_bytes = handle.read(8)
                if len(length_bytes) != 8:
                    raise ValueError(
                        f"safetensors shard has no header length: {shard_path}"
                    )
                header_length = int.from_bytes(length_bytes, "little", signed=False)
                header_bytes = handle.read(header_length)
            if len(header_bytes) != header_length:
                raise ValueError(f"safetensors header is truncated: {shard_path}")
            header = json.loads(header_bytes.decode("utf-8"))
            if not isinstance(header, Mapping):
                raise ValueError(f"safetensors header is not an object: {shard_path}")
            data_base = 8 + header_length
            for name, entry in header.items():
                if name == "__metadata__":
                    continue
                if self._weight_map.get(name) != shard_name:
                    raise ValueError(f"index assigns {name} to a different shard")
                if not isinstance(entry, Mapping):
                    raise ValueError(f"invalid safetensors header entry for {name}")
                location = self._parse_location(shard_path, data_base, name, entry)
                if name in self._locations:
                    raise ValueError(f"tensor appears in multiple shards: {name}")
                self._locations[name] = location

        index_names = set(self._weight_map)
        header_names = set(self._locations)
        if index_names != header_names:
            raise ValueError(
                "safetensors index and shard headers disagree: "
                f"missing={sorted(index_names - header_names)}, "
                f"unexpected={sorted(header_names - index_names)}"
            )
        self.checkpoint_kind = detect_checkpoint_kind(
            payload,
            expected_bf16_names=expected_names,
            tensor_specs=self.specs,
        )
        metadata = payload.get("metadata") or {}
        self.quantization_group_size: int | None = None
        if self.checkpoint_kind is CheckpointKind.W3A16:
            self.quantization_group_size = metadata.get(
                "group_size", W3A16_GROUP_SIZE
            )
        elif self.checkpoint_kind is CheckpointKind.W4A16:
            self.quantization_group_size = GROUP_SIZE
        self.tensor_storage_bytes = sum(
            location.data_end - location.data_start
            for location in self._locations.values()
        )
        declared_size = metadata.get("total_size")
        if declared_size is not None:
            if not isinstance(declared_size, int) or declared_size < 0:
                raise ValueError("safetensors index total_size must be a non-negative int")
            if declared_size != self.tensor_storage_bytes:
                raise ValueError(
                    "safetensors index total_size disagrees with tensor headers: "
                    f"index={declared_size}, headers={self.tensor_storage_bytes}"
                )
        if validate_real_layout:
            if self.checkpoint_kind is CheckpointKind.BF16:
                validate_real_checkpoint_layout(self.specs)
            elif self.checkpoint_kind is CheckpointKind.W4A16:
                validate_real_w4a16_layout(self.specs)
            else:
                validate_real_w3a16_layout(self.specs)

    @staticmethod
    def _parse_location(
        path: Path,
        data_base: int,
        name: str,
        entry: Mapping[str, Any],
    ) -> TensorLocation:
        shape = entry.get("shape")
        dtype = entry.get("dtype")
        offsets = entry.get("data_offsets")
        if not isinstance(shape, list) or not all(
            isinstance(dimension, int) and dimension >= 0 for dimension in shape
        ):
            raise ValueError(f"invalid safetensors shape for {name}")
        if dtype not in _DTYPES:
            raise ValueError(f"unsupported safetensors dtype for {name}: {dtype}")
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or not all(isinstance(offset, int) and offset >= 0 for offset in offsets)
            or offsets[1] < offsets[0]
        ):
            raise ValueError(f"invalid safetensors offsets for {name}")
        elements = prod(shape)
        expected_bytes = elements * _DTYPE_BYTES[dtype]
        if offsets[1] - offsets[0] != expected_bytes:
            raise ValueError(f"safetensors byte count disagrees for {name}")
        return TensorLocation(
            path,
            data_base + offsets[0],
            data_base + offsets[1],
            TensorSpec(tuple(shape), dtype),
        )

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self._weight_map)

    @property
    def specs(self) -> dict[str, TensorSpec]:
        return {name: location.spec for name, location in self._locations.items()}

    def spec(self, name: str) -> TensorSpec:
        try:
            return self._locations[name].spec
        except KeyError as error:
            raise KeyError(f"checkpoint tensor is absent: {name}") from error

    def validate(self, name: str, spec: TensorSpec) -> None:
        actual = self.spec(name)
        if actual != spec:
            raise ValueError(
                f"checkpoint manifest mismatch for {name}: expected {spec}, got {actual}"
            )

    @staticmethod
    def _read_location(
        name: str,
        location: TensorLocation,
        handle: BinaryIO,
        *,
        device: torch.device | str,
        dtype: torch.dtype | None,
    ) -> torch.Tensor:
        if str(device) == "meta":
            result_dtype = dtype or _DTYPES[location.spec.dtype]
            return torch.empty(location.spec.shape, device="meta", dtype=result_dtype)
        byte_count = location.data_end - location.data_start
        handle.seek(location.data_start)
        raw = bytearray(handle.read(byte_count))
        if len(raw) != byte_count:
            raise ValueError(f"tensor payload is truncated: {name}")
        storage_dtype = _DTYPES[location.spec.dtype]
        tensor = torch.frombuffer(raw, dtype=storage_dtype).clone()
        tensor = tensor.reshape(location.spec.shape)
        if dtype is not None and tensor.dtype != dtype:
            tensor = tensor.to(dtype=dtype)
        return tensor.to(device=device)

    def load(
        self,
        name: str,
        *,
        device: torch.device | str = "cpu",
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        try:
            location = self._locations[name]
        except KeyError as error:
            raise KeyError(f"checkpoint tensor is absent: {name}") from error
        with location.path.open("rb") as handle:
            return self._read_location(
                name, location, handle, device=device, dtype=dtype
            )

    def load_many(
        self,
        names: Collection[str],
        *,
        device: torch.device | str = "cpu",
        dtype: torch.dtype | None = None,
    ) -> dict[str, torch.Tensor]:
        requested = tuple(names)
        if len(requested) != len(set(requested)):
            raise ValueError("load_many received duplicate tensor names")
        missing = sorted(set(requested) - set(self._locations))
        if missing:
            raise KeyError(f"checkpoint tensors are absent: {missing}")
        by_path: dict[Path, list[str]] = defaultdict(list)
        for name in requested:
            by_path[self._locations[name].path].append(name)
        loaded: dict[str, torch.Tensor] = {}
        for path, path_names in by_path.items():
            with path.open("rb") as handle:
                for name in path_names:
                    loaded[name] = self._read_location(
                        name,
                        self._locations[name],
                        handle,
                        device=device,
                        dtype=dtype,
                    )
        return loaded

    def load_w4a16(
        self,
        name: str,
        original_shape: tuple[int, int],
        *,
        device: torch.device | str,
    ) -> W4A16Tensor:
        if self.checkpoint_kind is not CheckpointKind.W4A16:
            raise ValueError("the checkpoint is not W4A16")
        out_features, in_features = original_shape
        if in_features % GROUP_SIZE:
            raise ValueError(
                f"W4A16 reduction dimension must be divisible by {GROUP_SIZE}"
            )
        packed_name = name + _PACKED_SUFFIX
        scales_name = name + _SCALES_SUFFIX
        self.validate(
            packed_name, TensorSpec((out_features, in_features // 2), "U8")
        )
        self.validate(
            scales_name,
            TensorSpec((out_features, in_features // GROUP_SIZE), "BF16"),
        )
        return W4A16Tensor(
            packed=self.load(packed_name, device=device),
            scales=self.load(scales_name, device=device),
            original_shape=original_shape,
        )

    def load_w3a16(
        self,
        name: str,
        original_shape: tuple[int, int],
        *,
        device: torch.device | str,
    ) -> W3A16Tensor:
        if self.checkpoint_kind is not CheckpointKind.W3A16:
            raise ValueError("the checkpoint is not W3A16")
        group_size = self.quantization_group_size
        if group_size is None:
            raise RuntimeError("W3A16 checkpoint group size is unavailable")
        out_features, in_features = original_shape
        if in_features % 8:
            raise ValueError("W3A16 reduction dimension must be divisible by 8")
        if in_features % group_size:
            raise ValueError(
                f"W3A16 reduction dimension must be divisible by {group_size}"
            )
        packed_name = name + _W3A16_PACKED_SUFFIX
        scales_name = name + _W3A16_SCALES_SUFFIX
        self.validate(
            packed_name,
            TensorSpec((out_features, in_features // 8 * 3), "U8"),
        )
        self.validate(
            scales_name,
            TensorSpec((out_features, in_features // group_size), "BF16"),
        )
        return W3A16Tensor(
            packed=self.load(packed_name, device=device),
            scales=self.load(scales_name, device=device),
            original_shape=original_shape,
            group_size=group_size,
        )

    def load_linear_weight(
        self,
        name: str,
        original_shape: tuple[int, int],
        *,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> torch.Tensor | W3A16Tensor | W4A16Tensor:
        if name not in REAL_CHECKPOINT_MANIFEST:
            raise KeyError(f"linear tensor is outside the Kimi-Linear contract: {name}")
        if self.checkpoint_kind is CheckpointKind.W3A16 and name in _REAL_W3A16_NAMES:
            return self.load_w3a16(name, original_shape, device=device)
        if self.checkpoint_kind is CheckpointKind.W4A16 and name in _REAL_W4A16_NAMES:
            return self.load_w4a16(name, original_shape, device=device)
        self.validate(name, TensorSpec(original_shape, "BF16"))
        return self.load(name, device=device, dtype=dtype)

    def linear_factory(self) -> LinearFactory | None:
        if self.checkpoint_kind is CheckpointKind.BF16:
            return None

        def factory(
            tensor_name: str,
            in_features: int,
            out_features: int,
            *,
            device: torch.device | str | None,
            dtype: torch.dtype | None,
        ) -> nn.Module:
            try:
                spec = REAL_CHECKPOINT_MANIFEST[tensor_name]
            except KeyError as error:
                raise KeyError(
                    f"linear tensor is outside the Kimi-Linear contract: {tensor_name}"
                ) from error
            if spec.shape != (out_features, in_features):
                raise ValueError(
                    f"constructed shape for {tensor_name} is "
                    f"{(out_features, in_features)}, expected {spec.shape}"
                )
            if (
                self.checkpoint_kind is CheckpointKind.W3A16
                and tensor_name in _REAL_W3A16_NAMES
            ):
                group_size = self.quantization_group_size
                if group_size is None:
                    raise RuntimeError("W3A16 checkpoint group size is unavailable")
                return W3A16Linear(
                    in_features,
                    out_features,
                    group_size=group_size,
                    device=device,
                )
            if tensor_name in _REAL_W4A16_NAMES:
                return W4A16Linear(in_features, out_features, device=device)
            return nn.Linear(
                in_features,
                out_features,
                bias=False,
                device=device,
                dtype=dtype,
            )

        return factory


class SafetensorExpertProvider:
    """Load selected BF16 experts and retain a bounded device-resident LRU cache."""

    def __init__(self, store: SafetensorIndexStore, *, cache_entries: int = 256) -> None:
        if cache_entries < 0:
            raise ValueError("cache_entries cannot be negative")
        self.store = store
        self.cache_entries = cache_entries
        self._cache: OrderedDict[
            tuple[int, int, str, torch.dtype],
            tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        ] = OrderedDict()

    def __call__(
        self,
        layer_idx: int,
        expert_id: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not 1 <= layer_idx <= 26:
            raise IndexError("real Kimi-Linear experts only exist in layers 1..26")
        if not 0 <= expert_id < 256:
            raise IndexError("real Kimi-Linear expert id must be in 0..255")
        key = (layer_idx, expert_id, str(device), dtype)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        prefix = f"model.layers.{layer_idx}.block_sparse_moe.experts.{expert_id}"
        weights = tuple(
            self.store.load(
                f"{prefix}.{projection}.weight", device=device, dtype=dtype
            )
            for projection in ("w1", "w2", "w3")
        )
        result = (weights[0], weights[1], weights[2])
        if self.cache_entries:
            self._cache[key] = result
            self._cache.move_to_end(key)
            while len(self._cache) > self.cache_entries:
                self._cache.popitem(last=False)
        return result

    @property
    def resident_bytes(self) -> int:
        return sum(
            tensor.numel() * tensor.element_size()
            for weights in self._cache.values()
            for tensor in weights
        )

    def clear(self) -> None:
        self._cache.clear()


class W4A16ExpertProvider:
    """Keep every routed expert packed and resident for direct fused execution."""

    def __init__(
        self,
        store: SafetensorIndexStore,
        *,
        device: torch.device | str,
    ) -> None:
        if store.checkpoint_kind is not CheckpointKind.W4A16:
            raise ValueError("W4A16ExpertProvider requires a W4A16 checkpoint")
        bases: list[tuple[int, int, str, tuple[int, int]]] = []
        output_names: list[str] = []
        for layer_idx in range(1, 27):
            for expert_id in range(256):
                prefix = (
                    f"model.layers.{layer_idx}.block_sparse_moe.experts.{expert_id}"
                )
                for projection in ("w1", "w2", "w3"):
                    base = f"{prefix}.{projection}.weight"
                    template = (
                        f"block_sparse_moe.experts.{{expert}}.{projection}.weight"
                    )
                    shape = REAL_EXPERT_TEMPLATE_MANIFEST[template].shape
                    bases.append((layer_idx, expert_id, base, shape))
                    output_names.extend((base + _PACKED_SUFFIX, base + _SCALES_SUFFIX))

        payloads = store.load_many(output_names, device=device)
        grouped: dict[tuple[int, int], list[W4A16Linear]] = defaultdict(list)
        for layer_idx, expert_id, base, shape in bases:
            encoded = W4A16Tensor(
                packed=payloads[base + _PACKED_SUFFIX],
                scales=payloads[base + _SCALES_SUFFIX],
                original_shape=shape,
            )
            grouped[(layer_idx, expert_id)].append(W4A16Linear.from_encoded(encoded))
        self._weights = {
            key: (modules[0], modules[1], modules[2])
            for key, modules in grouped.items()
        }

    def __call__(
        self,
        layer_idx: int,
        expert_id: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[W4A16Linear, W4A16Linear, W4A16Linear]:
        try:
            weights = self._weights[(layer_idx, expert_id)]
        except KeyError as error:
            raise IndexError(
                f"real Kimi-Linear expert does not exist: layer={layer_idx}, "
                f"expert={expert_id}"
            ) from error
        if dtype != torch.bfloat16:
            raise TypeError("W4A16 experts require BF16 activations")
        resident_device = weights[0].packed_weight.device
        if resident_device != device:
            raise ValueError(
                f"W4A16 experts reside on {resident_device}, requested {device}"
            )
        return weights

    @property
    def resident_bytes(self) -> int:
        return sum(
            module.resident_bytes
            for weights in self._weights.values()
            for module in weights
        )


class W3A16ExpertProvider:
    """Keep every routed INT3 expert packed and resident for fused execution."""

    def __init__(
        self,
        store: SafetensorIndexStore,
        *,
        device: torch.device | str,
    ) -> None:
        if store.checkpoint_kind is not CheckpointKind.W3A16:
            raise ValueError("W3A16ExpertProvider requires a W3A16 checkpoint")
        group_size = store.quantization_group_size
        if group_size is None:
            raise RuntimeError("W3A16 checkpoint group size is unavailable")
        bases: list[tuple[int, int, str, tuple[int, int]]] = []
        output_names: list[str] = []
        for layer_idx in range(1, 27):
            for expert_id in range(256):
                prefix = (
                    f"model.layers.{layer_idx}.block_sparse_moe.experts.{expert_id}"
                )
                for projection in ("w1", "w2", "w3"):
                    base = f"{prefix}.{projection}.weight"
                    template = (
                        f"block_sparse_moe.experts.{{expert}}.{projection}.weight"
                    )
                    shape = REAL_EXPERT_TEMPLATE_MANIFEST[template].shape
                    bases.append((layer_idx, expert_id, base, shape))
                    output_names.extend(
                        (
                            base + _W3A16_PACKED_SUFFIX,
                            base + _W3A16_SCALES_SUFFIX,
                        )
                    )

        payloads = store.load_many(output_names, device=device)
        grouped: dict[tuple[int, int], list[W3A16Linear]] = defaultdict(list)
        for layer_idx, expert_id, base, shape in bases:
            encoded = W3A16Tensor(
                packed=payloads[base + _W3A16_PACKED_SUFFIX],
                scales=payloads[base + _W3A16_SCALES_SUFFIX],
                original_shape=shape,
                group_size=group_size,
            )
            grouped[(layer_idx, expert_id)].append(W3A16Linear.from_encoded(encoded))
        self._weights = {
            key: (modules[0], modules[1], modules[2])
            for key, modules in grouped.items()
        }

    def __call__(
        self,
        layer_idx: int,
        expert_id: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[W3A16Linear, W3A16Linear, W3A16Linear]:
        try:
            weights = self._weights[(layer_idx, expert_id)]
        except KeyError as error:
            raise IndexError(
                f"real Kimi-Linear expert does not exist: layer={layer_idx}, "
                f"expert={expert_id}"
            ) from error
        if dtype != torch.bfloat16:
            raise TypeError("W3A16 experts require BF16 activations")
        resident_device = weights[0].packed_weight.device
        if resident_device != device:
            raise ValueError(
                f"W3A16 experts reside on {resident_device}, requested {device}"
            )
        return weights

    @property
    def resident_bytes(self) -> int:
        return sum(
            module.resident_bytes
            for weights in self._weights.values()
            for module in weights
        )
