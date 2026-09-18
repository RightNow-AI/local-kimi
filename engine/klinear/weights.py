"""Low-level reader and lazy expert cache for sharded safetensors checkpoints."""

from __future__ import annotations

import json
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch

from .manifest import TensorSpec, validate_real_checkpoint_layout

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
        weight_map = payload.get("weight_map")
        if not isinstance(weight_map, Mapping) or not weight_map:
            raise ValueError("safetensors index has no weight_map")
        self._weight_map = dict(weight_map)
        self._locations: dict[str, TensorLocation] = {}

        for shard_name in sorted(set(self._weight_map.values())):
            shard_path = self.directory / shard_name
            if not shard_path.is_file():
                raise FileNotFoundError(f"safetensors shard is missing: {shard_path}")
            with shard_path.open("rb") as handle:
                length_bytes = handle.read(8)
                if len(length_bytes) != 8:
                    raise ValueError(f"safetensors shard has no header length: {shard_path}")
                header_length = int.from_bytes(length_bytes, "little", signed=False)
                header_bytes = handle.read(header_length)
            if len(header_bytes) != header_length:
                raise ValueError(f"safetensors header is truncated: {shard_path}")
            header = json.loads(header_bytes.decode("utf-8"))
            data_base = 8 + header_length
            for name, entry in header.items():
                if name == "__metadata__":
                    continue
                if self._weight_map.get(name) != shard_name:
                    raise ValueError(f"index assigns {name} to a different shard")
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
        if validate_real_layout:
            validate_real_checkpoint_layout(self.specs)

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
        elements = 1
        for dimension in shape:
            elements *= dimension
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
        if str(device) == "meta":
            result_dtype = dtype or _DTYPES[location.spec.dtype]
            return torch.empty(location.spec.shape, device="meta", dtype=result_dtype)
        byte_count = location.data_end - location.data_start
        with location.path.open("rb") as handle:
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


class SafetensorExpertProvider:
    """Load selected experts and retain a bounded device-resident LRU cache."""

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

    def clear(self) -> None:
        self._cache.clear()

