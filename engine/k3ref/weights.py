"""Reader for raw tensor files produced by engine.modal_app.fetch_layer."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from .dequant import dequantize_mxfp4
from .manifest import K3_EXPERT_CHECKPOINT_MANIFEST, TensorSpec

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


class RawTensorStore:
    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        if not self.directory.is_dir():
            raise FileNotFoundError(f"raw tensor directory does not exist: {self.directory}")
        self._entries: dict[str, tuple[Path, dict]] = {}
        for meta_path in self.directory.glob("*.bin.meta"):
            with meta_path.open("r", encoding="utf-8") as handle:
                metadata = json.load(handle)
            tensor_path = Path(str(meta_path)[: -len(".meta")])
            if not tensor_path.is_file():
                raise FileNotFoundError(f"tensor file missing for {meta_path.name}")
            self._entries[metadata["name"]] = (tensor_path, metadata)
        if not self._entries:
            raise FileNotFoundError(f"no .bin.meta files found in {self.directory}")

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self._entries)

    def find_name(self, suffix: str) -> str:
        matches = [name for name in self._entries if name.endswith(suffix)]
        if len(matches) != 1:
            raise KeyError(f"expected one tensor ending in {suffix!r}, found {len(matches)}")
        return matches[0]

    def metadata(self, suffix: str) -> dict:
        return self._entries[self.find_name(suffix)][1]

    def validate(self, suffix: str, spec: TensorSpec) -> None:
        metadata = self.metadata(suffix)
        actual_shape = tuple(metadata["shape"])
        actual_dtype = metadata["dtype"]
        if actual_shape != spec.shape or actual_dtype != spec.dtype:
            raise ValueError(
                f"checkpoint manifest mismatch for {suffix}: "
                f"expected {spec.shape} {spec.dtype}, "
                f"got {actual_shape} {actual_dtype}"
            )

    def load(
        self,
        suffix: str,
        *,
        device: torch.device | str = "cpu",
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        name = self.find_name(suffix)
        path, metadata = self._entries[name]
        storage_dtype = _DTYPES[metadata["dtype"]]
        shape = tuple(metadata["shape"])
        elements = 1
        for dimension in shape:
            elements *= dimension
        try:
            tensor = torch.from_file(
                str(path), shared=False, size=elements, dtype=storage_dtype
            ).clone()
        except (RuntimeError, OSError, NotImplementedError):
            # torch.from_file is unavailable on some Windows builds.
            raw = bytearray(path.read_bytes())
            tensor = torch.frombuffer(raw, dtype=storage_dtype, count=elements).clone()
        tensor = tensor.reshape(shape)
        if dtype is not None:
            tensor = tensor.to(dtype=dtype)
        return tensor.to(device=device)


class MXFP4ExpertProvider:
    def __init__(self, store: RawTensorStore, layer_idx: int) -> None:
        self.store = store
        self.layer_idx = layer_idx
        self._cache: dict[
            tuple[int, str, torch.dtype],
            tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        ] = {}

    def __call__(
        self,
        expert_id: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        key = (expert_id, str(device), dtype)
        if key not in self._cache:
            weights = []
            base = (
                f"layers.{self.layer_idx}.block_sparse_moe.experts.{expert_id}"
            )
            for projection in ("w1", "w2", "w3"):
                packed_key = (
                    f"block_sparse_moe.experts.{{expert}}.{projection}.weight_packed"
                )
                scale_key = (
                    f"block_sparse_moe.experts.{{expert}}.{projection}.weight_scale"
                )
                self.store.validate(
                    f"{base}.{projection}.weight_packed",
                    K3_EXPERT_CHECKPOINT_MANIFEST[packed_key],
                )
                self.store.validate(
                    f"{base}.{projection}.weight_scale",
                    K3_EXPERT_CHECKPOINT_MANIFEST[scale_key],
                )
                packed = self.store.load(
                    f"{base}.{projection}.weight_packed", device=device
                )
                scale = self.store.load(
                    f"{base}.{projection}.weight_scale", device=device
                )
                weights.append(dequantize_mxfp4(packed, scale, dtype=dtype))
            self._cache[key] = (weights[0], weights[1], weights[2])
        return self._cache[key]
