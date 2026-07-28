"""Runtime inspection of immutable safetensors weight artifacts."""

from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path
from typing import Any


def _safetensors_payload_bytes(path: Path) -> int:
    with path.open("rb") as handle:
        raw_length = handle.read(8)
        if len(raw_length) != 8:
            raise ValueError(f"{path} is too short to be a safetensors file")
        header_length = struct.unpack("<Q", raw_length)[0]
        header_raw = handle.read(header_length)
    if len(header_raw) != header_length:
        raise ValueError(f"{path} has a truncated safetensors header")
    header = json.loads(header_raw)
    total = 0
    for name, metadata in header.items():
        if name == "__metadata__":
            continue
        offsets = metadata.get("data_offsets") if isinstance(metadata, dict) else None
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or not all(isinstance(offset, int) for offset in offsets)
            or offsets[0] < 0
            or offsets[1] < offsets[0]
        ):
            raise ValueError(f"{path} tensor {name!r} has invalid data offsets")
        total += offsets[1] - offsets[0]
    return total


def _sha256_files(root: Path, paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        encoded_name = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(encoded_name).to_bytes(4, "big"))
        digest.update(encoded_name)
        digest.update(path.stat().st_size.to_bytes(8, "big"))
        with path.open("rb") as handle:
            while chunk := handle.read(8 * 1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest()


def inspect_weight_artifact(path: str | Path, *, compute_digest: bool) -> dict[str, Any]:
    """Measure exact tensor payload bytes and optionally hash all weight files."""

    root = Path(path).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"weight artifact directory does not exist: {root}")
    files = sorted(root.rglob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"weight artifact has no safetensors files: {root}")
    payload_bytes = sum(_safetensors_payload_bytes(file) for file in files)
    storage_bytes = sum(file.stat().st_size for file in files)
    result: dict[str, Any] = {
        "path": str(root),
        "file_count": len(files),
        "storage_bytes": storage_bytes,
        "weights_resident_bytes": payload_bytes,
        "measurement_method": "sum of safetensors tensor data_offsets at runtime",
        "residency_contract": (
            "the runtime command must load this complete artifact without CPU or disk offload"
        ),
    }
    if compute_digest:
        result["weights_digest_sha256"] = _sha256_files(root, files)
        result["digest_method"] = "sha256 over ordered file names, sizes, and complete bytes"
    else:
        result["weights_digest_sha256"] = None
        result["digest_method"] = "not computed because an immutable model revision is recorded"
    return result
