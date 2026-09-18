"""Stable cache keys for exact token prefixes and model configurations."""

from __future__ import annotations

import hashlib
import json
import struct
from collections.abc import Mapping, Sequence
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import torch


ModelFingerprint = str | int | float | bool | None | Mapping[str, Any] | Sequence[Any]


def token_tuple(token_ids: torch.Tensor | Sequence[int]) -> tuple[int, ...]:
    """Return one exact token sequence in a device-independent representation."""

    if isinstance(token_ids, torch.Tensor):
        if token_ids.ndim == 2:
            if token_ids.shape[0] != 1:
                raise ValueError("state-cache prefixes require batch size one")
            token_ids = token_ids[0]
        if token_ids.ndim != 1:
            raise ValueError("token_ids must have shape [sequence] or [1, sequence]")
        if token_ids.numel() == 0:
            raise ValueError("a cached token prefix cannot be empty")
        values = token_ids.detach().to(device="cpu", dtype=torch.int64).tolist()
    else:
        values = list(token_ids)
        if not values:
            raise ValueError("a cached token prefix cannot be empty")
    if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
        raise TypeError("token_ids must contain integers")
    if any(value < -(1 << 63) or value >= (1 << 63) for value in values):
        raise ValueError("token_ids must fit signed 64-bit integers")
    return tuple(values)


def canonical_fingerprint(model_fingerprint: ModelFingerprint) -> bytes:
    """Serialize a fingerprint with stable ordering and no object identity."""

    try:
        payload = json.dumps(
            model_fingerprint,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise TypeError("model_fingerprint must be canonical JSON data") from error
    return payload.encode("utf-8")


def fingerprint_digest(model_fingerprint: ModelFingerprint) -> str:
    """Return the digest used to separate prefix indexes by model identity."""

    return hashlib.sha256(canonical_fingerprint(model_fingerprint)).hexdigest()


def prefix_key(
    token_ids: torch.Tensor | Sequence[int],
    model_fingerprint: ModelFingerprint,
) -> str:
    """Hash exact token ids together with an exact model fingerprint.

    Prompt text is deliberately absent. Model state depends on token ids, and
    two distinct strings that tokenize identically must share one cache key.
    """

    tokens = token_tuple(token_ids)
    digest = hashlib.sha256()
    digest.update(b"klinear-statecache-v1\0")
    fingerprint = canonical_fingerprint(model_fingerprint)
    digest.update(struct.pack("<Q", len(fingerprint)))
    digest.update(fingerprint)
    digest.update(struct.pack("<Q", len(tokens)))
    for token_id in tokens:
        digest.update(struct.pack("<q", token_id))
    return digest.hexdigest()


def _checkpoint_index_sha256(directory: Path) -> str | None:
    index_path = directory / "model.safetensors.index.json"
    if not index_path.is_file():
        return None
    return hashlib.sha256(index_path.read_bytes()).hexdigest()


def _active_kernel_variants() -> dict[str, str]:
    try:
        from engine.kernels import (
            W4A16_DENSE,
            W4A16_GROUPED,
            W4A16_SWIGLU,
            registry,
        )
    except ImportError:
        return {}

    names: dict[str, str] = {}
    for operation in (
        W4A16_GROUPED,
        W4A16_DENSE,
        W4A16_SWIGLU,
    ):
        try:
            names[operation] = registry.active(operation)
        except KeyError:
            continue
    return names


def fingerprint_model(model: object) -> dict[str, Any]:
    """Build the model and kernel identity used by ``warm_prefill``.

    Loaded checkpoints include the resolved directory, directory name, index
    digest, checkpoint kind, resident weight byte count, checkpoint tensor byte
    count, model configuration, a process-local model identity, and every active
    registered kernel variant. The resolved directory and index digest make two
    same-sized checkpoints in different artifacts distinct. The process-local
    identity prevents two separately loaded model objects from sharing state
    even if an artifact was overwritten without changing its index or size.

    Callers with a release artifact digest may pass their own fingerprint to
    ``warm_prefill``. That is the strongest identity when checkpoint files can
    be overwritten in place without changing their index.
    """

    store = getattr(model, "_weight_store", None)
    raw_directory = getattr(store, "directory", None)
    directory = Path(raw_directory).resolve() if raw_directory is not None else None
    config = getattr(model, "config", None)
    if is_dataclass(config):
        config_payload: Any = asdict(config)
    elif config is None:
        config_payload = None
    else:
        config_payload = repr(config)

    resident = getattr(model, "resident_weight_bytes", None)
    if callable(resident):
        resident = resident()
    resident_identity = getattr(model, "_statecache_resident_weight_bytes", None)
    if resident_identity is None:
        resident_identity = resident
        try:
            setattr(model, "_statecache_resident_weight_bytes", resident_identity)
        except (AttributeError, TypeError):
            pass
    checkpoint_storage = getattr(model, "checkpoint_tensor_storage_bytes", None)
    checkpoint_kind = getattr(model, "checkpoint_kind", None)

    payload: dict[str, Any] = {
        "schema": "klinear-model-fingerprint-v1",
        "checkpoint_directory": str(directory) if directory is not None else None,
        "checkpoint_directory_name": directory.name if directory is not None else None,
        "checkpoint_index_sha256": (
            _checkpoint_index_sha256(directory) if directory is not None else None
        ),
        "checkpoint_kind": checkpoint_kind,
        "checkpoint_tensor_storage_bytes": checkpoint_storage,
        "resident_weight_bytes": resident_identity,
        "lm_head_quantized": getattr(model, "_lm_head_quantized", None),
        "model_instance_identity": id(model),
        "config": config_payload,
        "kernel_variants": _active_kernel_variants(),
    }
    return payload
