"""Offline Kimi checkpoint adaptation plan for ``engine.k3ref``.

This module reads only config JSON and safetensors headers. It does not load
tensor payloads. The returned requirements separate manifest generalisation
from module-shape changes that still need implementation outside this lane.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from engine.k3ref.config import K3LayerConfig
from engine.k3ref.manifest import (
    K3_EXPERT_CHECKPOINT_MANIFEST,
    K3_LAYER_TENSOR_MANIFEST,
    TensorSpec,
    expert_checkpoint_manifest,
    expert_runtime_manifest,
    layer_tensor_manifest,
    runtime_parameter_manifest,
)


@dataclass(frozen=True)
class K3RefAdaptation:
    config: K3LayerConfig
    layer_idx: int
    layer_manifest: dict[str, TensorSpec]
    expert_checkpoint_manifest: dict[str, TensorSpec]
    runtime_manifest: dict[str, TensorSpec]
    required_engine_changes: tuple[str, ...]

    @property
    def existing_loader_is_compatible(self) -> bool:
        return not self.required_engine_changes


def read_safetensors_header(path: str | Path) -> dict[str, Any]:
    """Read the JSON header without mapping or reading tensor payload bytes."""
    source = Path(path)
    with source.open("rb") as handle:
        length_bytes = handle.read(8)
        if len(length_bytes) != 8:
            raise ValueError(f"safetensors file has no complete header length: {source}")
        header_length = int.from_bytes(length_bytes, byteorder="little", signed=False)
        if header_length <= 0:
            raise ValueError(f"safetensors file has an invalid header length: {source}")
        header_bytes = handle.read(header_length)
    if len(header_bytes) != header_length:
        raise ValueError(f"safetensors file has a truncated header: {source}")
    payload = json.loads(header_bytes.decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"safetensors header is not an object: {source}")
    return payload


def merge_safetensors_headers(paths: Iterable[str | Path]) -> dict[str, Any]:
    """Merge shard headers while rejecting duplicate tensor ownership."""
    merged: dict[str, Any] = {}
    metadata: dict[str, Any] = {}
    for path in paths:
        header = read_safetensors_header(path)
        shard_metadata = header.get("__metadata__")
        if isinstance(shard_metadata, Mapping):
            metadata.update(shard_metadata)
        for name, entry in header.items():
            if name == "__metadata__":
                continue
            if name in merged:
                raise ValueError(f"tensor appears in more than one safetensors shard: {name}")
            merged[name] = entry
    if metadata:
        merged["__metadata__"] = metadata
    if not merged:
        raise ValueError("no safetensors headers were provided")
    return merged


def _required_engine_changes(
    config: K3LayerConfig,
    layer_idx: int,
    layer_manifest: Mapping[str, TensorSpec],
    checkpoint_experts: Mapping[str, TensorSpec],
) -> tuple[str, ...]:
    requirements: list[str] = []
    if dict(layer_manifest) != K3_LAYER_TENSOR_MANIFEST:
        requirements.append(
            "parameterize K3ReferenceLayer._load_raw_weights with this model-specific "
            "manifest instead of the global K3 layer-12 manifest"
        )

    if config.is_kda_layer(layer_idx):
        if not config.kda_use_full_rank_gate:
            requirements.append(
                "support the checkpoint's low-rank KDA g_a_proj/g_b_proj output gate"
            )
        a_log = layer_manifest.get("self_attn.A_log")
        if a_log is not None and a_log.shape != (config.kda_head_dim,):
            requirements.append(
                "support the checkpoint's per-head KDA A_log axis instead of K3's "
                "per-head-dimension axis"
            )
    if config.q_lora_rank is None and config.full_attention_layers:
        requirements.append(
            "support direct MLA q_proj weights when q_lora_rank is null"
        )

    if config.has_moe_layer(layer_idx) and config.routed_expert_hidden_size is None:
        requirements.append(
            "support non-latent routed experts without K3's down projection, latent norm, "
            "and up projection"
        )
    if (
        config.has_moe_layer(layer_idx)
        and dict(checkpoint_experts) != K3_EXPERT_CHECKPOINT_MANIFEST
    ):
        requirements.append(
            "parameterize the expert provider for this checkpoint format; Kimi-Linear "
            "stores ordinary expert weights rather than K3's fixed MXFP4 packed tensors"
        )
    return tuple(requirements)


def adapt_from_header(
    config_payload: Mapping[str, Any],
    safetensors_header: Mapping[str, Any],
    layer_idx: int,
    *,
    expert_id: int = 0,
) -> K3RefAdaptation:
    """Create the model-specific manifests and enumerate remaining loader work."""
    config = K3LayerConfig.from_mapping(config_payload)
    layer_manifest = layer_tensor_manifest(config, safetensors_header, layer_idx)
    checkpoint_experts: dict[str, TensorSpec] = {}
    runtime_experts: dict[str, TensorSpec] = {}
    if config.has_moe_layer(layer_idx):
        checkpoint_experts = expert_checkpoint_manifest(
            config,
            safetensors_header,
            layer_idx,
            expert_id,
        )
        runtime_experts = expert_runtime_manifest(config)
    runtime = runtime_parameter_manifest(
        config.num_experts if runtime_experts else 0,
        layer_manifest=layer_manifest,
        expert_manifest=runtime_experts,
    )
    requirements = _required_engine_changes(
        config,
        layer_idx,
        layer_manifest,
        checkpoint_experts,
    )
    return K3RefAdaptation(
        config=config,
        layer_idx=layer_idx,
        layer_manifest=layer_manifest,
        expert_checkpoint_manifest=checkpoint_experts,
        runtime_manifest=runtime,
        required_engine_changes=requirements,
    )


def adapt_from_files(
    config_path: str | Path,
    safetensors_paths: Iterable[str | Path],
    layer_idx: int,
    *,
    expert_id: int = 0,
) -> K3RefAdaptation:
    """Load config plus shard headers and build an offline adaptation report."""
    payload = json.loads(Path(config_path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("model config must be a JSON object")
    header = merge_safetensors_headers(safetensors_paths)
    return adapt_from_header(payload, header, layer_idx, expert_id=expert_id)
