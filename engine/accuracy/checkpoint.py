"""Build a BF16 checkpoint carrying only the existing W4A16 codec loss."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import shutil
import struct
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

EXPECTED_MODEL_ID = "moonshotai/Kimi-Linear-48B-A3B-Instruct"
EXPECTED_SOURCE_BYTES = 98_253_585_147
EXPECTED_SHARD_COUNT = 20
CHECKPOINT_MANIFEST = "accuracy_checkpoint_manifest.json"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_json(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _tree_files(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*") if path.is_file())


def _tree_size(root: Path) -> int:
    return sum(path.stat().st_size for path in _tree_files(root))


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return value


def _read_safetensors_header(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        length_bytes = handle.read(8)
        if len(length_bytes) != 8:
            raise ValueError(f"{path} is too short to be a safetensors file")
        (header_length,) = struct.unpack("<Q", length_bytes)
        header = handle.read(header_length)
    value = json.loads(header)
    if not isinstance(value, dict):
        raise TypeError(f"{path} safetensors header must be an object")
    return value


def _text_config(config: dict[str, Any]) -> dict[str, Any]:
    nested = config.get("text_config")
    return nested if isinstance(nested, dict) else config


def _validate_model_contract(config: dict[str, Any]) -> dict[str, int]:
    text = _text_config(config)
    facts = {
        "num_hidden_layers": int(text.get("num_hidden_layers", -1)),
        "hidden_size": int(text.get("hidden_size", -1)),
        "num_experts": int(text.get("num_experts", -1)),
        "num_experts_per_token": int(text.get("num_experts_per_token", -1)),
        "num_shared_experts": int(text.get("num_shared_experts", -1)),
        "moe_intermediate_size": int(text.get("moe_intermediate_size", -1)),
        "first_k_dense_replace": int(text.get("first_k_dense_replace", -1)),
        "vocab_size": int(text.get("vocab_size", -1)),
    }
    expected = {
        "num_hidden_layers": 27,
        "hidden_size": 2304,
        "num_experts": 256,
        "num_experts_per_token": 8,
        "num_shared_experts": 1,
        "moe_intermediate_size": 1024,
        "first_k_dense_replace": 1,
        "vocab_size": 163840,
    }
    if facts != expected:
        raise ValueError(f"checkpoint config changed: expected {expected}, got {facts}")

    linear = text.get("linear_attn_config")
    if not isinstance(linear, dict):
        raise ValueError("config.json is missing linear_attn_config")
    kda_layers = linear.get("kda_layers")
    full_layers = linear.get("full_attn_layers")
    if not isinstance(kda_layers, list) or not isinstance(full_layers, list):
        raise ValueError("linear_attn_config must list KDA and full-attention layers")
    if len(kda_layers) != 20 or len(full_layers) != 7:
        raise ValueError(
            f"expected 20 KDA and 7 full-attention layers, got {len(kda_layers)} and {len(full_layers)}"
        )
    return facts | {"kda_layer_count": 20, "full_attention_layer_count": 7}


def _tensor_specs(
    source_dir: Path,
    index: dict[str, Any],
):
    from engine.accuracy.plan import TensorSpec

    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError("model.safetensors.index.json has no weight_map")
    by_shard: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for shard_name in sorted(set(str(value) for value in weight_map.values())):
        header = _read_safetensors_header(source_dir / shard_name)
        for name, metadata in header.items():
            if name == "__metadata__":
                continue
            if not isinstance(metadata, dict):
                raise TypeError(f"invalid safetensors entry for {name}")
            by_shard[shard_name][name] = metadata

    specs = []
    for name, shard_value in sorted(weight_map.items()):
        shard = str(shard_value)
        try:
            metadata = by_shard[shard][name]
        except KeyError as exc:
            raise ValueError(f"index points to missing tensor {name} in {shard}") from exc
        specs.append(
            TensorSpec(
                name=name,
                shard=shard,
                shape=tuple(int(value) for value in metadata["shape"]),
                dtype=str(metadata["dtype"]),
            )
        )
    header_names = {name for shard in by_shard.values() for name in shard}
    extra = sorted(header_names - set(weight_map))
    if extra:
        raise ValueError(f"safetensors shards contain tensors absent from the index: {extra[:10]}")
    return tuple(specs)


def _copy_auxiliary_files(source_dir: Path, target_dir: Path) -> None:
    for source in _tree_files(source_dir):
        if source.suffix == ".safetensors":
            continue
        relative = source.relative_to(source_dir)
        target = target_dir / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def _tensor_sha256(tensor) -> str:
    import torch

    value = tensor.detach().cpu().contiguous().view(torch.uint8)
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()


def _validate_existing_checkpoint(
    output_dir: Path,
    *,
    plan_digest: str,
    source_index_sha256: str,
) -> dict[str, Any] | None:
    manifest_path = output_dir / CHECKPOINT_MANIFEST
    if not manifest_path.exists():
        return None
    manifest = _read_json(manifest_path)
    if not manifest.get("complete"):
        raise ValueError(f"existing checkpoint manifest at {manifest_path} is incomplete")
    if manifest.get("plan", {}).get("digest") != plan_digest:
        raise ValueError("existing checkpoint plan digest does not match the resolved plan")
    if manifest.get("source", {}).get("index_sha256") != source_index_sha256:
        raise ValueError("existing checkpoint source index digest does not match")
    for shard in manifest.get("dequantized_checkpoint", {}).get("shards", []):
        path = output_dir / shard["name"]
        if not path.is_file() or path.stat().st_size != shard["bytes"]:
            raise ValueError(f"existing dequantized shard is missing or truncated: {path}")
    return manifest


def build_dequantized_checkpoint(source_dir: Path, output_root: Path) -> dict[str, Any]:
    """Quantize and immediately dequantize every selected tensor shard by shard."""
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    from engine.accuracy.plan import is_router_tensor, resolve_plan
    from engine.quant.plan import w4a16_storage_bytes
    from engine.quant.verify import verify_round_trip
    from engine.quant.w4a16 import GROUP_SIZE, dequantise, quantise

    source_dir = source_dir.resolve()
    output_root = output_root.resolve()
    if not source_dir.is_dir():
        raise FileNotFoundError(source_dir)
    index_path = source_dir / "model.safetensors.index.json"
    config_path = source_dir / "config.json"
    if not index_path.is_file() or not config_path.is_file():
        raise FileNotFoundError("source checkpoint must contain config.json and the safetensors index")

    source_bytes = _tree_size(source_dir)
    if source_bytes != EXPECTED_SOURCE_BYTES:
        raise ValueError(
            f"source checkpoint byte count changed: expected {EXPECTED_SOURCE_BYTES}, got {source_bytes}"
        )
    index = _read_json(index_path)
    config = _read_json(config_path)
    model_facts = _validate_model_contract(config)
    shard_names = sorted(set(str(value) for value in index["weight_map"].values()))
    if len(shard_names) != EXPECTED_SHARD_COUNT:
        raise ValueError(
            f"expected {EXPECTED_SHARD_COUNT} safetensors shards, got {len(shard_names)}"
        )
    actual_shards = sorted(
        path.relative_to(source_dir).as_posix()
        for path in source_dir.rglob("*.safetensors")
        if path.is_file()
    )
    if actual_shards != shard_names:
        raise ValueError(
            "source safetensors files do not exactly match the checkpoint index: "
            f"indexed={shard_names}, actual={actual_shards}"
        )
    if not any(path.suffix == ".py" for path in source_dir.iterdir() if path.is_file()):
        raise ValueError("source checkpoint is missing its remote model code")

    specs = _tensor_specs(source_dir, index)
    plan = resolve_plan(
        specs,
        source_dir=source_dir,
        config=config,
        index=index,
    )
    decisions = {item.name: item for item in plan.decisions}
    output_dir = output_root / (
        f"Kimi-Linear-48B-A3B-Instruct-w4a16-dequant-{plan.digest[:12]}"
    )
    source_index_sha256 = _sha256_file(index_path)
    existing = _validate_existing_checkpoint(
        output_dir,
        plan_digest=plan.digest,
        source_index_sha256=source_index_sha256,
    )
    if existing is not None:
        return existing

    output_root.mkdir(parents=True, exist_ok=True)
    temporary = output_root / f".{output_dir.name}.partial-{uuid.uuid4().hex}"
    temporary.mkdir(parents=False, exist_ok=False)
    _copy_auxiliary_files(source_dir, temporary)

    selected_records = []
    selected_class_totals: dict[str, dict[str, int]] = defaultdict(
        lambda: {"tensors": 0, "bf16_bytes": 0, "w4a16_bytes": 0}
    )
    router_source_hashes: dict[str, str] = {}
    router_output_hashes: dict[str, str] = {}
    shard_records = []
    try:
        with torch.inference_mode():
            for shard_name in shard_names:
                source_shard = source_dir / shard_name
                output_shard = temporary / shard_name
                output_shard.parent.mkdir(parents=True, exist_ok=True)
                shard_tensors = {}
                with safe_open(source_shard, framework="pt", device="cpu") as source:
                    metadata = source.metadata()
                    source_keys = list(source.keys())
                    for name in source_keys:
                        tensor = source.get_tensor(name).contiguous()
                        decision = decisions[name]
                        if is_router_tensor(name):
                            router_source_hashes[name] = _tensor_sha256(tensor)
                        if not decision.quantize:
                            shard_tensors[name] = tensor
                            continue

                        weight = tensor.to(device="cuda", dtype=torch.bfloat16)
                        encoded = quantise(weight)
                        stats = verify_round_trip(decision.tensor_class, weight, encoded)
                        expected_storage = w4a16_storage_bytes(tuple(weight.shape))
                        if encoded.storage_bytes != expected_storage:
                            raise AssertionError(
                                f"codec storage mismatch for {name}: "
                                f"{encoded.storage_bytes} != {expected_storage}"
                            )
                        restored = dequantise(encoded).cpu().contiguous()
                        shard_tensors[name] = restored
                        bf16_bytes = tensor.numel() * tensor.element_size()
                        totals = selected_class_totals[decision.tensor_class]
                        totals["tensors"] += 1
                        totals["bf16_bytes"] += bf16_bytes
                        totals["w4a16_bytes"] += encoded.storage_bytes
                        selected_records.append(
                            {
                                "name": name,
                                "class": decision.tensor_class,
                                "shape": list(tensor.shape),
                                "bf16_bytes": bf16_bytes,
                                "w4a16_bytes": encoded.storage_bytes,
                                "round_trip": stats.as_dict(),
                            }
                        )
                        del weight, encoded, restored

                save_file(shard_tensors, output_shard, metadata=metadata)
                with safe_open(output_shard, framework="pt", device="cpu") as restored_file:
                    if list(restored_file.keys()) != source_keys:
                        raise ValueError(f"tensor-name drift while writing {shard_name}")
                    for name in source_keys:
                        if is_router_tensor(name):
                            router_output_hashes[name] = _tensor_sha256(
                                restored_file.get_tensor(name)
                            )
                shard_records.append(
                    {
                        "name": shard_name,
                        "source_bytes": source_shard.stat().st_size,
                        "source_sha256": _sha256_file(source_shard),
                        "bytes": output_shard.stat().st_size,
                        "sha256": _sha256_file(output_shard),
                    }
                )
                del shard_tensors
                gc.collect()
                torch.cuda.empty_cache()

        if not router_source_hashes:
            raise ValueError("router-tensor instrument found zero router tensors")
        if router_source_hashes != router_output_hashes:
            changed = sorted(
                name
                for name in set(router_source_hashes) | set(router_output_hashes)
                if router_source_hashes.get(name) != router_output_hashes.get(name)
            )
            raise ValueError(f"router tensors changed during checkpoint rewrite: {changed[:10]}")

        selected_source_bytes = sum(item["bf16_bytes"] for item in selected_records)
        selected_w4a16_bytes = sum(item["w4a16_bytes"] for item in selected_records)
        class_totals = {}
        for tensor_class, values in sorted(selected_class_totals.items()):
            class_totals[tensor_class] = values | {
                "saved_bytes": values["bf16_bytes"] - values["w4a16_bytes"],
                "saving_fraction": 1.0 - values["w4a16_bytes"] / values["bf16_bytes"],
            }

        manifest = {
            "schema_version": "runinfra.kimi_linear.dequantized_checkpoint.v1",
            "created_at": _utc_now(),
            "complete": True,
            "model_id": EXPECTED_MODEL_ID,
            "experimental_design": (
                "Selected BF16 tensors were encoded by engine.quant.w4a16.quantise "
                "and immediately restored by engine.quant.w4a16.dequantise. The saved "
                "checkpoint remains BF16 so both sides load through the same vLLM path."
            ),
            "source": {
                "path": str(source_dir),
                "total_bytes": source_bytes,
                "index_sha256": source_index_sha256,
                "config_sha256": _sha256_file(config_path),
                "model_facts": model_facts,
                "shard_count": len(shard_names),
            },
            "codec": {
                "module": "engine.quant.w4a16",
                "weight_bits": 4,
                "signed_range": [-7, 7],
                "group_size": GROUP_SIZE,
                "group_axis": "reduction_axis",
                "scale_dtype": "BF16",
                "saved_checkpoint_dtype": "BF16",
            },
            "plan": plan.as_dict(include_decisions=True),
            "selected_tensor_summary": {
                "tensor_count": len(selected_records),
                "bf16_bytes": selected_source_bytes,
                "simulated_w4a16_bytes": selected_w4a16_bytes,
                "simulated_saved_bytes": selected_source_bytes - selected_w4a16_bytes,
                "by_class": class_totals,
                "tensors": selected_records,
            },
            "router_preservation": {
                "tensor_count": len(router_source_hashes),
                "all_exact": True,
                "source_tensor_sha256": router_source_hashes,
                "dequantized_tensor_sha256": router_output_hashes,
            },
            "dequantized_checkpoint": {
                "path": str(output_dir),
                "shards": shard_records,
                "shard_manifest_sha256": _sha256_json(shard_records),
            },
        }
        with (temporary / CHECKPOINT_MANIFEST).open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, output_dir)
        return manifest
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--result-json", required=True, type=Path)
    args = parser.parse_args()

    result = build_dequantized_checkpoint(args.source_dir, args.output_root)
    args.result_json.parent.mkdir(parents=True, exist_ok=True)
    with args.result_json.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({"checkpoint": result["dequantized_checkpoint"], "plan": result["plan"]}))


if __name__ == "__main__":
    main()
