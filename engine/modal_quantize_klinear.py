"""Build and verify a real selective W4A16 Kimi-Linear checkpoint on Modal.

The source checkpoint is read only from the existing ``kimi-linear-weights``
volume. The output is written to a profile-specific directory on
``kimi-linear-quantized`` as safetensors plus an index and a per-tensor
evidence manifest. The default profile keeps the existing output path.

    modal run engine/modal_quantize_klinear.py
    modal run engine/modal_quantize_klinear.py --profile shared-experts-bf16
"""

from __future__ import annotations

import json
from pathlib import Path

import modal

app = modal.App("kimi-linear-w4a16-checkpoint")

SOURCE_VOLUME = modal.Volume.from_name("kimi-linear-weights", create_if_missing=False)
OUTPUT_VOLUME = modal.Volume.from_name("kimi-linear-quantized", create_if_missing=True)
SOURCE_MOUNT = "/source"
OUTPUT_MOUNT = "/output"
MODEL_NAME = "Kimi-Linear-48B-A3B-Instruct"
SOURCE_DIR = f"{SOURCE_MOUNT}/{MODEL_NAME}"
MANIFEST_NAME = "quantization-manifest.json"

# numpy is explicit. torch does not pull it, and without it torch degrades
# quietly at import ("Failed to initialize NumPy") and then dies later on the
# first conversion, far from the real cause.
IMAGE = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch>=2.5", "safetensors>=0.4.5", "numpy>=2.0")
    .add_local_dir(Path(__file__).parent, remote_path="/root/engine")
)


@app.function(
    image=IMAGE,
    gpu="H100",
    cpu=16.0,
    memory=65536,
    timeout=60 * 60 * 24,
    volumes={SOURCE_MOUNT: SOURCE_VOLUME, OUTPUT_MOUNT: OUTPUT_VOLUME},
)
def quantize_checkpoint(overwrite: bool = False, profile: str = "default") -> dict:
    import gc
    import math
    import os
    import re
    import shutil
    import uuid
    from collections import Counter, defaultdict
    from math import prod

    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    from engine.quant.klinear_plan import (
        DEFAULT_PROFILE_NAME,
        RESULTS_PROJECTION_BYTES,
        TensorMetadata,
        build_klinear_quantization_plan,
        get_klinear_quantization_profile,
    )
    from engine.quant.verify import (
        VerificationError,
        swapped_nibble_dequantise,
        verify_dequantizer,
        verify_round_trip,
        wrong_group_axis_dequantise,
        wrong_scale_dequantise,
    )
    from engine.quant.w4a16 import GROUP_SIZE, W4A16Tensor, dequantise, quantise

    selected_profile = get_klinear_quantization_profile(profile)
    output_directory_name = f"{MODEL_NAME}-W4A16"
    if selected_profile.name != DEFAULT_PROFILE_NAME:
        output_directory_name = f"{output_directory_name}-{selected_profile.name}"
    output_dir = f"{OUTPUT_MOUNT}/{output_directory_name}"

    if not os.path.isdir(SOURCE_DIR):
        raise FileNotFoundError(
            f"{SOURCE_DIR} is missing from the kimi-linear-weights volume"
        )
    source_index_path = os.path.join(SOURCE_DIR, "model.safetensors.index.json")
    if not os.path.isfile(source_index_path):
        raise FileNotFoundError(f"source checkpoint index is missing: {source_index_path}")
    config_path = os.path.join(SOURCE_DIR, "config.json")
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"source checkpoint config is missing: {config_path}")

    with open(config_path, encoding="utf-8") as handle:
        config = json.load(handle)
    with open(source_index_path, encoding="utf-8") as handle:
        source_index = json.load(handle)
    weight_map = source_index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError("source model.safetensors.index.json has no weight_map")
    if not all(
        isinstance(name, str) and name and isinstance(shard, str) and shard
        for name, shard in weight_map.items()
    ):
        raise ValueError("source weight_map must contain non-empty string pairs")

    dtype_bytes = {
        "BOOL": 1,
        "BF16": 2,
        "F16": 2,
        "F32": 4,
        "F64": 8,
        "I8": 1,
        "I16": 2,
        "I32": 4,
        "I64": 8,
        "U8": 1,
        "U16": 2,
        "U32": 4,
        "U64": 8,
    }
    torch_dtypes = {
        torch.bool: "BOOL",
        torch.bfloat16: "BF16",
        torch.float16: "F16",
        torch.float32: "F32",
        torch.float64: "F64",
        torch.int8: "I8",
        torch.int16: "I16",
        torch.int32: "I32",
        torch.int64: "I64",
        torch.uint8: "U8",
    }

    def read_header(path: str) -> dict:
        file_size = os.path.getsize(path)
        with open(path, "rb") as handle:
            raw_length = handle.read(8)
            if len(raw_length) != 8:
                raise ValueError(f"safetensors file has no complete header length: {path}")
            header_length = int.from_bytes(raw_length, byteorder="little", signed=False)
            if header_length <= 0 or header_length > file_size - 8:
                raise ValueError(
                    f"safetensors header length {header_length} is invalid for {path}"
                )
            raw_header = handle.read(header_length)
        try:
            header = json.loads(raw_header.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"safetensors header is invalid JSON: {path}") from exc
        if not isinstance(header, dict):
            raise ValueError(f"safetensors header is not an object: {path}")
        return header

    names_by_shard: dict[str, list[str]] = defaultdict(list)
    for tensor_name, shard_name in weight_map.items():
        names_by_shard[shard_name].append(tensor_name)

    tensor_metadata = []
    source_safetensors_file_bytes = 0
    for shard_name, mapped_names in sorted(names_by_shard.items()):
        shard_path = os.path.join(SOURCE_DIR, shard_name)
        if not os.path.isfile(shard_path):
            raise FileNotFoundError(f"index references missing shard {shard_path}")
        source_safetensors_file_bytes += os.path.getsize(shard_path)
        header = read_header(shard_path)
        header_names = {name for name in header if name != "__metadata__"}
        mapped_set = set(mapped_names)
        if header_names != mapped_set:
            missing = sorted(mapped_set - header_names)
            extra = sorted(header_names - mapped_set)
            raise ValueError(
                f"source index and shard {shard_name} disagree: "
                f"missing={missing[:10]}, extra={extra[:10]}"
            )
        for tensor_name in sorted(mapped_names):
            entry = header[tensor_name]
            if not isinstance(entry, dict):
                raise ValueError(f"invalid header entry for {tensor_name}")
            shape = entry.get("shape")
            dtype = entry.get("dtype")
            offsets = entry.get("data_offsets")
            if not isinstance(shape, list) or not all(
                isinstance(dimension, int) and dimension > 0 for dimension in shape
            ):
                raise ValueError(f"invalid shape in header for {tensor_name}: {shape}")
            if dtype not in dtype_bytes:
                raise ValueError(f"unsupported dtype in header for {tensor_name}: {dtype}")
            if (
                not isinstance(offsets, list)
                or len(offsets) != 2
                or not all(isinstance(offset, int) for offset in offsets)
                or offsets[0] < 0
                or offsets[1] < offsets[0]
            ):
                raise ValueError(f"invalid data offsets for {tensor_name}: {offsets}")
            span = offsets[1] - offsets[0]
            calculated = prod(shape) * dtype_bytes[dtype]
            if span != calculated:
                raise ValueError(
                    f"header byte span disagrees with shape for {tensor_name}: "
                    f"span={span}, calculated={calculated}"
                )
            tensor_metadata.append(
                TensorMetadata(
                    name=tensor_name,
                    shape=tuple(shape),
                    dtype=dtype,
                    source_file=shard_name,
                )
            )

    plan = build_klinear_quantization_plan(
        tensor_metadata,
        profile=selected_profile.name,
    )
    default_plan = (
        plan
        if selected_profile.name == DEFAULT_PROFILE_NAME
        else build_klinear_quantization_plan(
            tensor_metadata,
            profile=DEFAULT_PROFILE_NAME,
        )
    )
    profile_delta_bytes = plan.planned_bytes - default_plan.planned_bytes
    decisions = {decision.name: decision for decision in plan.tensors}
    if plan.quantized_tensor_count == 0:
        raise ValueError("the real checkpoint plan selected no tensors for W4A16")
    source_metadata = source_index.get("metadata")
    if isinstance(source_metadata, dict) and source_metadata.get("total_size") is not None:
        declared_total_size = source_metadata["total_size"]
        if declared_total_size != plan.original_bytes:
            raise ValueError(
                "source index total_size disagrees with its real tensor headers: "
                f"index={declared_total_size}, headers={plan.original_bytes}"
            )

    def validate_policy_coverage() -> dict:
        class_counts = Counter(decision.tensor_class for decision in plan.tensors)
        required_classes = {
            "attention projections",
            "dense layer 0 MLP projections",
            "KDA gates, state, and convolutions",
            "lm head",
            "router gate",
            "routed expert projections",
            "shared expert projections",
            "token embedding",
        }
        missing_classes = sorted(required_classes - set(class_counts))
        if missing_classes:
            raise ValueError(
                f"checkpoint plan is missing required policy classes: {missing_classes}"
            )
        unclassified_matrices = [
            decision.name
            for decision in plan.tensors
            if decision.tensor_class == "other checkpoint matrices"
        ]
        if unclassified_matrices:
            raise ValueError(
                "checkpoint contains matrix weights without an explicit policy: "
                f"{unclassified_matrices[:20]}"
            )

        num_layers = int(config["num_hidden_layers"])
        first_dense = int(config.get("first_k_dense_replace", 0))
        moe_frequency = int(config.get("moe_layer_freq", 1))
        num_experts = int(config["num_experts"])
        expected_moe_layers = [
            layer
            for layer in range(num_layers)
            if layer >= first_dense and layer % moe_frequency == 0
        ]
        layer_pattern = re.compile(r"(?:^|\.)layers\.(\d+)\.")

        def layers_for_class(tensor_class: str) -> list[int]:
            layers = set()
            for decision in plan.tensors:
                if decision.tensor_class != tensor_class:
                    continue
                match = layer_pattern.search(decision.name)
                if match:
                    layers.add(int(match.group(1)))
            return sorted(layers)

        attention_layers = layers_for_class("attention projections")
        if attention_layers != list(range(num_layers)):
            raise ValueError(
                "attention projection coverage does not include every layer: "
                f"found={attention_layers}, expected={list(range(num_layers))}"
            )
        dense_layer_decisions = [
            decision
            for decision in plan.tensors
            if decision.tensor_class == "dense layer 0 MLP projections"
        ]
        if len(dense_layer_decisions) < 2 or layers_for_class(
            "dense layer 0 MLP projections"
        ) != [0]:
            raise ValueError("the layer 0 dense MLP matrix bank is incomplete")
        for tensor_class in ("router gate", "shared expert projections"):
            covered_layers = layers_for_class(tensor_class)
            if covered_layers != expected_moe_layers:
                raise ValueError(
                    f"{tensor_class} coverage does not match configured MoE layers: "
                    f"found={covered_layers}, expected={expected_moe_layers}"
                )

        explicit = defaultdict(lambda: defaultdict(set))
        expert_pattern = re.compile(r"(?:^|\.)layers\.(\d+)\..*\.experts\.(\d+)\.")
        for decision in plan.tensors:
            if decision.tensor_class != "routed expert projections":
                continue
            match = expert_pattern.search(decision.name)
            if match:
                layer = int(match.group(1))
                expert = int(match.group(2))
                explicit[layer][expert].add(decision.name)
        if explicit:
            if sorted(explicit) != expected_moe_layers:
                raise ValueError(
                    "routed expert coverage does not match configured MoE layers: "
                    f"found={sorted(explicit)}, expected={expected_moe_layers}"
                )
            expected_ids = set(range(num_experts))
            projection_counts = set()
            for layer in expected_moe_layers:
                found_ids = set(explicit[layer])
                if found_ids != expected_ids:
                    missing_ids = sorted(expected_ids - found_ids)
                    extra_ids = sorted(found_ids - expected_ids)
                    raise ValueError(
                        f"layer {layer} routed expert coverage mismatch: "
                        f"missing={missing_ids[:10]}, extra={extra_ids[:10]}"
                    )
                projection_counts.update(
                    len(explicit[layer][expert]) for expert in sorted(expected_ids)
                )
            if len(projection_counts) != 1 or next(iter(projection_counts)) < 2:
                raise ValueError(
                    "routed experts do not expose a consistent complete matrix bank: "
                    f"counts={sorted(projection_counts)}"
                )
        else:
            fused_layers = set()
            for decision in plan.tensors:
                if decision.tensor_class != "routed expert projections":
                    continue
                match = re.search(r"(?:^|\.)layers\.(\d+)\.", decision.name)
                if match and num_experts in decision.shape[:-1]:
                    fused_layers.add(int(match.group(1)))
            if sorted(fused_layers) != expected_moe_layers:
                raise ValueError(
                    "could not prove all configured routed expert banks from the index: "
                    f"found={sorted(fused_layers)}, expected={expected_moe_layers}"
                )
        return {
            "class_counts": dict(sorted(class_counts.items())),
            "configured_moe_layers": expected_moe_layers,
            "configured_experts_per_layer": num_experts,
            "explicit_per_expert_layout": bool(explicit),
        }

    policy_coverage = validate_policy_coverage()

    staging_dir = f"{output_dir}.staging-{uuid.uuid4().hex}"
    output_exists = os.path.exists(output_dir)
    if output_exists and not overwrite:
        raise FileExistsError(
            f"{output_dir} already exists; pass overwrite=True to replace it"
        )
    os.makedirs(staging_dir, exist_ok=False)

    def copy_checkpoint_support_files() -> None:
        for root, directories, filenames in os.walk(SOURCE_DIR):
            directories[:] = [directory for directory in directories if directory != ".cache"]
            relative_root = os.path.relpath(root, SOURCE_DIR)
            target_root = (
                staging_dir
                if relative_root == "."
                else os.path.join(staging_dir, relative_root)
            )
            os.makedirs(target_root, exist_ok=True)
            for filename in filenames:
                if filename.endswith(".safetensors"):
                    continue
                if filename == "model.safetensors.index.json":
                    continue
                source = os.path.join(root, filename)
                target = os.path.join(target_root, filename)
                shutil.copy2(source, target)

    copy_checkpoint_support_files()

    control_decoders = {
        "wrong_group_axis": wrong_group_axis_dequantise,
        "wrong_scale": wrong_scale_dequantise,
        "swapped_nibbles": swapped_nibble_dequantise,
    }
    negative_controls: dict[str, dict] = {}

    def exercise_real_negative_controls(
        tensor_name: str,
        encoded: W4A16Tensor,
    ) -> None:
        if len(negative_controls) == len(control_decoders):
            return
        rows, reduction = encoded.original_shape
        if rows < 2:
            return
        sample_rows = min(rows, 8)
        sample = W4A16Tensor(
            packed=encoded.packed[:sample_rows],
            scales=encoded.scales[:sample_rows],
            original_shape=(sample_rows, reduction),
        )
        for control_name, decoder in control_decoders.items():
            if control_name in negative_controls:
                continue
            try:
                verify_dequantizer(sample, decoder, decoder_name=control_name)
            except VerificationError as exc:
                negative_controls[control_name] = {
                    "rejected": True,
                    "real_tensor": tensor_name,
                    "sample_shape": [sample_rows, reduction],
                    "failure": str(exc),
                }

    def round_trip_metrics(
        tensor_class: str,
        original: torch.Tensor,
        encoded: W4A16Tensor,
    ) -> dict:
        rows, reduction = encoded.original_shape
        max_elements = 8 * 1024 * 1024
        rows_per_chunk = max(1, max_elements // reduction)
        max_abs_error = 0.0
        max_allowed_abs_error = 0.0
        error_square_sum = 0.0
        original_square_sum = 0.0
        for row_start in range(0, rows, rows_per_chunk):
            row_end = min(rows, row_start + rows_per_chunk)
            original_chunk = original[row_start:row_end]
            encoded_chunk = W4A16Tensor(
                packed=encoded.packed[row_start:row_end],
                scales=encoded.scales[row_start:row_end],
                original_shape=(row_end - row_start, reduction),
            )
            verified = verify_round_trip(tensor_class, original_chunk, encoded_chunk)
            max_abs_error = max(max_abs_error, verified.max_abs_error)
            if verified.max_allowed_abs_error is not None:
                max_allowed_abs_error = max(
                    max_allowed_abs_error, verified.max_allowed_abs_error
                )
            restored = dequantise(encoded_chunk, dtype=torch.float32)
            source_float = original_chunk.float()
            error = restored - source_float
            error_square_sum += float(torch.sum(error.square(), dtype=torch.float64))
            original_square_sum += float(
                torch.sum(source_float.square(), dtype=torch.float64)
            )
            del error, source_float, restored, encoded_chunk, original_chunk
        if original_square_sum == 0.0:
            relative_frobenius = 0.0 if error_square_sum == 0.0 else math.inf
        else:
            relative_frobenius = math.sqrt(error_square_sum / original_square_sum)
        return {
            "max_absolute_error": max_abs_error,
            "relative_frobenius_error": relative_frobenius,
            "scale_aware_max_allowed_absolute_error": max_allowed_abs_error,
        }

    output_weight_map = {}
    tensor_manifest = []
    expected_outputs_by_shard = defaultdict(dict)
    output_safetensors_file_bytes = 0
    actual_tensor_storage_bytes = 0

    with torch.inference_mode():
        for shard_name, tensor_names in sorted(names_by_shard.items()):
            source_path = os.path.join(SOURCE_DIR, shard_name)
            output_path = os.path.join(staging_dir, shard_name)
            output_tensors = {}
            with safe_open(source_path, framework="pt", device="cpu") as source:
                for tensor_name in sorted(tensor_names):
                    decision = decisions[tensor_name]
                    source_tensor = source.get_tensor(tensor_name)
                    observed_dtype = torch_dtypes.get(source_tensor.dtype)
                    if observed_dtype != decision.dtype:
                        raise TypeError(
                            f"loaded dtype disagrees with header for {tensor_name}: "
                            f"loaded={source_tensor.dtype}, header={decision.dtype}"
                        )
                    if tuple(source_tensor.shape) != decision.shape:
                        raise ValueError(
                            f"loaded shape disagrees with header for {tensor_name}: "
                            f"loaded={tuple(source_tensor.shape)}, header={decision.shape}"
                        )
                    observed_bytes = source_tensor.numel() * source_tensor.element_size()
                    if observed_bytes != decision.original_bytes:
                        raise ValueError(
                            f"loaded bytes disagree with plan for {tensor_name}: "
                            f"loaded={observed_bytes}, plan={decision.original_bytes}"
                        )

                    if decision.quantize:
                        reduction = source_tensor.shape[-1]
                        flattened = source_tensor.reshape(-1, reduction).to("cuda")
                        encoded = quantise(flattened)
                        if encoded.storage_bytes != decision.planned_bytes:
                            raise ValueError(
                                f"actual W4A16 bytes disagree with plan for {tensor_name}: "
                                f"actual={encoded.storage_bytes}, plan={decision.planned_bytes}"
                            )
                        metrics = round_trip_metrics(
                            decision.tensor_class,
                            flattened,
                            encoded,
                        )
                        exercise_real_negative_controls(tensor_name, encoded)
                        packed = encoded.packed.cpu().contiguous()
                        scales = encoded.scales.cpu().contiguous()
                        packed_name = decision.packed_name
                        scales_name = decision.scales_name
                        if packed_name is None or scales_name is None:
                            raise AssertionError("quantized plan decision has no output names")
                        output_tensors[packed_name] = packed
                        output_tensors[scales_name] = scales
                        output_weight_map[packed_name] = shard_name
                        output_weight_map[scales_name] = shard_name
                        output_descriptors = [
                            {
                                "name": packed_name,
                                "shape": list(packed.shape),
                                "dtype": "U8",
                                "bytes": packed.numel() * packed.element_size(),
                            },
                            {
                                "name": scales_name,
                                "shape": list(scales.shape),
                                "dtype": "BF16",
                                "bytes": scales.numel() * scales.element_size(),
                            },
                        ]
                        resulting_bytes = sum(
                            descriptor["bytes"] for descriptor in output_descriptors
                        )
                        del encoded, flattened, packed, scales
                        torch.cuda.empty_cache()
                    else:
                        retained = source_tensor.contiguous()
                        output_tensors[tensor_name] = retained
                        output_weight_map[tensor_name] = shard_name
                        output_descriptors = [
                            {
                                "name": tensor_name,
                                "shape": list(retained.shape),
                                "dtype": decision.dtype,
                                "bytes": retained.numel() * retained.element_size(),
                            }
                        ]
                        resulting_bytes = output_descriptors[0]["bytes"]
                        metrics = None

                    if resulting_bytes != decision.planned_bytes:
                        raise ValueError(
                            f"resulting bytes disagree with plan for {tensor_name}: "
                            f"actual={resulting_bytes}, plan={decision.planned_bytes}"
                        )
                    actual_tensor_storage_bytes += resulting_bytes
                    tensor_manifest.append(
                        {
                            "original_name": tensor_name,
                            "shape": list(decision.shape),
                            "original_dtype": decision.dtype,
                            "original_bytes": decision.original_bytes,
                            "quantized": decision.quantize,
                            "tensor_class": decision.tensor_class,
                            "decision_reason": decision.reason,
                            "resulting_bytes": resulting_bytes,
                            "error_metrics": metrics,
                            "source_shard": shard_name,
                            "output_tensors": output_descriptors,
                        }
                    )
                    for descriptor in output_descriptors:
                        expected_outputs_by_shard[shard_name][descriptor["name"]] = descriptor
                    del source_tensor

            save_file(
                output_tensors,
                output_path,
                metadata={
                    "format": "pt",
                    "runinfra_quantization": "W4A16",
                    "runinfra_quantization_profile": plan.profile.name,
                    "runinfra_group_size": str(GROUP_SIZE),
                },
            )
            del output_tensors
            gc.collect()
            output_safetensors_file_bytes += os.path.getsize(output_path)

            expected = expected_outputs_by_shard[shard_name]
            with safe_open(output_path, framework="pt", device="cpu") as written:
                written_names = set(written.keys())
                if written_names != set(expected):
                    raise ValueError(
                        f"written shard {shard_name} keys disagree with manifest"
                    )
                for output_name, descriptor in expected.items():
                    written_tensor = written.get_tensor(output_name)
                    written_dtype = torch_dtypes.get(written_tensor.dtype)
                    if tuple(written_tensor.shape) != tuple(descriptor["shape"]):
                        raise ValueError(
                            f"written shape mismatch for {output_name}: "
                            f"{tuple(written_tensor.shape)} vs {tuple(descriptor['shape'])}"
                        )
                    if written_dtype != descriptor["dtype"]:
                        raise TypeError(
                            f"written dtype mismatch for {output_name}: "
                            f"{written_dtype} vs {descriptor['dtype']}"
                        )
                    del written_tensor

    missing_controls = sorted(set(control_decoders) - set(negative_controls))
    if missing_controls:
        raise VerificationError(
            "real checkpoint data did not reject every broken decoder: "
            f"{missing_controls}"
        )
    if actual_tensor_storage_bytes != plan.planned_bytes:
        raise ValueError(
            "total written tensor storage disagrees with the index-derived plan: "
            f"actual={actual_tensor_storage_bytes}, plan={plan.planned_bytes}"
        )

    output_index = {
        "metadata": {
            "total_size": actual_tensor_storage_bytes,
            "quantization": "W4A16",
            "quantization_profile": plan.profile.name,
            "group_size": GROUP_SIZE,
            "scale_dtype": "BF16",
        },
        "weight_map": dict(sorted(output_weight_map.items())),
    }
    output_index_path = os.path.join(staging_dir, "model.safetensors.index.json")
    with open(output_index_path, "w", encoding="utf-8") as handle:
        json.dump(output_index, handle, indent=2, sort_keys=True)
        handle.write("\n")

    with open(output_index_path, encoding="utf-8") as handle:
        reloaded_index = json.load(handle)
    if reloaded_index != output_index:
        raise ValueError("written safetensors index did not round trip exactly")
    expected_output_names = {
        descriptor["name"]
        for tensor in tensor_manifest
        for descriptor in tensor["output_tensors"]
    }
    if set(reloaded_index["weight_map"]) != expected_output_names:
        raise ValueError("written safetensors index does not cover every output tensor")

    quantized_entries = [
        tensor for tensor in tensor_manifest if tensor["quantized"]
    ]
    worst_max_absolute = sorted(
        quantized_entries,
        key=lambda tensor: tensor["error_metrics"]["max_absolute_error"],
        reverse=True,
    )[:20]
    worst_relative_frobenius = sorted(
        quantized_entries,
        key=lambda tensor: tensor["error_metrics"]["relative_frobenius_error"],
        reverse=True,
    )[:20]

    def error_summary(entries: list[dict], metric: str) -> list[dict]:
        return [
            {
                "name": entry["original_name"],
                "tensor_class": entry["tensor_class"],
                metric: entry["error_metrics"][metric],
            }
            for entry in entries
        ]

    source_directory_bytes = 0
    source_file_count = 0
    for root, _, filenames in os.walk(SOURCE_DIR):
        for filename in filenames:
            source_directory_bytes += os.path.getsize(os.path.join(root, filename))
            source_file_count += 1

    manifest = {
        "schema_version": 1,
        "model": "moonshotai/Kimi-Linear-48B-A3B-Instruct",
        "source": {
            "volume": "kimi-linear-weights",
            "path": f"/{MODEL_NAME}",
            "directory_bytes": source_directory_bytes,
            "file_count": source_file_count,
            "safetensors_file_bytes": source_safetensors_file_bytes,
            "safetensors_shard_count": len(names_by_shard),
            "tensor_storage_bytes": plan.original_bytes,
            "index": "model.safetensors.index.json",
        },
        "quantization": {
            "profile": plan.profile.name,
            "format": "symmetric signed INT4",
            "group_size": GROUP_SIZE,
            "group_axis": "final reduction axis after flattening leading dimensions",
            "scale_dtype": "BF16",
            "zero_point": None,
            "bits_per_quantized_parameter_including_scales": 4.5,
        },
        "policy": {
            "principle": "quantize for fit, not for speed",
            "profile": plan.as_dict()["profile"],
            "coverage": policy_coverage,
            "classes": plan.as_dict()["classes"],
        },
        "profile_comparison": {
            "baseline_profile": DEFAULT_PROFILE_NAME,
            "selected_profile": plan.profile.name,
            "default_planned_tensor_storage_bytes": default_plan.planned_bytes,
            "selected_planned_tensor_storage_bytes": plan.planned_bytes,
            "selected_minus_default_tensor_storage_bytes": profile_delta_bytes,
            "authority": "real source safetensors headers",
        },
        "output": {
            "volume": "kimi-linear-quantized",
            "path": f"/{output_directory_name}",
            "planned_tensor_storage_bytes": plan.planned_bytes,
            "actual_tensor_storage_bytes": actual_tensor_storage_bytes,
            "actual_safetensors_file_bytes": output_safetensors_file_bytes,
            "safetensors_header_overhead_bytes": (
                output_safetensors_file_bytes - actual_tensor_storage_bytes
            ),
            "checkpoint_directory_bytes": 0,
            "index": "model.safetensors.index.json",
        },
        "projection_comparison": {
            "engine_laptop_results_projection_bytes": RESULTS_PROJECTION_BYTES,
            "measured_safetensors_minus_projection_bytes": (
                output_safetensors_file_bytes - RESULTS_PROJECTION_BYTES
            ),
            "measured_safetensors_matches_projection": (
                output_safetensors_file_bytes == RESULTS_PROJECTION_BYTES
            ),
            "authority": "measured output safetensors file bytes",
        },
        "proof": {
            "scale_aware_round_trip_verified_for_every_quantized_tensor": True,
            "negative_controls_rejected_on_real_checkpoint_data": negative_controls,
            "written_checkpoint_reloaded": True,
            "written_shapes_and_dtypes_match_manifest": True,
            "worst_by_max_absolute_error": error_summary(
                worst_max_absolute, "max_absolute_error"
            ),
            "worst_by_relative_frobenius_error": error_summary(
                worst_relative_frobenius, "relative_frobenius_error"
            ),
        },
        "tensors": sorted(tensor_manifest, key=lambda tensor: tensor["original_name"]),
    }

    manifest_path = os.path.join(staging_dir, MANIFEST_NAME)

    def directory_bytes(path: str) -> int:
        return sum(
            os.path.getsize(os.path.join(root, filename))
            for root, _, filenames in os.walk(path)
            for filename in filenames
        )

    for _ in range(8):
        with open(manifest_path, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
        measured_directory_bytes = directory_bytes(staging_dir)
        if manifest["output"]["checkpoint_directory_bytes"] == measured_directory_bytes:
            break
        manifest["output"]["checkpoint_directory_bytes"] = measured_directory_bytes
    else:
        raise RuntimeError("checkpoint directory byte count did not converge in manifest")

    with open(manifest_path, encoding="utf-8") as handle:
        reloaded_manifest = json.load(handle)
    if reloaded_manifest["output"]["checkpoint_directory_bytes"] != directory_bytes(
        staging_dir
    ):
        raise ValueError("final checkpoint directory byte count is not self-consistent")

    backup_dir = None
    if output_exists:
        backup_dir = f"{output_dir}.previous-{uuid.uuid4().hex}"
        os.replace(output_dir, backup_dir)
    try:
        os.replace(staging_dir, output_dir)
        OUTPUT_VOLUME.commit()
    except Exception:
        if backup_dir is not None:
            if os.path.exists(output_dir):
                shutil.rmtree(output_dir)
            os.replace(backup_dir, output_dir)
            OUTPUT_VOLUME.commit()
        raise
    if backup_dir is not None:
        shutil.rmtree(backup_dir)
        OUTPUT_VOLUME.commit()

    written_shards = [
        {
            "name": shard_name,
            "bytes": os.path.getsize(os.path.join(output_dir, shard_name)),
        }
        for shard_name in sorted(names_by_shard)
    ]
    summary = {
        "profile": plan.profile.name,
        "output_volume": "kimi-linear-quantized",
        "output_path": f"/{output_directory_name}",
        "manifest_path": f"/{output_directory_name}/{MANIFEST_NAME}",
        "source_tensor_storage_bytes": plan.original_bytes,
        "default_profile_planned_tensor_storage_bytes": default_plan.planned_bytes,
        "planned_tensor_storage_bytes": plan.planned_bytes,
        "planned_tensor_storage_delta_from_default_bytes": profile_delta_bytes,
        "actual_tensor_storage_bytes": actual_tensor_storage_bytes,
        "actual_safetensors_file_bytes": output_safetensors_file_bytes,
        "actual_checkpoint_directory_bytes": reloaded_manifest["output"][
            "checkpoint_directory_bytes"
        ],
        "engine_laptop_results_projection_bytes": RESULTS_PROJECTION_BYTES,
        "actual_safetensors_minus_projection_bytes": (
            output_safetensors_file_bytes - RESULTS_PROJECTION_BYTES
        ),
        "quantized_tensor_count": plan.quantized_tensor_count,
        "retained_tensor_count": len(plan.tensors) - plan.quantized_tensor_count,
        "negative_controls": negative_controls,
        "written_checkpoint_reloaded": True,
        "written_shards": written_shards,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


@app.local_entrypoint()
def main(profile: str = "default", overwrite: bool = False):
    result = quantize_checkpoint.remote(overwrite=overwrite, profile=profile)
    print(json.dumps(result, indent=2, sort_keys=True))
