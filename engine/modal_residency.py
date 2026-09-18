"""Measure Kimi-Linear persistent state allocations on one Modal H100.

The default policy is vLLM's compressed latent cache. Pass ``expanded`` to
measure the Hugging Face reference layout instead. The default points end with
the disputed ``16 x 32K`` envelope. This harness intentionally does not load
model weights, so its measured comparison is against ``state_pool_bytes`` only.
Weight bytes come from measured artifact tensor storage but are not allocated by
this job. Operational headroom remains a projected policy field in the JSON.

Run only through the orchestrator:

    modal run engine/modal_residency.py
"""

from __future__ import annotations

import json

import modal

APP = modal.App("kimi-linear-residency")
IMAGE = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch>=2.5")
    .add_local_dir("engine", remote_path="/root/engine")
)

DEFAULT_POINTS = "1x32768,4x8192,16x2048,64x512,16x4096,16x32768"


def _parse_points(value: str) -> list[tuple[int, int]]:
    points = []
    for item in value.split(","):
        raw = item.strip().lower()
        if not raw:
            continue
        try:
            seqs_text, length_text = raw.split("x", maxsplit=1)
            point = (int(seqs_text), int(length_text))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"invalid point {item!r}; expected comma-separated SEQSxLENGTH values"
            ) from exc
        if point[0] <= 0 or point[1] <= 0:
            raise ValueError(f"point values must be positive: {item!r}")
        points.append(point)
    if not points:
        raise ValueError("at least one measurement point is required")
    return points


@APP.function(image=IMAGE, gpu="H100", timeout=60 * 30)
def measure_residency(
    points: list[tuple[int, int]],
    quantization_profile: str = "int4",
    mla_cache_policy: str = "compressed_latent",
    recurrent_dtype: str = "float32",
    conv_dtype: str = "bfloat16",
    mla_dtype: str = "bfloat16",
    activation_headroom_gib: int = 2,
    workspace_headroom_gib: int = 1,
) -> dict[str, object]:
    import gc

    import torch

    from engine.residency.budget import (
        BF16,
        FP16,
        FP32,
        GIB,
        KIMI_LINEAR_SHAPE,
        MLACachePolicy,
        MODEL_ID,
        MODEL_MAX_LENGTH,
        RuntimeHeadroom,
        StateDTypes,
        build_residency_budget,
        resolve_mla_cache_policy,
    )

    scalar_dtypes = {
        "bfloat16": BF16,
        "float16": FP16,
        "float32": FP32,
    }
    torch_dtypes = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    requested_dtypes = (recurrent_dtype, conv_dtype, mla_dtype)
    unknown = sorted(set(requested_dtypes) - set(scalar_dtypes))
    if unknown:
        raise ValueError(f"unsupported state dtypes: {unknown}")
    if activation_headroom_gib < 0 or workspace_headroom_gib < 0:
        raise ValueError("headroom GiB values cannot be negative")

    state_dtypes = StateDTypes(
        recurrent_state=scalar_dtypes[recurrent_dtype],
        short_conv_state=scalar_dtypes[conv_dtype],
        mla_kv_cache=scalar_dtypes[mla_dtype],
    )
    headroom = RuntimeHeadroom(
        activation_bytes=activation_headroom_gib * GIB,
        workspace_bytes=workspace_headroom_gib * GIB,
    )
    resolved_mla_cache_policy = resolve_mla_cache_policy(mla_cache_policy)
    device = torch.device("cuda")
    properties = torch.cuda.get_device_properties(0)

    result_points: list[dict[str, object]] = []
    for max_num_seqs, max_model_len in points:
        if max_model_len > MODEL_MAX_LENGTH:
            raise ValueError(
                f"measurement length {max_model_len} exceeds {MODEL_MAX_LENGTH}"
            )
        budget = build_residency_budget(
            quantization_profile,
            max_num_seqs,
            max_model_len,
            mla_cache_policy=resolved_mla_cache_policy,
            state_dtypes=state_dtypes,
            headroom=headroom,
        )

        allocations: list[torch.Tensor] = []
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        baseline_allocated = torch.cuda.memory_allocated(device)
        baseline_reserved = torch.cuda.memory_reserved(device)
        torch.cuda.reset_peak_memory_stats(device)

        point_result: dict[str, object] = {
            "status": "MEASURING",
            "evidence_status": "MEASURED",
            "max_num_seqs": max_num_seqs,
            "max_model_len": max_model_len,
            "predicted_state_pool_bytes": budget.state_pool_bytes,
            "predicted_total_residency_bytes": budget.total_bytes,
            "projected_breakdown": budget.as_dict(),
        }
        try:
            shape = KIMI_LINEAR_SHAPE
            for _ in range(shape.kda_layers):
                allocations.append(
                    torch.empty(
                        (
                            max_num_seqs,
                            shape.kda_num_heads,
                            shape.kda_key_head_dim,
                            shape.kda_value_head_dim,
                        ),
                        dtype=torch_dtypes[recurrent_dtype],
                        device=device,
                    )
                )
                allocations.append(
                    torch.empty(
                        (
                            max_num_seqs,
                            shape.kda_q_width,
                            shape.short_conv_kernel_size,
                        ),
                        dtype=torch_dtypes[conv_dtype],
                        device=device,
                    )
                )
                allocations.append(
                    torch.empty(
                        (
                            max_num_seqs,
                            shape.kda_k_width,
                            shape.short_conv_kernel_size,
                        ),
                        dtype=torch_dtypes[conv_dtype],
                        device=device,
                    )
                )
                allocations.append(
                    torch.empty(
                        (
                            max_num_seqs,
                            shape.kda_v_width,
                            shape.short_conv_kernel_size,
                        ),
                        dtype=torch_dtypes[conv_dtype],
                        device=device,
                    )
                )

            if resolved_mla_cache_policy is MLACachePolicy.EXPANDED:
                mla_key_width = (
                    shape.mla_qk_nope_head_dim + shape.mla_qk_rope_head_dim
                )
                for _ in range(shape.mla_layers):
                    allocations.append(
                        torch.empty(
                            (
                                max_num_seqs,
                                shape.mla_num_heads,
                                max_model_len,
                                mla_key_width,
                            ),
                            dtype=torch_dtypes[mla_dtype],
                            device=device,
                        )
                    )
                    allocations.append(
                        torch.empty(
                            (
                                max_num_seqs,
                                shape.mla_num_heads,
                                max_model_len,
                                shape.mla_value_head_dim,
                            ),
                            dtype=torch_dtypes[mla_dtype],
                            device=device,
                        )
                    )
            else:
                compressed_width = shape.mla_compressed_elements_per_token_per_layer
                for _ in range(shape.mla_layers):
                    allocations.append(
                        torch.empty(
                            (max_num_seqs, max_model_len, compressed_width),
                            dtype=torch_dtypes[mla_dtype],
                            device=device,
                        )
                    )

            torch.cuda.synchronize()
            tensor_storage_bytes = sum(
                tensor.numel() * tensor.element_size() for tensor in allocations
            )
            peak_allocated_delta = (
                torch.cuda.max_memory_allocated(device) - baseline_allocated
            )
            peak_reserved_delta = (
                torch.cuda.max_memory_reserved(device) - baseline_reserved
            )
            allocated_delta = peak_allocated_delta - budget.state_pool_bytes
            reserved_delta = peak_reserved_delta - budget.state_pool_bytes
            point_result.update(
                {
                    "status": (
                        "MATCH"
                        if tensor_storage_bytes == budget.state_pool_bytes
                        and allocated_delta == 0
                        else "MISMATCH"
                    ),
                    "tensor_count": len(allocations),
                    "tensor_storage_bytes": tensor_storage_bytes,
                    "torch_cuda_max_memory_allocated_delta_bytes": peak_allocated_delta,
                    "torch_cuda_max_memory_reserved_delta_bytes": peak_reserved_delta,
                    "allocated_minus_predicted_bytes": allocated_delta,
                    "reserved_minus_predicted_bytes": reserved_delta,
                    "allocated_delta_percent": round(
                        100.0 * allocated_delta / budget.state_pool_bytes,
                        6,
                    ),
                    "reserved_delta_percent": round(
                        100.0 * reserved_delta / budget.state_pool_bytes,
                        6,
                    ),
                }
            )
        except torch.OutOfMemoryError as exc:
            point_result.update(
                {
                    "status": "OOM",
                    "error": str(exc),
                    "torch_cuda_max_memory_allocated_delta_bytes": (
                        torch.cuda.max_memory_allocated(device) - baseline_allocated
                    ),
                    "torch_cuda_max_memory_reserved_delta_bytes": (
                        torch.cuda.max_memory_reserved(device) - baseline_reserved
                    ),
                }
            )
        finally:
            allocations.clear()
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        result_points.append(point_result)

    point_statuses = {point["status"] for point in result_points}
    if point_statuses == {"MATCH"}:
        overall_status = "MATCH"
    elif "OOM" in point_statuses:
        overall_status = "PARTIAL_WITH_OOM"
    else:
        overall_status = "MISMATCH"

    output = {
        "schema_version": 1,
        "status": overall_status,
        "evidence_status": "MEASURED",
        "model_id": MODEL_ID,
        "gpu": {
            "name": torch.cuda.get_device_name(0),
            "total_memory_bytes": properties.total_memory,
            "cuda_version": torch.version.cuda,
            "torch_version": torch.__version__,
        },
        "measurement_scope": (
            "persistent state tensors only; measured artifact weight bytes are not "
            "allocated; activations and workspace are projected policy reserves"
        ),
        "state_dtypes": {
            "recurrent": recurrent_dtype,
            "short_conv": conv_dtype,
            "mla_kv": mla_dtype,
            "matches_inspected_runtime": state_dtypes.matches_inspected_runtime,
        },
        "quantization_profile": quantization_profile,
        "mla_cache_policy": resolved_mla_cache_policy.value,
        "points": result_points,
    }
    return output


@APP.local_entrypoint()
def main(
    points: str = DEFAULT_POINTS,
    quantization_profile: str = "int4",
    mla_cache_policy: str = "compressed_latent",
    recurrent_dtype: str = "float32",
    conv_dtype: str = "bfloat16",
    mla_dtype: str = "bfloat16",
    activation_headroom_gib: int = 2,
    workspace_headroom_gib: int = 1,
) -> None:
    result = measure_residency.remote(
        points=_parse_points(points),
        quantization_profile=quantization_profile,
        mla_cache_policy=mla_cache_policy,
        recurrent_dtype=recurrent_dtype,
        conv_dtype=conv_dtype,
        mla_dtype=mla_dtype,
        activation_headroom_gib=activation_headroom_gib,
        workspace_headroom_gib=workspace_headroom_gib,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
