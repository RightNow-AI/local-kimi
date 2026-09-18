"""Orchestrate two sequential servers on one GPU and emit one strict record."""

from __future__ import annotations

import json
import statistics
import time
import traceback
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from engine.bench.artifacts import IMMUTABLE_REVISION

from .client import run_workload
from .record import SCHEMA_VERSION, validate_concurrency_levels, validate_measurement_record
from .runtime import (
    GpuMemoryProbe,
    ServerProcess,
    gpu_inventory,
    host_inventory,
    run_version_probe,
    wait_for_gpu_release,
)
from .weights import inspect_weight_artifact


@dataclass(frozen=True)
class RuntimeSpec:
    side: str
    name: str
    quantization_format: str
    command: list[str]
    version_command: list[str]
    weights_path: str
    compute_weights_digest: bool
    model_id: str
    requested_revision: str
    resolved_revision: str
    served_model_name: str
    tensor_parallel_size: int
    max_model_len: int
    port: int
    disclosures: tuple[str, ...] = ()


def load_download_manifest(
    path: str | Path,
    *,
    model_id: str,
    requested_revision: str,
) -> dict[str, Any]:
    """Reuse the existing bench download artifact without redownloading on GPU."""

    manifest_path = Path(path)
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"download manifest is missing: {manifest_path}; run the bench download action first"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("model_id") != model_id:
        raise ValueError("download manifest model does not match the measurement model")
    resolved = manifest.get("resolved_revision")
    if not isinstance(resolved, str) or IMMUTABLE_REVISION.fullmatch(resolved) is None:
        raise ValueError("download manifest is missing an immutable revision")
    if requested_revision not in {manifest.get("requested_revision"), resolved}:
        raise ValueError(
            f"download manifest resolves {resolved}, not requested revision {requested_revision}"
        )
    snapshot_path = manifest.get("snapshot_path")
    if not isinstance(snapshot_path, str) or not Path(snapshot_path).is_dir():
        raise FileNotFoundError("download manifest snapshot path is absent from the Volume")
    if Path(snapshot_path).name.lower() != resolved:
        raise ValueError("download manifest snapshot path and revision disagree")
    return manifest


def _failure(exc: BaseException, *, stage: str, partial: dict[str, Any] | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "status": "failed",
        "stage": stage,
        "error": {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        },
    }
    if partial:
        result["partial"] = partial
    return result


def _measure_side(
    spec: RuntimeSpec,
    *,
    gpu: dict[str, Any],
    prompt_set: dict[str, Any],
    concurrency_levels: tuple[int, ...],
    repetitions: int,
    warmup_requests: int,
    startup_timeout_seconds: float,
    request_timeout_seconds: float,
) -> dict[str, Any]:
    partial: dict[str, Any] = {
        "side": spec.side,
        "runtime": {
            "name": spec.name,
            "quantization_format": spec.quantization_format,
            "server_command": spec.command,
            "disclosures": list(spec.disclosures),
        },
        "model": {"weights_path": str(Path(spec.weights_path).resolve())},
    }
    server: ServerProcess | None = None
    probe: GpuMemoryProbe | None = None
    memory_samples: list[dict[str, Any]] = []
    stage = "inspect weights"
    try:
        weights = inspect_weight_artifact(
            spec.weights_path,
            compute_digest=spec.compute_weights_digest,
        )
        partial["weights"] = weights
        stage = "probe runtime version"
        version_probe = run_version_probe(spec.version_command)
        partial["version_probe"] = version_probe
        stage = "start memory sampler"
        probe = GpuMemoryProbe()
        idle_before = probe.used_bytes()
        probe.start()
        stage = "start server"
        server = ServerProcess(
            spec.command,
            base_url=f"http://127.0.0.1:{spec.port}",
            served_model_name=spec.served_model_name,
            startup_timeout_seconds=startup_timeout_seconds,
        )
        server.start()
        stage = "measure steady-state memory"
        steady_absolute = probe.collect_steady_state(samples=20, interval_seconds=0.25)
        steady_runtime = [max(0, reading - idle_before) for reading in steady_absolute]
        stage = "run serving sweep"
        workload = run_workload(
            base_url=f"http://127.0.0.1:{spec.port}",
            served_model_name=spec.served_model_name,
            prompt_set=prompt_set,
            concurrency_levels=concurrency_levels,
            repetitions=repetitions,
            warmup_requests=warmup_requests,
            request_timeout_seconds=request_timeout_seconds,
        )
        stage = "finalize memory"
        memory_samples = probe.stop()
        active_readings = [sample["device_used_bytes"] for sample in memory_samples]
        if not active_readings:
            raise RuntimeError("GPU memory sampler produced no readings")
        peak_device = max(active_readings)
        result = {
            "status": "ok",
            "prompt_set_id": prompt_set["id"],
            "prompt_token_counts": [
                {"id": prompt["id"], "prompt_tokens": prompt["prompt_tokens"]}
                for prompt in prompt_set["prompts"]
            ],
            "model": {
                "id": spec.model_id,
                "requested_revision": spec.requested_revision,
                "resolved_revision": spec.resolved_revision,
                "weights_path": weights["path"],
                "weights_digest_sha256": weights["weights_digest_sha256"],
                "weights_resident_bytes": weights["weights_resident_bytes"],
                "weights_storage_bytes": weights["storage_bytes"],
                "weights_file_count": weights["file_count"],
                "weights_measurement_method": weights["measurement_method"],
                "weights_residency_contract": weights["residency_contract"],
                "weights_digest_method": weights["digest_method"],
            },
            "runtime": {
                "name": spec.name,
                "version": version_probe["stdout"],
                "version_probe": version_probe,
                "quantization_format": spec.quantization_format,
                "tensor_parallel_size": spec.tensor_parallel_size,
                "max_model_len": spec.max_model_len,
                "served_model_name": spec.served_model_name,
                "server_command": spec.command,
                "server_flags": spec.command[1:],
                "disclosures": list(spec.disclosures),
            },
            "gpu": dict(gpu),
            "memory": {
                "idle_before_launch_bytes": idle_before,
                "steady_state_gpu_memory_bytes": float(statistics.median(steady_runtime)),
                "steady_state_device_used_bytes": float(statistics.median(steady_absolute)),
                "steady_state_samples_bytes": steady_runtime,
                "peak_gpu_memory_bytes": max(0, peak_device - idle_before),
                "peak_device_used_bytes": peak_device,
                "sampling_interval_ms": 100.0,
                "raw_device_samples": memory_samples,
                "measurement_method": (
                    "NVML device-used bytes minus the prelaunch idle reading on the one visible GPU"
                ),
            },
            "warmup": workload["warmup"],
            "measurements": workload["measurements"],
            "server_log_sha256": server.log_digest(),
        }
        return result
    except Exception as exc:
        return _failure(exc, stage=stage, partial=partial)
    finally:
        if probe is not None and not memory_samples:
            try:
                probe.stop()
            except Exception:
                pass
        if server is not None:
            try:
                server.stop()
                partial["server_log_tail"] = server.log_tail()
                partial["server_log_sha256"] = server.log_digest()
            except Exception as cleanup_exc:
                partial["cleanup_error"] = f"{type(cleanup_exc).__name__}: {cleanup_exc}"
        if probe is not None:
            try:
                probe.close()
            except Exception:
                pass


def run_both_sides(
    *,
    baseline: RuntimeSpec,
    candidate: RuntimeSpec,
    prompt_set: dict[str, Any],
    concurrency_levels: tuple[int, ...],
    repetitions: int,
    warmup_requests: int,
    startup_timeout_seconds: float,
    request_timeout_seconds: float,
) -> dict[str, Any]:
    """Measure both runtimes sequentially inside one GPU function invocation."""

    levels = validate_concurrency_levels(concurrency_levels)
    run_id = str(uuid.uuid4())
    started_wall = datetime.now(timezone.utc)
    started = time.perf_counter()
    gpu = gpu_inventory()
    initial_probe = GpuMemoryProbe()
    try:
        initial_gpu_used_bytes = initial_probe.used_bytes()
    finally:
        initial_probe.close()
    environment = {
        "run_id": run_id,
        "started_at_utc": started_wall.isoformat(),
        "host": host_inventory(),
        "gpu": gpu,
        "initial_gpu_used_bytes": initial_gpu_used_bytes,
        "execution": "one Modal function invocation, sequential baseline then candidate",
        "server_order": ["baseline", "candidate"],
    }
    baseline_result = _measure_side(
        baseline,
        gpu=gpu,
        prompt_set=prompt_set,
        concurrency_levels=levels,
        repetitions=repetitions,
        warmup_requests=warmup_requests,
        startup_timeout_seconds=startup_timeout_seconds,
        request_timeout_seconds=request_timeout_seconds,
    )

    release_probe: GpuMemoryProbe | None = None
    try:
        release_probe = GpuMemoryProbe()
        release_limit = initial_gpu_used_bytes + 256 * 1024 * 1024
        wait_for_gpu_release(
            release_probe,
            at_most_bytes=release_limit,
            timeout_seconds=300.0,
        )
        candidate_result = _measure_side(
            candidate,
            gpu=gpu,
            prompt_set=prompt_set,
            concurrency_levels=levels,
            repetitions=repetitions,
            warmup_requests=warmup_requests,
            startup_timeout_seconds=startup_timeout_seconds,
            request_timeout_seconds=request_timeout_seconds,
        )
    except Exception as exc:
        candidate_result = _failure(exc, stage="verify clean GPU before candidate")
    finally:
        if release_probe is not None:
            release_probe.close()

    complete = baseline_result.get("status") == "ok" and candidate_result.get("status") == "ok"
    finished_wall = datetime.now(timezone.utc)
    environment["finished_at_utc"] = finished_wall.isoformat()
    environment["invocation_seconds"] = time.perf_counter() - started
    record: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete" if complete else "failed",
        "environment": environment,
        "measurement_protocol": {
            "claim": (
                "matched serving latency, throughput, and footprint on the recorded prompt set "
                "and environment"
            ),
            "does_not_establish": [
                "quality or accuracy",
                "customer workload performance",
                "performance on another GPU, model revision, prompt set, or concurrency",
                "performance of a partial reference implementation",
            ],
            "percentile_method": "nearest-rank",
            "warmups_are_discarded": True,
            "raw_request_samples_retained": True,
            "baseline_tuning_disclosure": (
                "vLLM 0.26.0 reports that H200 has no tuned fused-MoE "
                "configuration for E=256 and N=1024, so it uses a default "
                "configuration. This record does not describe that fused-MoE "
                "path as tuned."
            ),
        },
        "prompt_set": prompt_set,
        "concurrency_levels": list(levels),
        "sides": {"baseline": baseline_result, "candidate": candidate_result},
        "comparison": (
            {
                "eligible": True,
                "gpu_device_uuids": gpu["device_uuids"],
                "prompt_set_id": prompt_set["id"],
                "concurrency_levels": list(levels),
            }
            if complete
            else None
        ),
    }
    return validate_measurement_record(record)
