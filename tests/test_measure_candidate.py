from __future__ import annotations

from copy import deepcopy

import pytest

from engine.measure import harness
from engine.measure.candidate_server import (
    DEFAULT_INT4_WEIGHTS_PATH,
    DEFAULT_QUANTIZATION_FORMAT,
    DEFAULT_RUNTIME_NAME,
    candidate_runtime_spec,
)
from engine.measure.compare import ComparisonRefused, compare_record
from engine.measure.harness import RuntimeSpec


def _gpu(uuid: str = "GPU-synthetic-1") -> dict:
    return {
        "count": 1,
        "names": ["Synthetic H200"],
        "device_uuids": [uuid],
        "total_memory_bytes": [150_000_000_000],
        "driver_version": "synthetic-driver",
        "cuda_driver_version": "synthetic-cuda-driver",
        "cuda_runtime_version": "synthetic-cuda-runtime",
    }


def _spec(side: str, weights_path: str) -> RuntimeSpec:
    return RuntimeSpec(
        side=side,
        name=DEFAULT_RUNTIME_NAME if side == "candidate" else "vLLM",
        quantization_format=(
            DEFAULT_QUANTIZATION_FORMAT if side == "candidate" else "BF16"
        ),
        command=["python", "-m", "synthetic.server", "--weights", weights_path],
        version_command=["python", "-c", "print('synthetic-version')"],
        weights_path=weights_path,
        compute_weights_digest=side == "candidate",
        model_id="moonshotai/Kimi-Linear-48B-A3B-Instruct",
        requested_revision="main",
        resolved_revision="a" * 40,
        served_model_name="moonshotai/Kimi-Linear-48B-A3B-Instruct",
        tensor_parallel_size=1,
        max_model_len=4096,
        port=8000,
    )


def test_candidate_descriptor_records_int4_artifact_and_measured_bytes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    int4_measurement = {
        "path": DEFAULT_INT4_WEIGHTS_PATH,
        "file_count": 20,
        "storage_bytes": 29,
        "weights_resident_bytes": 23,
        "measurement_method": "synthetic runtime safetensors inspection",
        "residency_contract": "synthetic complete artifact",
        "weights_digest_sha256": "b" * 64,
        "digest_method": "synthetic digest",
    }
    bf16_measurement = {
        **int4_measurement,
        "path": "/weights/Kimi-Linear-48B-A3B-Instruct",
        "storage_bytes": 97,
        "weights_resident_bytes": 89,
    }

    def inspect_weights(path: str, *, compute_digest: bool) -> dict:
        assert path == DEFAULT_INT4_WEIGHTS_PATH
        assert compute_digest is True
        return deepcopy(int4_measurement)

    class Probe:
        def __init__(self) -> None:
            self.samples = [
                {"offset_ms": 0.0, "device_used_bytes": 31},
                {"offset_ms": 1.0, "device_used_bytes": 37},
            ]

        def used_bytes(self) -> int:
            return 0

        def start(self) -> None:
            return None

        def collect_steady_state(self, *, samples: int, interval_seconds: float) -> list[int]:
            assert samples == 20
            assert interval_seconds == 0.25
            return [31] * samples

        def stop(self) -> list[dict]:
            return list(self.samples)

        def close(self) -> None:
            return None

    class Server:
        def __init__(self, command: list[str], **kwargs: object) -> None:
            self.command = command

        def start(self) -> None:
            return None

        def stop(self) -> None:
            return None

        def log_tail(self) -> str:
            return ""

        def log_digest(self) -> str:
            return "c" * 64

    monkeypatch.setattr(harness, "inspect_weight_artifact", inspect_weights)
    monkeypatch.setattr(harness, "run_version_probe", lambda command: {
        "command": command,
        "stdout": "synthetic-version",
        "stderr": "",
        "returncode": 0,
    })
    monkeypatch.setattr(harness, "GpuMemoryProbe", Probe)
    monkeypatch.setattr(harness, "ServerProcess", Server)
    monkeypatch.setattr(harness, "run_workload", lambda **kwargs: {
        "warmup": {"requested": 1, "dropped": 1, "mode": "synthetic"},
        "measurements": [],
    })

    descriptor = candidate_runtime_spec(
        runtime_name=DEFAULT_RUNTIME_NAME,
        quantization_format=DEFAULT_QUANTIZATION_FORMAT,
        command=["python", "-m", "engine.measure.candidate_server"],
        version_command=[
            "python",
            "-m",
            "engine.measure.candidate_server",
            "--version",
        ],
        weights_path=DEFAULT_INT4_WEIGHTS_PATH,
        model_id="moonshotai/Kimi-Linear-48B-A3B-Instruct",
        requested_revision="main",
        resolved_revision="a" * 40,
        served_model_name="moonshotai/Kimi-Linear-48B-A3B-Instruct",
        max_model_len=4096,
        port=8000,
    )
    result = harness._measure_side(
        descriptor,
        gpu=_gpu(),
        prompt_set={"id": "sha256:synthetic", "prompts": []},
        concurrency_levels=(1, 4, 16, 64),
        repetitions=20,
        warmup_requests=1,
        startup_timeout_seconds=1.0,
        request_timeout_seconds=1.0,
    )

    assert result["status"] == "ok"
    assert result["runtime"]["name"] == DEFAULT_RUNTIME_NAME
    assert result["runtime"]["quantization_format"] == DEFAULT_QUANTIZATION_FORMAT
    assert result["model"]["weights_path"] == DEFAULT_INT4_WEIGHTS_PATH
    assert result["model"]["weights_resident_bytes"] == int4_measurement[
        "weights_resident_bytes"
    ]
    assert result["model"]["weights_resident_bytes"] != bf16_measurement[
        "weights_resident_bytes"
    ]


def test_candidate_record_with_different_gpu_is_refused() -> None:
    baseline = {"status": "ok", "gpu": _gpu("GPU-baseline")}
    candidate = {"status": "ok", "gpu": _gpu("GPU-candidate")}
    record = {
        "status": "complete",
        "sides": {"baseline": baseline, "candidate": candidate},
    }

    with pytest.raises(ComparisonRefused, match="GPU mismatch"):
        compare_record(record)


def test_candidate_start_failure_emits_no_comparison(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline_result = {"status": "ok"}
    candidate_result = {
        "status": "failed",
        "stage": "start server",
        "error": {
            "type": "SyntheticStartError",
            "message": "candidate server did not start",
            "traceback": "synthetic traceback",
        },
    }

    class Probe:
        def used_bytes(self) -> int:
            return 0

        def close(self) -> None:
            return None

    monkeypatch.setattr(harness, "gpu_inventory", _gpu)
    monkeypatch.setattr(harness, "host_inventory", lambda: {"host": "synthetic"})
    monkeypatch.setattr(harness, "GpuMemoryProbe", Probe)
    monkeypatch.setattr(harness, "wait_for_gpu_release", lambda *args, **kwargs: 0)
    monkeypatch.setattr(
        harness,
        "_measure_side",
        lambda spec, **kwargs: (
            baseline_result if spec.side == "baseline" else candidate_result
        ),
    )
    monkeypatch.setattr(harness, "validate_measurement_record", lambda record: record)

    record = harness.run_both_sides(
        baseline=_spec("baseline", "/weights/Kimi-Linear-48B-A3B-Instruct"),
        candidate=_spec("candidate", DEFAULT_INT4_WEIGHTS_PATH),
        prompt_set={"id": "sha256:synthetic"},
        concurrency_levels=(1, 4, 16, 64),
        repetitions=20,
        warmup_requests=1,
        startup_timeout_seconds=1.0,
        request_timeout_seconds=1.0,
    )

    assert record["status"] == "failed"
    assert record["sides"]["candidate"] == candidate_result
    assert record["comparison"] is None
