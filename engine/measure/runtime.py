"""Process, GPU identity, and memory measurement primitives."""

from __future__ import annotations

import hashlib
import os
import platform
import signal
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any


class ServerStartError(RuntimeError):
    """Raised when a measured runtime does not become ready."""


def _text(value: str | bytes) -> str:
    return value.decode("utf-8", errors="replace") if isinstance(value, bytes) else value


def _cuda_version(value: int) -> str:
    major = value // 1000
    minor = (value % 1000) // 10
    return f"{major}.{minor}"


def gpu_inventory() -> dict[str, Any]:
    """Identify the one visible GPU without creating a CUDA context."""

    import pynvml
    import torch

    pynvml.nvmlInit()
    try:
        count = pynvml.nvmlDeviceGetCount()
        devices = []
        for index in range(count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(index)
            memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
            devices.append(
                {
                    "index": index,
                    "name": _text(pynvml.nvmlDeviceGetName(handle)),
                    "uuid": _text(pynvml.nvmlDeviceGetUUID(handle)),
                    "total_memory_bytes": int(memory.total),
                }
            )
        if count != 1:
            raise RuntimeError(f"measurement requires one visible GPU, observed {count}")
        return {
            "count": count,
            "names": [device["name"] for device in devices],
            "device_uuids": [device["uuid"] for device in devices],
            "total_memory_bytes": [device["total_memory_bytes"] for device in devices],
            "driver_version": _text(pynvml.nvmlSystemGetDriverVersion()),
            "cuda_driver_version": _cuda_version(
                int(
                    getattr(
                        pynvml,
                        "nvmlSystemGetCudaDriverVersion_v2",
                        pynvml.nvmlSystemGetCudaDriverVersion,
                    )()
                )
            ),
            "cuda_runtime_version": str(torch.version.cuda),
        }
    finally:
        pynvml.nvmlShutdown()


def host_inventory() -> dict[str, Any]:
    return {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
    }


class GpuMemoryProbe:
    """Sample device-used bytes for the single visible GPU."""

    def __init__(self, interval_seconds: float = 0.1) -> None:
        if interval_seconds <= 0.0:
            raise ValueError("memory sampling interval must be positive")
        import pynvml

        self._pynvml = pynvml
        self._pynvml.nvmlInit()
        if self._pynvml.nvmlDeviceGetCount() != 1:
            self._pynvml.nvmlShutdown()
            raise RuntimeError("memory probe requires one visible GPU")
        self._handle = self._pynvml.nvmlDeviceGetHandleByIndex(0)
        self._interval_seconds = interval_seconds
        self._origin = time.perf_counter()
        self._samples: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def used_bytes(self) -> int:
        return int(self._pynvml.nvmlDeviceGetMemoryInfo(self._handle).used)

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("memory probe already started")

        def sample() -> None:
            while not self._stop.is_set():
                reading = {
                    "offset_ms": (time.perf_counter() - self._origin) * 1000.0,
                    "device_used_bytes": self.used_bytes(),
                }
                with self._lock:
                    self._samples.append(reading)
                self._stop.wait(self._interval_seconds)

        self._thread = threading.Thread(target=sample, daemon=True)
        self._thread.start()

    def collect_steady_state(self, *, samples: int, interval_seconds: float) -> list[int]:
        if samples < 2 or interval_seconds <= 0.0:
            raise ValueError("steady-state sampling requires multiple positive-interval samples")
        readings = []
        for _ in range(samples):
            readings.append(self.used_bytes())
            time.sleep(interval_seconds)
        return readings

    def stop(self) -> list[dict[str, Any]]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self._interval_seconds * 4))
        with self._lock:
            return list(self._samples)

    def close(self) -> None:
        self._pynvml.nvmlShutdown()


class ServerProcess:
    """Launch one server in its own process group and retain its complete log."""

    def __init__(
        self,
        command: list[str],
        *,
        base_url: str,
        served_model_name: str,
        startup_timeout_seconds: float,
        environment: dict[str, str] | None = None,
    ) -> None:
        if not command:
            raise ValueError("server command must not be empty")
        self.command = list(command)
        self.base_url = base_url.rstrip("/")
        self.served_model_name = served_model_name
        self.startup_timeout_seconds = startup_timeout_seconds
        self.environment = environment
        descriptor, log_path = tempfile.mkstemp(prefix="kimi-measure-server-", suffix=".log")
        os.close(descriptor)
        self.log_path = Path(log_path)
        self._log_handle: Any = None
        self.process: subprocess.Popen[bytes] | None = None

    def start(self) -> None:
        self._log_handle = self.log_path.open("wb")
        env = os.environ.copy()
        if self.environment:
            env.update(self.environment)
        self.process = subprocess.Popen(
            self.command,
            stdout=self._log_handle,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
        )
        self._wait_ready()

    def _wait_ready(self) -> None:
        import httpx

        if self.process is None:
            raise RuntimeError("server process has not been launched")
        deadline = time.monotonic() + self.startup_timeout_seconds
        last_error = "no response"
        with httpx.Client(timeout=5.0) as client:
            while time.monotonic() < deadline:
                return_code = self.process.poll()
                if return_code is not None:
                    raise ServerStartError(
                        f"server exited with code {return_code}: {self.log_tail()}"
                    )
                try:
                    response = client.get(f"{self.base_url}/v1/models")
                    if response.status_code == 200:
                        body = response.json()
                        model_ids = [
                            item.get("id")
                            for item in body.get("data", [])
                            if isinstance(item, dict)
                        ]
                        if self.served_model_name in model_ids:
                            return
                        last_error = (
                            f"ready endpoint omitted served model {self.served_model_name!r}: "
                            f"{model_ids}"
                        )
                    else:
                        last_error = f"HTTP {response.status_code}"
                except Exception as exc:
                    last_error = f"{type(exc).__name__}: {exc}"
                time.sleep(2.0)
        raise ServerStartError(
            f"server did not become ready: {last_error}; log tail: {self.log_tail()}"
        )

    def stop(self) -> None:
        process = self.process
        if process is not None and process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=30.0)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait(timeout=30.0)
        if self._log_handle is not None:
            self._log_handle.close()
            self._log_handle = None

    def log_tail(self, limit: int = 8000) -> str:
        if self._log_handle is not None:
            self._log_handle.flush()
        if not self.log_path.exists():
            return ""
        data = self.log_path.read_bytes()
        return data[-limit:].decode("utf-8", errors="replace")

    def log_digest(self) -> str:
        if self._log_handle is not None:
            self._log_handle.flush()
        digest = hashlib.sha256()
        with self.log_path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
        return digest.hexdigest()


def run_version_probe(command: list[str], *, timeout_seconds: float = 60.0) -> dict[str, Any]:
    """Execute the runtime's own version command and retain exact output."""

    if not command:
        raise ValueError("version probe command must not be empty")
    completed = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=timeout_seconds,
    )
    stdout = completed.stdout.decode("utf-8", errors="replace").strip()
    stderr = completed.stderr.decode("utf-8", errors="replace").strip()
    if completed.returncode != 0:
        raise RuntimeError(
            f"version probe failed with code {completed.returncode}: {stderr or stdout}"
        )
    if not stdout:
        raise RuntimeError("version probe returned no version text")
    return {
        "command": command,
        "stdout": stdout,
        "stderr": stderr,
        "returncode": completed.returncode,
    }


def expand_command_template(
    template: list[str],
    values: dict[str, Any],
    *,
    required_placeholders: tuple[str, ...],
) -> list[str]:
    """Expand a candidate command while requiring fairness-critical arguments."""

    if not template or not all(isinstance(item, str) and item for item in template):
        raise ValueError("command template must be a nonempty string array")
    joined = "\n".join(template)
    missing = [name for name in required_placeholders if "{" + name + "}" not in joined]
    if missing:
        raise ValueError(f"candidate command is missing required placeholders: {missing}")
    try:
        command = [item.format_map(values) for item in template]
    except KeyError as exc:
        raise ValueError(f"candidate command uses unknown placeholder {exc.args[0]!r}") from exc
    offload_flags = (
        "--cpu-offload-gb",
        "--offload-folder",
        "--swap-space",
    )
    for index, item in enumerate(command):
        if any(item == flag or item.startswith(flag + "=") for flag in offload_flags):
            raise ValueError("candidate command requests weight or cache offload")
        if item == "--load-format" and index + 1 < len(command):
            if command[index + 1].lower() == "dummy":
                raise ValueError("candidate command requests dummy weights")
        if item.lower() == "--load-format=dummy":
            raise ValueError("candidate command requests dummy weights")
    return command


def wait_for_gpu_release(
    probe: GpuMemoryProbe,
    *,
    at_most_bytes: int,
    timeout_seconds: float,
) -> int:
    """Require the prior runtime to release the shared GPU before the next side."""

    deadline = time.monotonic() + timeout_seconds
    last = probe.used_bytes()
    while time.monotonic() < deadline:
        last = probe.used_bytes()
        if last <= at_most_bytes:
            return last
        time.sleep(1.0)
    raise RuntimeError(
        f"GPU memory did not return below {at_most_bytes} bytes; last reading was {last}"
    )


def python_package_version_command(package: str) -> list[str]:
    return [
        sys.executable,
        "-c",
        "import importlib.metadata as m; print(m.version(" + repr(package) + "))",
    ]
