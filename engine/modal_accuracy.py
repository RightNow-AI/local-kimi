"""Run the controlled Kimi Linear BF16 versus INT4-round-trip experiment.

The source checkpoint is already present on ``kimi-linear-weights``. This job
does not download it. It creates a BF16 dequantized checkpoint on a separate
volume and a config-owning BF16 view over the source shards. Both served configs
carry the same vLLM router-capture alias. Each vLLM side runs in a fresh child
process so process exit fully releases GPU state between loads.

    modal run engine/modal_accuracy.py
"""

from __future__ import annotations

import json
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import modal

app = modal.App("kimi-linear-int4-accuracy")

SOURCE_VOLUME = modal.Volume.from_name("kimi-linear-weights", create_if_missing=False)
ACCURACY_VOLUME = modal.Volume.from_name("kimi-linear-accuracy", create_if_missing=True)
SOURCE_MOUNT = "/weights"
ACCURACY_MOUNT = "/accuracy"
SOURCE_MODEL = f"{SOURCE_MOUNT}/Kimi-Linear-48B-A3B-Instruct"

# The official vLLM image is the right base: it carries a correctly built vLLM
# and the CUDA toolchain this model needs at RUN time, because Kimi-Linear's KDA
# path JIT-compiles kernels at startup. It ships python3 but no `python` on
# PATH, which breaks Modal twice: pip_install shells out to `python -m pip` and
# fails the build with "python: not found", and Modal separately introspects
# `python` to determine the image Python version and fails with a ConflictError
# before the function starts. The symlink fixes both.
#
# Do NOT replace this with add_python. That installs a second interpreter which
# does not have vLLM in it. This exact fix has now been lost once to a file
# overwrite, so it is spelled out rather than left as a one-line incantation.
IMAGE = (
    modal.Image.from_registry("vllm/vllm-openai:v0.26.0")
    .entrypoint([])
    .run_commands("ln -sf \"$(command -v python3)\" /usr/local/bin/python")
    .pip_install("safetensors>=0.5,<1", "numpy>=2,<3")
    .env({"VLLM_USE_V1": "1"})
    .add_local_dir(Path(__file__).parent, remote_path="/root/engine")
)


def _run(command: list[str]) -> None:
    subprocess.run(command, check=True)


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _side_command(
    *,
    model_path: str,
    side: str,
    raw_dir: Path,
    result_path: Path,
    router_reason_path: Path | None = None,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "engine.accuracy.vllm_runner",
        "--model-path",
        model_path,
        "--side",
        side,
        "--output-dir",
        str(raw_dir),
        "--result-json",
        str(result_path),
    ]
    if router_reason_path is not None:
        command.extend(["--router-unavailable-reason-json", str(router_reason_path)])
    return command


def _attempt_router_enabled_side(command: list[str], *, side: str) -> dict | None:
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    if completed.returncode == 0:
        if completed.stdout:
            print(completed.stdout, end="")
        return None
    return {
        "side": side,
        "returncode": completed.returncode,
        "stdout_tail": completed.stdout[-4000:],
        "stderr_tail": completed.stderr[-8000:],
    }


def _run_matched_sides(
    *,
    bf16_model: str,
    dequantized_model: str,
    raw_dir: Path,
    bf16_result: Path,
    dequantized_result: Path,
) -> dict | None:
    """Try capture on both sides, then rerun both without it if either fails."""
    bf16_enabled = _side_command(
        model_path=bf16_model,
        side="bf16",
        raw_dir=raw_dir,
        result_path=bf16_result,
    )
    candidate_enabled = _side_command(
        model_path=dequantized_model,
        side="int4_dequantized",
        raw_dir=raw_dir,
        result_path=dequantized_result,
    )
    failures = []
    failure = _attempt_router_enabled_side(bf16_enabled, side="bf16")
    if failure is not None:
        failures.append(failure)
    else:
        failure = _attempt_router_enabled_side(
            candidate_enabled,
            side="int4_dequantized",
        )
        if failure is not None:
            failures.append(failure)
    if not failures:
        return None

    fallback = {
        "available": False,
        "code": "ROUTER_ENABLED_SIDE_RUN_FAILED",
        "reason": (
            "At least one router-enabled side process failed after config preflight. "
            "Both sides were rerun with router capture disabled so greedy, perplexity, "
            "and next-token distribution metrics still use identical vLLM arguments. "
            "Router agreement remains unavailable and the overall verdict must fail."
        ),
        "failed_attempts": failures,
    }
    reason_path = raw_dir / "router-unavailable.json"
    _write_json(reason_path, fallback)
    for command in (
        _side_command(
            model_path=bf16_model,
            side="bf16",
            raw_dir=raw_dir,
            result_path=bf16_result,
            router_reason_path=reason_path,
        ),
        _side_command(
            model_path=dequantized_model,
            side="int4_dequantized",
            raw_dir=raw_dir,
            result_path=dequantized_result,
            router_reason_path=reason_path,
        ),
    ):
        _run(command)
    return fallback


@app.function(
    image=IMAGE,
    # The BF16 checkpoint is 91.51 GiB before runtime state and therefore does
    # not fit an H100 80GB. One H200 keeps both sides on the same physical GPU.
    gpu="H200",
    volumes={SOURCE_MOUNT: SOURCE_VOLUME, ACCURACY_MOUNT: ACCURACY_VOLUME},
    cpu=16.0,
    memory=262144,
    timeout=60 * 60 * 12,
)
def measure() -> dict:
    from engine.accuracy.analyze import build_evidence, verify_evidence_record
    from engine.accuracy.router_compat import validate_vllm_router_capture_config

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + f"-{uuid.uuid4().hex[:8]}"
    run_root = Path(ACCURACY_MOUNT) / "runs" / run_id
    raw_dir = run_root / "raw"
    raw_dir.mkdir(parents=True, exist_ok=False)

    checkpoint_result = raw_dir / "checkpoint.json"
    _run(
        [
            sys.executable,
            "-m",
            "engine.accuracy.checkpoint",
            "--source-dir",
            SOURCE_MODEL,
            "--output-root",
            f"{ACCURACY_MOUNT}/checkpoints",
            "--result-json",
            str(checkpoint_result),
        ]
    )
    ACCURACY_VOLUME.commit()
    with checkpoint_result.open("r", encoding="utf-8") as handle:
        checkpoint = json.load(handle)
    bf16_model = checkpoint["served_bf16_checkpoint"]["path"]
    dequantized_model = checkpoint["dequantized_checkpoint"]["path"]

    # This guard runs before either vLLM process. The H200 job must not spend a
    # multi-minute model load discovering that routed_experts_capturer cannot
    # read the served config. Both derived configs must satisfy the same rule.
    for side, model_path in (
        ("bf16", bf16_model),
        ("int4_dequantized", dequantized_model),
    ):
        config_path = Path(model_path) / "config.json"
        with config_path.open("r", encoding="utf-8") as handle:
            served_config = json.load(handle)
        validate_vllm_router_capture_config(served_config, config_path=config_path)
    if (
        checkpoint["served_bf16_checkpoint"]["config_sha256"]
        != checkpoint["dequantized_checkpoint"]["config_sha256"]
    ):
        raise ValueError("served BF16 and INT4-dequantized config bytes differ")

    bf16_result = raw_dir / "bf16.json"
    dequantized_result = raw_dir / "int4-dequantized.json"
    router_fallback = _run_matched_sides(
        bf16_model=bf16_model,
        dequantized_model=dequantized_model,
        raw_dir=raw_dir,
        bf16_result=bf16_result,
        dequantized_result=dequantized_result,
    )
    ACCURACY_VOLUME.commit()

    evidence_path = run_root / "evidence.json"
    evidence = build_evidence(
        reference_path=bf16_result,
        candidate_path=dequantized_result,
        checkpoint_path=checkpoint_result,
        output_path=evidence_path,
        run_root=run_root,
    )
    offline_verification = verify_evidence_record(evidence_path, artifact_root=run_root)
    ACCURACY_VOLUME.commit()

    result = {
        "run_id": run_id,
        "evidence_path": str(evidence_path),
        "verdict": evidence["verdict"],
        "metrics": evidence["metrics"],
        "plan_reconciliation": evidence["checkpoints"]["plan"]["reconciliation_status"],
        "router_fallback": router_fallback,
        "offline_verification": offline_verification,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


@app.local_entrypoint()
def main():
    print(json.dumps(measure.remote(), indent=2, sort_keys=True))
