"""Run the controlled Kimi Linear BF16 versus INT4-round-trip experiment.

The source checkpoint is already present on ``kimi-linear-weights``. This job
does not download it. It creates a BF16 dequantized checkpoint on a separate
volume, then launches each vLLM side in a fresh child process so process exit
fully releases GPU state between loads.

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

# The official vLLM image already carries a correctly built vLLM and the CUDA
# toolchain this model needs at run time, so it is the right base. It ships
# python3 but no `python` on PATH, and Modal's pip_install shells out to
# `python -m pip`, which fails the build with "python: not found". Installing
# through python3 explicitly keeps the image's own interpreter, which is the one
# vLLM is installed into. Do not swap this for add_python: that would add a
# second interpreter without vLLM in it.
# The official vLLM image already carries a correctly built vLLM and the CUDA
# toolchain this model needs at RUN time, so it is the right base. It ships
# python3 but no `python` on PATH, which breaks Modal twice: pip_install shells
# out to `python -m pip`, and Modal separately introspects `python` to determine
# the image's Python version, failing with a ConflictError before the function
# ever starts. Symlinking first fixes both without adding a second interpreter.
# Do NOT use add_python here: that installs a fresh interpreter that does not
# have vLLM in it.
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
    dequantized_model = checkpoint["dequantized_checkpoint"]["path"]

    bf16_result = raw_dir / "bf16.json"
    _run(
        [
            sys.executable,
            "-m",
            "engine.accuracy.vllm_runner",
            "--model-path",
            SOURCE_MODEL,
            "--side",
            "bf16",
            "--output-dir",
            str(raw_dir),
            "--result-json",
            str(bf16_result),
        ]
    )
    ACCURACY_VOLUME.commit()

    dequantized_result = raw_dir / "int4-dequantized.json"
    _run(
        [
            sys.executable,
            "-m",
            "engine.accuracy.vllm_runner",
            "--model-path",
            dequantized_model,
            "--side",
            "int4_dequantized",
            "--output-dir",
            str(raw_dir),
            "--result-json",
            str(dequantized_result),
        ]
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
        "offline_verification": offline_verification,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


@app.local_entrypoint()
def main():
    print(json.dumps(measure.remote(), indent=2, sort_keys=True))
