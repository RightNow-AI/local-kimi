"""Run BF16 vLLM versus engine.klinear implementation validation on one H200.

Both child processes receive one protocol file containing the exact same prompt
token IDs and the exact same BF16 checkpoint identity. The child processes run
sequentially so each implementation releases all GPU state before the other
loads, while Modal keeps them on the same physical H200.

    modal run engine/modal_validate.py
"""

from __future__ import annotations

import json
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import modal

app = modal.App("kimi-linear-bf16-engine-validation")

SOURCE_VOLUME = modal.Volume.from_name("kimi-linear-weights", create_if_missing=False)
VALIDATION_VOLUME = modal.Volume.from_name(
    "kimi-linear-validation", create_if_missing=True
)
SOURCE_MOUNT = "/weights"
VALIDATION_MOUNT = "/validation"
MODEL_DIR = f"{SOURCE_MOUNT}/Kimi-Linear-48B-A3B-Instruct"

# KDA JIT-compiles CUDA at runtime, so the image must retain a CUDA toolchain.
# The official vLLM 0.26.0 image supplies both that toolchain and the pinned
# reference implementation. It has python3 but no `python` on PATH, so the
# symlink must be created before pip_install and Modal image introspection.
IMAGE = (
    modal.Image.from_registry("vllm/vllm-openai:v0.26.0")
    .entrypoint([])
    .run_commands('ln -sf "$(command -v python3)" /usr/local/bin/python')
    .pip_install(
        "safetensors>=0.5,<1",
        "numpy>=2,<3",
        "tiktoken>=0.9,<1",
        "blobfile>=3,<4",
    )
    .env({"VLLM_USE_V1": "1", "CUDA_HOME": "/usr/local/cuda"})
    .add_local_dir(Path(__file__).parent, remote_path="/root/engine")
    # k3 is required by engine/serve, which reuses k3/toolcalls.py for the
    # K2-family tool-call parser rather than carrying a third copy of it. Every
    # Modal job that imports engine.serve needs this mount, and forgetting it
    # fails only inside the container.
    .add_local_dir(Path(__file__).parent.parent / "k3", remote_path="/root/k3")
)


def _run(command: list[str]) -> None:
    subprocess.run(command, check=True)


def _side_command(
    *,
    side: str,
    protocol_path: Path,
    raw_dir: Path,
    result_path: Path,
) -> list[str]:
    return [
        sys.executable,
        "-m",
        "engine.validate.runner",
        "--side",
        side,
        "--model-path",
        MODEL_DIR,
        "--protocol-json",
        str(protocol_path),
        "--output-dir",
        str(raw_dir),
        "--result-json",
        str(result_path),
    ]


@app.function(
    image=IMAGE,
    gpu="H200",
    volumes={
        SOURCE_MOUNT: SOURCE_VOLUME,
        VALIDATION_MOUNT: VALIDATION_VOLUME,
    },
    cpu=16.0,
    memory=262144,
    timeout=60 * 60 * 12,
)
def measure() -> dict:
    from engine.validate.analyze import build_evidence
    from engine.validate.protocol import build_protocol, write_json

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id += f"-{uuid.uuid4().hex[:8]}"
    run_root = Path(VALIDATION_MOUNT) / "runs" / run_id
    raw_dir = run_root / "raw"
    raw_dir.mkdir(parents=True, exist_ok=False)

    # This records the threshold and exact tokenized inputs before either model
    # loads. Both side runners independently recheck the checkpoint identity.
    protocol_path = raw_dir / "protocol.json"
    protocol = build_protocol(Path(MODEL_DIR))
    write_json(protocol_path, protocol)
    VALIDATION_VOLUME.commit()

    candidate_path = raw_dir / "klinear-candidate.json"
    reference_path = raw_dir / "vllm-reference.json"
    _run(
        _side_command(
            side="klinear_candidate",
            protocol_path=protocol_path,
            raw_dir=raw_dir,
            result_path=candidate_path,
        )
    )
    VALIDATION_VOLUME.commit()
    _run(
        _side_command(
            side="vllm_reference",
            protocol_path=protocol_path,
            raw_dir=raw_dir,
            result_path=reference_path,
        )
    )
    VALIDATION_VOLUME.commit()

    evidence_path = run_root / "evidence.json"
    report_path = run_root / "RESULTS.md"
    evidence = build_evidence(
        protocol_path=protocol_path,
        reference_path=reference_path,
        candidate_path=candidate_path,
        output_path=evidence_path,
        report_path=report_path,
    )
    VALIDATION_VOLUME.commit()

    result = {
        "run_id": run_id,
        "verdict": evidence["verdict"],
        "evidence_path": str(evidence_path),
        "report_path": str(report_path),
        "threshold_checks": evidence["threshold_checks"],
        "greedy_summary": evidence["metrics"]["greedy_token_identity"]["summary"],
        "first_token_summary": evidence["metrics"]["first_token_distribution"]
        ["summary"],
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


@app.local_entrypoint()
def main() -> None:
    print(json.dumps(measure.remote(), indent=2, sort_keys=True))
