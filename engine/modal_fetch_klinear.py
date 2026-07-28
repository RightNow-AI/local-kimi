"""Fetch Kimi-Linear-48B-A3B-Instruct onto a Modal volume.

The measurement harness needs both sides on one GPU in one invocation, and both
sides need the weights already local. Pulling 91.5 GiB inside the timed job
would put network variance inside the measurement, so it happens here instead.

    modal run engine/modal_fetch_klinear.py

Idempotent: hf snapshot_download resumes, so a re-run after an interruption
costs only the missing shards.
"""

from __future__ import annotations

import modal

app = modal.App("k3-fetch-klinear")

MODEL_ID = "moonshotai/Kimi-Linear-48B-A3B-Instruct"
VOLUME = modal.Volume.from_name("kimi-linear-weights", create_if_missing=True)
MOUNT = "/weights"

IMAGE = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("huggingface_hub[hf_transfer]>=0.26")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)


@app.function(
    image=IMAGE,
    volumes={MOUNT: VOLUME},
    timeout=60 * 60 * 4,
    cpu=8.0,
    memory=16384,
)
def fetch() -> dict:
    """Snapshot the repo and report what actually landed, byte for byte."""
    import os

    from huggingface_hub import snapshot_download

    target = f"{MOUNT}/{MODEL_ID.split('/')[-1]}"
    path = snapshot_download(
        repo_id=MODEL_ID,
        local_dir=target,
        max_workers=8,
    )

    files: list[tuple[str, int]] = []
    total = 0
    for root, _, names in os.walk(path):
        for name in names:
            full = os.path.join(root, name)
            size = os.path.getsize(full)
            total += size
            files.append((os.path.relpath(full, path), size))
    files.sort()

    VOLUME.commit()

    shards = [f for f, _ in files if f.endswith(".safetensors")]
    report = {
        "path": path,
        "total_bytes": total,
        "total_gib": round(total / 2**30, 2),
        "file_count": len(files),
        "safetensors_shards": len(shards),
        "has_index": any(f.endswith("model.safetensors.index.json") for f, _ in files),
        "has_remote_code": sorted(
            f for f, _ in files if f in ("modeling_kimi.py", "configuration_kimi.py")
        ),
        "license_files": sorted(
            f for f, _ in files if f.upper().startswith(("LICENSE", "NOTICE"))
        ),
        "largest": files and max(files, key=lambda p: p[1]),
    }
    print(report)
    return report


@app.local_entrypoint()
def main():
    print(fetch.remote())
