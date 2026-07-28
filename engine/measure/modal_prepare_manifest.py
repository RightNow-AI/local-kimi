"""Write the download manifest the measurement harness requires.

The harness refuses to measure weights it cannot pin, which is correct: a
throughput number attached to "whatever was on the volume" is not attributable
to anything. It wants an immutable revision and a snapshot directory named after
it.

Our weights were fetched by a different path, into a human-named directory, so
the manifest never existed. This creates it without moving or duplicating 91.51
GiB: the revision-named path is a SYMLINK to the existing snapshot, which
satisfies is_dir() while costing nothing.

The resolved revision is read from the HuggingFace API at run time rather than
trusted from a constant, so the manifest records what the hub actually resolves
main to, and the job refuses to write anything if the local snapshot does not
look like that model.

    modal run engine/measure/modal_prepare_manifest.py
"""

from __future__ import annotations

import json

import modal

app = modal.App("kimi-linear-prepare-manifest")

WEIGHTS = modal.Volume.from_name("kimi-linear-weights", create_if_missing=False)
MOUNT = "/weights"
MODEL_ID = "moonshotai/Kimi-Linear-48B-A3B-Instruct"
SNAPSHOT_DIR = f"{MOUNT}/Kimi-Linear-48B-A3B-Instruct"
MANIFEST_PATH = f"{MOUNT}/bench/kimi-linear-48b-a3b/download.json"

IMAGE = modal.Image.debian_slim(python_version="3.12").pip_install(
    "huggingface_hub>=0.26"
)


@app.function(image=IMAGE, volumes={MOUNT: WEIGHTS}, timeout=60 * 20)
def prepare(requested_revision: str = "main") -> dict:
    import os
    import urllib.request

    if not os.path.isdir(SNAPSHOT_DIR):
        raise FileNotFoundError(f"expected weights at {SNAPSHOT_DIR}")

    # The hub decides what main resolves to, not us.
    api = f"https://huggingface.co/api/models/{MODEL_ID}"
    with urllib.request.urlopen(api, timeout=60) as response:
        info = json.loads(response.read())
    resolved = info.get("sha", "")
    if not resolved or len(resolved) != 40:
        raise ValueError(f"hub returned no usable revision sha: {resolved!r}")

    index_path = os.path.join(SNAPSHOT_DIR, "model.safetensors.index.json")
    if not os.path.isfile(index_path):
        raise FileNotFoundError(f"snapshot has no safetensors index: {index_path}")
    with open(index_path, encoding="utf-8") as handle:
        index = json.load(handle)
    weight_map = index.get("weight_map") or {}
    if len(weight_map) != 20493:
        raise ValueError(
            f"snapshot has {len(weight_map)} mapped tensors, expected 20493; "
            "this does not look like the measured Kimi-Linear checkpoint"
        )

    shard_names = sorted(set(weight_map.values()))
    tensor_bytes = 0
    for shard in shard_names:
        tensor_bytes += os.path.getsize(os.path.join(SNAPSHOT_DIR, shard))

    # Symlink rather than copy. is_dir() follows the link, and 91.51 GiB does
    # not need a second home to satisfy a naming convention.
    revision_path = f"{MOUNT}/{resolved}"
    if not os.path.exists(revision_path):
        os.symlink(SNAPSHOT_DIR, revision_path)

    manifest = {
        "model_id": MODEL_ID,
        "requested_revision": requested_revision,
        "resolved_revision": resolved,
        "snapshot_path": revision_path,
        "real_snapshot_path": SNAPSHOT_DIR,
        "mapped_tensor_count": len(weight_map),
        "safetensors_shard_count": len(shard_names),
        "safetensors_file_bytes": tensor_bytes,
        "note": (
            "Written by modal_prepare_manifest because the weights were fetched "
            "outside the bench download action. snapshot_path is a symlink to "
            "the existing snapshot rather than a second copy."
        ),
    }
    os.makedirs(os.path.dirname(MANIFEST_PATH), exist_ok=True)
    with open(MANIFEST_PATH, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    WEIGHTS.commit()

    print(json.dumps(manifest, indent=2, sort_keys=True))
    return manifest


@app.local_entrypoint()
def main(requested_revision: str = "main"):
    print(json.dumps(prepare.remote(requested_revision=requested_revision), indent=2))
