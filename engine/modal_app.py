"""Kimi K3 local inference engine - Modal harness.

Nothing here runs on a laptop. Modal gives us GPUs and, more importantly, a
persistent Volume so the 1.4 TB of weights is fetched once and reused.

Cost discipline: every function declares the smallest GPU that can do its job.
`probe` and `fetch_layer` need no GPU at all. Only `run_layer` and above touch
one, and the full-model functions are gated behind an explicit flag so nobody
starts an 8xH200 container by autocomplete.

    modal run engine/modal_app.py::probe
    modal run engine/modal_app.py::fetch_layer --layer 12
    modal run engine/modal_app.py::run_layer --layer 12
    modal run engine/modal_app.py::run_reference_layer --layer 12
"""

from __future__ import annotations

import json
import os
import struct
import time
from pathlib import Path

import modal

APP = modal.App("k3-engine")

# torch only where a GPU is actually used; the fetch path stays tiny and cheap.
BASE_IMAGE = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("numpy>=2.0", "httpx>=0.27", "huggingface_hub>=0.26")
)
GPU_IMAGE = BASE_IMAGE.pip_install("torch>=2.5").add_local_dir(
    Path(__file__).parent / "k3ref",
    remote_path="/root/k3ref",
).add_local_file(
    Path(__file__).parents[1] / "reference" / "config.json",
    remote_path="/root/reference/config.json",
)

WEIGHTS = modal.Volume.from_name("k3-weights", create_if_missing=True)
VOL = "/weights"

REPO = "moonshotai/Kimi-K3"
BASE_URL = f"https://huggingface.co/{REPO}/resolve/main"

# The MXFP4 codec and the raw-tensor reader live in engine/k3ref so the Modal
# harness and the reference implementation cannot drift apart.


def _http(url: str, rng: tuple[int, int] | None = None, retries: int = 5) -> bytes:
    import httpx

    headers = {"user-agent": "k3-engine"}
    if rng:
        headers["range"] = f"bytes={rng[0]}-{rng[1]}"
    last = None
    for attempt in range(retries):
        try:
            r = httpx.get(url, headers=headers, timeout=300.0, follow_redirects=True)
            r.raise_for_status()
            return r.content
        except Exception as exc:
            last = exc
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"GET {url} failed: {last}")


def _index() -> dict:
    """weight_map, cached on the Volume so we fetch it once."""
    p = f"{VOL}/model.safetensors.index.json"
    if not os.path.exists(p):
        os.makedirs(VOL, exist_ok=True)
        with open(p, "wb") as fh:
            fh.write(_http(f"{BASE_URL}/model.safetensors.index.json"))
        WEIGHTS.commit()
    with open(p, "r", encoding="utf-8") as fh:
        return json.load(fh)["weight_map"]


def _header(shard: str) -> tuple[dict, int]:
    p = f"{VOL}/{shard}.header.json"
    if os.path.exists(p):
        with open(p, "r", encoding="utf-8") as fh:
            d = json.load(fh)
        return d["header"], d["data_start"]
    n = struct.unpack("<Q", _http(f"{BASE_URL}/{shard}", (0, 7)))[0]
    header = json.loads(_http(f"{BASE_URL}/{shard}", (8, 8 + n - 1)))
    header.pop("__metadata__", None)
    with open(p, "w", encoding="utf-8") as fh:
        json.dump({"header": header, "data_start": 8 + n}, fh)
    WEIGHTS.commit()
    return header, 8 + n


@APP.function(image=BASE_IMAGE, volumes={VOL: WEIGHTS}, timeout=900)
def probe() -> dict:
    """Cheapest possible check: can we reach HF, read a header, and write the Volume."""
    t0 = time.time()
    wm = _index()
    layers = sorted(
        {
            int(k.split(".layers.")[1].split(".")[0])
            for k in wm
            if ".layers." in k and "block_sparse_moe.experts.0.w1.weight_packed" in k
        }
    )
    shard = wm["language_model.model.layers.12.block_sparse_moe.experts.0.w1.weight_packed"]
    header, start = _header(shard)
    expert_tensors = [k for k in header if "experts." in k and k.endswith("weight_packed")]
    out = {
        "tensors_in_index": len(wm),
        "moe_layers": len(layers),
        "moe_layer_range": [layers[0], layers[-1]] if layers else [],
        "probe_shard": shard,
        "tensors_in_shard": len(header),
        "packed_expert_tensors_in_shard": len(expert_tensors),
        "data_start": start,
        "seconds": round(time.time() - t0, 1),
    }
    print(json.dumps(out, indent=2))
    return out


@APP.function(image=BASE_IMAGE, volumes={VOL: WEIGHTS}, timeout=60 * 60 * 4, cpu=8.0)
def fetch_layer(layer: int = 12, experts: int = 896) -> dict:
    """Pull one full MoE layer onto the Volume: all experts plus the shared skeleton.

    One layer is the unit of work for the reference implementation, and at
    ~15.7 GB of experts it is cheap enough to iterate on.
    """
    from concurrent.futures import ThreadPoolExecutor

    wm = _index()
    layer_prefix = f"language_model.model.layers.{layer}."
    dest = f"{VOL}/layer{layer}"
    os.makedirs(dest, exist_ok=True)

    names: list[str] = []
    for name in wm:
        if not name.startswith(layer_prefix):
            continue
        if ".experts." in name:
            expert_id = int(name.split(".experts.", 1)[1].split(".", 1)[0])
            if expert_id >= experts:
                continue
        names.append(name)
    names.sort()

    # A K3 layer can cross shard boundaries, so resolve every layer shard first.
    shard_data = {shard: _header(shard) for shard in {wm[name] for name in names}}

    def grab(name: str) -> int:
        relative_name = name[len(layer_prefix):]
        out = f"{dest}/{relative_name.replace('.', '__')}.bin"
        if os.path.exists(out) and os.path.exists(out + ".meta"):
            return 0
        shard = wm[name]
        header, start = shard_data[shard]
        meta = header[name]
        a, b = meta["data_offsets"]
        raw = _http(f"{BASE_URL}/{shard}", (start + a, start + b - 1))
        with open(out + ".meta", "w", encoding="utf-8") as fh:
            json.dump({"shape": meta["shape"], "dtype": meta["dtype"], "name": name}, fh)
        with open(out, "wb") as fh:
            fh.write(raw)
        return len(raw)

    t0 = time.time()
    total = 0
    with ThreadPoolExecutor(max_workers=32) as pool:
        for i, got in enumerate(pool.map(grab, names)):
            total += got
            if (i + 1) % 512 == 0:
                el = time.time() - t0
                print(f"  {i+1}/{len(names)}  {total/1e9:.1f} GB  {total/1e6/max(el,1e-9):.0f} MB/s",
                      flush=True)
    WEIGHTS.commit()
    out = {
        "layer": layer,
        "tensors": len(names),
        "bytes": total,
        "gb": round(total / 1e9, 2),
        "seconds": round(time.time() - t0, 1),
    }
    print(json.dumps(out, indent=2))
    return out


@APP.function(image=GPU_IMAGE, gpu="A10G", volumes={VOL: WEIGHTS}, timeout=60 * 30)
def run_layer(layer: int = 12, experts: int = 8) -> dict:
    """Dequantize on GPU and sanity-check the MXFP4 decode against the real weights.

    This is the first rung of the correctness ladder: before any kernel work we
    must prove we can turn K3's packed bytes back into numbers that look like
    trained weights rather than noise.
    """
    import torch

    from k3ref.dequant import dequantize_mxfp4
    from k3ref.weights import RawTensorStore

    dest = f"{VOL}/layer{layer}"
    if not os.path.isdir(dest):
        return {"error": f"{dest} missing; run fetch_layer --layer {layer} first"}

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    store = RawTensorStore(dest)

    def dequant(expert: int) -> torch.Tensor:
        stem = f"layers.{layer}.block_sparse_moe.experts.{expert}.w1"
        packed = store.load(f"{stem}.weight_packed", device=dev)
        scale = store.load(f"{stem}.weight_scale", device=dev)
        return dequantize_mxfp4(packed, scale)

    stats = []
    for e in range(experts):
        w1 = dequant(e)
        stats.append(
            {
                "expert": e,
                "shape": list(w1.shape),
                "absmax": float(w1.abs().max()),
                "rms": float(w1.pow(2).mean().sqrt()),
                "zero_frac": float((w1 == 0).float().mean()),
            }
        )

    rms = [s["rms"] for s in stats]
    out = {
        "device": torch.cuda.get_device_name(0) if dev == "cuda" else "cpu",
        "experts_checked": len(stats),
        "shape": stats[0]["shape"] if stats else None,
        "rms_mean": round(sum(rms) / len(rms), 6),
        "rms_spread": round(max(rms) - min(rms), 6),
        "zero_frac_mean": round(sum(s["zero_frac"] for s in stats) / len(stats), 4),
        "first": stats[0] if stats else None,
    }
    print(json.dumps(out, indent=2))
    return out


@APP.function(image=GPU_IMAGE, gpu="H100", volumes={VOL: WEIGHTS}, timeout=60 * 60)
def run_reference_layer(layer: int = 12, sequence_length: int = 2, seed: int = 0) -> dict:
    """Run the plain PyTorch layer against fetched real weights and report activations."""
    import torch

    from k3ref.config import K3LayerConfig
    from k3ref.layer import K3ReferenceLayer

    dest = f"{VOL}/layer{layer}"
    if not os.path.isdir(dest):
        return {"error": f"{dest} missing; run fetch_layer --layer {layer} first"}
    if not torch.cuda.is_available():
        return {"error": "run_reference_layer requires a CUDA GPU"}

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.reset_peak_memory_stats()
    config = K3LayerConfig.from_json("/root/reference/config.json")
    load_started = time.time()
    reference_layer = K3ReferenceLayer.from_directory(
        dest,
        layer,
        config=config,
        device="cuda",
        dtype=torch.bfloat16,
    )
    load_seconds = time.time() - load_started
    hidden_states = torch.randn(
        1,
        sequence_length,
        config.hidden_size,
        device="cuda",
        dtype=torch.bfloat16,
    )
    attention_mask = torch.ones(
        1, sequence_length, device="cuda", dtype=torch.long
    )

    def activation_stats(tensor: torch.Tensor) -> dict:
        values = tensor.float()
        return {
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "mean": float(values.mean()),
            "std": float(values.std()),
            "rms": float(values.square().mean().sqrt()),
            "absmax": float(values.abs().max()),
            "finite": bool(torch.isfinite(values).all()),
        }

    forward_started = time.time()
    with torch.inference_mode():
        result = reference_layer(
            hidden_states,
            attention_mask=attention_mask,
            return_aux=True,
        )
    torch.cuda.synchronize()
    forward_seconds = time.time() - forward_started
    selected = result.router_indices
    router_weights = result.router_weights
    out = {
        "layer": layer,
        "attention": "kda" if reference_layer.is_kda else "mla",
        "device": torch.cuda.get_device_name(0),
        "sequence_length": sequence_length,
        "load_seconds": round(load_seconds, 3),
        "forward_seconds": round(forward_seconds, 3),
        "peak_allocated_gb": round(torch.cuda.max_memory_allocated() / 1e9, 3),
        "input": activation_stats(hidden_states),
        "output": activation_stats(result.hidden_states),
        "router": {
            "shape": list(selected.shape),
            "unique_experts": int(selected.unique().numel()),
            "weight_min": float(router_weights.min()),
            "weight_max": float(router_weights.max()),
            "weight_sum_min": float(router_weights.sum(-1).min()),
            "weight_sum_max": float(router_weights.sum(-1).max()),
        },
        "block_residual_shape": (
            list(result.block_residual.shape)
            if result.block_residual is not None
            else None
        ),
    }
    if hasattr(result.attention_state, "recurrent"):
        out["recurrent_state"] = activation_stats(result.attention_state.recurrent)
    print(json.dumps(out, indent=2))
    return out


@APP.local_entrypoint()
def main(
    action: str = "probe",
    layer: int = 12,
    experts: int = 8,
    sequence_length: int = 2,
):
    if action == "probe":
        probe.remote()
    elif action == "fetch":
        fetch_layer.remote(layer=layer)
    elif action == "run":
        run_layer.remote(layer=layer, experts=experts)
    elif action == "reference":
        run_reference_layer.remote(layer=layer, sequence_length=sequence_length)
    else:
        print(f"unknown action {action!r}; use probe | fetch | run | reference")
