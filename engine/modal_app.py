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
"""

from __future__ import annotations

import json
import os
import struct
import time

import modal

APP = modal.App("k3-engine")

# torch only where a GPU is actually used; the fetch path stays tiny and cheap.
BASE_IMAGE = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("numpy>=2.0", "httpx>=0.27", "huggingface_hub>=0.26")
)
GPU_IMAGE = BASE_IMAGE.pip_install("torch>=2.5")

WEIGHTS = modal.Volume.from_name("k3-weights", create_if_missing=True)
VOL = "/weights"

REPO = "moonshotai/Kimi-K3"
BASE_URL = f"https://huggingface.co/{REPO}/resolve/main"

# 4-bit E2M1 codes; 8..15 are the negatives of 0..7. One U8 E8M0 exponent per 32.
E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]


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


def _fname(tensor_name: str) -> str:
    """Volume filename for a tensor. One definition, used by writer and reader,
    because deriving it twice is how the two halves drift apart."""
    return tensor_name.split(".", 3)[-1].replace(".", "__") + ".bin"


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
    stem = f"language_model.model.layers.{layer}.block_sparse_moe"
    shard = wm[f"{stem}.experts.0.w1.weight_packed"]
    header, start = _header(shard)
    dest = f"{VOL}/layer{layer}"
    os.makedirs(dest, exist_ok=True)

    names: list[str] = []
    for e in range(experts):
        for proj in ("w1", "w2", "w3"):
            for suffix in ("weight_packed", "weight_scale"):
                n = f"{stem}.experts.{e}.{proj}.{suffix}"
                if n in header:
                    names.append(n)
    # The per-layer skeleton: latent projections, router, shared experts, norms.
    for n in header:
        if ".experts." not in n and f".layers.{layer}." in n:
            names.append(n)

    def grab(name: str) -> int:
        out = f"{dest}/{_fname(name)}"
        if os.path.exists(out):
            return 0
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
    import numpy as np
    import torch

    dest = f"{VOL}/layer{layer}"
    if not os.path.isdir(dest):
        return {"error": f"{dest} missing; run fetch_layer --layer {layer} first"}

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    codes = torch.tensor(E2M1 + [-v for v in E2M1], dtype=torch.float32, device=dev)

    stem = f"language_model.model.layers.{layer}.block_sparse_moe"

    def load(tensor_name: str) -> tuple[torch.Tensor, dict]:
        path = f"{dest}/{_fname(tensor_name)}"
        with open(path + ".meta", "r", encoding="utf-8") as fh:
            meta = json.load(fh)
        raw = np.fromfile(path, dtype=np.uint8)
        return torch.from_numpy(raw.copy()).to(dev).reshape(meta["shape"]), meta

    def dequant(tensor: str) -> torch.Tensor:
        packed, _ = load(f"{tensor}.weight_packed")
        scale, _ = load(f"{tensor}.weight_scale")
        rows, half = packed.shape
        vals = torch.empty((rows, half * 2), dtype=torch.long, device=dev)
        vals[:, 0::2] = (packed & 0x0F).long()
        vals[:, 1::2] = (packed >> 4).long()
        w = codes[vals]
        exp = torch.exp2(scale.to(torch.int16).float() - 127.0)
        return w * exp.repeat_interleave(w.shape[1] // scale.shape[1], dim=1)

    stats = []
    for e in range(experts):
        w1 = dequant(f"{stem}.experts.{e}.w1")
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


@APP.local_entrypoint()
def main(action: str = "probe", layer: int = 12, experts: int = 8):
    if action == "probe":
        probe.remote()
    elif action == "fetch":
        fetch_layer.remote(layer=layer)
    elif action == "run":
        run_layer.remote(layer=layer, experts=experts)
    else:
        print(f"unknown action {action!r}; use probe | fetch | run")
