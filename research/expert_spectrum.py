"""WARNING: legacy diagnostic only; this script must not be cited as evidence.

Only the routed-expert matrices studied here are MXFP4. K3's shared experts,
attention, latent projections, and embeddings remain BF16 and total 114.4 GB;
two shared experts participate alongside 16 routed experts per token.

This legacy raw-weight sketch ignores the hidden-unit permutation gauge described
in ``expert_spectrum_v2.py``. Its apparent spectrum cannot decide whether a shared
basis exists. The script is retained only to reproduce historical diagnostics.

Method, so it runs on a laptop against a 1.5 TB model:
  * read only the tensors we need, by HTTP range against the safetensors shard,
    using byte offsets from the shard header;
  * dequantize the on-hub MXFP4-style format (4-bit E2M1 elements, one U8 E8M0
    exponent per 32-element group);
  * never hold 896 full matrices - project each to a small two-sided sketch
    S_e = P^T W_e R, which preserves inner products up to JL variance;
  * the Gram gives the spectrum of the flattened sketches only, not the rank of
    hidden-unit-aligned factor stacks;
  * compare against a Marchenko-Pastur baseline built from norm-matched random
    matrices. This historical control does not repair the permutation gauge.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import struct
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
dequantize_mxfp4 = importlib.import_module(
    "engine.k3ref.dequant"
).dequantize_mxfp4

BASE = "https://huggingface.co/moonshotai/Kimi-K3/resolve/main"
INDEX = "model.safetensors.index.json"

def http(url: str, rng: tuple[int, int] | None = None, retries: int = 4) -> bytes:
    headers = {"User-Agent": "k3-spectrum"}
    if rng:
        headers["Range"] = f"bytes={rng[0]}-{rng[1]}"
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=headers)
            return urllib.request.urlopen(req, timeout=180).read()
        except Exception as exc:  # transient CDN failures are common at this size
            last = exc
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"GET failed after {retries}: {last}")


def load_header(shard: str, cache: Path) -> tuple[dict, int]:
    """Safetensors header plus the byte offset where the data block starts."""
    hp = cache / f"{shard}.header.json"
    if hp.exists():
        d = json.loads(hp.read_text(encoding="utf-8"))
        return d["header"], d["data_start"]
    n = struct.unpack("<Q", http(f"{BASE}/{shard}", (0, 7)))[0]
    header = json.loads(http(f"{BASE}/{shard}", (8, 8 + n - 1)))
    header.pop("__metadata__", None)
    hp.write_text(json.dumps({"header": header, "data_start": 8 + n}), encoding="utf-8")
    return header, 8 + n


def dequant_mxfp4(packed: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """NumPy adapter around the canonical PyTorch MXFP4 codec."""
    packed_tensor = torch.tensor(packed, dtype=torch.uint8)
    scale_tensor = torch.tensor(scale, dtype=torch.uint8)
    return dequantize_mxfp4(packed_tensor, scale_tensor).cpu().numpy()


def raw_cache_path(cache: Path, tensor_name: str) -> Path:
    return cache / (tensor_name.replace(".", "_") + ".npy")


def fetch_expert(shard: str, header: dict, data_start: int, name: str, cache: Path) -> np.ndarray:
    """Range-read one packed tensor and its scale, return the dequantized matrix."""
    out = cache / (name.replace(".", "_") + ".npy")
    mats = {}
    for suffix in ("weight_packed", "weight_scale"):
        tensor_name = f"{name}.{suffix}"
        cached = raw_cache_path(cache, tensor_name)
        if cached.exists():
            mats[suffix] = np.load(cached)
            continue
        meta = header[tensor_name]
        a, b = meta["data_offsets"]
        raw = http(f"{BASE}/{shard}", (data_start + a, data_start + b - 1))
        mats[suffix] = np.frombuffer(raw, dtype=np.uint8).reshape(meta["shape"])
        np.save(cached, mats[suffix])
    if out.exists():
        return np.load(out)
    w = dequant_mxfp4(mats["weight_packed"], mats["weight_scale"])
    np.save(out, w.astype(np.float16))
    return w.astype(np.float16)


def effective_rank(ev: np.ndarray, frac: float) -> int:
    c = np.cumsum(ev) / ev.sum()
    return int(np.searchsorted(c, frac) + 1)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--layer", type=int, default=12)
    ap.add_argument("--proj", default="w1", choices=["w1", "w2", "w3"])
    ap.add_argument("--experts", type=int, default=896)
    ap.add_argument("--sketch", type=int, default=128, help="two-sided sketch dim")
    ap.add_argument("--cache", default="research/.cache")
    args = ap.parse_args()

    cache = Path(args.cache)
    cache.mkdir(parents=True, exist_ok=True)

    print("fetching weight index ...", flush=True)
    ip = cache / INDEX
    if not ip.exists():
        ip.write_bytes(http(f"{BASE}/{INDEX}"))
    wm = json.loads(ip.read_text(encoding="utf-8"))["weight_map"]

    stem = f"language_model.model.layers.{args.layer}.block_sparse_moe.experts"
    key0 = f"{stem}.0.{args.proj}.weight_packed"
    if key0 not in wm:
        print(f"no such tensor: {key0}")
        return 1
    shard = wm[key0]
    print(f"layer {args.layer} {args.proj} lives in {shard}", flush=True)

    header, data_start = load_header(shard, cache)

    rng = np.random.default_rng(20260728)
    P = R = None
    sketches = []
    t0 = time.time()
    bytes_pulled = 0

    for e in range(args.experts):
        name = f"{stem}.{e}.{args.proj}"
        if f"{name}.weight_packed" not in header:
            print(f"  expert {e} not in this shard; stopping at {len(sketches)}")
            break
        w = fetch_expert(shard, header, data_start, name, cache).astype(np.float32)
        if P is None:
            rows, cols = w.shape
            P = rng.standard_normal((rows, args.sketch), dtype=np.float32) / np.sqrt(rows)
            R = rng.standard_normal((cols, args.sketch), dtype=np.float32) / np.sqrt(cols)
            print(f"  expert shape {w.shape}, sketching to {args.sketch}x{args.sketch}")
        sketches.append((P.T @ w @ R).ravel())
        bytes_pulled += w.size  # counted in elements; reported below as packed bytes
        if (e + 1) % 64 == 0:
            el = time.time() - t0
            print(f"  {e+1}/{args.experts} experts  {el:6.1f}s  "
                  f"({(e+1)/max(el,1e-9):.1f}/s)", flush=True)

    M = np.stack(sketches)
    n = M.shape[0]
    print(f"\nsketch matrix: {M.shape}  ({M.nbytes/1e6:.0f} MB)")

    # Row norms tell us whether experts differ mostly in scale or in direction.
    norms = np.linalg.norm(M, axis=1)
    Mn = M / norms[:, None]

    for label, X in (("RAW (scale included)", M), ("NORMALIZED (direction only)", Mn)):
        G = X @ X.T
        ev = np.linalg.eigvalsh(G)[::-1]
        ev = np.clip(ev, 0, None)
        print(f"\n=== {label} ===")
        print("  top 12 eigenvalue fractions:",
              np.array2string(ev[:12] / ev.sum(), precision=4, suppress_small=True))
        for f in (0.50, 0.90, 0.95, 0.99):
            print(f"  effective rank @ {int(f*100)}% energy: {effective_rank(ev, f)} / {n}")
        part = ev / ev.sum()
        print(f"  participation ratio: {1.0/np.sum(part**2):.1f}")

    # Marchenko-Pastur control: same shape, same row norms, no structure.
    ctrl = rng.standard_normal(M.shape, dtype=np.float32)
    ctrl = ctrl / np.linalg.norm(ctrl, axis=1)[:, None]
    ev_c = np.clip(np.linalg.eigvalsh(ctrl @ ctrl.T)[::-1], 0, None)
    print("\n=== RANDOM CONTROL (Marchenko-Pastur) ===")
    for f in (0.50, 0.90, 0.95, 0.99):
        print(f"  effective rank @ {int(f*100)}% energy: {effective_rank(ev_c, f)} / {n}")
    print(f"  participation ratio: {1.0/np.sum((ev_c/ev_c.sum())**2):.1f}")

    print("\nWARNING: NO COMPRESSION VERDICT")
    print("  Raw expert sketches are not aligned across hidden-unit permutations.")
    print("  Do not cite this spectrum as evidence for or against factorization.")

    np.save(cache / f"spectrum_layer{args.layer}_{args.proj}.npy", M)
    print(f"\nsketches saved -> {cache}/spectrum_layer{args.layer}_{args.proj}.npy")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
