"""Measure K3's routing distribution from the real router weights.

The concurrency model assumes uniform routing, which is an upper bound on the
expert union and therefore a lower bound on throughput. That assumption has been
carrying a lot of weight on the strength of a single observation (31 unique
experts from 32 draws). This replaces it with the real thing.

We cannot produce true hidden states without running the whole model, but we can
use the REAL learned router - gate.weight and the noaux_tc correction bias - and
drive it with hidden states drawn to match the statistics observed in an actual
layer run (rms ~1.0 after input_layernorm). Whatever specialisation the router
learned is baked into those weights, so if experts are unequally reachable it
will show here.

Reports the utilisation distribution, its entropy against the uniform maximum,
and the union growth curve that decides aggregate throughput.
"""

from __future__ import annotations

import argparse
import json
import struct
import urllib.request
from pathlib import Path

import numpy as np

BASE = "https://huggingface.co/moonshotai/Kimi-K3/resolve/main"
CACHE = Path("research/.cache")
TOP_K = 16
N_EXPERTS = 896


def http(url: str, rng=None) -> bytes:
    h = {"User-Agent": "k3-routing"}
    if rng:
        h["Range"] = f"bytes={rng[0]}-{rng[1]}"
    return urllib.request.urlopen(urllib.request.Request(url, headers=h), timeout=180).read()


def header(shard: str) -> tuple[dict, int]:
    p = CACHE / f"{shard}.header.json"
    if p.exists():
        d = json.loads(p.read_text(encoding="utf-8"))
        return d["header"], d["data_start"]
    n = struct.unpack("<Q", http(f"{BASE}/{shard}", (0, 7)))[0]
    hdr = json.loads(http(f"{BASE}/{shard}", (8, 8 + n - 1)))
    hdr.pop("__metadata__", None)
    p.write_text(json.dumps({"header": hdr, "data_start": 8 + n}), encoding="utf-8")
    return hdr, 8 + n


def bf16_to_f32(raw: bytes, shape) -> np.ndarray:
    u16 = np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16
    return u16.view(np.float32).reshape(shape)


def load_router(layer: int, wm: dict) -> tuple[np.ndarray, np.ndarray]:
    stem = f"language_model.model.layers.{layer}.block_sparse_moe.gate"
    shard = wm[f"{stem}.weight"]
    hdr, start = header(shard)
    out = []
    for name, conv in ((f"{stem}.weight", "bf16"), (f"{stem}.e_score_correction_bias", "f32")):
        meta = hdr[name]
        a, b = meta["data_offsets"]
        raw = http(f"{BASE}/{shard}", (start + a, start + b - 1))
        if conv == "bf16":
            out.append(bf16_to_f32(raw, tuple(meta["shape"])))
        else:
            out.append(np.frombuffer(raw, dtype=np.float32).reshape(meta["shape"]))
    return out[0], out[1]


def route(hidden: np.ndarray, gate: np.ndarray, bias: np.ndarray) -> np.ndarray:
    """noaux_tc: sigmoid scores pick by score+bias, weights come from score alone."""
    logits = hidden @ gate.T
    scores = 1.0 / (1.0 + np.exp(-logits))
    picked = np.argpartition(-(scores + bias), TOP_K - 1, axis=1)[:, :TOP_K]
    return picked


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", default="1,12,46,92")
    ap.add_argument("--tokens", type=int, default=4096)
    args = ap.parse_args()

    CACHE.mkdir(parents=True, exist_ok=True)
    wm = json.loads((CACHE / "model.safetensors.index.json").read_text(encoding="utf-8"))["weight_map"]
    rng = np.random.default_rng(20260728)

    print(f"driving the REAL router with {args.tokens} hidden states (rms 1.0, as observed)\n")
    all_curves = {}
    all_counts: dict[int, list[int]] = {}

    for layer in [int(x) for x in args.layers.split(",")]:
        key = f"language_model.model.layers.{layer}.block_sparse_moe.gate.weight"
        if key not in wm:
            print(f"layer {layer}: no router (dense layer)")
            continue
        gate, bias = load_router(layer, wm)
        hidden = rng.standard_normal((args.tokens, gate.shape[1])).astype(np.float32)
        hidden /= np.linalg.norm(hidden, axis=1, keepdims=True) / np.sqrt(gate.shape[1])

        picked = route(hidden, gate, bias)
        counts = np.bincount(picked.ravel(), minlength=N_EXPERTS)
        share = counts / counts.sum()
        nz = share[share > 0]
        entropy = -(nz * np.log(nz)).sum()
        max_entropy = np.log(N_EXPERTS)
        top10 = np.sort(share)[::-1][:10].sum()
        never = int((counts == 0).sum())

        # Union growth: how many distinct experts does a batch of B tokens touch?
        curve = {}
        for b in (1, 2, 4, 8, 16, 32, 64, 128):
            trials = [len(np.unique(picked[rng.choice(len(picked), b, replace=False)]))
                      for _ in range(64)]
            curve[b] = float(np.mean(trials))
        all_curves[layer] = curve
        all_counts[layer] = counts.tolist()

        uniform = {b: N_EXPERTS * (1 - (1 - TOP_K / N_EXPERTS) ** b) for b in curve}
        print(f"layer {layer:3d}  entropy {entropy:.3f} / {max_entropy:.3f} "
              f"({100*entropy/max_entropy:.1f}% of uniform)   "
              f"top-10 experts take {100*top10:.1f}%   never used: {never}")
        print("            union:  " + "  ".join(
            f"B={b}:{curve[b]:.0f}(u{uniform[b]:.0f})" for b in (2, 8, 32, 128)))

    print("\nINTERPRETATION")
    print("  entropy at ~100% of uniform means the learned router spreads load evenly, so the")
    print("  uniform prior in engine/batching is right and there is no free overlap to exploit.")
    print("  A measured union BELOW the uniform column is real skew and would raise throughput.")
    out = CACHE / "routing_prior.json"
    out.write_text(json.dumps(all_curves, indent=2), encoding="utf-8")
    print(f"\n  union curves saved -> {out}")

    # The per-expert popularity curve is what a tiered residency design needs:
    # it says how much traffic a hot set of size k actually captures. Committed
    # to the repo so downstream work has real data instead of a prior.
    dist = Path("engine/residency/measured_routing.json")
    dist.parent.mkdir(parents=True, exist_ok=True)
    dist.write_text(json.dumps(all_counts, indent=1), encoding="utf-8")
    print(f"  per-expert counts saved -> {dist}")
    for layer, counts in all_counts.items():
        c = np.array(counts, dtype=np.float64)
        c = np.sort(c)[::-1] / c.sum()
        cum = np.cumsum(c)
        hits = {k: round(float(cum[k - 1]), 4) for k in (16, 32, 64, 128, 256) if k <= len(cum)}
        print(f"    layer {layer}: hot-set coverage " +
              "  ".join(f"top{k}={100*v:.1f}%" for k, v in hits.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

