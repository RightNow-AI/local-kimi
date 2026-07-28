"""Cross-expert rank of one Kimi K3 MoE layer, measured gauge-invariantly.

v1 of this script was WRONG in a way that would have produced a confident false
negative. Three defects, all found by adversarial review and all fixed here:

1. GAUGE. A SwiGLU expert computes  y = w2 @ (silu(w1 x) * (w3 x)).  Permuting the
   3072 hidden units - w1 -> P w1, w3 -> P w3, w2 -> w2 P^T - leaves the expert's
   function EXACTLY unchanged. Experts are independently initialised, so each one
   sits in an arbitrary permutation gauge. A sketch of the raw w1 therefore compares
   two functionally identical experts as if they were unrelated (measured cosine
   ~0.008). The fix: compare the gauge-invariant products
       M1_e = w2_e @ w1_e     M3_e = w2_e @ w3_e      (both 3584 x 3584)
   in which P^T P cancels identically. We never form them densely; sketching each
   side first is equivalent because  P^T w2 (w1 R) = P^T (w2 w1) R.

2. ENERGY vs ERROR. Gram eigenvalues are SQUARED singular values, so "95% energy"
   is sqrt(0.05) = 22.4% relative Frobenius error, not 5%. We report the rank needed
   to hit a stated RECONSTRUCTION ERROR, which is the quantity the product depends on.

3. MEAN. A large component shared by every expert makes any stack look rank-1. We
   report both centred and uncentred, and the centred number is the one that decides.

A POSITIVE CONTROL runs through the identical pipeline: a synthetic stack of known
rank 8. If the pipeline cannot recover rank 8 from that, the measurement is blind and
no conclusion may be drawn from it.
"""

from __future__ import annotations

import argparse
import json
import struct
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

BASE = "https://huggingface.co/moonshotai/Kimi-K3/resolve/main"
E2M1 = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)
E2M1_FULL = np.concatenate([E2M1, -E2M1])


def http(url: str, rng: tuple[int, int] | None = None, retries: int = 5) -> bytes:
    headers = {"User-Agent": "k3-spectrum"}
    if rng:
        headers["Range"] = f"bytes={rng[0]}-{rng[1]}"
    last = None
    for attempt in range(retries):
        try:
            return urllib.request.urlopen(
                urllib.request.Request(url, headers=headers), timeout=240
            ).read()
        except Exception as exc:
            last = exc
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"GET failed: {last}")


def dequant(packed: np.ndarray, scale: np.ndarray) -> np.ndarray:
    cols = packed.shape[1] * 2
    vals = np.empty((packed.shape[0], cols), dtype=np.uint8)
    vals[:, 0::2] = packed & 0x0F
    vals[:, 1::2] = packed >> 4
    w = E2M1_FULL[vals]
    exp = np.exp2(scale.astype(np.int16) - 127).astype(np.float32)
    return w * np.repeat(exp, cols // scale.shape[1], axis=1)


def get_tensor(shard: str, header: dict, start: int, name: str, cache: Path) -> np.ndarray:
    out = cache / (name.replace(".", "_") + ".npy")
    if out.exists():
        try:
            return np.load(out)
        except Exception:
            out.unlink(missing_ok=True)
    parts = {}
    for suffix in ("weight_packed", "weight_scale"):
        meta = header[f"{name}.{suffix}"]
        a, b = meta["data_offsets"]
        raw = http(f"{BASE}/{shard}", (start + a, start + b - 1))
        parts[suffix] = np.frombuffer(raw, dtype=np.uint8).reshape(meta["shape"])
    w = dequant(parts["weight_packed"], parts["weight_scale"]).astype(np.float16)
    np.save(out, w)
    return w


def rank_for_error(ev: np.ndarray, err: float) -> int:
    """Smallest rank whose truncation leaves <= err relative Frobenius error."""
    tail = 1.0 - np.cumsum(ev) / ev.sum()
    tail = np.clip(tail, 0.0, None)
    idx = np.searchsorted(-np.sqrt(tail), -err)
    return int(min(idx + 1, len(ev)))


def report(name: str, S: np.ndarray, errs=(0.01, 0.03, 0.05, 0.10)) -> dict:
    n = S.shape[0]
    out = {}
    for label, X in (("uncentred", S), ("CENTRED", S - S.mean(axis=0, keepdims=True))):
        ev = np.clip(np.linalg.eigvalsh(X @ X.T)[::-1], 0, None)
        ranks = {f"{int(e*100)}%": rank_for_error(ev, e) for e in errs}
        pr = 1.0 / np.sum((ev / ev.sum()) ** 2)
        out[label] = {"ranks": ranks, "participation": float(pr)}
        line = "  ".join(f"err<={k}: r={v:>4d}" for k, v in ranks.items())
        print(f"  {name:22s} {label:10s} n={n:<4d} {line}   PR={pr:6.1f}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=12)
    ap.add_argument("--experts", type=int, default=896)
    ap.add_argument("--sketch", type=int, default=192)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--cache", default="research/.cache")
    args = ap.parse_args()

    cache = Path(args.cache)
    cache.mkdir(parents=True, exist_ok=True)

    wm = json.loads((cache / "model.safetensors.index.json").read_text(encoding="utf-8"))["weight_map"]
    stem = f"language_model.model.layers.{args.layer}.block_sparse_moe.experts"
    shard = wm[f"{stem}.0.w1.weight_packed"]
    hp = cache / f"{shard}.header.json"
    if hp.exists():
        d = json.loads(hp.read_text(encoding="utf-8"))
        header, start = d["header"], d["data_start"]
    else:
        n = struct.unpack("<Q", http(f"{BASE}/{shard}", (0, 7)))[0]
        header = json.loads(http(f"{BASE}/{shard}", (8, 8 + n - 1)))
        header.pop("__metadata__", None)
        start = 8 + n
        hp.write_text(json.dumps({"header": header, "data_start": start}), encoding="utf-8")

    rng = np.random.default_rng(20260728)
    s = args.sketch
    P = (rng.standard_normal((3584, s), dtype=np.float32) / np.sqrt(3584))   # left,  on w2 rows
    R1 = (rng.standard_normal((3584, s), dtype=np.float32) / np.sqrt(3584))  # right, on w1 cols
    R3 = (rng.standard_normal((3584, s), dtype=np.float32) / np.sqrt(3584))

    print(f"layer {args.layer}, {args.experts} experts, sketch {s}, {args.threads} threads")
    print("gauge-invariant products: M1 = w2@w1, M3 = w2@w3\n")

    def one(e: int):
        w1 = get_tensor(shard, header, start, f"{stem}.{e}.w1", cache).astype(np.float32)
        w2 = get_tensor(shard, header, start, f"{stem}.{e}.w2", cache).astype(np.float32)
        w3 = get_tensor(shard, header, start, f"{stem}.{e}.w3", cache).astype(np.float32)
        A = P.T @ w2                      # (s, 3072)   P^T w2
        return (A @ (w1 @ R1)).ravel(), (A @ (w3 @ R3)).ravel()

    t0 = time.time()
    S1, S3 = [], []
    with ThreadPoolExecutor(max_workers=args.threads) as pool:
        for i, (a, b) in enumerate(pool.map(one, range(args.experts))):
            S1.append(a)
            S3.append(b)
            if (i + 1) % 64 == 0:
                el = time.time() - t0
                print(f"  {i+1}/{args.experts}  {el:6.0f}s  ({(i+1)/max(el,1e-9):.2f}/s)", flush=True)

    S1 = np.stack(S1)
    S3 = np.stack(S3)
    print(f"\nsketches: {S1.shape} each\n")
    print("RANK NEEDED FOR A GIVEN RELATIVE FROBENIUS RECONSTRUCTION ERROR")
    print("(the CENTRED row is the one that decides the thesis)\n")

    results = {
        "M1_w2w1": report("M1 = w2@w1", S1),
        "M3_w2w3": report("M3 = w2@w3", S3),
    }

    n, d = S1.shape
    print()
    ctrl = rng.standard_normal((n, d), dtype=np.float32)
    results["negative_control"] = report("random control", ctrl)

    # Positive control: a stack that genuinely has rank 8. If the pipeline cannot
    # see this, it cannot see anything and no conclusion may be drawn.
    basis = rng.standard_normal((8, d), dtype=np.float32)
    coef = rng.standard_normal((n, 8), dtype=np.float32)
    results["positive_control_rank8"] = report("synthetic rank-8", coef @ basis)

    out = Path(args.cache) / f"spectrum_v2_layer{args.layer}.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")

    c = results["M1_w2w1"]["CENTRED"]["ranks"]
    pos = results["positive_control_rank8"]["CENTRED"]["ranks"]["1%"]
    neg = results["negative_control"]["CENTRED"]["ranks"]["5%"]
    print("\n" + "=" * 68)
    print("VERDICT")
    print("=" * 68)
    if pos > 12:
        print(f"  MEASUREMENT IS BLIND: positive control needs r={pos} for 1% error,")
        print("  but it is rank 8 by construction. Do not draw a conclusion.")
        return 2
    print(f"  pipeline validated: synthetic rank-8 recovered at r={pos} (1% error)")
    print(f"  random control needs r={neg} for 5% error out of n={n}")
    print(f"  K3 experts (centred, w2@w1) need r={c['5%']} for 5% error, r={c['1%']} for 1%")
    print()
    if c["5%"] <= 16:
        print(f"  GO. r={c['5%']} <= 16 = num_experts_per_token, so a shared basis beats")
        print("  plain top-16 routing on bandwidth AND makes the bank resident.")
    elif c["5%"] <= 64:
        print(f"  PARTIAL. r={c['5%']} is well below n={n} so real structure exists, but it is")
        print("  above the 16 break-even: this buys capacity, not per-token bandwidth.")
    else:
        print(f"  NO-GO on this route. r={c['5%']} is too high; pivot to pruning/clustering.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
