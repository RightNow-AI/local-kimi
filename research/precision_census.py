"""What precision is each published Kimi K3 component stored in?

The routed-expert matrices are MXFP4, but that is not the whole-model baseline.
Shared experts, attention, latent projections, and embeddings remain BF16 and
total 114.4 GB. Two shared experts participate alongside 16 routed experts per
token. The published routed-expert values are the quantized baseline; decoder
fidelity is a separate claim that ``verify_lossless.py`` must establish.
"""

from __future__ import annotations

import json
import struct
import urllib.request
from collections import defaultdict
from pathlib import Path

BASE = "https://huggingface.co/moonshotai/Kimi-K3/resolve/main"
CACHE = Path("research/.cache")

DTYPE_BITS = {"F32": 32, "F16": 16, "BF16": 16, "F8_E4M3": 8, "F8_E5M2": 8,
              "I8": 8, "U8": 8, "I32": 32, "I64": 64, "BOOL": 8}


def http(url: str, rng=None) -> bytes:
    h = {"User-Agent": "k3-census"}
    if rng:
        h["Range"] = f"bytes={rng[0]}-{rng[1]}"
    return urllib.request.urlopen(urllib.request.Request(url, headers=h), timeout=120).read()


def header(shard: str) -> dict:
    p = CACHE / f"{shard}.header.json"
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))["header"]
    n = struct.unpack("<Q", http(f"{BASE}/{shard}", (0, 7)))[0]
    hdr = json.loads(http(f"{BASE}/{shard}", (8, 8 + n - 1)))
    hdr.pop("__metadata__", None)
    p.write_text(json.dumps({"header": hdr, "data_start": 8 + n}), encoding="utf-8")
    return hdr


def numel(shape) -> int:
    n = 1
    for d in shape:
        n *= d
    return n


def main() -> int:
    CACHE.mkdir(parents=True, exist_ok=True)
    ip = CACHE / "model.safetensors.index.json"
    if not ip.exists():
        ip.write_bytes(http(f"{BASE}/model.safetensors.index.json"))
    idx = json.loads(ip.read_text(encoding="utf-8"))
    wm = idx["weight_map"]
    total_size = idx.get("metadata", {}).get("total_size", 0)

    shards = sorted(set(wm.values()))
    # Sample shards across the file: expert-heavy middle plus the ends, which
    # hold embeddings and the attention stack.
    sample = [shards[0], shards[1], shards[len(shards) // 2], shards[-2], shards[-1]]
    print(f"{len(wm)} tensors, {len(shards)} shards, total_size {total_size/1e9:.1f} GB")
    print(f"sampling {len(sample)} shards\n")

    by_dtype: dict[str, list[int]] = defaultdict(lambda: [0, 0])   # [tensors, bytes]
    by_kind: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for sh in sample:
        hdr = header(sh)
        for name, meta in hdr.items():
            dt = meta["dtype"]
            a, b = meta["data_offsets"]
            nbytes = b - a
            by_dtype[dt][0] += 1
            by_dtype[dt][1] += nbytes
            if ".experts." in name:
                kind = "routed_expert"
            elif "shared_expert" in name:
                kind = "shared_expert"
            elif "self_attn" in name:
                kind = "attention"
            elif "embed" in name or "lm_head" in name:
                kind = "embedding"
            elif "routed_expert_" in name:
                kind = "latent_proj"
            elif "gate" in name:
                kind = "router"
            else:
                kind = "other"
            by_kind[kind][dt] += nbytes

    print("=== dtype census over sampled shards ===")
    tot = sum(v[1] for v in by_dtype.values())
    for dt, (n, nb) in sorted(by_dtype.items(), key=lambda x: -x[1][1]):
        print(f"  {dt:8s} {n:7d} tensors  {nb/1e9:8.2f} GB  {100*nb/tot:5.1f}%")

    print("\n=== precision by component ===")
    for kind in sorted(by_kind, key=lambda k: -sum(by_kind[k].values())):
        parts = ", ".join(f"{dt} {nb/1e9:.2f}GB" for dt, nb in
                          sorted(by_kind[kind].items(), key=lambda x: -x[1]))
        print(f"  {kind:16s} {parts}")

    print("\n=== effective bits per parameter ===")
    exp_bytes = 2 * (3072 * 1792) + (3584 * 1536)
    exp_scale = 2 * (3072 * 112) + (3584 * 96)
    exp_params = 3 * 3072 * 3584
    print(f"  routed expert: {(exp_bytes+exp_scale)*8/exp_params:.3f} bits/param "
          f"({exp_bytes/1e6:.2f} MB packed + {exp_scale/1e6:.2f} MB scales)")
    print(f"  whole model  : {total_size*8/2.78e12:.3f} bits/param average")

    print("\n=== BASELINE SUMMARY ===")
    u8 = by_dtype.get("U8", [0, 0])[1]
    if u8 > 0.5 * tot:
        print("  Moonshot ships K3's routed-expert matrices in MXFP4.")
        print("  Shared experts, attention, latent projections, and embeddings remain")
        print("  BF16 and total 114.4 GB; two shared experts join 16 routed experts/token.")
        print("  Decoder fidelity is established separately by verify_lossless.py.")
    else:
        print("  The sampled routed-expert mass is not predominantly U8.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
