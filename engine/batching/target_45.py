"""What actually reaches 40-50 tok/s on Kimi K3, from measured bytes.

Decode is memory bound: a step cannot finish faster than the time to move its
bytes across the bus. So the question "can kernel fusion get us to 45 tok/s" has
an arithmetic answer, and the answer is no on its own - fusion removes
materialisation and launch overhead, it does not remove bytes.

This computes, for each hardware configuration, the physical ceiling and whether
the working set even fits, under three weight profiles.

Every input is measured or read from the checkpoint this session.
"""

from __future__ import annotations

# Measured / read from the checkpoint.
EXPERT_BYTES = 17_547_264          # per routed expert, MXFP4 at 4.250 bits/param
EXPERTS_PER_TOKEN = 16
MOE_LAYERS = 92
ROUTED_BYTES = EXPERT_BYTES * EXPERTS_PER_TOKEN * MOE_LAYERS      # 25.83 GB

# Non-routed mass pinned exactly: (1560.9e9 hub - 1446.46e9 experts) / 2 bytes.
DENSE_PARAMS = 57.222e9
DENSE_BF16 = DENSE_PARAMS * 2      # 114.44 GB, and NOTHING has quantised it

#: Fraction of a decode step's bytes that Moonshot left in BF16.
DENSE_SHARE = DENSE_BF16 / (ROUTED_BYTES + DENSE_BF16)

PROFILES = {
    "as shipped (BF16 dense)": DENSE_BF16,
    "FP8 dense (weight-only)": DENSE_PARAMS * 1,
    "INT4 dense (weight-only)": DENSE_PARAMS * 0.5,
}

# name -> (aggregate TB/s, aggregate VRAM GB)
HARDWARE = {
    "RTX 5090": (1.79, 32),
    "2x RTX 5090": (3.58, 64),
    "RTX PRO 6000 96GB": (1.80, 96),
    "H100 80GB": (3.35, 80),
    "H200 141GB": (4.80, 141),
    "2x H200": (9.60, 282),
    "B200 180GB": (8.00, 180),
}

#: Fraction of the physical ceiling a well-engineered system actually captures.
#: Ours measured 60% (1.861x captured of a 3.09x ceiling) via tools/headroom.py.
ATTAINMENT = 0.60
TARGET = 45.0


def main() -> None:
    print(f"Routed experts per token : {ROUTED_BYTES/1e9:8.2f} GB   (MXFP4, already 4-bit)")
    print(f"Dense skeleton per token : {DENSE_BF16/1e9:8.2f} GB   (BF16, UNTOUCHED)")
    print(f"Dense share of the step  : {100*DENSE_SHARE:8.1f} %\n")
    print(f"For {TARGET:.0f} tok/s as shipped you need "
          f"{(ROUTED_BYTES + DENSE_BF16) * TARGET / 1e12:.2f} TB/s of effective bandwidth.\n")

    header = f"{'hardware':<20}{'VRAM':>6}" + "".join(f"{p[:22]:>26}" for p in PROFILES)
    print(header)
    print("-" * len(header))
    for hw, (tbs, vram) in HARDWARE.items():
        row = f"{hw:<20}{vram:>4} GB"
        for _, dense in PROFILES.items():
            total = ROUTED_BYTES + dense
            ceiling = tbs * 1e12 / total
            realistic = ceiling * ATTAINMENT
            fits = total / 1e9 < vram * 0.88   # leave headroom for KV, state, activations
            mark = "fits" if fits else "NO FIT"
            row += f"{ceiling:>10.0f}/{realistic:>5.0f} {mark:>7}"
        print(row)

    print("\n  columns are  physical-ceiling / realistic-at-60%-attainment  tok/s")
    print("\nCONCLUSION")
    print(f"  Kernel fusion cannot reach {TARGET:.0f} tok/s on its own: it removes overhead,")
    print("  not bytes, and the byte floor is what binds. The lever that moves the floor")
    print(f"  is the {100*DENSE_SHARE:.0f}% of the step nobody has quantised - Moonshot compressed the")
    print("  experts and left attention, shared experts, latent projections and the LM")
    print("  head in BF16. Weight-only quantisation there pays no per-token activation")
    print("  tax, so it does not invert with concurrency the way FP8 activation quant does.")


if __name__ == "__main__":
    main()
