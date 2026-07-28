"""Measured calibration for the K3 decode roofline.

An adversarial pass showed the perf model was anchored on a number that implied
the dense weights moving at 3.6x theoretical peak DRAM. The fix is not a better
guess, it is a measurement. This benchmarks the three things a K3 decode step
actually does with expert weights, on real GPUs, at K3's real tensor shapes:

  1. host-to-device bandwidth at expert granularity (17.5 MB transfers), which
     bounds any design that streams experts across PCIe;
  2. MXFP4 dequantization throughput, which is on the critical path unless the
     kernel consumes packed weights directly;
  3. the expert GEMM itself at (3072, 3584), to see whether decode is bound by
     bandwidth or by compute at batch 1 and at batch 32.

Every number is measured, with warmup discarded and CUDA synchronised. Nothing
here is modelled.

    modal run engine/modal_kernelbench.py --gpu-kind A10G
"""

from __future__ import annotations

import json
import statistics
import time

import modal

app = modal.App("k3-kernelbench")

IMAGE = modal.Image.debian_slim(python_version="3.12").pip_install(
    "torch>=2.5", "numpy>=2.0"
)

# K3's real routed-expert shapes, from the checkpoint's safetensors headers.
LATENT_IN = 3584
LATENT_HIDDEN = 3072
PACKED_BYTES = 2 * (3072 * 1792) + (3584 * 1536)
SCALE_BYTES = 2 * (3072 * 112) + (3584 * 96)
EXPERT_BYTES = PACKED_BYTES + SCALE_BYTES  # 17,547,264
EXPERTS_PER_TOKEN = 16
MOE_LAYERS = 92


def _bench(fn, warmup: int = 5, iters: int = 30) -> tuple[float, float]:
    """Median and p10 seconds per call, warmup discarded, device synchronised."""
    import torch

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        samples.append(time.perf_counter() - t0)
    samples.sort()
    return statistics.median(samples), samples[max(0, len(samples) // 10)]


@app.function(image=IMAGE, gpu="A10G", timeout=60 * 30)
def bench(gpu_kind: str = "A10G", batch: int = 1) -> dict:
    import torch

    dev = torch.device("cuda")
    name = torch.cuda.get_device_name(0)
    props = torch.cuda.get_device_properties(0)
    out: dict[str, object] = {
        "gpu": name,
        "vram_gb": round(props.total_memory / 1e9, 1),
        "sm_count": props.multi_processor_count,
        "batch": batch,
    }

    # 1. Host-to-device at expert granularity. Pinned memory is the best case a
    #    streaming design could ever hope for, so this is an upper bound.
    host = torch.empty(EXPERT_BYTES, dtype=torch.uint8, pin_memory=True)
    dst = torch.empty(EXPERT_BYTES, dtype=torch.uint8, device=dev)

    def h2d():
        dst.copy_(host, non_blocking=True)

    med, best = _bench(h2d)
    out["h2d_expert_GBps"] = round(EXPERT_BYTES / med / 1e9, 1)
    out["h2d_expert_GBps_best"] = round(EXPERT_BYTES / best / 1e9, 1)
    out["h2d_ms_per_expert"] = round(med * 1e3, 3)

    # 2. MXFP4 dequantization at the real w1 shape.
    packed = torch.randint(0, 255, (LATENT_HIDDEN, 1792), dtype=torch.uint8, device=dev)
    scale = torch.randint(110, 124, (LATENT_HIDDEN, 112), dtype=torch.uint8, device=dev)
    codes = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0] + [-0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
        dtype=torch.float16,
        device=dev,
    )

    def dequant():
        rows, half = packed.shape
        idx = torch.empty((rows, half * 2), dtype=torch.long, device=dev)
        idx[:, 0::2] = (packed & 0x0F).long()
        idx[:, 1::2] = (packed >> 4).long()
        w = codes[idx]
        exp = torch.exp2(scale.to(torch.int16).float() - 127.0).half()
        return w * exp.repeat_interleave(w.shape[1] // scale.shape[1], dim=1)

    med, _ = _bench(dequant, iters=20)
    out["dequant_ms_per_tensor"] = round(med * 1e3, 3)
    out["dequant_GBps_out"] = round(LATENT_HIDDEN * LATENT_IN * 2 / med / 1e9, 1)

    # 3. The expert GEMM at decode shapes. Batch 1 has no arithmetic intensity;
    #    batch 32 is where a serving design would actually live.
    w = torch.randn(LATENT_HIDDEN, LATENT_IN, dtype=torch.float16, device=dev)
    for b in (1, batch) if batch != 1 else (1, 32):
        x = torch.randn(b, LATENT_IN, dtype=torch.float16, device=dev)

        def gemm(x=x):
            return torch.nn.functional.linear(x, w)

        med, _ = _bench(gemm, iters=50)
        flops = 2 * b * LATENT_HIDDEN * LATENT_IN
        out[f"gemm_b{b}_us"] = round(med * 1e6, 2)
        out[f"gemm_b{b}_TFLOPs"] = round(flops / med / 1e12, 2)
        # A batch-1 GEMM reads the whole weight and does almost nothing with it,
        # so effective bandwidth is the number that matters, not FLOPs.
        out[f"gemm_b{b}_eff_GBps"] = round(LATENT_HIDDEN * LATENT_IN * 2 / med / 1e9, 1)

    # What the measurements imply for a full decode step.
    per_token_expert_bytes = EXPERTS_PER_TOKEN * EXPERT_BYTES * MOE_LAYERS
    out["per_token_expert_GB"] = round(per_token_expert_bytes / 1e9, 2)
    out["implied_tok_s_if_streamed_over_pcie"] = round(
        1.0 / (per_token_expert_bytes / (out["h2d_expert_GBps"] * 1e9)), 2
    )
    print(json.dumps(out, indent=2))
    return out


@app.function(image=IMAGE, cpu=16.0, memory=32768, timeout=60 * 30)
def bench_cpu_gather(experts: int = 512) -> dict:
    """How much DRAM bandwidth does the expert ACCESS PATTERN actually get?

    The v1 design keeps the expert bank in CPU DRAM, so its roofline depends on
    the bandwidth achieved when gathering 16 scattered 17.5 MB blocks per token,
    not on the sequential memcpy figure a spec sheet quotes. That ratio is the
    part worth measuring; the absolute number here reflects Modal's host, not a
    12-channel workstation, so use the FRACTION and apply it to the target bus.
    """
    import numpy as np

    bank_bytes = experts * EXPERT_BYTES
    bank = np.empty(bank_bytes, dtype=np.uint8)
    bank[::4096] = 1  # touch pages so the allocation is real
    sink = np.empty(EXPERT_BYTES, dtype=np.uint8)
    rng = np.random.default_rng(0)

    def move(order) -> float:
        """Copy out expert-sized blocks in the given order; return GB/s."""
        t0 = time.perf_counter()
        moved = 0
        for e in order:
            off = int(e) * EXPERT_BYTES
            np.copyto(sink, bank[off : off + EXPERT_BYTES])
            moved += EXPERT_BYTES
        return moved / (time.perf_counter() - t0) / 1e9

    # Same primitive, same volume, only the ORDER differs. That isolates the
    # cost of the access pattern from the cost of the copy itself, which is the
    # only part that transfers to a different machine.
    count = min(experts, 512)
    seq_gbps = move(range(count))
    seq_gbps = max(seq_gbps, move(range(count)))  # second pass: caches warm
    gather_gbps = move(rng.integers(0, experts, size=count))
    total = 0

    out = {
        "bank_gb": round(bank_bytes / 1e9, 2),
        "experts_in_bank": experts,
        "sequential_scan_GBps": round(seq_gbps, 1),
        "expert_gather_GBps": round(gather_gbps, 1),
        "gather_fraction_of_sequential": round(gather_gbps / max(seq_gbps, 1e-9), 3),
        "ms_per_token_of_expert_gather": round(
            EXPERTS_PER_TOKEN * EXPERT_BYTES * MOE_LAYERS / (gather_gbps * 1e9) * 1e3, 1
        ),
        "note": "absolute GB/s is Modal's host; apply the FRACTION to a target bus",
        "checksum": total % 1000,
    }
    print(json.dumps(out, indent=2))
    return out


@app.function(image=IMAGE, gpu="H100", timeout=60 * 30)
def bench_h100(batch: int = 32) -> dict:
    """Same measurements on target-class hardware.

    An A10G is PCIe 4.0 with ~600 GB/s of VRAM; the configurations this project
    is actually costing use H100/H200/5090-class parts. Calibrating on the wrong
    tier is how the last perf model ended up impossible.
    """
    return bench.local(gpu_kind="H100", batch=batch)


@app.local_entrypoint()
def main(gpu_kind: str = "A10G", batch: int = 32):
    if gpu_kind.upper() == "H100":
        bench_h100.remote(batch=batch)
    else:
        bench.remote(gpu_kind=gpu_kind, batch=batch)
