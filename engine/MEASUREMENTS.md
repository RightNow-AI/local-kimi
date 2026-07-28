# Measured calibration

Every number here was measured on real GPUs via `engine/modal_kernelbench.py`,
at Kimi K3's real routed-expert shapes, with warmup discarded and CUDA
synchronised. Nothing on this page is modelled.

This exists because an adversarial pass showed the perf model was anchored on a
figure that implied the dense weights moving at 3.6x theoretical peak DRAM. The
fix for a bad guess is not a better guess.

## Results

| | A10G | H100 80GB HBM3 |
|---|---:|---:|
| Host-to-device, 17.5 MB expert-sized transfers | 13.3 GB/s | **53.7 GB/s** |
| Time to move one expert over PCIe | 1.316 ms | 0.327 ms |
| MXFP4 dequant, one w1 tensor (naive PyTorch) | 1.703 ms | **0.342 ms** |
| Expert GEMM (3072x3584), batch 1 | 74.2 us | **23.3 us** |
| Expert GEMM, batch 32 | 81.8 us | **22.6 us** |
| GEMM batch 32 throughput | 8.62 TFLOP/s | 31.19 TFLOP/s |

## What the measurements establish

### PCIe expert streaming is dead, measured

At 53.7 GB/s measured on an H100, moving 25.83 GB of routed experts per token
gives **2.08 tok/s**. An independent adversarial review predicted a ~2.1 tok/s
cap from first principles; the measurement lands on it. Any design that streams
expert weights across PCIe per token is bounded here regardless of how good the
kernels are.

### Dequantization dominates the expert path by an order of magnitude

On an H100 a naive MXFP4 dequant of one expert tensor costs **0.342 ms** while
the GEMM it feeds costs **0.023 ms**. Dequant is **15x the matmul** (23x on
A10G). Extended to a full decode step - 3 tensors per expert, 16 routed experts,
92 MoE layers - unfused dequantization alone would cost on the order of a second
per token. It is not a tax on the expert path, it *is* the expert path.

This is the single most valuable kernel target in the project, and it is exactly
the work that makes the difference between a research build and a sellable one:

- fuse dequant into the GEMM prologue so packed weights are never materialised
  in fp16;
- or, on Blackwell, feed the packed MXFP4 weights to the tensor cores directly,
  since K3 ships in the OCP microscaling format the hardware consumes natively.

### There is no scatter penalty for MoE expert access

Copying expert-sized blocks out of a DRAM-resident bank, same primitive and same
volume, only the order differing:

| | GB/s |
|---|---:|
| Sequential expert order | 25.8 |
| Random expert order | 26.2 |
| **Ratio** | **1.013** |

At 17.5 MB per expert the blocks are far larger than any DRAM page or prefetch
window, so routing to 16 arbitrary experts costs exactly what reading 16
adjacent ones costs. MoE designs often assume a gather penalty here; at K3's
expert granularity there is none.

This matters for the v1 roofline: the expert bank can be modelled at the full
sequential DRAM figure of the target bus with no scatter discount. The absolute
25-26 GB/s above is single-threaded on Modal's host and does not transfer; the
ratio is what does.

### Batching is nearly free on the compute side

Batch 1 and batch 32 cost the same wall time on an H100 - 23.3 us versus 22.6 us
- so batch 32 delivers **33x the tokens for the same GEMM**. Decode at batch 1
has no arithmetic intensity; the GEMM reads the entire weight and does almost
nothing with it, reaching only 944 GB/s of ~3.35 TB/s peak.

The caveat that matters for this product: this is the *compute* side amortizing.
K3 is sparse MoE, so the *weight* side does not amortize the same way, because B
concurrent tokens touch a growing union of experts. The two effects pull in
opposite directions and the union model in `engine/batching/` is what resolves
them.

## Reproducing

```bash
modal run engine/modal_kernelbench.py                  # A10G
modal run engine/modal_kernelbench.py --gpu-kind H100  # target class
```

Roughly a minute of GPU time each.
