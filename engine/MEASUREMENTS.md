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

## Real tokens from real Kimi K3 weights

The first end-to-end generation, on an H100 80GB HBM3 from the actual Moonshot
checkpoint:

```
layers            [11, 12, 13]      real K3 weights off the Modal volume
prompt_token_ids  [100, 200, 300, 400]
generated_token_ids  [72628, 50873, 113280, 67093]
load_seconds      20.197
generation_seconds 11.010            4 tokens
peak_allocated_gb 25.245
```

Every generated id falls inside K3's 163,840-token vocabulary, and the whole
path executed: embeddings, KDA and MLA attention with their separate state
objects, the `noaux_tc` router, MXFP4 expert dequantization, the latent MoE,
the final norm, the LM head, and sampling.

**What this does not show.** Three layers of ninety-three ran, so the tokens are
not semantically meaningful - the model is not there, only the machinery. And
the timing is a reference implementation reading from a network volume with no
fusion, so 2.75 s/token across 3 layers must NOT be extrapolated to 93. It is a
correctness result, not a throughput result.

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

### The real router is strongly skewed, not uniform

Driving K3's actual learned router - `gate.weight` plus the `noaux_tc` correction
bias, read from the checkpoint - with 4,096 isotropic hidden states matched to
the rms observed in a real layer run:

| layer | routing entropy | experts never selected | union at B=32 | union at B=128 |
|---:|---:|---:|---:|---:|
| 1 | 72.3% of uniform | 660 of 896 | **142** (uniform 393) | **175** (uniform 807) |
| 12 | 65.7% | 717 | **100** (393) | **120** (807) |
| 46 | 77.0% | 538 | 179 (393) | 239 (807) |
| 92 | 77.6% | 471 | 181 (393) | 251 (807) |

Across 65,536 draws a uniform router would leave essentially no expert unused.
The real one leaves 471 to 717 of 896 untouched, and the batch union comes in
**2.2x to 6.7x below** the uniform prediction. Since decode cost is set by
distinct experts touched, that is a direct multiplier on aggregate throughput.

**Both of these are bounds, and the truth is between them.** The uniform prior
used in `engine/batching/` is a pessimistic upper bound on the union. This
measurement is an optimistic lower bound, because isotropic inputs are not real
hidden states: they probe the router's intrinsic bias rather than the spread
that genuinely diverse text would produce. Only routing traces from the full
model settle it, and neither number should be quoted alone.

Skew also varies with depth - layer 12 is the most concentrated, the late layers
the least - so a scheduler cannot assume one profile for the whole stack.

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
