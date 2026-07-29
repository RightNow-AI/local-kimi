# Does retaining the shared experts in BF16 fix the routing divergence

A controlled experiment with a stated hypothesis, run because the first accuracy
result failed on router agreement rather than on perplexity, and named a suspect.

## The hypothesis

From `engine/quant/QUANTIZATION-RESULTS.md`: the worst tensor class by relative
Frobenius error is the shared experts at 11.70 percent, and the shared expert is
the only expert that runs on EVERY token, while a routed expert sees roughly 8
of 256. Its error is therefore not diluted by sparsity, and it perturbs exactly
the hidden state the router reads.

Predicted: retaining all 78 shared-expert tensors in BF16 raises router set
agreement materially, for 264,536,064 bytes, 0.918 percent of the artifact.

## The two artifacts

Both built by the same job from the same source at revision `e1df551a`, both
with planned bytes equal to actual bytes, both with all three negative-control
decoders rejected on real data.

| profile | tensor bytes | quantized | retained |
|---|---:|---:|---:|
| `default` | 28,803,304,448 | 20,150 | 343 |
| `shared-experts-bf16` | 29,067,840,512 | 20,072 | 421 |

The only difference is those 78 tensors.

## Result

Both sides of each run served by the same vLLM on the same H200, identical
prompts, identical sampling, byte-identical served configs.

| metric | `default` | `shared-experts-bf16` | change |
|---|---:|---:|---|
| Router set agreement | 34.30% | 36.74% | +2.44 points |
| Greedy output identity | 37.69% | 40.77% | +3.08 points |
| Next-token top-1 agreement | 85.16% | 91.41% | **+6.25 points** |
| Mean KL(BF16 \|\| candidate) | 0.0555 nats | 0.0398 nats | **-28%** |
| Median first divergence | token 13 | token 14 | +1 |
| Teacher-forced perplexity | 12.815 | 12.960 | **worse** |
| Perplexity increase vs BF16 | +0.81% | +1.95% | **worse** |
| **Verdict** | **FAIL** | **FAIL** | unchanged |

## Verdict: FAIL

Both profiles fail the thresholds declared before the run. The retained-shared-
expert profile improves distributional agreement substantially and moves router
agreement only 2.44 points, which is not enough to call the quantized model
behaviourally equivalent to the original. Stated here in one line because a
verdict a reader has to extract from a comparison table is a verdict that can be
misread.

## What this establishes

**The hypothesis is partially confirmed and the suspect is largely ruled out.**

Distributional agreement improved substantially. Top-1 agreement rose 6.25
points and mean KL fell by more than a quarter, so on the next-token
distribution the retained-shared-expert artifact is meaningfully closer to the
original.

Router agreement rose only 2.44 points, from 34.30 to 36.74 percent. That is a
real improvement and it is nowhere near enough. The two checkpoints still select
different experts for roughly 63 percent of tokens. Whatever dominates the
routing divergence, it is not the shared experts.

The remaining candidates, in the order I would test them: the attention
projections, which carry the largest max absolute errors measured (0.19 on
`layers.16.self_attn.o_proj`) and feed the residual stream the router reads; and
the routed experts themselves, whose error enters the residual stream on every
layer even though each individual expert is rarely selected.

## The anomaly, stated rather than smoothed

The retained artifact moved CLOSER to the original in distribution while moving
FURTHER away in teacher-forced perplexity. Those two should normally move
together, and they did not.

Two explanations are consistent with the data and this experiment cannot
separate them:

1. The perplexity sample is 1,023 positions, which may be too small to
   distinguish 12.815 from 12.960 when the baseline is 12.712. The distribution
   metrics cover 128 positions across the full 163,840-token vocabulary, which
   is far more signal per position.
2. Mixing full-precision shared experts with quantized routed experts may
   introduce a genuine precision mismatch in the summed MoE output, which the
   uniformly quantized artifact does not have.

Until one of those is settled, neither perplexity figure should be quoted as the
quality cost of this profile. The honest statement today is that both profiles
FAIL the pre-declared thresholds, and that the failure is driven by routing
divergence in both cases.

## Cost of learning this

One H100 artifact build and one H200 accuracy run. The negative half of the
result is the valuable half: it removes the most plausible explanation, so the
next experiment does not start by retesting it.

## Reproducing

```bash
modal run engine/modal_quantize_klinear.py --profile shared-experts-bf16
modal run engine/modal_accuracy.py --profile shared-experts-bf16
```
