# What INT4 costs Kimi-Linear, measured

Run `20260728T213950Z-35971f7c` on one H200. Both sides served by the same vLLM
0.26.0, on the same physical GPU (`GPU-37947044-3026-cb43-c9fe-b12bb75ee75b`),
with identical prompts, identical sampling and byte-identical served configs.
The only difference between the two sides is the weight values.

The candidate is the original BF16 weights passed through this repo's INT4 codec
and straight back out, so it carries exactly the information loss INT4 imposes
while still loading in the same runtime. That isolates quantization damage from
our engine's numerics and from any bug in our engine, which a naive
ours-versus-vLLM comparison would have confounded beyond interpretation.

## Verdict: FAIL

Against thresholds declared in `engine/accuracy/thresholds.py` before the run.
They are not revised here. Choosing what counts as acceptable after seeing the
numbers is the failure this whole design exists to prevent.

## The numbers

| Metric | Value |
|---|---:|
| Teacher-forced perplexity, BF16 | 12.712 |
| Teacher-forced perplexity, INT4 round trip | 12.815 |
| **Perplexity increase** | **+0.81%** |
| Next-token top-1 agreement | 85.2% |
| Mean KL(BF16 \|\| INT4), exact full vocabulary | 0.0555 nats |
| Max KL | 0.322 nats |
| Greedy output identity | 37.7% (49 of 130 prompts) |
| Median first divergence | token 13 |
| Minimum first divergence | token 0 |
| Mean common prefix fraction | 51.5% |
| **Router set agreement** | **34.3%** |

Distribution metrics cover 128 positions over the full 163,840-token vocabulary,
not a truncated top-k. Perplexity covers 1,023 positions. Router agreement is
averaged over 26 MoE layers and compares expert SETS, so it is not fooled by an
ordering difference.

## What this means

**The predictions barely move. The behaviour moves a lot.**

An 0.81% perplexity increase is a good result for four-bit weights. Top-1
agreement of 85.2% and a mean KL of 0.055 nats are modest divergences in the
output distribution.

Router set agreement of 34.3% is not modest. The two checkpoints select
different experts for roughly two thirds of tokens. The router gate is
deliberately NOT quantized, so this is not gate error: it is the gate reacting
to perturbed inputs. Selecting 8 of 256 experts is a discrete decision at the
edge of a ranking, and a small change in the hidden state flips it.

The consequence is that these are two different computations that happen to be
about equally good at next-token prediction. That is a coherent thing for a
sparse MoE to do, because the experts carry overlapping capability, but it means
the INT4 artifact must NOT be described as behaviourally equivalent to the
original. Greedy identity of 37.7% says the same thing from the output side.

## Which package class this permits

- **RECIPE**, whose claim is that outputs are unchanged: refused by this
  evidence. 37.7% greedy identity is not identity.
- **OPTIMIZED_WEIGHTS**, whose claim is modified weights with a measured and
  disclosed quality cost: supported, with +0.81% perplexity as the headline cost
  and the routing divergence disclosed alongside it. A buyer must be told that
  outputs differ from the reference model, not merely that they are slightly
  worse on average.

## The next experiment, with a stated hypothesis

`engine/quant/QUANTIZATION-RESULTS.md` already identified the prime suspect
before this run: the highest-error tensor class by relative Frobenius norm is
the shared experts, at 11.70%, and the shared expert runs on EVERY token while a
routed expert sees roughly 8 of 256. Its error is therefore not diluted by
sparsity, and it perturbs the hidden state that the router reads.

**Hypothesis:** holding all 78 shared-expert tensors in BF16 raises router
agreement materially, at a cost of roughly 264 MB against a 28.8 GB artifact,
under one percent of the footprint.

This is worth running before any listing, because router agreement is the metric
that decides whether we can honestly call this the same model.

## Known limits of this measurement

- Per-layer-depth KL is unavailable. vLLM 0.26.0 exposes final next-token
  distributions but not intermediate layer logits without changing the measured
  runtime path, and changing that path would have invalidated the comparison.
- Two vLLM incompatibilities had to be worked around and are disclosed in the
  evidence record: this model publishes `num_experts_per_token` while vLLM's
  routed-expert capture accepts only `num_experts_per_tok` or `top_k_experts`,
  so an additive alias was injected into the SERVED config copies of both sides
  identically, leaving the shared source checkpoint untouched.
- This measures quantization damage, not our engine. Our engine's own
  correctness against a reference implementation remains unmeasured.

## Reproducing

```bash
modal run engine/modal_accuracy.py
```

Evidence record at `/accuracy/runs/20260728T213950Z-35971f7c/evidence.json` on
the `kimi-linear-accuracy` volume, with raw per-prompt samples retained so every
figure above can be recomputed.
