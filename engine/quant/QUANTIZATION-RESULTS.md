# Kimi-Linear INT4 checkpoint, measured

Produced by `engine/modal_quantize_klinear.py` on one H100 from the real BF16
checkpoint on the `kimi-linear-weights` Modal volume. Every number on this page
was read from the tensors or from the files that were written. The full
per-tensor manifest lives beside the artifact as `quantization-manifest.json`
and is not committed here because it is 23.9 MB.

Artifact: volume `kimi-linear-quantized`, path
`Kimi-Linear-48B-A3B-Instruct-W4A16`.

## Size

| | bytes | GiB |
|---|---:|---:|
| Source tensor storage, BF16 as shipped | 98,245,528,576 | 91.50 |
| Output tensor storage, INT4 selective | 28,803,304,448 | 26.83 |
| Output safetensors files | 28,809,016,344 | 26.83 |
| Output directory including support files | 28,840,764,112 | 26.86 |

That is a **3.41x reduction in weight bytes**, MEASURED.

The plan projected 28,803,304,448 bytes and the artifact is 28,803,304,448
bytes. Planned and actual agree exactly, because both are computed from real
safetensors header shapes rather than from a bits-per-parameter estimate.

For the record, `engine/laptop/RESULTS.md` previously projected 24,561,340,864
bytes. That figure assumed a flat 4.0 bits per parameter across the whole model.
It is superseded and is retained there only so the correction is auditable.

## Format

| | |
|---|---|
| Format | symmetric signed INT4 |
| Group size | 32 |
| Group axis | final reduction axis after flattening leading dimensions |
| Scale dtype | BF16 |
| Zero point | none |
| Cost | 4.5 bits per quantized parameter including scales |

## What was quantized, and what was deliberately not

20,150 tensors quantized, 343 retained in source precision.

| Class | count | quantized | why |
|---|---:|:---:|---|
| routed expert projections | 19,968 | yes | dominate resident bytes, the primary fit target |
| attention projections | 101 | yes | large matrices on the token path |
| shared expert projections | 78 | yes | large token-path projections |
| dense layer 0 MLP | 3 | yes | a large matrix bank |
| KDA gates, state, convolutions | 200 | no | control the recurrent state; small byte share |
| normalization | 82 | no | tiny, and quantizing them buys nothing |
| router gate | 26 | no | routing is discrete, so error changes which computation runs |
| biases | 26 | no | tiny |
| MLA latent down-projection | 7 | no | writes the cached KV latent, so error persists for a whole sequence |
| token embedding | 1 | no | quality sensitive |
| lm head | 1 | no | output logits are quality sensitive |

Coverage was validated rather than assumed: all 26 configured MoE layers and all
256 routed experts per layer are present, with an explicit per-expert layout,
and the job REFUSES to run if any matrix weight lacks an explicit policy. It did
refuse on the first attempt, for the seven `kv_a_proj_with_mqa` matrices, which
is how that class came to have a stated decision instead of a default.

## Proof

| check | result |
|---|---|
| Scale-aware round trip verified for every quantized tensor | PASS |
| Written checkpoint reopened and reloaded | PASS |
| Written shapes and dtypes match the manifest | PASS |
| `wrong_group_axis` decoder rejected on real data | REJECTED as required |
| `wrong_scale` decoder rejected on real data | REJECTED as required |
| `swapped_nibbles` decoder rejected on real data | REJECTED as required |

The three negative controls matter more than the positive ones. An earlier
verifier in this project was accidentally a tautology that would have passed any
decoder. These are run against a real checkpoint tensor,
`model.layers.0.mlp.down_proj.weight`, and each wrong decoder is required to
FAIL. Their measured max absolute deviations were 0.107, 0.174 and 0.244.

## Error profile, and the one thing it points at

Worst tensors by max absolute error, all attention output projections:

| tensor | max abs error |
|---|---:|
| `layers.16.self_attn.o_proj.weight` | 0.1914 |
| `layers.20.self_attn.o_proj.weight` | 0.1816 |
| `layers.25.self_attn.o_proj.weight` | 0.1758 |

Worst tensors by relative Frobenius error, all shared experts:

| tensor | relative Frobenius error |
|---|---:|
| `layers.26...shared_experts.down_proj.weight` | 11.70% |
| `layers.25...shared_experts.down_proj.weight` | 11.09% |
| `layers.24...shared_experts.down_proj.weight` | 10.83% |

**The highest-error class is also the one that runs most often.** A routed expert
sees roughly 8 of every 256 tokens. The shared expert runs on EVERY token, so
its error is not diluted by sparsity the way routed-expert error is.

There are 78 shared-expert tensors. Holding them in BF16 instead would cost on
the order of 264 MB against a 28.8 GB artifact, under one percent, and would
remove the worst-measured error class from the path every token takes. That is
exactly the trade the fit-not-speed principle argues for.

This page does NOT claim that change is necessary, because weight-space error is
not model quality and the two can diverge in both directions. `engine/accuracy`
exists to answer the question properly, by serving both checkpoints through the
same vLLM and measuring identity, perplexity, KL and router agreement. This is
the first hypothesis that run should test.

## Reproducing

```bash
modal run engine/modal_quantize_klinear.py
```

Reads `kimi-linear-weights`, writes `kimi-linear-quantized`. The job is
fail-closed: it refuses unclassified matrices, refuses a reduction dimension not
divisible by the group size rather than padding, and refuses to commit the
output volume unless every written shard reopens and matches the manifest.
