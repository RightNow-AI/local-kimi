# Kimi-Linear-48B served from INT4 by this engine, measured

Produced by `engine/modal_klinear_int4.py` on one NVIDIA H100 80GB HBM3, loading
the W4A16 artifact built by `engine/modal_quantize_klinear.py`. Every number here
was read from the device or from the checkpoint. Nothing is projected.

This is the result the product argument rests on, because it is the first time
the two halves of this repository met: a verified INT4 artifact and an engine
that can actually run it.

## What ran

| | |
|---|---|
| GPU | NVIDIA H100 80GB HBM3 |
| Checkpoint kind, detected from the index | `w4a16` |
| Layers | all 27 |
| Experts | all 256 per MoE layer, 8 routed plus 1 shared per token |
| Weight path | fused Triton W4A16 GEMM on packed weights, nothing expanded to BF16 |
| Load time | 65.98 s |

## Memory, measured on the device

| | bytes | GiB |
|---|---:|---:|
| Resident weight bytes | 28,803,304,448 | 26.83 |
| Checkpoint tensor storage | 28,803,304,448 | 26.83 |
| Allocated after load | 29,908,238,336 | 27.85 |
| Reserved after load | 29,972,496,384 | 27.91 |
| **Peak allocated after generation** | **30,265,326,080** | **28.19** |
| **Peak reserved after generation** | **30,511,464,448** | **28.42** |

`resident_matches_checkpoint` is `true`: the model's own accounting of what it
holds equals the checkpoint's tensor storage exactly. The loader fails closed if
those disagree, so the artifact cannot quietly be a different size from the one
that was measured.

The gap between resident weights and peak reserved, 1.71 GiB, is the allocator's
working set for this short run: activations, the KDA recurrent state, the MLA
compressed-latent cache and workspace. It is not a serving envelope. For the
envelope as a function of `max_num_seqs` and `max_model_len`, see
`engine/residency/REPORT.md`.

## Generation

```
prompt        "The main benefit of lower model memory is"
continuation  " that it allows for more efficient use of"
token ids     [473, 483, 6846, 395, 1070, 11021, 1328, 318]
```

The continuation is grammatical, on topic, and continues the prompt sensibly.
That is a meaningful signal because it is the whole model rather than a slice:
an earlier result in this project generated valid token ids from three of
ninety-three K3 layers and was explicitly labelled a correctness result about
machinery, not a working model. This is not that.

It is still one short greedy continuation from one prompt. It shows the stack is
wired correctly end to end. It is NOT a quality measurement, and it says nothing
about how much the INT4 quantization costs. `engine/accuracy/` exists to answer
that by serving the original and a dequantized checkpoint through the same vLLM
and measuring identity, perplexity, KL and router agreement.

## What this does and does not establish

Establishes, measured:

- This engine loads and runs the selective INT4 artifact.
- Serving the full model peaks at 30,511,464,448 bytes of reserved device memory
  under this short workload, which is 28.42 GiB.
- The fused path works on real weights: packed INT4 feeds the GEMM directly and
  no BF16 weight matrix is materialised.

Does not establish:

- Any throughput number. Eight tokens were generated. Nothing here should be
  extrapolated to tokens per second, and no timing on this page is a performance
  claim.
- Any quality result.
- That a 32 GiB consumer card runs this. The measurement was taken on an H100
  80GB. That 28.42 GiB fits inside 32 GiB is arithmetic on a measured figure,
  not an observation on a 32 GiB device, and a smaller card differs in more than
  capacity.

## Why the comparison matters

Stock vLLM 0.26.0 refuses this model below BF16. Measured on an H200 in the same
session: the as-shipped checkpoint loads in 377.87 s and generates, and the
bitsandbytes 4-bit path is refused at load with

```
Model KimiLinearForCausalLM does not support BitsAndBytes quantization yet.
No 'packed_modules_mapping' found.
```

So the alternative a buyer already has needs 98,245,528,576 bytes of weights.
This needs 28,803,304,448, measured, and produces coherent text.

That is a footprint statement, not a speed statement. No speed comparison
against vLLM has been measured, and none is claimed.

## Reproducing

```bash
modal run engine/modal_quantize_klinear.py   # build the artifact, once
modal run engine/modal_klinear_int4.py       # load it and generate
```
