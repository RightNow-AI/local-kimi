# Kimi-Linear laptop configuration

## SUPERSEDED FIGURES, read this first

Two numbers below have been overtaken by later work and are kept only so the
change is auditable. Do not quote them.

1. Weight residence. The 24,561,340,864 byte figure is whole-model arithmetic at
   a flat 4.0 bits per parameter. The actual codec costs 4.5 bits per quantized
   parameter once BF16 group scales are counted, and the quantization plan in
   `engine/quant/klinear_plan.py` deliberately leaves the embedding, the LM head,
   the router, all norms and biases, the KDA controls and the MLA latent
   down-projection in source precision. That plan projects 28,789,785,344 bytes
   from real safetensors headers. The authoritative number is the one
   `engine/modal_quantize_klinear.py` measures from the written shards.
   Consequently the "remaining 32 GiB capacity" line below is also too generous.

2. Runtime state. This page models weights only, which is not a server claim.
   `engine/residency/REPORT.md` carries the live budget, including the KDA
   recurrent pool that is sized by `max_num_seqs` rather than by live sequences.

The decode projections remain projections. The 60 percent attainment factor is
transferred from a K3 measurement and has never been measured on this model or
on a laptop.

## Recommended configuration

- Model: `moonshotai/Kimi-Linear-48B-A3B-Instruct`
- Weight format: INT4 weight-only
- Memory target: 32 GiB minimum
- Weight residence: 24,561,340,864 bytes, or 24.561 GB / 22.875 GiB
- Remaining 32 GiB capacity before runtime state: 9.125 GiB
- Projected batch-1 decode at 100 GB/s DDR5: 38.62 tok/s at 60% attainment
- Projected batch-1 decode at 900 GB/s resident dGPU memory: 347.61 tok/s at 60% attainment

The dGPU row applies only when the complete weight bank and runtime state are resident in the
dGPU memory path. A laptop with 32 GiB of system RAM but insufficient VRAM is governed by the
DDR5 row. A 24 GiB device can hold the packed weights arithmetically, but its 1.125 GiB remaining
capacity is not an honest runtime allowance. The supported target is therefore 32 GiB, not 24
GiB.

These tok/s values are projected, not measured on a laptop. The 60% factor is a transferred
calibration from the K3 measurement named in the lane brief. It is not a Kimi-Linear laptop
measurement.

## Parameter arithmetic

The config-level total is 49,122,681,728 parameters, not exactly 48 billion:

```text
token embedding                 163,840 * 2,304             =    377,487,360
untied LM head                  163,840 * 2,304             =    377,487,360
20 KDA layers                   20 * 39,514,272             =    790,285,440
7 MLA layers                    7 * 29,114,880              =    203,804,160
1 dense FFN                     3 * 2,304 * 9,216           =     63,700,992
26 complete MoE banks           26 * 1,819,607,296          = 47,309,789,696
27 pairs of layer norms         27 * 2 * 2,304              =        124,416
final norm                                                    +         2,304
                                                               --------------
total                                                           49,122,681,728
```

The active path is 3,106,974,848 parameters per generated token:

```text
one selected embedding row                                      2,304
untied LM head                                             377,487,360
all KDA and MLA weights                                    994,089,600
the dense FFN                                               63,700,992
26 MoE layers, top-8 plus router and shared expert       1,671,567,872
layer and final norms                                          126,720
                                                         --------------
active per token                                         3,106,974,848
```

For one MoE layer, the active term is:

```text
8 routed experts * 3 matrices * 2,304 * 1,024 = 56,623,104
router: 256 * 2,304 + 256                       =    590,080
one shared expert: 3 * 2,304 * 1,024           =  7,077,888
                                                    ----------
                                                    64,291,072
```

The byte model counts the full untied LM head because every output token scores the vocabulary.
It counts one embedding row rather than the complete embedding table for active traffic.

## Memory fit

Hardware capacities below are binary GiB. Checkpoint and bandwidth GB are decimal.

| Format | Resident bytes | Resident GiB | Active bytes/token | 16 GiB | 24 GiB | 32 GiB | 64 GiB |
|---|---:|---:|---:|:---:|:---:|:---:|:---:|
| BF16 | 98,245,363,456 | 91.498 | 6,213,949,696 | NO | NO | NO | NO |
| FP8 | 49,122,681,728 | 45.749 | 3,106,974,848 | NO | NO | NO | YES |
| INT4 weight-only | 24,561,340,864 | 22.875 | 1,553,487,424 | NO | YES* | YES | YES |

`YES*` means the packed weights alone fit. It does not claim enough headroom for the engine,
KV state, KDA recurrent state, allocator workspace, or the operating system.

INT4 is exactly half the FP8 bytes and one quarter of the BF16 bytes for the same parameter
count.

## Bandwidth roofline

Each physical ceiling is `bandwidth bytes/s / active bytes/token`. Each 60% row is that ceiling
multiplied by 0.60.

| Format | 900 GB/s ceiling | 900 GB/s at 60%, projected | 100 GB/s ceiling | 100 GB/s at 60%, projected |
|---|---:|---:|---:|---:|
| BF16 | 144.84 tok/s | 86.90 tok/s | 16.09 tok/s | 9.66 tok/s |
| FP8 | 289.67 tok/s | 173.80 tok/s | 32.19 tok/s | 19.31 tok/s |
| INT4 weight-only | 579.34 tok/s | 347.61 tok/s | 64.37 tok/s | 38.62 tok/s |

The exact system-memory projection is 38.62 tok/s, below the rounded 40 tok/s expectation in the
brief. Compute, dequantization, cache traffic, and dispatch overhead can only lower a pure
bandwidth roofline unless batching or reuse reduces effective weight traffic.

## Accuracy position

Weight-only INT4 changes model numerics. It is not free. A release configuration owes a paired
accuracy gate against the pinned BF16 checkpoint on the same prompts, tokenizer, decoding
settings, and scoring code. No INT4 accuracy result was measured in this lane.

A reproducible quantized artifact also needs an immutable model revision plus the quantizer,
group size, scale type, zero-point policy, calibration data, and packed layout. This lane provides
the deterministic size and roofline configuration, not a signed or accuracy-gated INT4 artifact.

## What `engine.k3ref` needs

`engine.k3ref.config.K3LayerConfig` can now read both K3's nested config and direct Kimi-Linear
configs. `engine.k3ref.manifest` can build a layer and expert manifest from that config plus merged
safetensors headers. The existing literal K3 layer-12 and MXFP4 manifests remain unchanged, so
the prior K3 shape contract remains available exactly as before.

The broad KDA, MLA, and sparse-MoE family match is not by itself a drop-in loader guarantee. The
offline adapter reports these model-specific requirements when the config/header contains them:

- inject the model-specific manifest into `K3ReferenceLayer` instead of the global K3 layer-12
  constant;
- support low-rank KDA `g_a_proj` / `g_b_proj` when the checkpoint does not use K3's full-rank
  `g_proj`;
- support the safetensors header's `A_log` axis when it differs from K3;
- support direct MLA `q_proj` when `q_lora_rank` is null;
- support non-latent routed experts when the model has no K3 latent down/norm/up projections;
- parameterize the expert provider for ordinary checkpoint weights instead of K3's fixed MXFP4
  packed tensors.

Those module and loader changes live outside this lane's allowed additive edits to
`engine/k3ref/{config,manifest}.py`. The header-derived contract is implemented, but the existing
K3 loader is not yet honestly claimable as a complete Kimi-Linear loader.

## Why full K3 is closed on 32 GB

Using the lane's decimal-GB figures:

```text
full K3:          1,561 GB / 32 GB = 48.78 times the entire memory budget
top-64 variant:     132 GB / 32 GB =  4.125 times the entire memory budget
top-64 overage:     132 GB - 32 GB = 100 GB
```

The top-64 experiment captured 86% of routing at layer 12 in the prior measurement, but 132 GB
still cannot fit in 32 GB. It also discards routed capacity and therefore would owe an accuracy
gate even if it fit. Full K3 and the top-64 variant are both closed for this laptop target.

## Evidence status

| Item | Status |
|---|---|
| Config dimensions and parameter arithmetic | Verified offline from pinned fields |
| BF16 / FP8 / INT4 byte ratios | Computed exactly |
| K3 exact layer-12 manifest preservation | Test written, not run |
| 60% attainment | Prior K3 measurement supplied by the brief |
| Laptop tok/s | Projected only, not measured |
| INT4 paired accuracy | Not measured |
| Full Kimi-Linear load through the existing K3 loader | Not complete for the requirements above |
