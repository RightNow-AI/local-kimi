# Kimi-Linear live residency budget

## Verdict

The weights-only 32 GiB claim is not a server claim. Under the inspected runtime layout,
INT4 weights, FP32 KDA recurrent state, BF16 convolution state, BF16 MLA cache, and the
explicit 3 GiB operational reserve used by this report:

| Envelope | Total bytes | Total GiB | Fits 32 GiB | Status |
|---|---:|---:|:---:|---|
| `max_num_seqs=1`, `max_model_len=32768` | 32,524,095,936 | 30.290 | YES | PROJECTED, NOT MEASURED |
| `max_num_seqs=2`, `max_model_len=32768` | 37,265,625,536 | 34.706 | NO | PROJECTED, NOT MEASURED |
| `max_num_seqs=16`, `max_model_len=32768` | 103,647,039,936 | 96.529 | NO | PROJECTED, NOT MEASURED |
| `max_num_seqs=64`, `max_model_len=32768` | 331,240,460,736 | 308.492 | NO | PROJECTED, NOT MEASURED |

The precise 32 GiB frontier at one sequence is 45,572 tokens. A 32K context fits only
at `max_num_seqs=1` under this policy. The proposed `16 x 32K` point is not close to
fitting because its expanded MLA cache alone is 70 GiB.

No card in the requested set reaches the advertised 1,048,576-token context with the
current expanded MLA cache, even at `max_num_seqs=1`. On INT4, that cache alone is
exactly 140 GiB. The full projected single-sequence total is 165.915 GiB.

## Evidence status

- `SOURCE-DERIVED` means an exact tensor shape or dtype was read from pinned source.
- `PROJECTED, NOT MEASURED` means byte arithmetic or a policy reserve was applied.
- `MEASURED` is reserved for output from `engine/modal_residency.py`.
- The Modal harness has not been run in this lane, so this report contains no measured row.

## Authoritative source trail

The Hugging Face files were fetched with Python `urllib` from model revision
`e1df551a447157d4658b573f9a695d57658590e9`.

| Artifact | SHA256 | Status |
|---|---|---|
| `modeling_kimi.py` | `d79b365e37378881b9f1585007a56e236ca27a414920943cb85d1dacb75dda99` | SOURCE-DERIVED |
| `configuration_kimi.py` | `79422aca3ee6c89d201e0c15c4c9a6db517ba83d87ecdc4e41fa0f71297238d9` | SOURCE-DERIVED |
| `config.json` | `a6ac3c2c4b5aa72370f9727f49ffa4432715d20061889acdb37c688be853096e` | SOURCE-DERIVED |

The exact `modeling_kimi.py` lines used were:

- Lines 451 to 453 read convolution width 4, head dimension 128, and 32 heads.
- Lines 462 to 484 set equal Q, K, and V widths and construct three short convolutions.
- Lines 563 to 565 reshape Q, K, and V into the 32 by 128 head layout.
- Lines 568 to 594 request and retain the recurrent final state and all three conv states.
- Lines 397 to 414 expand MLA into per-head keys and values before updating the cache.

Short exact excerpts from those lines are:

> `projection_k_size = self.head_k_dim * self.num_k_heads`

> `projection_size = self.head_dim * self.num_heads`

> `key_states, value_states = past_key_values.update`

> `output_final_state=True`

The model delegates the actual KDA and convolution allocations to unpinned `fla-core`.
The allocation was therefore traced into FLA commit
`9c8e42e762fce087c27b673af4922795d9edb85e`, dated 2026-07-27:

- `fla/ops/common/chunk_delta_h.py` lines 690 to 707 derive `N`, `HV`, `K`, and `V`,
  then allocate the final state in FP32.
- `fla/ops/kda/fused_recurrent.py` lines 271 to 278 independently allocate the same
  FP32 final-state shape for recurrent decode.
- `fla/modules/conv/short_conv.py` lines 211 to 217 allocate each cache with sequence
  count `N`, channel count `D`, full kernel width `W`, and the input projection dtype.

The decisive FLA excerpts are:

> `final_state = k.new_zeros(N, HV, K, V, dtype=torch.float32)`

> `cache = x.new_zeros(N, D, W)`

## Exact byte model

The budget treats `max_num_seqs` as the fixed server pool capacity. The Hugging Face
dynamic cache uses actual batch size `N`; a serving engine that preallocates capacity
must substitute `max_num_seqs` for `N`.

| Component | Exact formula | Rate | Status |
|---|---|---:|---|
| INT4 weights | pinned profile | 24,561,340,864 bytes | PROJECTED, NOT MEASURED |
| BF16 weights | 49,122,681,728 parameters times 2 | 98,245,363,456 bytes | PROJECTED, NOT MEASURED |
| KDA recurrent pool | `20 * seqs * 32 * 128 * 128 * 4` | 41,943,040 bytes per sequence | SOURCE-DERIVED, NOT MEASURED |
| Short conv pool | `20 * seqs * 3 * 4096 * 4 * 2` | 1,966,080 bytes per sequence | SOURCE-DERIVED, NOT MEASURED |
| Expanded MLA key | `7 * seqs * tokens * 32 * 192 * 2` | 86,016 bytes per token per sequence | SOURCE-DERIVED, NOT MEASURED |
| Expanded MLA value | `7 * seqs * tokens * 32 * 128 * 2` | 57,344 bytes per token per sequence | SOURCE-DERIVED, NOT MEASURED |
| Expanded MLA total | key plus value | 143,360 bytes per token per sequence | SOURCE-DERIVED, NOT MEASURED |
| Activation reserve | explicit report policy | 2,147,483,648 bytes | PROJECTED, NOT MEASURED |
| Workspace reserve | explicit report policy | 1,073,741,824 bytes | PROJECTED, NOT MEASURED |

The 3 GiB reserve is a visible policy input, not a hidden fudge factor and not a
measurement. Change it through `RuntimeHeadroom` and recompute the frontier. The Modal
harness measures state allocation only and reports allocator reservation separately.

## Corrections to the first-pass hypotheses

| Hypothesis | Result | Why | Status |
|---|---|---|---|
| KDA is `32 * 128 * 128` FP32 elements per layer per sequence | CORRECT | Both FLA prefill and recurrent paths allocate that final-state shape in FP32. | SOURCE-DERIVED |
| KDA is 40 MiB per sequence across 20 layers | CORRECT | `20 * 32 * 128 * 128 * 4 = 41,943,040` bytes. | SOURCE-DERIVED, NOT MEASURED |
| Conv holds only 3 prior positions | WRONG FOR ALLOCATION | FLA allocates full width `W=4`, not `W-1`. | SOURCE-DERIVED, NOT MEASURED |
| Unknown number of convolved projections | RESOLVED | Q, K, and V are all convolved. Each projection width is 4096. | SOURCE-DERIVED, NOT MEASURED |
| MLA caches 512 latent plus 64 rotary elements | WRONG FOR THIS CODE | The model expands to 32 per-head keys of width 192 and values of width 128 before cache update. | SOURCE-DERIVED, NOT MEASURED |
| MLA costs 8,064 bytes per token across seven layers | WRONG | The shipped cache costs 143,360 bytes, exactly 17.7778 times larger. | SOURCE-DERIVED, NOT MEASURED |
| `16 x 32K` INT4 is about 27.5 GiB | WRONG | The corrected total with explicit headroom is 96.529 GiB. | PROJECTED, NOT MEASURED |
| `64 x 32K` MLA is about 16.9 GB | WRONG | The corrected MLA cache alone is 300,647,710,720 bytes, or 280 GiB. | PROJECTED, NOT MEASURED |

A compressed 576-element MLA cache would be a different engine implementation. It is
not what the inspected remote code stores, so it is not used in the product claim.

## Projected frontier

Policy for every row:

- Card labels are treated as binary GiB capacities.
- State dtypes are FP32 recurrent, BF16 conv, and BF16 MLA.
- Operational reserve is 2 GiB activation plus 1 GiB workspace.
- Sequence-pool candidates are `1, 2, 4, 8, 16, 32, 64, 128, 256`.
- Each length is the largest exact integer that fits for that sequence-pool candidate.
- The model length is capped at 1,048,576.

### 24 GiB

| Weight profile | Frontier | Reason | Status |
|---|---|---|---|
| INT4 | No envelope | Weights plus the 3 GiB reserve already require 25.875 GiB before state. | PROJECTED, NOT MEASURED |
| BF16 | No envelope | Weights alone require 91.498 GiB. | PROJECTED, NOT MEASURED |

### 32 GiB

BF16 has no envelope because weights alone exceed capacity.

| Weight profile | max_num_seqs | Maximum max_model_len | Status |
|---|---:|---:|---|
| INT4 | 1 | 45,572 | PROJECTED, NOT MEASURED |
| INT4 | 2 | 22,633 | PROJECTED, NOT MEASURED |
| INT4 | 4 | 11,163 | PROJECTED, NOT MEASURED |
| INT4 | 8 | 5,428 | PROJECTED, NOT MEASURED |
| INT4 | 16 | 2,561 | PROJECTED, NOT MEASURED |
| INT4 | 32 | 1,127 | PROJECTED, NOT MEASURED |
| INT4 | 64 | 410 | PROJECTED, NOT MEASURED |
| INT4 | 128 | 52 | PROJECTED, NOT MEASURED |
| BF16 | none | none | PROJECTED, NOT MEASURED |

### 48 GiB

BF16 has no envelope because weights alone exceed capacity.

| Weight profile | max_num_seqs | Maximum max_model_len | Status |
|---|---:|---:|---|
| INT4 | 1 | 165,409 | PROJECTED, NOT MEASURED |
| INT4 | 2 | 82,551 | PROJECTED, NOT MEASURED |
| INT4 | 4 | 41,122 | PROJECTED, NOT MEASURED |
| INT4 | 8 | 20,408 | PROJECTED, NOT MEASURED |
| INT4 | 16 | 10,050 | PROJECTED, NOT MEASURED |
| INT4 | 32 | 4,872 | PROJECTED, NOT MEASURED |
| INT4 | 64 | 2,283 | PROJECTED, NOT MEASURED |
| INT4 | 128 | 988 | PROJECTED, NOT MEASURED |
| INT4 | 256 | 341 | PROJECTED, NOT MEASURED |
| BF16 | none | none | PROJECTED, NOT MEASURED |

### 80 GiB

BF16 has no envelope because weights alone exceed capacity.

| Weight profile | max_num_seqs | Maximum max_model_len | Status |
|---|---:|---:|---|
| INT4 | 1 | 405,084 | PROJECTED, NOT MEASURED |
| INT4 | 2 | 202,388 | PROJECTED, NOT MEASURED |
| INT4 | 4 | 101,041 | PROJECTED, NOT MEASURED |
| INT4 | 8 | 50,367 | PROJECTED, NOT MEASURED |
| INT4 | 16 | 25,030 | PROJECTED, NOT MEASURED |
| INT4 | 32 | 12,362 | PROJECTED, NOT MEASURED |
| INT4 | 64 | 6,027 | PROJECTED, NOT MEASURED |
| INT4 | 128 | 2,860 | PROJECTED, NOT MEASURED |
| INT4 | 256 | 1,277 | PROJECTED, NOT MEASURED |
| BF16 | none | none | PROJECTED, NOT MEASURED |

### 141 GiB

| Weight profile | max_num_seqs | Maximum max_model_len | Status |
|---|---:|---:|---|
| INT4 | 1 | 861,963 | PROJECTED, NOT MEASURED |
| INT4 | 2 | 430,828 | PROJECTED, NOT MEASURED |
| INT4 | 4 | 215,261 | PROJECTED, NOT MEASURED |
| INT4 | 8 | 107,477 | PROJECTED, NOT MEASURED |
| INT4 | 16 | 53,585 | PROJECTED, NOT MEASURED |
| INT4 | 32 | 26,639 | PROJECTED, NOT MEASURED |
| INT4 | 64 | 13,166 | PROJECTED, NOT MEASURED |
| INT4 | 128 | 6,430 | PROJECTED, NOT MEASURED |
| INT4 | 256 | 3,061 | PROJECTED, NOT MEASURED |
| BF16 | 1 | 347,984 | PROJECTED, NOT MEASURED |
| BF16 | 2 | 173,839 | PROJECTED, NOT MEASURED |
| BF16 | 4 | 86,766 | PROJECTED, NOT MEASURED |
| BF16 | 8 | 43,230 | PROJECTED, NOT MEASURED |
| BF16 | 16 | 21,461 | PROJECTED, NOT MEASURED |
| BF16 | 32 | 10,577 | PROJECTED, NOT MEASURED |
| BF16 | 64 | 5,135 | PROJECTED, NOT MEASURED |
| BF16 | 128 | 2,414 | PROJECTED, NOT MEASURED |
| BF16 | 256 | 1,054 | PROJECTED, NOT MEASURED |

## Advertised 1M context

| Capacity | Weight profile | Best single-sequence length | 1,048,576 reachable | Status |
|---:|---|---:|:---:|---|
| 24 GiB | INT4 | none | NO | PROJECTED, NOT MEASURED |
| 24 GiB | BF16 | none | NO | PROJECTED, NOT MEASURED |
| 32 GiB | INT4 | 45,572 | NO | PROJECTED, NOT MEASURED |
| 32 GiB | BF16 | none | NO | PROJECTED, NOT MEASURED |
| 48 GiB | INT4 | 165,409 | NO | PROJECTED, NOT MEASURED |
| 48 GiB | BF16 | none | NO | PROJECTED, NOT MEASURED |
| 80 GiB | INT4 | 405,084 | NO | PROJECTED, NOT MEASURED |
| 80 GiB | BF16 | none | NO | PROJECTED, NOT MEASURED |
| 141 GiB | INT4 | 861,963 | NO | PROJECTED, NOT MEASURED |
| 141 GiB | BF16 | 347,984 | NO | PROJECTED, NOT MEASURED |

## Modal measurement contract

`engine/modal_residency.py` allocates the exact persistent state structures as separate
GPU tensors at several points. It emits JSON containing:

- predicted state-pool bytes from `budget.py`;
- tensor storage bytes from `numel * element_size`;
- measured `torch.cuda.max_memory_allocated` delta;
- measured `torch.cuda.max_memory_reserved` delta;
- signed allocated and reserved deltas from prediction;
- `MATCH`, `MISMATCH`, or `OOM` for every point.

The default point set ends with `max_num_seqs=16`, `max_model_len=32768`, so the
central rejected 32 GiB hypothesis is measured directly on the H100 when the
orchestrator runs the job.

The harness does not load weights. Its state comparison is still the critical check of
the shape model, while weight bytes and the operational reserve remain explicitly
projected. A nonzero allocated delta is surfaced as `MISMATCH`; allocator reservation
overhead is reported separately and is never smoothed into the prediction.

## Remaining risks

- Modal was not run, so no measured allocation can yet certify these projections.
- The model asks users to install the latest `fla-core` rather than pinning a version.
  A future allocation-layout change can invalidate the source-derived state contract.
- The INT4 profile is the existing exact four-bit arithmetic. A real packed artifact
  can add scales, zero points, alignment, and metadata unless its measured resident
  bytes are substituted into `QuantizationProfile`.
- The 2 GiB activation reserve and 1 GiB workspace reserve are policy values, not a
  measured peak for a production engine. The solver makes them explicit so the
  orchestrator can replace them after measurement.
- Frontier labels follow the repo's binary GiB convention. Production gating should
  pass the device's actual byte capacity, because a marketed GB label can be smaller.
- A production allocator may reserve more than tensor storage. The Modal JSON reports
  both allocated and reserved peaks so that difference remains visible.
