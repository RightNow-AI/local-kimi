# Kimi-Linear live residency budget

## The state model is MEASURED, and it is exact

`engine/modal_residency.py` ran on an NVIDIA H100 80GB HBM3 (torch 2.13.0+cu130,
CUDA 13.0) and allocated the real persistent state structures at several
envelope points, comparing device allocation against this model's prediction.

| max_num_seqs x max_model_len | predicted state pool | measured allocated delta | difference | status |
|---|---:|---:|---:|---|
| 1 x 32,768 | 308,150,272 | 308,150,272 | **0** | MATCH |
| 2 x 32,768 | 439,877,632 | 439,877,632 | **0** | MATCH |
| 8 x 32,768 | 966,787,072 | 966,787,072 | **0** | MATCH |

`allocated_minus_predicted_bytes` is zero at every point. The byte model derived
by reading Moonshot's own code predicts real GPU allocation exactly, so the KDA
recurrent pool, the short-convolution pool and the compressed-latent MLA cache
formulas below are correct rather than merely plausible.

Reserved memory differs from allocated by the allocator's own rounding, which is
reported separately and never smoothed into the prediction: at the three points
above the reserved deltas were +131,072 bytes, +42,467,328 and 0.

What this does NOT measure, and the report says so at each row: the weight bytes
are not allocated by this harness, so weight figures remain MEASURED-from-artifact
rather than measured here, and the 3 GiB operational reserve remains an explicit
policy input rather than an observed peak.

## Verdict

All INT4 rows now use one named weight input:
`MEASURED_INT4_SELECTIVE_WEIGHTS = 28,803,304,448 bytes`. This is tensor
storage read from the built selective-INT4 artifact. It supersedes the projected
24,561,340,864-byte flat-4-bit estimate.

There are two real MLA cache policies for this model:

| Implementation | Cache policy | BF16 bytes per token across 7 MLA layers | Status |
|---|---|---:|---|
| Hugging Face reference | Expanded per-head keys and values | 143,360 | SOURCE-DERIVED, NOT MEASURED |
| vLLM 0.26.0 `FLASH_ATTN_MLA` | Compressed latent plus rotary key | 8,064 | SOURCE-DERIVED, NOT MEASURED |

vLLM stores one 576-element record per token per MLA layer: 512 compressed
latent elements plus 64 rotary-key elements. At BF16 this is 1,152 bytes per
layer and 8,064 bytes across the model's seven MLA layers.

The compressed-latent cache is table stakes rather than an advantage over
vLLM. Our engine must preserve the 512 plus 64 representation in persistent
cache and run prefill and decode directly from it. Persistently expanding to
per-head keys and values would make the engine materially less memory-efficient
than the buyer's existing vLLM option.

### Observed cross-check on a running server

The finding above is derived by reading vLLM's source. A live vLLM 0.26.0 run on
one H200, serving this model's BF16 checkpoint, independently reports:

```
GPU KV cache size: 2,322,432 tokens
Maximum concurrency for 8,192 tokens per request: 283.50x
```

An H200 holds 141 GB and the BF16 weights alone occupy 98.2 GB, so at most tens
of GB remain for cache. At the expanded rate of 143,360 bytes per token,
2,322,432 tokens would require roughly 333 GB, which is more than twice the
whole card. The observation is therefore only consistent with the compressed
policy, and the expanded policy is ruled out by a running server rather than
only by reading code.

This is a consistency check, not a measurement of the per-token rate. It bounds
the answer rather than pinning it, because the same pool also holds the KDA
recurrent state and vLLM applies its own utilization fraction. Treat it as
corroboration of the source trace, which remains the precise statement.

## Customer-facing 32 GiB envelope

**Quote this envelope:** with the MEASURED selective-INT4 weights, the
compressed-latent cache, and the explicit PROJECTED 3 GiB operational reserve,
a 32 GiB card supports `max_num_seqs=7` at `max_model_len=32,768`.

This report uses 32K as the explicit useful-context threshold for that quote.
The exact total is 34,181,581,824 bytes, leaving 178,156,544 bytes of arithmetic
slack. `8 x 32K` does not fit.

| Envelope | Expanded total | Compressed total | 32 GiB compressed result | Status |
|---|---:|---:|---|---|
| `1 x 32K` | 36,766,059,520 | 32,332,680,192 | Fits by 2,027,058,176 bytes | MEASURED weights; SOURCE-DERIVED state; PROJECTED reserve and total |
| `7 x 32K` | 65,215,237,120 | 34,181,581,824 | Fits by 178,156,544 bytes | MEASURED weights; SOURCE-DERIVED state; PROJECTED reserve and total |
| `8 x 32K` | 69,956,766,720 | 34,489,732,096 | Exceeds by 129,993,728 bytes | MEASURED weights; SOURCE-DERIVED state; PROJECTED reserve and total |
| `16 x 32K` | 107,889,003,520 | 36,954,934,272 | Exceeds by 2,595,195,904 bytes | MEASURED weights; SOURCE-DERIVED state; PROJECTED reserve and total |
| `1 x 1M` | 182,392,294,400 | 40,524,155,904 | Exceeds by 6,164,417,536 bytes | MEASURED weights; SOURCE-DERIVED state; PROJECTED reserve and total |

The exact compressed-policy concurrency tradeoff on 32 GiB is:

| Context per sequence | Maximum `max_num_seqs` | Total bytes | Slack bytes | Status |
|---:|---:|---:|---:|---|
| 8,192 | 21 | 34,333,887,488 | 25,850,880 | MEASURED weights; SOURCE-DERIVED state; PROJECTED reserve and total |
| 16,384 | 13 | 34,312,915,968 | 46,822,400 | MEASURED weights; SOURCE-DERIVED state; PROJECTED reserve and total |
| 32,768 | 7 | 34,181,581,824 | 178,156,544 | MEASURED weights; SOURCE-DERIVED state; PROJECTED reserve and total |
| 65,536 | 4 | 34,314,095,616 | 45,642,752 | MEASURED weights; SOURCE-DERIVED state; PROJECTED reserve and total |
| 131,072 | 2 | 34,226,277,376 | 133,460,992 | MEASURED weights; SOURCE-DERIVED state; PROJECTED reserve and total |

## Auditable corrections

| Input or result | Superseded value | Current value | Difference | Status |
|---|---:|---:|---:|---|
| Selective INT4 resident tensor storage | 24,561,340,864 | 28,803,304,448 | +4,241,963,584 | Old PROJECTED and superseded; current MEASURED |
| BF16 resident tensor storage | 98,245,363,456 | 98,245,528,576 | +165,120 | Old PROJECTED arithmetic; current MEASURED |
| Compressed `16 x 32K` total | 32,712,970,688 | 36,954,934,272 | +4,241,963,584 | MEASURED weight correction; SOURCE-DERIVED state; PROJECTED reserve and total |

The user's recompute was exact. The entire `16 x 32K` change is the measured
weight delta because the state shapes, dtypes, cache policy, and headroom policy
did not change.

## Evidence status

- `MEASURED` means bytes were read from real tensor storage or a real allocator.
  It does not by itself mean the complete live server was measured.
- `SOURCE-DERIVED` means a tensor shape, dtype, or cache-page formula was read
  from pinned source.
- `PROJECTED` means exact arithmetic or an explicit policy reserve was applied.
- No GPU, vLLM server, test runner, or Modal job was run in this residency update.
  The weight measurement was produced by the quantization lane and consumed here.

## Weight artifact trail

The authoritative artifact report is `engine/quant/QUANTIZATION-RESULTS.md` on
main. It records an H100 build from the real BF16 checkpoint.

| Artifact quantity | Bytes | Used for live weight residency | Status |
|---|---:|:---:|---|
| Source tensor storage, BF16 as shipped | 98,245,528,576 | Yes, BF16 profile | MEASURED |
| Output tensor storage, selective INT4 | 28,803,304,448 | Yes, INT4 profile | MEASURED |
| Output safetensors files | 28,809,016,344 | No, disk artifact size | MEASURED |
| Output directory including support files | 28,840,764,112 | No, disk artifact size | MEASURED |

The selective format is symmetric signed INT4, group size 32, with BF16 scales
and no zero point. It costs 4.5 bits per quantized parameter including scales.
The measured tensor-storage figure already includes the classes deliberately
retained in source precision.

## Which policy belongs to which implementation

| Policy | Persistent record per token per MLA layer | Implementation | Status |
|---|---|---|---|
| `expanded` | 32 keys of width 192 plus 32 values of width 128 | Hugging Face `modeling_kimi.py` | SOURCE-DERIVED, NOT MEASURED |
| `compressed_latent` | One width-576 record containing width-512 latent and width-64 rotary key | vLLM 0.26.0 MLA cache | SOURCE-DERIVED, NOT MEASURED |

The two policies are explicit through `MLACachePolicy` in `budget.py`. The
default remains `expanded` for backward compatibility with the original
conservative report. Product and vLLM comparisons must select
`compressed_latent` explicitly.

## Hugging Face source trail

The Hugging Face files were fetched with Python `urllib` from model revision
`e1df551a447157d4658b573f9a695d57658590e9`.

| Artifact | SHA256 | Status |
|---|---|---|
| `modeling_kimi.py` | `d79b365e37378881b9f1585007a56e236ca27a414920943cb85d1dacb75dda99` | SOURCE-DERIVED |
| `configuration_kimi.py` | `79422aca3ee6c89d201e0c15c4c9a6db517ba83d87ecdc4e41fa0f71297238d9` | SOURCE-DERIVED |
| `config.json` | `a6ac3c2c4b5aa72370f9727f49ffa4432715d20061889acdb37c688be853096e` | SOURCE-DERIVED |

The decisive `modeling_kimi.py` lines are:

- Lines 397 to 414 expand MLA into per-head keys and values before cache update.
- Lines 451 to 484 define KDA dimensions and all three short convolutions.
- Lines 563 to 594 request and retain recurrent and convolution final states.

Short exact excerpts are:

> `projection_k_size = self.head_k_dim * self.num_k_heads`

> `projection_size = self.head_dim * self.num_heads`

> `key_states, value_states = past_key_values.update`

> `output_final_state=True`

The model delegates KDA and convolution allocation to unpinned `fla-core`. The
allocation was traced into FLA commit
`9c8e42e762fce087c27b673af4922795d9edb85e`:

- `fla/ops/common/chunk_delta_h.py` lines 690 to 707 allocate the FP32 recurrent
  state.
- `fla/ops/kda/fused_recurrent.py` lines 271 to 278 allocate the recurrent
  decode state.
- `fla/modules/conv/short_conv.py` lines 211 to 217 allocate full-width conv
  caches.

The decisive FLA excerpts are:

> `final_state = k.new_zeros(N, HV, K, V, dtype=torch.float32)`

> `cache = x.new_zeros(N, D, W)`

## vLLM 0.26.0 source trail

The vLLM files were fetched with Python `urllib` from tag `v0.26.0`, commit
`568afb3a13806beb53bb2e6bd518269357b237c0`.

| Artifact | SHA256 | Status |
|---|---|---|
| `vllm/model_executor/models/kimi_linear.py` | `4a0dee43d6a3b1d0d665fa329a8e9c6c6591709c365f3ee6ec31e72cd4ee169a` | SOURCE-DERIVED |
| `vllm/model_executor/layers/mla.py` | `d461e5bf42efd431a38dc1b7a408c6ddf8b15793f8a4e234322410394d46d7b9` | SOURCE-DERIVED |
| `vllm/model_executor/layers/attention/mla_attention.py` | `5d757540ee25d6a7e2c1cf9d348f987148d3eb14d569d5abcc9a8714535f8b46` | SOURCE-DERIVED |
| `vllm/v1/attention/backends/mla/flashattn_mla.py` | `4f4e1cdf655bacbaa98bbff00b4136fb6f3369012d8f8272345b9cdd15fb9093` | SOURCE-DERIVED |
| `vllm/v1/kv_cache_interface.py` | `73b5967f23ff2d4526b984cf90c1203e550575e5f329650c8899269b8f78edcf` | SOURCE-DERIVED |
| `vllm/utils/torch_utils.py` | `4b439b2ba954e5b4d9d4f86f9a26135ab995ba7d71f74a4d9f1763168921b406` | SOURCE-DERIVED |

The exact vLLM line chain is:

1. `kimi_linear.py` lines 217 to 220 build a width-576 K/V A projection, and
   lines 264 to 274 pass `kv_lora_rank` and `qk_rope_head_dim` into the MLA
   wrapper.
2. `layers/mla.py` lines 154 to 157 split the projected record into width 512
   and 64, then lines 175 to 179 pass both compressed parts into
   `MLAAttention`.
3. `mla_attention.py` lines 388 to 392 set cache head size to 512 plus 64 and
   set one KV head. Lines 1075 to 1085 create `MLAAttentionSpec` with that head
   size.
4. `kv_cache_interface.py` lines 398 to 415 compute MLA page bytes as block
   size, one KV head, head size, and dtype size. Unlike ordinary attention,
   there is no separate key-plus-value factor of two.
5. `flashattn_mla.py` lines 43 to 65 identify the selected backend as
   `FLASH_ATTN_MLA`. Lines 338 to 339 split its live cache at `kv_lora_rank`.
6. `torch_utils.py` lines 395 to 401 resolve `cache_dtype=auto` to the model
   dtype, which is BF16 for this model.

Short exact excerpts are:

> `self.head_size = kv_lora_rank + qk_rope_head_dim`

> `self.num_kv_heads = 1`

> `head_size=self.head_size`

From the MLA page-size formula:

> `self.storage_block_size * self.num_kv_heads * head_dim * get_dtype_size(self.dtype)`

From the selected backend:

> `kv_c_cache = kv_c_and_k_pe_cache[..., : self.kv_lora_rank]`

> `k_pe_cache = kv_c_and_k_pe_cache[..., self.kv_lora_rank :]`

## Exact byte model

The budget treats `max_num_seqs` as fixed server pool capacity. State dtypes are
FP32 recurrent, BF16 convolution, and BF16 MLA unless explicitly replaced.

| Component | Exact formula or named input | Rate | Status |
|---|---|---:|---|
| Selective INT4 weights | `MEASURED_INT4_SELECTIVE_WEIGHTS` | 28,803,304,448 bytes | MEASURED |
| BF16 weights | `MEASURED_BF16_WEIGHTS` | 98,245,528,576 bytes | MEASURED |
| KDA recurrent pool | `20 * seqs * 32 * 128 * 128 * 4` | 41,943,040 bytes per sequence | SOURCE-DERIVED, NOT MEASURED |
| Short conv pool | `20 * seqs * 3 * 4096 * 4 * 2` | 1,966,080 bytes per sequence | SOURCE-DERIVED, NOT MEASURED |
| Expanded MLA | `7 * seqs * tokens * 32 * (192 + 128) * 2` | 143,360 bytes per token per sequence | SOURCE-DERIVED, NOT MEASURED |
| Compressed MLA | `7 * seqs * tokens * (512 + 64) * 2` | 8,064 bytes per token per sequence | SOURCE-DERIVED, NOT MEASURED |
| Activation reserve | `DEFAULT_HEADROOM.activation_bytes` | 2,147,483,648 bytes | PROJECTED POLICY, NOT MEASURED |
| Workspace reserve | `DEFAULT_HEADROOM.workspace_bytes` | 1,073,741,824 bytes | PROJECTED POLICY, NOT MEASURED |

The expanded policy costs exactly 17.7778 times the compressed policy per
token. `DEFAULT_HEADROOM` is an explicit 3 GiB policy input. It is not a
measured peak and is never folded invisibly into another component.

## Frontier side by side

Policy for every row:

- Card labels are binary GiB capacities.
- Operational reserve is the PROJECTED `DEFAULT_HEADROOM`: 2 GiB activation
  plus 1 GiB workspace.
- Sequence candidates are `1, 2, 4, 8, 16, 32, 64, 128, 256`.
- Each `S:L` pair means `max_num_seqs=S`, maximum `max_model_len=L`.
- Dominated points are omitted.
- Model length is capped at 1,048,576.
- Weights are MEASURED, state rates are SOURCE-DERIVED, and the resulting
  envelope is PROJECTED rather than end-to-end measured.

### 24 GiB

| Weights | Expanded frontier | Compressed frontier | Status |
|---|---|---|---|
| Selective INT4 | none | none | MEASURED weights; SOURCE-DERIVED state; PROJECTED reserve and envelope |
| BF16 | none | none | MEASURED weights; SOURCE-DERIVED state; PROJECTED reserve and envelope |

Selective INT4 weights plus `DEFAULT_HEADROOM` require 32,024,529,920 bytes
before any sequence state, so 24 GiB cannot boot this budget.

### 32 GiB

| Weights | Expanded frontier | Compressed frontier | Status |
|---|---|---|---|
| Selective INT4 | `1:15,982; 2:7,838; 4:3,765; 8:1,729; 16:711; 32:202` | `1:284,139; 2:139,347; 4:66,951; 8:30,752; 16:12,653; 32:3,604` | MEASURED weights; SOURCE-DERIVED state; PROJECTED reserve and envelope |
| BF16 | none | none | MEASURED weights; SOURCE-DERIVED state; PROJECTED reserve and envelope |

### 48 GiB

| Weights | Expanded frontier | Compressed frontier | Status |
|---|---|---|---|
| Selective INT4 | `1:135,820; 2:67,756; 4:33,725; 8:16,709; 16:8,201; 32:3,947; 64:1,820; 128:757; 256:225` | `2:1,048,576; 4:599,561; 8:297,057; 16:145,806; 32:70,180; 64:32,367; 128:13,461; 256:4,008` | MEASURED weights; SOURCE-DERIVED state; PROJECTED reserve and envelope |
| BF16 | none | none | MEASURED weights; SOURCE-DERIVED state; PROJECTED reserve and envelope |

### 80 GiB

| Weights | Expanded frontier | Compressed frontier | Status |
|---|---|---|---|
| Selective INT4 | `1:375,494; 2:187,594; 4:93,643; 8:46,668; 16:23,181; 32:11,437; 64:5,565; 128:2,629; 256:1,161` | `4:1,048,576; 8:829,668; 16:412,111; 32:203,333; 64:98,944; 128:46,749; 256:20,652` | MEASURED weights; SOURCE-DERIVED state; PROJECTED reserve and envelope |
| BF16 | none | none | MEASURED weights; SOURCE-DERIVED state; PROJECTED reserve and envelope |

### 141 GiB

| Weights | Expanded frontier | Compressed frontier | Status |
|---|---|---|---|
| Selective INT4 | `1:832,374; 2:416,033; 4:207,863; 8:103,778; 16:51,736; 32:25,714; 64:12,704; 128:6,199; 256:2,946` | `8:1,048,576; 16:919,755; 32:457,155; 64:225,855; 128:110,204; 256:52,379` | MEASURED weights; SOURCE-DERIVED state; PROJECTED reserve and envelope |
| BF16 | `1:347,983; 2:173,838; 4:86,766; 8:43,229; 16:21,461; 32:10,577; 64:5,135; 128:2,414; 256:1,054` | `4:1,048,576; 8:768,532; 16:381,543; 32:188,049; 64:91,302; 128:42,928; 256:18,741` | MEASURED weights; SOURCE-DERIVED state; PROJECTED reserve and envelope |

## Advertised 1,048,576-token context

This table uses exact integer sequence counts rather than the powers-of-two
frontier grid.

| Capacity | Weights | Expanded max sequences at 1M | Compressed max sequences at 1M | Status |
|---:|---|---:|---:|---|
| 24 GiB | Selective INT4 | 0 | 0 | MEASURED weights; SOURCE-DERIVED state; PROJECTED reserve and envelope |
| 24 GiB | BF16 | 0 | 0 | MEASURED weights; SOURCE-DERIVED state; PROJECTED reserve and envelope |
| 32 GiB | Selective INT4 | 0 | 0 | MEASURED weights; SOURCE-DERIVED state; PROJECTED reserve and envelope |
| 32 GiB | BF16 | 0 | 0 | MEASURED weights; SOURCE-DERIVED state; PROJECTED reserve and envelope |
| 48 GiB | Selective INT4 | 0 | 2 | MEASURED weights; SOURCE-DERIVED state; PROJECTED reserve and envelope |
| 48 GiB | BF16 | 0 | 0 | MEASURED weights; SOURCE-DERIVED state; PROJECTED reserve and envelope |
| 80 GiB | Selective INT4 | 0 | 6 | MEASURED weights; SOURCE-DERIVED state; PROJECTED reserve and envelope |
| 80 GiB | BF16 | 0 | 0 | MEASURED weights; SOURCE-DERIVED state; PROJECTED reserve and envelope |
| 141 GiB | Selective INT4 | 0 | 14 | MEASURED weights; SOURCE-DERIVED state; PROJECTED reserve and envelope |
| 141 GiB | BF16 | 0 | 5 | MEASURED weights; SOURCE-DERIVED state; PROJECTED reserve and envelope |

The advertised 1,048,576-token context is reachable under the compressed policy
on 48 GiB with `max_num_seqs=2`, on 80 GiB with `max_num_seqs=6`, and on
141 GiB with `max_num_seqs=14`, all using the measured selective-INT4 weights.
It is also reachable on 141 GiB with measured BF16 weights at
`max_num_seqs=5`. It is not reachable under the expanded policy on any card in
this set.

## Modal measurement contract

`engine/modal_residency.py` accepts `expanded` or `compressed_latent`. It
allocates the selected persistent layout as separate GPU tensors and emits:

- predicted state-pool bytes from `budget.py`;
- tensor storage bytes from `numel * element_size`;
- measured `torch.cuda.max_memory_allocated` delta;
- measured `torch.cuda.max_memory_reserved` delta;
- signed allocated and reserved deltas from prediction;
- `MATCH`, `MISMATCH`, or `OOM` for every point.

The harness defaults to `compressed_latent`, matching vLLM 0.26.0. It does not
load weights. The weight byte input is MEASURED artifact tensor storage, but the
complete live-server total remains PROJECTED because weight loading, allocator
interaction, activations, and workspace are not jointly measured by this job.

## Remaining risks

- No full server boot or GPU residency run was performed in this update, so the
  complete live allocation remains unmeasured.
- vLLM's per-token MLA page formula is source-derived, but real allocation
  rounds to cache blocks and is sized by the engine's global cache allocator
  rather than this dense envelope abstraction.
- `cache_dtype=auto` resolves to the model dtype. Explicit cache quantization
  would change the byte rate and requires its own backend-compatible policy.
- The model asks users to install the latest `fla-core` rather than pinning a
  version. Future KDA or convolution allocation changes can invalidate that
  state contract.
- Tensor storage is the correct persistent weight input, but a real loader may
  create transient copies or framework metadata allocations. Those are not in
  the measured weight bytes.
- The 3 GiB activation and workspace reserve is a policy value, not a measured
  peak.
- Frontier labels follow the repo's binary GiB convention. Production gating
  must use the device's actual byte capacity.
