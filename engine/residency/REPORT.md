# Kimi-Linear live residency budget

## Verdict

There are two real MLA cache policies for this model:

| Implementation | Cache policy | BF16 bytes per token across 7 MLA layers | Status |
|---|---|---:|---|
| Hugging Face reference | Expanded per-head keys and values | 143,360 | SOURCE-DERIVED, NOT MEASURED |
| vLLM 0.26.0 `FLASH_ATTN_MLA` | Compressed latent plus rotary key | 8,064 | SOURCE-DERIVED, NOT MEASURED |

vLLM stores one 576-element record per token per MLA layer: 512 compressed latent
elements plus 64 rotary-key elements. At BF16 this is 1,152 bytes per layer and
8,064 bytes across the model's seven MLA layers.

The compressed-latent cache is table stakes rather than an advantage over vLLM.
Our engine must preserve the 512 plus 64 representation in persistent cache and run
prefill and decode directly from it. Persistently expanding to per-head keys and values
would make the engine materially less memory-efficient than the buyer's existing vLLM
option.

Under INT4 weights and the explicit 3 GiB operational reserve:

| Envelope | Expanded total | Compressed total | 32 GiB result | Status |
|---|---:|---:|---|---|
| `1 x 32K` | 32,524,095,936 bytes | 28,090,716,608 bytes | Both fit | PROJECTED, NOT MEASURED |
| `16 x 32K` | 103,647,039,936 bytes | 32,712,970,688 bytes | Compressed only | PROJECTED, NOT MEASURED |
| `21 x 32K` | 127,354,687,936 bytes | 34,253,722,048 bytes | Compressed only, near limit | PROJECTED, NOT MEASURED |
| `1 x 1M` | 178,150,330,816 bytes | 36,282,192,320 bytes | Neither fits | PROJECTED, NOT MEASURED |

The exact compressed-policy 32 GiB capacity at 32K is 21 sequences under this
headroom policy. A single 1M sequence still misses 32 GiB, but reaches 48 GiB.

## Evidence status

- `SOURCE-DERIVED` means a tensor shape, dtype, or page formula was read from pinned source.
- `PROJECTED, NOT MEASURED` means exact arithmetic or an explicit policy reserve was applied.
- `MEASURED` is reserved for output from `engine/modal_residency.py`.
- No GPU, vLLM server, test runner, or Modal job was run in this lane.

## Which policy belongs to which implementation

| Policy | Persistent record per token per MLA layer | Implementation | Status |
|---|---|---|---|
| `expanded` | 32 keys of width 192 plus 32 values of width 128 | Hugging Face `modeling_kimi.py` | SOURCE-DERIVED, NOT MEASURED |
| `compressed_latent` | One width-576 record containing width-512 latent and width-64 rotary key | vLLM 0.26.0 MLA cache | SOURCE-DERIVED, NOT MEASURED |

The two policies are exposed explicitly through `MLACachePolicy` in `budget.py`.
The default remains `expanded` for backward compatibility with the original conservative
report. Product and vLLM comparisons must select `compressed_latent` explicitly.

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

- `fla/ops/common/chunk_delta_h.py` lines 690 to 707 allocate the FP32 recurrent state.
- `fla/ops/kda/fused_recurrent.py` lines 271 to 278 allocate the recurrent decode state.
- `fla/modules/conv/short_conv.py` lines 211 to 217 allocate full-width conv caches.

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

1. `kimi_linear.py` lines 217 to 220 build a width-576 K/V A projection, and lines
   264 to 274 pass `kv_lora_rank` and `qk_rope_head_dim` into the MLA wrapper.
2. `layers/mla.py` lines 154 to 157 split the projected record into width 512 and 64,
   then lines 175 to 179 pass both compressed parts into `MLAAttention`.
3. `mla_attention.py` lines 388 to 392 set cache head size to 512 plus 64 and set one
   KV head. Lines 1075 to 1085 create `MLAAttentionSpec` with that head size.
4. `kv_cache_interface.py` lines 398 to 415 compute MLA page bytes as block size,
   one KV head, head size, and dtype size. Unlike ordinary attention, there is no
   separate key-plus-value factor of two.
5. `flashattn_mla.py` lines 43 to 65 identify the selected backend as
   `FLASH_ATTN_MLA`. Lines 338 to 339 split its live cache at `kv_lora_rank`.
6. `torch_utils.py` lines 395 to 401 resolve `cache_dtype=auto` to the model dtype,
   which is BF16 for this model.

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

| Component | Exact formula | Rate | Status |
|---|---|---:|---|
| INT4 weights | pinned profile | 24,561,340,864 bytes | PROJECTED, NOT MEASURED |
| BF16 weights | 49,122,681,728 parameters times 2 | 98,245,363,456 bytes | PROJECTED, NOT MEASURED |
| KDA recurrent pool | `20 * seqs * 32 * 128 * 128 * 4` | 41,943,040 bytes per sequence | SOURCE-DERIVED, NOT MEASURED |
| Short conv pool | `20 * seqs * 3 * 4096 * 4 * 2` | 1,966,080 bytes per sequence | SOURCE-DERIVED, NOT MEASURED |
| Expanded MLA | `7 * seqs * tokens * 32 * (192 + 128) * 2` | 143,360 bytes per token per sequence | SOURCE-DERIVED, NOT MEASURED |
| Compressed MLA | `7 * seqs * tokens * (512 + 64) * 2` | 8,064 bytes per token per sequence | SOURCE-DERIVED, NOT MEASURED |
| Activation reserve | explicit report policy | 2,147,483,648 bytes | PROJECTED, NOT MEASURED |
| Workspace reserve | explicit report policy | 1,073,741,824 bytes | PROJECTED, NOT MEASURED |

The expanded policy costs exactly 17.7778 times the compressed policy per token.
The 3 GiB operational reserve is a visible policy input, not a measurement.

## Projected frontier side by side

Policy for every row:

- Card labels are binary GiB capacities.
- Operational reserve is 2 GiB activation plus 1 GiB workspace.
- Sequence candidates are `1, 2, 4, 8, 16, 32, 64, 128, 256`.
- Each `S:L` pair means `max_num_seqs=S`, maximum `max_model_len=L`.
- Dominated points are omitted.
- Model length is capped at 1,048,576.

### 24 GiB

| Weights | Expanded frontier | Compressed frontier | Status |
|---|---|---|---|
| INT4 | none | none | PROJECTED, NOT MEASURED |
| BF16 | none | none | PROJECTED, NOT MEASURED |

INT4 weights plus the operational reserve already require 25.875 GiB before state.

### 32 GiB

| Weights | Expanded frontier | Compressed frontier | Status |
|---|---|---|---|
| INT4 | `1:45,572; 2:22,633; 4:11,163; 8:5,428; 16:2,561; 32:1,127; 64:410; 128:52` | `1:810,176; 2:402,365; 4:198,460; 8:96,507; 16:45,531; 32:20,043; 64:7,299; 128:926` | PROJECTED, NOT MEASURED |
| BF16 | none | none | PROJECTED, NOT MEASURED |

### 48 GiB

| Weights | Expanded frontier | Compressed frontier | Status |
|---|---|---|---|
| INT4 | `1:165,409; 2:82,551; 4:41,122; 8:20,408; 16:10,050; 32:4,872; 64:2,283; 128:988; 256:341` | `2:1,048,576; 4:731,070; 8:362,812; 16:178,683; 32:86,619; 64:40,587; 128:17,571; 256:6,062` | PROJECTED, NOT MEASURED |
| BF16 | none | none | PROJECTED, NOT MEASURED |

### 80 GiB

| Weights | Expanded frontier | Compressed frontier | Status |
|---|---|---|---|
| INT4 | `1:405,084; 2:202,388; 4:101,041; 8:50,367; 16:25,030; 32:12,362; 64:6,027; 128:2,860; 256:1,277` | `4:1,048,576; 8:895,422; 16:444,988; 32:219,771; 64:107,163; 128:50,859; 256:22,707` | PROJECTED, NOT MEASURED |
| BF16 | none | none | PROJECTED, NOT MEASURED |

### 141 GiB

| Weights | Expanded frontier | Compressed frontier | Status |
|---|---|---|---|
| INT4 | `1:861,963; 2:430,828; 4:215,261; 8:107,477; 16:53,585; 32:26,639; 64:13,166; 128:6,430; 256:3,061` | `8:1,048,576; 16:952,632; 32:473,593; 64:234,074; 128:114,314; 256:54,434` | PROJECTED, NOT MEASURED |
| BF16 | `1:347,984; 2:173,839; 4:86,766; 8:43,230; 16:21,461; 32:10,577; 64:5,135; 128:2,414; 256:1,054` | `4:1,048,576; 8:768,535; 16:381,545; 32:188,049; 64:91,302; 128:42,928; 256:18,741` | PROJECTED, NOT MEASURED |

## Advertised 1M context

This table uses exact integer sequence counts rather than the powers-of-two frontier grid.

| Capacity | Weights | Expanded max sequences at 1M | Compressed max sequences at 1M | Status |
|---:|---|---:|---:|---|
| 24 GiB | INT4 | 0 | 0 | PROJECTED, NOT MEASURED |
| 24 GiB | BF16 | 0 | 0 | PROJECTED, NOT MEASURED |
| 32 GiB | INT4 | 0 | 0 | PROJECTED, NOT MEASURED |
| 32 GiB | BF16 | 0 | 0 | PROJECTED, NOT MEASURED |
| 48 GiB | INT4 | 0 | 2 | PROJECTED, NOT MEASURED |
| 48 GiB | BF16 | 0 | 0 | PROJECTED, NOT MEASURED |
| 80 GiB | INT4 | 0 | 6 | PROJECTED, NOT MEASURED |
| 80 GiB | BF16 | 0 | 0 | PROJECTED, NOT MEASURED |
| 141 GiB | INT4 | 0 | 14 | PROJECTED, NOT MEASURED |
| 141 GiB | BF16 | 0 | 5 | PROJECTED, NOT MEASURED |

The 1M context is therefore reachable under the compressed policy on 48 GiB and larger
INT4 configurations in this set, and on 141 GiB with BF16 weights. It is not reachable
under the expanded policy on any requested card.

## Modal measurement contract

`engine/modal_residency.py` now accepts `expanded` or `compressed_latent`. It allocates
the selected persistent layout as separate GPU tensors and emits:

- predicted state-pool bytes from `budget.py`;
- tensor storage bytes from `numel * element_size`;
- measured `torch.cuda.max_memory_allocated` delta;
- measured `torch.cuda.max_memory_reserved` delta;
- signed allocated and reserved deltas from prediction;
- `MATCH`, `MISMATCH`, or `OOM` for every point.

The harness defaults to `compressed_latent`, matching vLLM 0.26.0. It does not load
weights, so weight bytes and operational reserves remain projected.

## Remaining risks

- No GPU job was run, so allocator behavior and operational headroom remain unmeasured.
- vLLM's per-token MLA page formula is source-derived, but real allocation rounds to
  cache blocks and is sized by the engine's global cache allocator rather than this
  dense envelope abstraction.
- `cache_dtype=auto` resolves to the model dtype. Explicit cache quantization would
  change the byte rate and requires its own backend-compatible policy.
- The model asks users to install the latest `fla-core` rather than pinning a version.
  Future KDA or convolution allocation changes can invalidate that state contract.
- The INT4 profile is exact four-bit parameter arithmetic. A real artifact can add
  scales, zero points, alignment, and metadata.
- The activation and workspace reserves are policy values, not measured peaks.
- Frontier labels follow the repo's binary GiB convention. Production gating must use
  the device's actual byte capacity.
