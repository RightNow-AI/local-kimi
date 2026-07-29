# Decode throughput, measured, with output equivalence proved

Produced by `engine/modal_decode_bench.py` on one NVIDIA L40S, loading the
selective INT4 artifact with the process hard-capped at 32 GiB. 8-token prompt,
64 generated tokens, greedy, 3 repeats.

## Result

| path | tok/s | token ids identical to reference |
|---|---:|:---:|
| `growing_eager` (pre-optimisation reference) | 9.02 | reference |
| `preallocated_eager` | 24.07 | **yes** |
| streaming, end to end incl. prefill and graph setup | 32.33 | **yes** |
| **CUDA graph, median of 3** | **35.71** | **yes** |
| CUDA graph, best of 3 | 35.72 | yes |

**3.96x** over this benchmark's own baseline, with byte-identical output.

Equivalence is the gate, not a footnote. All three optimised paths were compared
against the growing-state reference with `torch.equal` on the generated token
ids, and all three matched. A faster engine that answers differently is a
different product, so a speedup without this column means nothing.

## What made the difference

In the order the analysis pointed at them:

1. **MLA was reprojecting the entire KV prefix on every token**, which is O(n^2)
   work where decode should be O(n). It now caches projected keys and values and
   projects only the current latent token.
2. **Fixed-capacity state buffers mutated in place**, with device-resident
   positions, so no tensor grows per token.
3. **Grouped expert execution.** The MoE was scanning all 256 experts with
   dynamic `torch.where` and then launching 27 separate GEMMs per layer. Experts
   now live in contiguous per-layer banks and the path uses fixed nine-slot
   routing with three grouped launches for w1, w3 and w2. No expert-id `.item()`,
   `.cpu()` or `.tolist()` remains in decode, so the loop no longer drains the
   GPU every token.
4. **CUDA graph capture** on the fixed decode step, replayed thereafter.

## The cost, which is not free

| | bytes | GiB |
|---|---:|---:|
| Budget enforced | 34,359,738,368 | 32.00 |
| Peak reserved, optimised | 34,349,252,608 | 31.99 |
| **Slack** | **10,485,760** | **0.01** |

Before this work the same 32 GiB budget had **3.58 GiB** of headroom. The
preallocated buffers, the contiguous expert banks and the graph capture bought
speed with memory, and 32 GiB is now tight rather than comfortable. A card with
slightly different driver or allocator overhead may not fit. Treat 32 GiB as the
floor for this configuration, not a comfortable target, and reduce
`max_num_seqs` or context before assuming it will run.

## What this does NOT establish

- **Not comparable to the earlier 0.67 to 3.39 figure.** That came from
  `engine/modal_consumer_card.py`, a different harness with longer prompts and
  per-prompt overhead. The honest speedup is 3.96x against the baseline measured
  in the same run, not a cross-harness ratio.
- **Not a comparison against llama.cpp.** llama.cpp PR 17592 reports roughly 32
  tok/s on an RTX 3090; this is an L40S. Different card, different harness,
  different prompt. The numbers are not a like-for-like comparison and must not
  be presented as one.
- **Not a serving throughput claim.** Single stream, one prompt, no concurrency,
  no continuous batching.
- **Not a quality claim.** `engine/accuracy/RESULTS.md` records what INT4 costs.

## Reproducing

```bash
modal run engine/modal_decode_bench.py
```
