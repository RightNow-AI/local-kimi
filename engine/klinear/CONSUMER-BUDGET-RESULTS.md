# Kimi-Linear-48B serves inside a 32 GiB budget, measured

This is the result the product rests on, and it is a CAPABILITY result rather
than a speed one.

## The claim

Stock vLLM 0.26.0 will not serve this model below BF16. Measured on an H200 in
this project: the as-shipped checkpoint loads and generates, while the
bitsandbytes 4-bit path is refused at load because `KimiLinearForCausalLM`
declares no `packed_modules_mapping`. So on vLLM the weights floor is
98,245,528,576 bytes, and **a 32 GB card cannot run this model at any speed**.

Our selective INT4 artifact is 28,803,304,448 bytes and this engine runs it
inside 32 GiB with room to spare. The difference is binary, not incremental.

## How the budget was enforced

A peak-memory reading on a large card would prove nothing about a small one.
`torch.cuda.set_per_process_memory_fraction` capped the process at exactly 32
GiB of the device, computed from its real total. Every allocation past that
raises rather than spilling, so a model that did not fit would have produced an
`OutOfMemoryError`, not a flattering number.

The card is an L40S because it is the cheapest Modal GPU above 32 GiB. The extra
capacity is deliberately made unusable by the cap.

## Result

| | bytes | GiB |
|---|---:|---:|
| Budget enforced | 34,359,738,368 | 32.00 |
| Resident weights | 28,803,304,448 | 26.83 |
| Peak reserved after load | 29,972,496,384 | 27.91 |
| **Peak reserved after generation** | **30,515,658,752** | **28.42** |
| **Headroom inside the budget** | **3,844,079,616** | **3.58** |

`fits: true`. All 27 layers, all 256 experts per MoE layer, packed INT4 weights
fed to the fused Triton GEMM with nothing expanded to BF16 in memory.

## What it generated

Three prompts, greedy, 48 new tokens each.

> **The main benefit of running a large model locally is** data privacy and
> control, you keep sensitive data on your own hardware, avoid sending it over
> the internet, and reduce reliance on external services.

> **Write a Python function that reverses a linked list.**
> Below is a concise, production-ready implementation of a singly-linked list
> and a function that reverses it in-place. The key idea: 1. Keep three
> pointers: prev, curr, and next_node.

> **Explain in two sentences why memory bandwidth limits decoding speed.**
> Decoding video frames requires repeatedly reading compressed data from memory
> and writing uncompressed frames back out. If the memory bandwidth is
> insufficient to keep up with this read and write demand, the decoder becomes
> bottlenecked and cannot process frames faster than the bandwidth allows.

The first two are coherent, correctly formatted and on topic, with the code
answer laying out the right algorithm. The third answered about VIDEO decoding
rather than token decoding, which is a defensible reading of an ambiguous
prompt but is recorded here rather than quietly dropped.

## What this does NOT establish

- **Not a throughput claim.** Measured tokens per second were 0.67, 2.53 and
  3.39 on the three prompts. That is a single-stream reference implementation
  under a hard memory cap, with no batching, no CUDA graphs and no kernel
  tuning. It must not be quoted as serving throughput, and no throughput
  comparison against vLLM has been measured or is claimed.
- **Not a measurement on consumer silicon.** This is a demonstration under a
  hard budget on a datacenter card. It establishes that the working set fits the
  budget, not that a specific consumer GPU reaches any particular speed.
- **Not a quality claim.** `engine/accuracy/RESULTS.md` records that the INT4
  artifact differs from the original: perplexity rises 0.81 percent, but router
  set agreement is 34.3 percent, so outputs differ from the reference model.
  That is disclosed, not hidden.
- **Not a serving envelope.** This is one sequence at short context.
  `engine/residency/REPORT.md` carries the envelope as a function of
  `max_num_seqs` and `max_model_len`.

## Reproducing

```bash
modal run engine/modal_consumer_card.py
modal run engine/modal_consumer_card.py --budget-gib 24
```

The second form is worth running: it should FAIL, and a budget that cannot hold
the artifact is as informative as one that can.
