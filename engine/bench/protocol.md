# Kimi engine quality benchmark protocol

## Purpose

This protocol measures whether a disclosed candidate path reproduces HuggingFace Transformers on `moonshotai/Kimi-Linear-48B-A3B-Instruct`. Kimi-Linear is used because it exercises the same KDA linear-attention and sparse latent-MoE architecture family while fitting available benchmark hardware.

This protocol does not measure full Kimi K3 end-to-end quality. Full K3 cannot fit the available Modal configurations described for this project, so a passing Kimi-Linear result must not be presented as a full-K3 quality result.

## Download, reference, and candidate

The download action stores `moonshotai/Kimi-Linear-48B-A3B-Instruct` in the dedicated Modal Volume `kimi-linear-weights`. It must not use or mount `k3-weights`. The download manifest records the requested revision, immutable revision resolved by HuggingFace, snapshot path, Volume name, and elapsed time.

The reference is `AutoModelForCausalLM.from_pretrained` with Moonshot's model code, BF16 weights, Flash Attention 2, `trust_remote_code=True`, and `local_files_only=True` against that saved snapshot. The artifact records both the requested revision and the resolved commit hash. An audit run is invalid if the resolved revision is absent, the prompt fingerprint differs, the model ID differs, or the artifact schema differs.

The CLI GPU parameter accepts `H100:2`, `H200`, `H200:2`, `B200`, or `B200:2`. It defaults to one H200, the smallest listed single-GPU shape that can hold the roughly 96 GB BF16 checkpoint. One H100 is rejected and no 8-GPU shape is selected. Each artifact records the requested GPU shape, observed GPU count, and device names.

The candidate is supplied through `module.path:function_name`. The factory receives the same model ID, resolved revision, saved snapshot path, Volume cache path, and canonical `engine.k3ref.dequant.dequantize_mxfp4` function. It must return a runner whose `run` method accepts the saved token IDs and attention mask and returns:

- Full logits with shape `[batch, tokens, vocabulary]`.
- A routing mapping keyed by the same router module names as the reference.
- One expert-index set for every routed token at every MoE layer.

The runner must expose a coverage mapping with an exact list of router keys installed by the candidate and whether it is a full-model candidate. Routing agreement is computed only over those candidate-owned keys so unchanged HuggingFace routers cannot inflate the score.

The built-in `engine.bench.candidate:build_kimi_linear_runner` is a partial mixed-path adapter. It keeps the HuggingFace model shell and replaces every compatible KDA and latent-MoE module with `engine.k3ref`. If a full latent-MoE replacement is incompatible, it may replace only the router, but it must record that downgrade per layer. It fails if no engine component can be installed. Embeddings, dense layers, MLA attention, residual plumbing, final normalization, and the LM head remain HuggingFace and are reported as uncovered candidate scope.

There is no fallback that runs an unchanged HuggingFace model as the candidate. A missing factory, failed import, failed model load, or zero installed engine components fails closed.

## Fixed samples

The logit and routing corpus has 8 fixed prompts:

1. Arithmetic with a requested check.
2. Python code generation.
3. Structured JSON output.
4. Exact-count instruction following.
5. Short factual explanation.
6. Arabic explanation.
7. Elementary probability reasoning.
8. Ordered long-context retrieval.

For each prompt, the reference stores at most the final 8 token distributions. Therefore the largest prompt-logit comparison has `8 prompts * 8 positions = 64 token distributions`. The artifact records the actual sum because tokenization or truncation can make it smaller. Routing is captured for every input token and every MoE layer, not only the final 8 positions.

Perplexity uses 4 fixed, self-contained held-out texts covering science, service engineering, deductive reasoning, and Arabic. Each text is truncated to at most 64 tokens. A text with `T` tokens contributes `T - 1` next-token predictions, and the artifact records `sum(T - 1)` across all 4 texts. Covered routing decisions from the held-out texts are included in the aggregate routing metric.

The samples are deterministic regression fixtures. They are not a random sample from customer traffic or a public benchmark distribution.

## Metrics and arithmetic

All logit metrics compare identical token IDs. The causal shift is applied before perplexity.

### Token KL divergence

For token `t`, let `p_t` be the reference softmax and `q_t` the candidate softmax:

`KL_t = sum_v p_t(v) * (log p_t(v) - log q_t(v))`

The reported value is `sum_t KL_t / N`, where `N` is the number of scored token positions. It is measured in nats. Identical distributions produce zero. A changed distribution produces a positive value unless the logits differ only by a per-token additive constant, which leaves the distribution unchanged.

### Top-1 agreement

For each scored position, compare the reference and candidate argmax token:

`matching argmax positions / N`

### Routing agreement

Moonshot's router returns the selected expert indices before expert execution. The harness records that output directly. Expert order is ignored, but the selected set must be identical. Each token at each candidate-owned MoE layer contributes one token-layer routing decision:

`identical selected expert sets / all covered token-layer routing decisions`

The expected set size is read from the reference checkpoint and must be consistent across its routed layers. Missing layers, extra layers, different token counts, duplicate experts, or a different set size are errors rather than partial scores. Routing agreement is a first-class gate because expert selection is discrete and one changed expert sends work through different parameters. Full K3 selects 16 experts per token, but this Kimi-Linear run proves top-16 behavior only if its recorded checkpoint configuration also selects 16.

### Perplexity

For each held-out target token, compute its negative log likelihood under the corresponding preceding-token logits. With `M` non-ignored targets:

`perplexity = exp(sum negative_log_likelihood / M)`

The report includes reference perplexity, candidate perplexity, and relative delta:

`(candidate perplexity - reference perplexity) / reference perplexity`

## Acceptance policy

The defaults are policy limits, not measurements and not entries in the loss ledger:

- Mean token KL must be at most `0.0001` nats.
- Top-1 agreement must be at least `0.999`.
- Routing agreement must equal `1.0`.
- Absolute perplexity relative delta must be at most `0.001`.

The report records configured limits and arithmetic for every measured result. Changing a limit changes release policy and must not be described as changing measured quality.

## Loss ledger

Every quality-affecting transformation must have a ledger entry. The required inventory is skeleton quantization, expert requantization below the published 4.25 bits per weight, reduced top-k, kernel or engine reference-path numerics, and speculative decoding.

The completed comparison writes MEASURED entries for mean token KL, top-1 agreement, routing agreement, and perplexity relative delta. These metrics have different units and are not added together. Each numeric entry records what changed, its metric, its reference, and the arithmetic that produced the number. A missing measurement renders as `UNMEASURED`, never zero. A total for a metric is unavailable while any contributing entry for that metric is unmeasured. A total containing any modelled input remains labelled `MODELLED`.

The published K3 expert unpack is not a lossy transformation. The candidate factory receives the canonical `engine.k3ref.dequant.dequantize_mxfp4` function. This Kimi-Linear BF16 candidate does not invoke that decoder and records `mxfp4_decoder_used: false`. Any later requantization below published expert precision is a separate ledger entry and starts unmeasured.

## Statistical claim

A run justifies only a deterministic regression claim on the recorded checkpoint and fixed corpus: the candidate did or did not meet configured agreement limits for these inputs. The protocol does not justify a population accuracy estimate, confidence interval, task-level capability claim, or customer-workload quality claim. Fixed observations are correlated across tokens and layers, so treating them as independent samples would overstate evidence.

Repeated runs on the same hardware can reveal nondeterminism, but this version does not convert repeated runs into a confidence interval. Any extension must record the number of runs, aggregation rule, and observed dispersion.

## What a passing run can prove

- The candidate accepts the same token IDs as the HuggingFace reference.
- Its mixed-path Kimi-Linear logits meet configured KL, top-1, and perplexity limits on the fixed corpus.
- Its selected expert set agrees at every candidate-owned measured token-layer decision when the routing gate passes.
- Exact `engine.k3ref` components and layers listed in the coverage record can participate in an end-to-end Kimi-Linear forward within measured limits.

## What a passing run cannot prove

- Full Kimi K3 end-to-end quality or multimodal behavior.
- A full custom Kimi-Linear engine, because the coverage record lists retained HuggingFace components.
- Correctness of every K3 layer, tensor mapping, kernel, or distributed execution path.
- Accuracy on public benchmarks or customer workloads.
- Performance, throughput, latency, cost, memory safety, or fault tolerance.
- Quality of an unmeasured transformation in the loss ledger.
- Speculative decoding quality unless that path is enabled, measured, and recorded separately.

The separately proven K3 MXFP4 decoder establishes only its own codec behavior. It does not bridge these uncovered end-to-end claims.
