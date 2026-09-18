# Kimi Linear BF16 engine validation

Status: **NOT RUN**

No correctness measurement is claimed in this repository file. The Modal job has been implemented but was not run in this lane, as required by the brief.

## Controlled experiment

- Reference: vLLM 0.26.0 on `Kimi-Linear-48B-A3B-Instruct` in BF16.
- Candidate: direct `engine.klinear` on the same checkpoint directory in BF16.
- Hardware: both child processes run sequentially on the same Modal H200 allocation.
- Inputs: one protocol file supplies identical Moonshot chat-template token IDs to both implementations.
- Greedy comparison: 32 stratified prompts, temperature 0, exactly 32 generated tokens, token IDs only.
- Distribution comparison: exact full-vocabulary next-token log probabilities at the first generated position for every prompt.
- KL direction: `KL(vLLM reference || engine.klinear candidate)`.

Direct `engine.klinear` was selected instead of the serving adapter because the adapter delegates generation to the same engine functions but does not expose logits. The candidate runner calls the existing `prefill`, `sample_logits`, and `decode` functions used by `generate_tokens`, so it captures first-token logits without changing the model implementation being measured.

## Predeclared PASS threshold

| Check | Requirement |
| --- | ---: |
| Prompt count | exactly 32 |
| First-token top-1 agreement | at least 0.95 |
| Mean first-token KL | at most 0.02 nats |
| Maximum single-prompt first-token KL | at most 0.10 nats |
| Token-zero greedy divergence rate | at most 0.05 |

The top-1 floor permits one close-call flip among 32 prompts but rejects two or more first-position disagreements. The mean KL ceiling permits ordinary BF16 kernel and reduction-order noise while rejecting broad distribution drift. The maximum KL ceiling prevents the mean from hiding one severe prompt-local discrepancy. Exact long-generation identity is reported but is deliberately not a PASS gate.

## Interpretation rules

- Exact greedy token identity is not expected, and its absence alone is not proof of a defect.
- Different kernels, reduction orders, and attention backends can create small floating-point differences that greedy decoding amplifies after a close decision.
- High first-token logit agreement with divergent longer generations is normal and healthy.
- Low first-token top-1 agreement, large first-token KL, or token-zero divergence on most prompts would indicate a real defect.
- Per-layer hidden states are skipped because vLLM 0.26.0 does not expose matching intermediate states through this unmodified runtime path. Adding hooks or a model fork would perturb the comparison.

## Run command

```text
modal run engine/modal_validate.py
```

Each run writes `protocol.json`, both raw side records, both full-vocabulary log-probability arrays, `evidence.json`, and a measured `RESULTS.md` under the `kimi-linear-validation` Modal volume.
