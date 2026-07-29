"""Predeclared acceptance policy for BF16 engine implementation parity."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class EngineParityThresholds:
    name: str
    prompt_count: int
    greedy_max_tokens: int
    min_first_token_top1_agreement: float
    max_mean_first_token_kl_nats: float
    max_single_prompt_first_token_kl_nats: float
    max_token_zero_divergence_rate: float
    reasoning: tuple[str, ...]

    def as_dict(self) -> dict:
        """Return the threshold in JSON-native types.

        asdict preserves the tuple, and a tuple survives json.dump as a list, so
        comparing a freshly built dict against one that has been through a file
        raised "runtime threshold differs from the predeclared source threshold"
        on a run where nothing had actually changed. The guard is right to exist
        and was firing on a serialisation artifact, so the fix is to make both
        sides the same shape rather than to loosen the comparison.
        """
        data = asdict(self)
        data["reasoning"] = list(data["reasoning"])
        return data


ENGINE_PARITY_V1 = EngineParityThresholds(
    name="kimi-linear-bf16-engine-parity-v1",
    prompt_count=32,
    greedy_max_tokens=32,
    min_first_token_top1_agreement=0.95,
    max_mean_first_token_kl_nats=0.02,
    max_single_prompt_first_token_kl_nats=0.10,
    max_token_zero_divergence_rate=0.05,
    reasoning=(
        "The threshold is declared in source before any validation job is run.",
        "Thirty-two stratified prompts cover factual, reasoning, code, instruction, and long-context behavior while keeping the direct BF16 engine run practical.",
        "A 0.95 first-token top-1 floor permits one close-call flip among 32 prompts but rejects two or more first-position disagreements.",
        "A mean KL ceiling of 0.02 nats permits ordinary BF16 reduction-order noise while rejecting broad distribution drift.",
        "A per-prompt KL ceiling of 0.10 nats prevents the mean from hiding one severe prompt-local discrepancy.",
        "Greedy exact identity is reported but is not a PASS gate because small floating-point differences can be amplified after the first close argmax decision.",
    ),
)


INTERPRETATION_RULES = (
    "Exact greedy token identity is not expected, and its absence alone is not proof of a defect.",
    "Different kernels, reduction orders, and attention backends can produce small floating-point differences that greedy decoding amplifies after a close decision.",
    "High first-token logit agreement with divergent longer generations is normal and healthy.",
    "Low first-token top-1 agreement, large first-token KL, token-zero divergence on most prompts, or a discontinuous per-layer jump would indicate a real defect.",
    "Per-layer hidden states are skipped when the reference cannot expose them without hooks or another runtime path.",
)


__all__ = ["ENGINE_PARITY_V1", "EngineParityThresholds", "INTERPRETATION_RULES"]
