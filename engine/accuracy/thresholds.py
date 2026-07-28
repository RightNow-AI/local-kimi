"""Predeclared acceptance policy for the Kimi Linear INT4 damage screen."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class AccuracyThresholds:
    name: str
    min_prompt_count: int
    min_greedy_identity_rate: float
    min_median_first_divergence_index: int
    max_perplexity_relative_increase: float
    min_top1_agreement: float
    max_mean_kl_nats: float
    min_router_set_agreement: float
    distribution_positions: int
    min_teacher_forced_positions: int
    teacher_forced_max_tokens: int
    reasoning: tuple[str, ...]

    def as_dict(self) -> dict:
        return asdict(self)


ACCURACY_SCREEN_V1 = AccuracyThresholds(
    name="kimi-linear-w4a16-screen-v1",
    min_prompt_count=128,
    min_greedy_identity_rate=0.90,
    min_median_first_divergence_index=32,
    max_perplexity_relative_increase=0.02,
    min_top1_agreement=0.99,
    max_mean_kl_nats=0.01,
    min_router_set_agreement=1.0,
    distribution_positions=128,
    min_teacher_forced_positions=511,
    teacher_forced_max_tokens=1024,
    reasoning=(
        "Router set agreement is exact because the router is not quantized and any routing flip amplifies weight error through a different expert path.",
        "If routed-expert capture is unavailable, the other metrics remain evidence but the screen fails because routing identity was not established.",
        "A two percent perplexity ceiling matches the conservative end of the expected four-bit quality-loss band and is only a screening gate.",
        "Top-1 and full-vocabulary KL jointly reject broad local distribution drift, while neither metric is allowed to replace later task-level evaluation.",
        "Greedy identity and first-divergence depth reject early behavioral changes that an average continuous metric can hide.",
        "A PASS here is not release certification. It only says this controlled weight-only experiment cleared its predeclared screen.",
    ),
)
