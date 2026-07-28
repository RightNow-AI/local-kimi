import math

import pytest

from engine.bench.metrics import (
    perplexity,
    routing_agreement,
    token_kl_divergence,
    top1_agreement,
)


def test_kl_is_zero_for_identical_distributions_and_positive_otherwise():
    reference = [[2.0, 0.0, -1.0], [0.0, 1.0, 0.0]]
    assert token_kl_divergence(reference, reference) == 0.0
    assert token_kl_divergence(reference, [[0.0, 2.0, -1.0], [1.0, 0.0, 0.0]]) > 0.0


def test_top1_agreement_counts_token_positions():
    reference = [[3.0, 1.0], [0.0, 2.0]]
    candidate = [[2.0, 1.0], [4.0, 1.0]]
    assert top1_agreement(reference, candidate) == 0.5


def test_routing_agreement_is_set_based_and_degrades_per_token():
    reference = [[0, 1, 2], [3, 4, 5]]
    reordered = [[2, 0, 1], [5, 3, 4]]
    one_expert_changed = [[2, 0, 1], [5, 3, 6]]
    assert routing_agreement(reference, reordered, expected_experts_per_token=3) == 1.0
    assert routing_agreement(reference, one_expert_changed, expected_experts_per_token=3) == 0.5


def test_perplexity_matches_hand_computed_value():
    # P(targets) is [3/4, 1/2]. PPL = exp(-(ln(3/4)+ln(1/2))/2) = sqrt(8/3).
    logits = [[math.log(3.0), 0.0], [0.0, 0.0]]
    assert perplexity(logits, [0, 1]) == pytest.approx(math.sqrt(8.0 / 3.0))
