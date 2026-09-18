from __future__ import annotations

import math

import pytest

from engine.accuracy.metrics import (
    first_divergence_index,
    greedy_identity_rate,
    router_set_agreement,
    token_kl_divergence,
)


def test_identity_rate_uses_token_ids_not_decoded_strings() -> None:
    decoded_reference = ["same text", "unchanged"]
    decoded_candidate = ["same text", "unchanged"]
    reference_token_ids = [[11, 12], [21, 22]]
    candidate_token_ids = [[11, 99], [21, 22]]

    assert decoded_reference == decoded_candidate
    assert greedy_identity_rate(reference_token_ids, candidate_token_ids) == 0.5


def test_first_divergence_index_is_zero_based_and_handles_prefixes() -> None:
    assert first_divergence_index([10, 20, 30, 40], [10, 20, 99, 40]) == 2
    assert first_divergence_index([10, 20], [10, 20, 30]) == 2
    assert first_divergence_index([10, 20], [10, 20]) is None


def test_kl_is_reference_to_candidate_and_zero_for_identical_inputs() -> None:
    reference = [[math.log(0.9), math.log(0.1)]]
    candidate = [[math.log(0.5), math.log(0.5)]]
    expected_reference_to_candidate = 0.9 * math.log(0.9 / 0.5) + 0.1 * math.log(
        0.1 / 0.5
    )
    reverse_direction = 0.5 * math.log(0.5 / 0.9) + 0.5 * math.log(0.5 / 0.1)

    actual = token_kl_divergence(reference, candidate)

    assert actual == pytest.approx(expected_reference_to_candidate)
    assert actual != pytest.approx(reverse_direction)
    assert token_kl_divergence(reference, reference) == pytest.approx(0.0, abs=1e-12)


def test_router_agreement_compares_expert_sets_not_order() -> None:
    reference = [[[3, 8, 13], [2, 5, 7]]]
    same_sets_permuted = [[[13, 3, 8], [7, 2, 5]]]
    changed_set = [[[13, 3, 9], [7, 2, 5]]]

    assert router_set_agreement(reference, same_sets_permuted) == 1.0
    assert router_set_agreement(reference, changed_set) == 0.5
