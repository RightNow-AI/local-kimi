from __future__ import annotations

import math

import numpy as np
import pytest

from engine.accuracy.metrics import first_divergence_index
from engine.validate.analyze import (
    first_token_metrics,
    greedy_metrics,
    validate_matched_sides,
)


def test_greedy_identity_uses_token_ids_not_decoded_strings():
    reference = [
        {
            "prompt_id": "known",
            "category": "pin",
            "output_token_ids": [10, 20, 30],
            "output_text": "same decoded text",
        }
    ]
    candidate = [
        {
            "prompt_id": "known",
            "category": "pin",
            "output_token_ids": [10, 21, 30],
            "output_text": "same decoded text",
        }
    ]

    measured = greedy_metrics(reference, candidate)

    assert measured["summary"]["identity_rate"] == 0.0
    assert measured["raw_per_prompt"][0]["identical_token_ids"] is False
    assert measured["raw_per_prompt"][0]["first_divergence_index"] == 1
    assert measured["summary"]["identity_basis"] == (
        "generated token IDs, never decoded strings"
    )


def test_first_divergence_index_is_correct_for_known_pairs():
    assert first_divergence_index([4, 5, 6, 7], [4, 5, 9, 7]) == 2
    assert first_divergence_index([4, 5], [4, 5, 6]) == 2
    assert first_divergence_index([4, 5, 6], [4, 5, 6]) is None


def test_first_token_kl_is_zero_for_identical_distributions():
    logprobs = np.log(np.asarray([[0.75, 0.20, 0.05]], dtype=np.float64))

    measured = first_token_metrics(logprobs, logprobs, ["known"])

    assert measured["summary"]["top1_agreement"] == 1.0
    assert measured["summary"]["mean_kl_reference_to_candidate_nats"] == pytest.approx(
        0.0, abs=1e-12
    )


def test_first_token_kl_uses_reference_to_candidate_direction():
    reference = np.log(np.asarray([[0.9, 0.1]], dtype=np.float64))
    candidate = np.log(np.asarray([[0.5, 0.5]], dtype=np.float64))
    expected = 0.9 * math.log(0.9 / 0.5) + 0.1 * math.log(0.1 / 0.5)

    measured = first_token_metrics(reference, candidate, ["known"])

    assert measured["summary"]["mean_kl_reference_to_candidate_nats"] == pytest.approx(
        expected
    )
    assert measured["summary"]["kl_direction"] == (
        "KL(vLLM reference || engine.klinear candidate)"
    )


def test_comparator_refuses_different_checkpoint_directories(tmp_path):
    reference_dir = tmp_path / "reference"
    candidate_dir = tmp_path / "candidate"
    reference_dir.mkdir()
    candidate_dir.mkdir()
    shared = {
        "kind": "bf16",
        "config_sha256": "config",
        "index_sha256": "index",
        "tensor_count": 1,
        "declared_total_size_bytes": 2,
    }
    reference = {
        "side": "vllm_reference",
        "protocol_fingerprint": "same",
        "checkpoint": shared | {"directory": str(reference_dir)},
    }
    candidate = {
        "side": "klinear_candidate",
        "protocol_fingerprint": "same",
        "checkpoint": shared | {"directory": str(candidate_dir)},
    }

    with pytest.raises(ValueError, match="same checkpoint directory"):
        validate_matched_sides(reference, candidate)
