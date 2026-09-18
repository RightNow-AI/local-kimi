from __future__ import annotations

import numpy as np
import pytest

from engine.router_analysis.metrics import (
    compare_routing_runs,
    probability_mass_overlap,
    rank_position_histogram,
    top1_expert_agreement,
)
from engine.router_analysis.records import (
    LayerRoutingTrace,
    PromptRoutingTrace,
    RoutingRun,
)


def test_probability_mass_overlap_is_one_for_identical_selection() -> None:
    overlap = probability_mass_overlap(
        [7, 3, 11],
        [1.223, 0.815, 0.408],
        [7, 3, 11],
        [1.223, 0.815, 0.408],
    )

    assert overlap == pytest.approx(1.0)


def test_probability_mass_overlap_uses_weighted_shared_mass() -> None:
    overlap = probability_mass_overlap(
        [10, 20],
        [0.75, 0.25],
        [10, 30],
        [0.50, 0.50],
    )

    # The only shared expert is 10. Its overlapping probability mass is
    # min(0.75, 0.50), not one shared ID divided by two selected IDs.
    assert overlap == pytest.approx(0.50)


def test_top1_agreement_ignores_everything_below_the_highest_weight() -> None:
    agreement = top1_expert_agreement(
        [[2, 1, 4], [8, 7, 6]],
        [[0.10, 0.80, 0.10], [0.70, 0.20, 0.10]],
        [[1, 3, 5], [9, 7, 8]],
        [[0.75, 0.15, 0.10], [0.20, 0.10, 0.70]],
    )

    assert agreement == pytest.approx(1.0)


def test_rank_histogram_uses_weight_rank_not_unsorted_topk_position() -> None:
    histogram = rank_position_histogram(
        [[10, 11, 12, 13, 14, 15, 16, 17]],
        [[0.80, 0.02, 0.60, 0.50, 0.40, 0.30, 0.20, 0.01]],
        [[10, 18, 12, 13, 14, 15, 16, 17]],
        [[0.80, 0.02, 0.60, 0.50, 0.40, 0.30, 0.20, 0.01]],
    )

    # Experts 11 and 18 are each seventh by weight even though they occupy the
    # second slot returned by an unsorted top-k operation.
    assert histogram == {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 2, 8: 0}


def test_comparison_refuses_different_prompt_sets() -> None:
    reference = _run(prompt_set_sha256="prompt-set-a")
    candidate = _run(prompt_set_sha256="prompt-set-b")

    with pytest.raises(ValueError, match="different prompt sets"):
        compare_routing_runs(reference, candidate)


def _run(*, prompt_set_sha256: str) -> RoutingRun:
    layer = LayerRoutingTrace(
        layer_index=1,
        expert_ids=np.asarray([[1, 2]], dtype=np.int16),
        expert_weights=np.asarray([[1.8, 0.646]], dtype=np.float32),
    )
    prompt = PromptRoutingTrace(
        prompt_id="factual-000",
        token_ids=(101,),
        layers=(layer,),
    )
    return RoutingRun(
        checkpoint="checkpoint",
        prompt_set_sha256=prompt_set_sha256,
        prompts=(prompt,),
        router_config={
            "num_experts": 256,
            "num_experts_per_token": 2,
            "activation": "sigmoid",
            "renormalize": True,
            "routed_scaling_factor": 2.446,
            "use_grouped_topk": True,
            "num_expert_group": 1,
            "topk_group": 1,
        },
    )
