from __future__ import annotations

import math

from engine.batching.union_model import ExpertUnionModel, HardwareConfig
from engine.scheduling.compose import PendingToken, compare_composers, compose_batch
from engine.scheduling.metrics import (
    compare_throughput,
    deferred_rounds,
    measure_fairness,
    measure_union,
    measure_union_reduction,
)


def _token(token_id: str, *layers: set[int]) -> PendingToken:
    return PendingToken(
        token_id=token_id,
        layer_experts=tuple(frozenset(layer) for layer in layers),
    )


def test_identical_routes_have_union_sixteen_at_every_batch_size() -> None:
    route = set(range(16))
    pool = [_token(f"token-{index}", route) for index in range(8)]

    for batch_size in (1, 2, 4, 8):
        composed = compose_batch(pool, batch_size, strategy="greedy", seed=7)

        assert measure_union(composed).total_union == 16


def test_fully_disjoint_routes_report_zero_reduction() -> None:
    pool = [
        _token(f"token-{index}", set(range(index * 16, (index + 1) * 16)))
        for index in range(8)
    ]
    decision = compare_composers(pool, 4, seed=19)
    reduction = measure_union_reduction(
        decision.selected,
        decision.random_baseline,
    )

    assert decision.selected_total_union == 4 * 16
    assert decision.random_total_union == 4 * 16
    assert reduction.experts_saved_across_layers == 0
    assert reduction.reduction_fraction == 0.0
    assert reduction.bytes_saved_per_token == 0.0


def test_guarded_greedy_never_exceeds_random_on_the_same_pool() -> None:
    pool = [
        _token("a", {0, 1, 2, 3}, {10, 11, 12, 13}),
        _token("b", {0, 1, 4, 5}, {10, 11, 14, 15}),
        _token("c", {6, 7, 8, 9}, {16, 17, 18, 19}),
        _token("d", {0, 2, 6, 8}, {10, 12, 16, 18}),
        _token("e", {1, 3, 7, 9}, {11, 13, 17, 19}),
        _token("f", {20, 21, 22, 23}, {30, 31, 32, 33}),
    ]

    for seed in range(20):
        decision = compare_composers(pool, 4, seed=seed)

        assert decision.selected_total_union <= decision.random_total_union
        assert measure_union(decision.selected).total_union <= measure_union(
            decision.random_baseline
        ).total_union


def test_composer_selects_unique_tokens_and_respects_batch_limit() -> None:
    pool = [_token(str(index), set(range(index, index + 16))) for index in range(12)]

    for strategy in ("greedy", "random"):
        selected = compose_batch(pool, 5, strategy=strategy, seed=23)
        selected_ids = [token.token_id for token in selected]

        assert len(selected) == 5
        assert len(selected_ids) == len(set(selected_ids))
        assert all(token in pool for token in selected)


def test_greedy_objective_sums_layers_instead_of_optimizing_one_layer() -> None:
    pool = [
        _token("anchor", {0}, {10, 11, 12, 13}),
        _token("layer-one-overlap", {0}, {20, 21, 22, 23}),
        _token("lower-total-cost", {1}, {10, 11, 12, 13}),
    ]

    selected = compose_batch(pool, 2, strategy="greedy", seed=0)

    assert [token.token_id for token in selected] == ["anchor", "lower-total-cost"]
    assert measure_union(selected).per_layer_union == (2, 4)


def test_fairness_reports_exact_deferred_rounds_including_outstanding_tokens() -> None:
    assert deferred_rounds(arrival_round=4, selected_round=9) == 5

    fairness = measure_fairness(
        [0, 2, 5],
        outstanding_deferred_rounds=[3],
        served_wait_seconds=[0.0, 0.2, 0.5],
        outstanding_wait_seconds=[0.3],
    )

    assert fairness.average_deferred_rounds == 2.5
    assert fairness.max_served_deferred_rounds == 5
    assert fairness.max_outstanding_deferred_rounds == 3
    assert fairness.worst_case_deferred_rounds == 5
    assert fairness.worst_case_wait_seconds == 0.5


def test_byte_savings_count_expert_layer_pairs_once() -> None:
    shared = set(range(16))
    composed = [_token("a", shared), _token("b", shared)]
    random = [_token("c", shared), _token("d", set(range(16, 32)))]

    reduction = measure_union_reduction(composed, random, expert_bytes=100)

    assert reduction.composed.total_union == 16
    assert reduction.random.total_union == 32
    assert reduction.experts_saved_across_layers == 16
    assert reduction.bytes_saved_per_batch == 16 * 100
    assert reduction.bytes_saved_per_token == 16 * 100 / 2


def test_observed_union_is_fed_back_into_existing_throughput_terms() -> None:
    model = ExpertUnionModel(
        total_experts=8,
        experts_per_token=2,
        moe_layers=3,
        expert_bytes=5,
    )
    hardware = HardwareConfig(
        key="test",
        label="test",
        routed_bandwidth_gb_s=1.0,
        dense_bytes=1_000_000_000,
        per_token_compute_seconds=0.1,
        old_batch1_tokens_per_second=1.0,
        calibration="synthetic",
    )

    feedback = compare_throughput(
        hardware,
        model,
        batch_size=2,
        random_mean_union_per_layer=4.0,
        composed_mean_union_per_layer=2.0,
    )

    expected_seconds_saved = (4.0 - 2.0) * 5 * 3 / 1e9
    assert math.isclose(
        feedback.seconds_saved_per_batch,
        expected_seconds_saved,
        rel_tol=0.0,
        abs_tol=1e-15,
    )
    assert feedback.composed.aggregate_tokens_per_second > (
        feedback.random.aggregate_tokens_per_second
    )
    assert feedback.composed.dense_seconds_per_batch == (
        feedback.random.dense_seconds_per_batch
    )
    assert feedback.composed.token_compute_seconds_per_batch == (
        feedback.random.token_compute_seconds_per_batch
    )
