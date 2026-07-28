from __future__ import annotations

from engine.batching.union_model import ExpertUnionModel, HardwareConfig
from engine.scheduling.simulate import simulate_comparison


def test_pool_equal_to_batch_has_no_composition_choice() -> None:
    model = ExpertUnionModel(
        total_experts=32,
        experts_per_token=4,
        moe_layers=3,
        expert_bytes=100,
    )
    hardware = HardwareConfig(
        key="test",
        label="test",
        routed_bandwidth_gb_s=1.0,
        dense_bytes=1_000_000,
        per_token_compute_seconds=0.0,
        old_batch1_tokens_per_second=1.0,
        calibration="synthetic",
    )

    result = simulate_comparison(
        model=model,
        hardware=hardware,
        prior=None,
        routing_prior_name="uniform",
        arrival_process="saturated",
        pool_size=4,
        max_batch_size=4,
        rounds=5,
        seed=17,
    )

    assert result.random.tokens_served == 20
    assert result.greedy.tokens_served == 20
    assert result.same_pool_union_reduction_fraction == 0.0
    assert result.end_to_end_union_reduction_fraction == 0.0
    assert result.bytes_saved_per_token == 0.0
    assert result.throughput_gain_fraction == 0.0
    assert result.greedy.fairness.worst_case_deferred_rounds == 0


def test_sparse_arrivals_do_not_invent_a_full_composition_pool() -> None:
    model = ExpertUnionModel(
        total_experts=32,
        experts_per_token=4,
        moe_layers=3,
        expert_bytes=100,
    )
    hardware = HardwareConfig(
        key="test",
        label="test",
        routed_bandwidth_gb_s=1.0,
        dense_bytes=1_000_000,
        per_token_compute_seconds=0.0,
        old_batch1_tokens_per_second=1.0,
        calibration="synthetic",
    )

    result = simulate_comparison(
        model=model,
        hardware=hardware,
        prior=None,
        routing_prior_name="uniform",
        arrival_process="sparse",
        pool_size=16,
        max_batch_size=4,
        rounds=5,
        seed=17,
    )

    assert result.random.tokens_served == 10
    assert result.greedy.tokens_served == 10
    assert result.end_to_end_union_reduction_fraction == 0.0
    assert result.greedy.mean_batch_size == 2.0
    assert result.greedy.fairness.worst_case_deferred_rounds == 0
