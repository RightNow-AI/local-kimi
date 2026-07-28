from __future__ import annotations

import math

from engine.batching.report import build_decision_rows
from engine.batching.union_model import ExpertUnionModel, HardwareConfig, zipf_prior


def test_uniform_union_matches_closed_form_and_converges() -> None:
    model = ExpertUnionModel()

    assert model.expected_uniform_union(1) == 16.0
    expected = 896 * (1 - (1 - 16 / 896) ** 32)
    assert math.isclose(model.expected_uniform_union(32), expected, rel_tol=1e-12)
    assert model.expected_uniform_union(10_000) > 895.999999


def test_skewed_routing_has_smaller_union_than_uniform() -> None:
    model = ExpertUnionModel()
    skewed = zipf_prior(exponent=1.0)

    assert model.expected_union(32, skewed) < model.expected_uniform_union(32)


def test_per_token_traffic_is_monotone_and_batch1_is_25_83_gb() -> None:
    model = ExpertUnionModel()
    concurrencies = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512)
    traffic = [model.routed_traffic_bytes_per_token(b) for b in concurrencies]

    assert traffic[0] == 16 * 17_547_264 * 92
    assert traffic[0] / 1e9 == 25.829572608
    assert all(right <= left for left, right in zip(traffic, traffic[1:]))


def test_report_rows_are_generated_from_supplied_model() -> None:
    model = ExpertUnionModel(
        total_experts=8,
        experts_per_token=2,
        moe_layers=3,
        expert_bytes=5,
    )
    hardware = HardwareConfig(
        key="test",
        label="test hardware",
        routed_bandwidth_gb_s=1.0,
        non_union_seconds_per_token=0.25,
        calibration="test calibration",
    )

    rows = build_decision_rows(
        model=model,
        hardware=hardware,
        concurrencies=(1, 3),
    )

    for row in rows:
        prediction = hardware.predict(model, row.concurrency)
        assert row.expected_union == prediction.expected_union
        assert row.routed_traffic_gb_per_token == prediction.routed_traffic_gb_per_token
        assert row.aggregate_tokens_per_second == prediction.aggregate_tokens_per_second

