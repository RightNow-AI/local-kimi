from __future__ import annotations

import math

import pytest

from engine.batching.report import build_decision_rows
from engine.batching.union_model import (
    DEFAULT_DENSE_BYTES,
    ExpertUnionModel,
    HardwareConfig,
    default_hardware_configs,
    zipf_prior,
)


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
        dense_bytes=1_000_000_000,
        per_token_compute_seconds=0.25,
        old_batch1_tokens_per_second=0.5,
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
        assert row.dense_milliseconds_per_token == (
            prediction.dense_seconds_per_token * 1_000.0
        )
        assert row.aggregate_tokens_per_second == prediction.aggregate_tokens_per_second


def test_batch1_calibration_rejects_impossible_nine_tok_s_anchor() -> None:
    model = ExpertUnionModel()

    with pytest.raises(ValueError) as raised:
        HardwareConfig.calibrated_batch1(
            key="impossible",
            label="impossible 9 tok/s anchor",
            routed_bandwidth_gb_s=450.0,
            batch1_tokens_per_second=9.0,
            model=model,
            calibration="must fail",
        )

    message = str(raised.value)
    assert "physically impossible batch-1 calibration" in message
    assert "requested=9.000000 tok/s" in message
    assert "true_weight_roofline=3.208017 tok/s" in message
    assert "dense=114.444000000 GB" in message
    assert "bandwidth=450.000000 GB/s" in message


def test_batch1_calibration_rejects_impossible_pcie_anchor() -> None:
    model = ExpertUnionModel()

    with pytest.raises(ValueError, match="requested=2.100000 tok/s"):
        HardwareConfig.calibrated_batch1(
            key="impossible-pcie",
            label="impossible PCIe anchor",
            routed_bandwidth_gb_s=55.0,
            batch1_tokens_per_second=2.1,
            model=model,
            calibration="must fail",
        )


def test_physics_derived_batch1_includes_routed_and_dense_bytes() -> None:
    model = ExpertUnionModel()
    hardware = default_hardware_configs(model)[0]
    batch1 = hardware.predict(model, 1)
    expected_seconds = (
        model.batch1_routed_traffic_bytes + DEFAULT_DENSE_BYTES
    ) / 1e9 / 450.0

    assert hardware.per_token_compute_seconds == 0.0
    assert math.isclose(batch1.weight_seconds_per_batch, expected_seconds, rel_tol=1e-12)
    assert math.isclose(batch1.total_seconds_per_batch, expected_seconds, rel_tol=1e-12)
    assert math.isclose(
        batch1.aggregate_tokens_per_second, 1.0 / expected_seconds, rel_tol=1e-12
    )


def test_dense_cost_amortizes_exactly_across_the_batch() -> None:
    model = ExpertUnionModel()
    hardware = default_hardware_configs(model)[0]
    batch1 = hardware.predict(model, 1)
    batch32 = hardware.predict(model, 32)

    assert batch32.dense_seconds_per_batch == batch1.dense_seconds_per_batch
    assert math.isclose(
        batch32.dense_seconds_per_token,
        batch1.dense_seconds_per_token / 32.0,
        rel_tol=1e-12,
    )


def test_kernel_scale_changes_only_per_token_compute_term() -> None:
    model = ExpertUnionModel()
    hardware = HardwareConfig(
        key="compute-test",
        label="compute test",
        routed_bandwidth_gb_s=450.0,
        dense_bytes=DEFAULT_DENSE_BYTES,
        per_token_compute_seconds=0.02,
        old_batch1_tokens_per_second=1.0,
        calibration="synthetic nonzero compute",
    )
    base = hardware.predict(model, 32, compute_scale=1.0)
    fused = hardware.predict(model, 32, compute_scale=0.5)

    assert fused.union_seconds_per_batch == base.union_seconds_per_batch
    assert fused.dense_seconds_per_batch == base.dense_seconds_per_batch
    assert fused.token_compute_seconds_per_batch == (
        base.token_compute_seconds_per_batch * 0.5
    )
    assert fused.aggregate_tokens_per_second > base.aggregate_tokens_per_second
