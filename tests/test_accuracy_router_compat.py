from __future__ import annotations

import copy

import pytest

from engine.accuracy.analyze import _router_metrics, _threshold_verdict
from engine.accuracy.checkpoint import _write_served_config
from engine.accuracy.router_compat import (
    inject_vllm_router_alias,
    validate_vllm_router_capture_config,
)


def _published_config() -> dict:
    return {
        "text_config": {
            "num_experts": 256,
            "num_experts_per_token": 8,
        }
    }


def test_router_alias_is_additive_and_does_not_mutate_published_config() -> None:
    published = _published_config()
    original = copy.deepcopy(published)

    served, metadata = inject_vllm_router_alias(published)

    assert published == original
    assert "num_experts_per_tok" not in published["text_config"]
    assert served["text_config"]["num_experts_per_token"] == 8
    assert served["text_config"]["num_experts_per_tok"] == 8
    assert metadata["injected"] is True
    assert metadata["weight_values_changed"] is False


def test_router_capture_preflight_names_vllm_keys_before_model_load() -> None:
    with pytest.raises(ValueError) as error:
        validate_vllm_router_capture_config(_published_config())

    message = str(error.value)
    assert "num_experts_per_tok" in message
    assert "top_k_experts" in message
    assert "num_experts_per_token" in message


def test_both_served_sides_receive_byte_identical_config(tmp_path) -> None:
    served, _ = inject_vllm_router_alias(_published_config())
    bf16_dir = tmp_path / "bf16"
    dequantized_dir = tmp_path / "int4-dequantized"

    bf16_sha256 = _write_served_config(bf16_dir, served)
    dequantized_sha256 = _write_served_config(dequantized_dir, served)

    assert bf16_sha256 == dequantized_sha256
    assert (bf16_dir / "config.json").read_bytes() == (
        dequantized_dir / "config.json"
    ).read_bytes()


@pytest.mark.parametrize("accepted_key", ["num_experts_per_tok", "top_k_experts"])
def test_router_capture_preflight_accepts_each_vllm_key(accepted_key: str) -> None:
    config = _published_config()
    config["text_config"][accepted_key] = 8

    result = validate_vllm_router_capture_config(config)

    assert result["valid"] is True
    assert result["accepted_keys_present"] == [accepted_key]
    assert result["experts_per_token"] == 8


def test_missing_router_capture_is_explicit_and_forces_fail() -> None:
    reason = {"code": "ROUTER_ENABLED_SIDE_RUN_FAILED", "reason": "capture failed"}
    side = {
        "model_config": {"num_experts_per_token": 8},
        "teacher_forced": {
            "router_capture": {
                "available": False,
                "requested": False,
                "reason": reason,
            },
            "router_layer_indices": [],
            "routed_experts": None,
        },
    }
    router = _router_metrics(side, copy.deepcopy(side))
    assert router["summary"]["available"] is False
    assert router["summary"]["set_agreement"] is None
    assert router["summary"]["reason"] == reason
    assert router["raw"]["bf16"] is None

    verdict, checks = _threshold_verdict(
        greedy={
            "summary": {
                "prompt_count": 128,
                "identity_rate": 1.0,
                "median_first_divergence_index": None,
            }
        },
        teacher={"summary": {"relative_increase": 0.0, "positions": 511}},
        distribution={
            "summary": {
                "positions": 128,
                "top1_agreement": 1.0,
                "mean_kl_bf16_to_int4_dequantized_nats": 0.0,
            }
        },
        router=router,
        checkpoint={
            "plan": {
                "reconciliation_status": "RECONCILED_CANONICAL_MATCHES_LOCAL_FALLBACK",
                "publication_allowed": True,
            },
            "router_preservation": {"all_exact": True},
        },
    )

    router_check = next(item for item in checks if item["name"] == "router_set_agreement")
    assert router_check["actual"] is None
    assert router_check["pass"] is False
    assert verdict == "FAIL"
