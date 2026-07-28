"""Compatibility contract for vLLM routed-expert capture on Kimi Linear."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Mapping

MODEL_EXPERTS_PER_TOKEN_KEY = "num_experts_per_token"
VLLM_EXPERTS_PER_TOKEN_ALIAS = "num_experts_per_tok"
VLLM_ROUTER_CAPTURE_KEYS = (VLLM_EXPERTS_PER_TOKEN_ALIAS, "top_k_experts")
EXPECTED_EXPERTS_PER_TOKEN = 8

ROUTER_ALIAS_REASON = (
    "Kimi-Linear config.json uses num_experts_per_token, while vLLM 0.26.0 "
    "routed_experts_capturer._get_num_experts_per_tok accepts only "
    "num_experts_per_tok or top_k_experts. The additive alias is applied "
    "identically to both served checkpoints so router capture can start without "
    "changing any weight or creating a between-side configuration difference."
)


def text_config(config: Mapping[str, Any]) -> Mapping[str, Any]:
    nested = config.get("text_config")
    return nested if isinstance(nested, Mapping) else config


def inject_vllm_router_alias(config: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return a copied config with the vLLM alias added to its text config."""
    served = copy.deepcopy(dict(config))
    target = served.get("text_config")
    if target is None:
        target = served
    if not isinstance(target, dict):
        raise TypeError("config.json text_config must be a JSON object")

    if MODEL_EXPERTS_PER_TOKEN_KEY not in target:
        raise ValueError(
            f"config.json is missing required model key {MODEL_EXPERTS_PER_TOKEN_KEY!r}"
        )
    source_value = int(target[MODEL_EXPERTS_PER_TOKEN_KEY])
    if source_value != EXPECTED_EXPERTS_PER_TOKEN:
        raise ValueError(
            f"expected {MODEL_EXPERTS_PER_TOKEN_KEY}={EXPECTED_EXPERTS_PER_TOKEN}, "
            f"got {source_value}"
        )

    existing = target.get(VLLM_EXPERTS_PER_TOKEN_ALIAS)
    if existing is not None and int(existing) != source_value:
        raise ValueError(
            f"config.json has conflicting {MODEL_EXPERTS_PER_TOKEN_KEY}={source_value} "
            f"and {VLLM_EXPERTS_PER_TOKEN_ALIAS}={existing}"
        )
    target[VLLM_EXPERTS_PER_TOKEN_ALIAS] = source_value
    metadata = {
        "injected": existing is None,
        "source_key": MODEL_EXPERTS_PER_TOKEN_KEY,
        "alias_key": VLLM_EXPERTS_PER_TOKEN_ALIAS,
        "value": source_value,
        "reason": ROUTER_ALIAS_REASON,
        "weight_values_changed": False,
    }
    return served, metadata


def validate_vllm_router_capture_config(
    config: Mapping[str, Any],
    *,
    config_path: Path | None = None,
) -> dict[str, Any]:
    """Fail before model loading if vLLM's capturer cannot read top-k."""
    target = text_config(config)
    present = [key for key in VLLM_ROUTER_CAPTURE_KEYS if key in target]
    location = f" at {config_path}" if config_path is not None else ""
    if not present:
        accepted = " or ".join(repr(key) for key in VLLM_ROUTER_CAPTURE_KEYS)
        raise ValueError(
            "vLLM 0.26.0 routed-expert capture preflight failed"
            f"{location}: served config must define {accepted}; the published "
            f"Kimi-Linear key {MODEL_EXPERTS_PER_TOKEN_KEY!r} is not accepted by "
            "routed_experts_capturer._get_num_experts_per_tok"
        )

    values = {key: int(target[key]) for key in present}
    if any(value != EXPECTED_EXPERTS_PER_TOKEN for value in values.values()):
        raise ValueError(
            "vLLM routed-expert capture preflight found an unexpected top-k"
            f"{location}: expected {EXPECTED_EXPERTS_PER_TOKEN}, got {values}"
        )
    source_value = target.get(MODEL_EXPERTS_PER_TOKEN_KEY)
    if source_value is not None and int(source_value) != EXPECTED_EXPERTS_PER_TOKEN:
        raise ValueError(
            f"served config{location} has {MODEL_EXPERTS_PER_TOKEN_KEY}={source_value}, "
            f"expected {EXPECTED_EXPERTS_PER_TOKEN}"
        )
    return {
        "valid": True,
        "accepted_keys_present": present,
        "experts_per_token": EXPECTED_EXPERTS_PER_TOKEN,
        "config_path": str(config_path) if config_path is not None else None,
    }


def compatibility_findings() -> list[dict[str, Any]]:
    """Return measured vLLM compatibility gaps that explain this harness design."""
    return [
        {
            "code": "VLLM_0_26_KIMI_LINEAR_NATIVE_INT4_LOAD_GAP",
            "status": "OBSERVED_IN_PRIOR_LANE_RUN",
            "finding": (
                "vLLM 0.26.0 refused to load Kimi-Linear in 4-bit form. This "
                "experiment therefore writes dequantized BF16 weights and serves "
                "both sides through the same BF16 runtime path."
            ),
            "effect_on_experiment": (
                "Direct INT4 serving is deliberately outside this measurement so "
                "engine support cannot be confused with quantization damage."
            ),
        },
        {
            "code": "VLLM_0_26_KIMI_LINEAR_ROUTER_CONFIG_KEY_GAP",
            "status": "OBSERVED_ON_H200_ACCURACY_RUN",
            "component": (
                "vllm/model_executor/layers/fused_moe/"
                "routed_experts_capturer.py::_get_num_experts_per_tok"
            ),
            "published_config_key": MODEL_EXPERTS_PER_TOKEN_KEY,
            "accepted_vllm_keys": list(VLLM_ROUTER_CAPTURE_KEYS),
            "observed_failure": (
                "ValueError: Cannot determine num_experts_per_tok: HF config has "
                "neither 'num_experts_per_tok' nor 'top_k_experts'"
            ),
            "workaround": ROUTER_ALIAS_REASON,
        },
    ]
