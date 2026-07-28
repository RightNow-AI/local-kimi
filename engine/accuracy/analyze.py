"""Build and independently verify the controlled accuracy evidence record."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from engine.accuracy.metrics import (
    first_divergence_index,
    greedy_identity_rate,
    kl_per_position_from_logprobs,
    perplexity_from_gold_logprobs,
    router_set_agreement,
)
from engine.accuracy.prompts import build_prompt_set, teacher_forced_text
from engine.accuracy.thresholds import ACCURACY_SCREEN_V1


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _close(left: float, right: float, *, tolerance: float = 1e-10) -> bool:
    return math.isclose(left, right, rel_tol=tolerance, abs_tol=tolerance)


def _validate_matched_protocol(reference: dict[str, Any], candidate: dict[str, Any]) -> None:
    if reference.get("side") != "bf16" or candidate.get("side") != "int4_dequantized":
        raise ValueError("expected bf16 and int4_dequantized side records")
    if reference["protocol_fingerprint"] != candidate["protocol_fingerprint"]:
        raise ValueError("the two sides used different protocol fingerprints")
    for key in ("gpu_uuid", "gpu_name", "driver_version", "vllm_version"):
        if reference["environment"].get(key) != candidate["environment"].get(key):
            raise ValueError(f"the two sides differ on environment field {key}")
    if reference["model_config"] != candidate["model_config"]:
        raise ValueError("the two sides loaded different model architecture metadata")

    reference_greedy = reference["greedy"]
    candidate_greedy = candidate["greedy"]
    if len(reference_greedy) != len(candidate_greedy):
        raise ValueError("the two sides returned different prompt counts")
    for baseline, dequantized in zip(reference_greedy, candidate_greedy, strict=True):
        if baseline["prompt_id"] != dequantized["prompt_id"]:
            raise ValueError("greedy prompt order differs between sides")
        if baseline["prompt_token_ids"] != dequantized["prompt_token_ids"]:
            raise ValueError(f"rendered prompt tokens differ for {baseline['prompt_id']}")

    if (
        reference["teacher_forced"]["token_ids"]
        != candidate["teacher_forced"]["token_ids"]
    ):
        raise ValueError("teacher-forced token IDs differ between sides")
    if reference["distribution"]["token_ids"] != candidate["distribution"]["token_ids"]:
        raise ValueError("distribution-probe token IDs differ between sides")
    reference_capture = reference["teacher_forced"]["router_capture"]
    candidate_capture = candidate["teacher_forced"]["router_capture"]
    for key in ("available", "requested"):
        if reference_capture.get(key) != candidate_capture.get(key):
            raise ValueError(f"router capture differs between sides on {key}")
    if reference_capture.get("reason") != candidate_capture.get("reason"):
        raise ValueError("router unavailability reason differs between sides")
    if reference_capture.get("available"):
        if (
            reference["teacher_forced"]["router_layer_indices"]
            != candidate["teacher_forced"]["router_layer_indices"]
        ):
            raise ValueError("router layer selection differs between sides")
    else:
        for side in (reference, candidate):
            teacher = side["teacher_forced"]
            if teacher["router_layer_indices"] or teacher["routed_experts"] is not None:
                raise ValueError("unavailable router capture contains route samples")


def _greedy_metrics(reference: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    reference_sequences = [item["output_token_ids"] for item in reference["greedy"]]
    candidate_sequences = [item["output_token_ids"] for item in candidate["greedy"]]
    rate = greedy_identity_rate(reference_sequences, candidate_sequences)
    raw = []
    divergence_positions = []
    prefix_fractions = []
    for baseline, dequantized in zip(reference["greedy"], candidate["greedy"], strict=True):
        divergence = first_divergence_index(
            baseline["output_token_ids"],
            dequantized["output_token_ids"],
        )
        if divergence is not None:
            divergence_positions.append(divergence)
        denominator = max(
            len(baseline["output_token_ids"]),
            len(dequantized["output_token_ids"]),
            1,
        )
        common_prefix = divergence if divergence is not None else denominator
        prefix_fractions.append(common_prefix / denominator)
        raw.append(
            {
                "prompt_id": baseline["prompt_id"],
                "category": baseline["category"],
                "prompt_token_ids": baseline["prompt_token_ids"],
                "bf16_output_token_ids": baseline["output_token_ids"],
                "int4_dequantized_output_token_ids": dequantized["output_token_ids"],
                "bf16_output_text": baseline["output_text"],
                "int4_dequantized_output_text": dequantized["output_text"],
                "identical_token_ids": divergence is None,
                "first_divergence_index": divergence,
                "bf16_finish_reason": baseline["finish_reason"],
                "int4_dequantized_finish_reason": dequantized["finish_reason"],
            }
        )
    return {
        "summary": {
            "prompt_count": len(raw),
            "identical_prompt_count": sum(item["identical_token_ids"] for item in raw),
            "identity_rate": rate,
            "diverged_prompt_count": len(divergence_positions),
            "median_first_divergence_index": (
                statistics.median(divergence_positions) if divergence_positions else None
            ),
            "min_first_divergence_index": (
                min(divergence_positions) if divergence_positions else None
            ),
            "mean_common_prefix_fraction": statistics.fmean(prefix_fractions),
            "identity_basis": "generated token IDs, never decoded strings",
        },
        "raw_per_prompt": raw,
    }


def _teacher_metrics(reference: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    reference_logprobs = [float(value) for value in reference["teacher_forced"]["gold_logprobs"]]
    candidate_logprobs = [float(value) for value in candidate["teacher_forced"]["gold_logprobs"]]
    if len(reference_logprobs) != len(candidate_logprobs):
        raise ValueError("teacher-forced logprob counts differ")
    token_ids = reference["teacher_forced"]["token_ids"]
    if len(reference_logprobs) != len(token_ids) - 1:
        raise ValueError("teacher-forced logprobs do not cover every causal target")
    reference_ppl = perplexity_from_gold_logprobs(reference_logprobs)
    candidate_ppl = perplexity_from_gold_logprobs(candidate_logprobs)
    raw_positions = [
        {
            "position": position,
            "target_token_id": token_ids[position],
            "bf16_gold_logprob": reference_logprobs[position - 1],
            "int4_dequantized_gold_logprob": candidate_logprobs[position - 1],
        }
        for position in range(1, len(token_ids))
    ]
    return {
        "summary": {
            "positions": len(reference_logprobs),
            "bf16_perplexity": reference_ppl,
            "int4_dequantized_perplexity": candidate_ppl,
            "absolute_increase": candidate_ppl - reference_ppl,
            "relative_increase": candidate_ppl / reference_ppl - 1.0,
        },
        "raw_per_position": raw_positions,
    }


def _resolve_artifact(side: dict[str, Any]) -> Path:
    path = Path(side["distribution"]["artifact_path"])
    if not path.is_file():
        raise FileNotFoundError(path)
    if _sha256_file(path) != side["distribution"]["artifact_sha256"]:
        raise ValueError(f"distribution artifact digest mismatch: {path}")
    return path


def _distribution_metrics(reference: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    reference_path = _resolve_artifact(reference)
    candidate_path = _resolve_artifact(candidate)
    reference_values = np.load(reference_path, allow_pickle=False)
    candidate_values = np.load(candidate_path, allow_pickle=False)
    if reference_values.shape != candidate_values.shape:
        raise ValueError("distribution artifact shapes differ")
    values = kl_per_position_from_logprobs(reference_values, candidate_values)
    reference_top1 = np.argmax(reference_values, axis=-1)
    candidate_top1 = np.argmax(candidate_values, axis=-1)
    matches = reference_top1 == candidate_top1
    token_ids = reference["distribution"]["token_ids"]
    raw_positions = [
        {
            "position": position,
            "target_token_id": token_ids[position],
            "bf16_top1_token_id": int(reference_top1[position - 1]),
            "int4_dequantized_top1_token_id": int(candidate_top1[position - 1]),
            "top1_agrees": bool(matches[position - 1]),
            "kl_bf16_to_int4_dequantized_nats": float(values[position - 1]),
        }
        for position in range(1, len(token_ids))
    ]
    return {
        "summary": {
            "positions": int(values.shape[0]),
            "vocab_size": int(reference_values.shape[1]),
            "top1_agreement": float(np.mean(matches)),
            "mean_kl_bf16_to_int4_dequantized_nats": float(np.mean(values)),
            "max_kl_bf16_to_int4_dequantized_nats": float(np.max(values)),
            "kl_direction": "KL(BF16 || INT4-dequantized)",
            "kl_scope": "exact full vocabulary, not truncated top-k",
            "per_layer_depth": {
                "available": False,
                "reason": (
                    "vLLM 0.26.0 exposes final next-token distributions but not "
                    "intermediate layer logits without changing the measured runtime path."
                ),
            },
        },
        "raw_per_position": raw_positions,
        "artifacts": {
            "bf16": reference["distribution"],
            "int4_dequantized": candidate["distribution"],
        },
    }


def _router_metrics(reference: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    reference_capture = reference["teacher_forced"]["router_capture"]
    candidate_capture = candidate["teacher_forced"]["router_capture"]
    if not reference_capture["available"] or not candidate_capture["available"]:
        return {
            "summary": {
                "available": False,
                "token_count": None,
                "moe_layer_count": None,
                "experts_per_token": int(
                    reference["model_config"]["num_experts_per_token"]
                ),
                "set_agreement": None,
                "comparison_basis": "unordered expert sets per token and MoE layer",
                "reason": reference_capture["reason"],
                "per_layer": [],
            },
            "raw": {
                "available": False,
                "reason": reference_capture["reason"],
                "layer_indices": [],
                "bf16": None,
                "int4_dequantized": None,
            },
        }

    reference_routes = np.asarray(reference["teacher_forced"]["routed_experts"], dtype=np.int64)
    candidate_routes = np.asarray(candidate["teacher_forced"]["routed_experts"], dtype=np.int64)
    if reference_routes.shape != candidate_routes.shape:
        raise ValueError("router-capture shapes differ")
    layer_indices = reference["teacher_forced"]["router_layer_indices"]
    topk = int(reference["model_config"]["num_experts_per_token"])
    selected_reference = reference_routes[:, layer_indices, :]
    selected_candidate = candidate_routes[:, layer_indices, :]
    overall = router_set_agreement(
        selected_reference.tolist(),
        selected_candidate.tolist(),
        expected_experts_per_token=topk,
    )
    per_layer = []
    for layer_index in layer_indices:
        layer_reference = reference_routes[:, layer_index, :]
        layer_candidate = candidate_routes[:, layer_index, :]
        agreement = router_set_agreement(
            layer_reference.tolist(),
            layer_candidate.tolist(),
            expected_experts_per_token=topk,
        )
        per_layer.append(
            {
                "layer_index": layer_index,
                "token_count": int(layer_reference.shape[0]),
                "set_agreement": agreement,
                "changed_token_count": int(round((1.0 - agreement) * layer_reference.shape[0])),
            }
        )
    return {
        "summary": {
            "available": True,
            "token_count": int(reference_routes.shape[0]),
            "moe_layer_count": len(layer_indices),
            "experts_per_token": topk,
            "set_agreement": overall,
            "comparison_basis": "unordered expert sets per token and MoE layer",
            "per_layer": per_layer,
        },
        "raw": {
            "available": True,
            "layer_indices": layer_indices,
            "bf16": reference_routes.tolist(),
            "int4_dequantized": candidate_routes.tolist(),
        },
    }


def _threshold_verdict(
    *,
    greedy: dict[str, Any],
    teacher: dict[str, Any],
    distribution: dict[str, Any],
    router: dict[str, Any],
    checkpoint: dict[str, Any],
) -> tuple[str, list[dict[str, Any]]]:
    threshold = ACCURACY_SCREEN_V1
    median_divergence = greedy["summary"]["median_first_divergence_index"]
    checks = [
        {
            "name": "prompt_count",
            "actual": greedy["summary"]["prompt_count"],
            "operator": ">=",
            "threshold": threshold.min_prompt_count,
            "pass": greedy["summary"]["prompt_count"] >= threshold.min_prompt_count,
        },
        {
            "name": "greedy_identity_rate",
            "actual": greedy["summary"]["identity_rate"],
            "operator": ">=",
            "threshold": threshold.min_greedy_identity_rate,
            "pass": greedy["summary"]["identity_rate"] >= threshold.min_greedy_identity_rate,
        },
        {
            "name": "median_first_divergence_index",
            "actual": median_divergence,
            "operator": ">= or no divergences",
            "threshold": threshold.min_median_first_divergence_index,
            "pass": (
                median_divergence is None
                or median_divergence >= threshold.min_median_first_divergence_index
            ),
        },
        {
            "name": "perplexity_relative_increase",
            "actual": teacher["summary"]["relative_increase"],
            "operator": "<=",
            "threshold": threshold.max_perplexity_relative_increase,
            "pass": (
                teacher["summary"]["relative_increase"]
                <= threshold.max_perplexity_relative_increase
            ),
        },
        {
            "name": "teacher_forced_positions",
            "actual": teacher["summary"]["positions"],
            "operator": ">=",
            "threshold": threshold.min_teacher_forced_positions,
            "pass": teacher["summary"]["positions"] >= threshold.min_teacher_forced_positions,
        },
        {
            "name": "distribution_positions",
            "actual": distribution["summary"]["positions"],
            "operator": "==",
            "threshold": threshold.distribution_positions,
            "pass": distribution["summary"]["positions"] == threshold.distribution_positions,
        },
        {
            "name": "next_token_top1_agreement",
            "actual": distribution["summary"]["top1_agreement"],
            "operator": ">=",
            "threshold": threshold.min_top1_agreement,
            "pass": distribution["summary"]["top1_agreement"] >= threshold.min_top1_agreement,
        },
        {
            "name": "mean_kl_bf16_to_int4_dequantized_nats",
            "actual": distribution["summary"]["mean_kl_bf16_to_int4_dequantized_nats"],
            "operator": "<=",
            "threshold": threshold.max_mean_kl_nats,
            "pass": (
                distribution["summary"]["mean_kl_bf16_to_int4_dequantized_nats"]
                <= threshold.max_mean_kl_nats
            ),
        },
        {
            "name": "router_set_agreement",
            "actual": router["summary"]["set_agreement"],
            "operator": "available and >=",
            "threshold": threshold.min_router_set_agreement,
            "pass": bool(router["summary"]["available"])
            and router["summary"]["set_agreement"] >= threshold.min_router_set_agreement,
        },
        {
            "name": "canonical_plan_reconciled",
            "actual": checkpoint["plan"]["reconciliation_status"],
            "operator": "publication_allowed is true",
            "threshold": True,
            "pass": bool(checkpoint["plan"]["publication_allowed"]),
        },
        {
            "name": "router_tensors_preserved_exactly",
            "actual": checkpoint["router_preservation"]["all_exact"],
            "operator": "==",
            "threshold": True,
            "pass": checkpoint["router_preservation"]["all_exact"] is True,
        },
    ]
    return ("PASS" if all(item["pass"] for item in checks) else "FAIL", checks)


def build_evidence(
    *,
    reference_path: Path,
    candidate_path: Path,
    checkpoint_path: Path,
    output_path: Path,
    run_root: Path,
) -> dict[str, Any]:
    reference = _read_json(reference_path)
    candidate = _read_json(candidate_path)
    checkpoint = _read_json(checkpoint_path)
    _validate_matched_protocol(reference, candidate)
    served_config_sha256 = reference["protocol"]["served_config_sha256"]
    if served_config_sha256 != checkpoint["served_bf16_checkpoint"]["config_sha256"]:
        raise ValueError("BF16 side did not serve the checkpoint manifest's config bytes")
    if served_config_sha256 != checkpoint["dequantized_checkpoint"]["config_sha256"]:
        raise ValueError(
            "INT4-dequantized side did not serve the checkpoint manifest's config bytes"
        )

    greedy = _greedy_metrics(reference, candidate)
    teacher = _teacher_metrics(reference, candidate)
    distribution = _distribution_metrics(reference, candidate)
    router = _router_metrics(reference, candidate)
    verdict, checks = _threshold_verdict(
        greedy=greedy,
        teacher=teacher,
        distribution=distribution,
        router=router,
        checkpoint=checkpoint,
    )
    router_available = bool(router["summary"]["available"])
    router_valid = router_available and router["summary"]["set_agreement"] == 1.0

    for artifact in distribution["artifacts"].values():
        artifact_path = Path(artifact["artifact_path"])
        artifact["relative_path"] = artifact_path.relative_to(run_root).as_posix()

    prompts = build_prompt_set()
    record = {
        "schema_version": "runinfra.kimi_linear.int4_accuracy.v2",
        "created_at": _utc_now(),
        "verdict": verdict,
        "thresholds_stated_before_measurement": ACCURACY_SCREEN_V1.as_dict(),
        "threshold_checks": checks,
        "experimental_design": {
            "controlled_variable": "weight values after INT4 encode and immediate BF16 decode",
            "held_constant": [
                "vLLM version",
                "physical GPU",
                "engine arguments",
                "tokenizer and chat template",
                "prompt token IDs",
                "sampling parameters",
                "teacher-forced token IDs",
            ],
            "causal_interpretation": (
                "VALID_QUANTIZATION_ONLY"
                if router_valid
                else (
                    "INCOMPLETE_ROUTER_AGREEMENT_UNAVAILABLE_NO_CLEAN_PASS"
                    if not router_available
                    else "INVALID_ROUTING_CHANGED_OUTPUT_METRICS_NOT_CAUSALLY_INTERPRETABLE"
                )
            ),
            "release_scope": (
                "This is a controlled quantization-damage screen. It is not a task-level "
                "release certification and it does not compare serving engines."
            ),
        },
        "model": {
            "id": checkpoint["model_id"],
            "config": reference["model_config"],
        },
        "checkpoints": {
            "published_source": checkpoint["source"],
            "bf16": checkpoint["served_bf16_checkpoint"],
            "int4_dequantized": checkpoint["dequantized_checkpoint"],
            "served_config_compatibility": checkpoint["served_config_compatibility"],
            "compatibility_findings": checkpoint["compatibility_findings"],
            "codec": checkpoint["codec"],
            "plan": checkpoint["plan"],
            "router_preservation": checkpoint["router_preservation"],
            "selected_tensor_summary": checkpoint["selected_tensor_summary"],
        },
        "environment": {
            "bf16": reference["environment"],
            "int4_dequantized": candidate["environment"],
            "same_gpu_uuid": (
                reference["environment"]["gpu_uuid"]
                == candidate["environment"]["gpu_uuid"]
            ),
            "same_vllm_version": (
                reference["environment"]["vllm_version"]
                == candidate["environment"]["vllm_version"]
            ),
        },
        "protocol": reference["protocol"],
        "protocol_fingerprint": reference["protocol_fingerprint"],
        "prompt_set": {
            "sha256": reference["protocol"]["prompt_set_sha256"],
            "count": len(prompts),
            "prompts": [prompt.as_dict() for prompt in prompts],
        },
        "teacher_forced_sample": {
            "sha256": reference["protocol"]["teacher_text_sha256"],
            "text": teacher_forced_text(),
            "token_ids": reference["teacher_forced"]["token_ids"],
        },
        "metrics": {
            "greedy_output_identity": greedy["summary"],
            "teacher_forced_perplexity": teacher["summary"],
            "next_token_distribution": distribution["summary"],
            "router_agreement": router["summary"],
        },
        "raw": {
            "greedy_per_prompt": greedy["raw_per_prompt"],
            "teacher_forced_per_position": teacher["raw_per_position"],
            "distribution_per_position": distribution["raw_per_position"],
            "distribution_artifacts": distribution["artifacts"],
            "router": router["raw"],
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return record


def verify_evidence_record(record_path: Path, *, artifact_root: Path | None = None) -> dict[str, Any]:
    record = _read_json(record_path)
    raw = record["raw"]
    reference_sequences = [item["bf16_output_token_ids"] for item in raw["greedy_per_prompt"]]
    candidate_sequences = [
        item["int4_dequantized_output_token_ids"] for item in raw["greedy_per_prompt"]
    ]
    identity = greedy_identity_rate(reference_sequences, candidate_sequences)
    if not _close(identity, record["metrics"]["greedy_output_identity"]["identity_rate"]):
        raise ValueError("greedy identity does not recompute from raw token IDs")
    for item in raw["greedy_per_prompt"]:
        divergence = first_divergence_index(
            item["bf16_output_token_ids"],
            item["int4_dequantized_output_token_ids"],
        )
        if divergence != item["first_divergence_index"]:
            raise ValueError(f"first divergence does not recompute for {item['prompt_id']}")

    baseline_gold = [item["bf16_gold_logprob"] for item in raw["teacher_forced_per_position"]]
    candidate_gold = [
        item["int4_dequantized_gold_logprob"] for item in raw["teacher_forced_per_position"]
    ]
    baseline_ppl = perplexity_from_gold_logprobs(baseline_gold)
    candidate_ppl = perplexity_from_gold_logprobs(candidate_gold)
    if not _close(baseline_ppl, record["metrics"]["teacher_forced_perplexity"]["bf16_perplexity"]):
        raise ValueError("BF16 perplexity does not recompute")
    if not _close(
        candidate_ppl,
        record["metrics"]["teacher_forced_perplexity"]["int4_dequantized_perplexity"],
    ):
        raise ValueError("INT4-dequantized perplexity does not recompute")

    router = raw["router"]
    router_summary = record["metrics"]["router_agreement"]
    if router_summary["available"]:
        layers = router["layer_indices"]
        reference_routes = np.asarray(router["bf16"], dtype=np.int64)[:, layers, :]
        candidate_routes = np.asarray(router["int4_dequantized"], dtype=np.int64)[:, layers, :]
        router_agreement = router_set_agreement(
            reference_routes.tolist(),
            candidate_routes.tolist(),
            expected_experts_per_token=record["model"]["config"]["num_experts_per_token"],
        )
        if not _close(router_agreement, router_summary["set_agreement"]):
            raise ValueError("router agreement does not recompute from raw expert sets")
    else:
        if router.get("available") is not False:
            raise ValueError("router summary is unavailable but raw status is not")
        if router.get("bf16") is not None or router.get("int4_dequantized") is not None:
            raise ValueError("unavailable router metric contains fabricated route samples")
        if router_summary["set_agreement"] is not None or not router_summary.get("reason"):
            raise ValueError("unavailable router metric must carry no value and a reason")
        router_agreement = None

    compatibility = record["checkpoints"]["served_config_compatibility"]
    if compatibility["source_checkpoint_modified"] is not False:
        raise ValueError("evidence claims the shared source checkpoint was modified")
    if compatibility["weight_values_changed"] is not False:
        raise ValueError("router config alias must not be represented as a weight change")
    if not compatibility["byte_identical_between_sides"]:
        raise ValueError("served configs were not byte-identical between sides")
    if compatibility["injected"] and compatibility["byte_identical_to_published_source"]:
        raise ValueError("an injected router alias cannot be byte-identical to the source config")
    if (
        compatibility["bf16_served_config_sha256"]
        != compatibility["int4_dequantized_served_config_sha256"]
    ):
        raise ValueError("served config digests differ between sides")
    if (
        record["protocol"]["served_config_sha256"]
        != compatibility["bf16_served_config_sha256"]
    ):
        raise ValueError("protocol config digest does not match checkpoint evidence")

    root = record_path.parent if artifact_root is None else artifact_root
    artifacts = raw["distribution_artifacts"]
    arrays = {}
    for side in ("bf16", "int4_dequantized"):
        relative = artifacts[side].get("relative_path")
        path = root / relative if relative else Path(artifacts[side]["artifact_path"])
        if _sha256_file(path) != artifacts[side]["artifact_sha256"]:
            raise ValueError(f"distribution artifact digest mismatch for {side}")
        arrays[side] = np.load(path, allow_pickle=False)
    kl_values = kl_per_position_from_logprobs(arrays["bf16"], arrays["int4_dequantized"])
    top1 = float(
        np.mean(np.argmax(arrays["bf16"], axis=-1) == np.argmax(arrays["int4_dequantized"], axis=-1))
    )
    distribution_summary = record["metrics"]["next_token_distribution"]
    if not _close(float(np.mean(kl_values)), distribution_summary["mean_kl_bf16_to_int4_dequantized_nats"]):
        raise ValueError("mean KL does not recompute from full-vocabulary artifacts")
    if not _close(top1, distribution_summary["top1_agreement"]):
        raise ValueError("top-1 agreement does not recompute")
    for index, item in enumerate(raw["distribution_per_position"]):
        reference_top1 = int(np.argmax(arrays["bf16"][index]))
        candidate_top1 = int(np.argmax(arrays["int4_dequantized"][index]))
        if reference_top1 != item["bf16_top1_token_id"]:
            raise ValueError(f"BF16 top-1 does not recompute at position {index + 1}")
        if candidate_top1 != item["int4_dequantized_top1_token_id"]:
            raise ValueError(f"INT4-dequantized top-1 does not recompute at position {index + 1}")
        if not _close(
            float(kl_values[index]),
            item["kl_bf16_to_int4_dequantized_nats"],
        ):
            raise ValueError(f"per-position KL does not recompute at position {index + 1}")
    router_check = next(
        item for item in record["threshold_checks"] if item["name"] == "router_set_agreement"
    )
    if not router_summary["available"] and router_check["pass"]:
        raise ValueError("unavailable router agreement cannot pass its threshold")
    recomputed_verdict = "PASS" if all(item["pass"] for item in record["threshold_checks"]) else "FAIL"
    if recomputed_verdict != record["verdict"]:
        raise ValueError("record verdict disagrees with its predeclared threshold checks")
    return {
        "verified": True,
        "verdict": record["verdict"],
        "identity_rate": identity,
        "bf16_perplexity": baseline_ppl,
        "int4_dequantized_perplexity": candidate_ppl,
        "top1_agreement": top1,
        "mean_kl_nats": float(np.mean(kl_values)),
        "router_set_agreement": router_agreement,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recompute a Kimi Linear INT4 accuracy record from its raw evidence."
    )
    parser.add_argument("record", type=Path)
    parser.add_argument("--artifact-root", type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            verify_evidence_record(args.record, artifact_root=args.artifact_root),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
