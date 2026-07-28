"""Compare vLLM and engine.klinear outputs and build quantitative evidence."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from engine.accuracy.metrics import (
    first_divergence_index,
    greedy_identity_rate,
    kl_per_position_from_logprobs,
)
from engine.validate.protocol import (
    read_json,
    sha256_file,
    sha256_json,
    utc_now,
    write_json,
)
from engine.validate.thresholds import ENGINE_PARITY_V1, INTERPRETATION_RULES


def _resolved_directory(record: dict[str, Any]) -> str:
    directory = record.get("directory")
    if not isinstance(directory, str) or not directory:
        raise ValueError("checkpoint record has no directory")
    return str(Path(directory).resolve())


def validate_matched_sides(
    reference: dict[str, Any],
    candidate: dict[str, Any],
) -> None:
    """Fail closed unless implementation is the only intended variable."""
    if reference.get("side") != "vllm_reference":
        raise ValueError("reference record must be the vLLM side")
    if candidate.get("side") != "klinear_candidate":
        raise ValueError("candidate record must be the engine.klinear side")
    if reference.get("protocol_fingerprint") != candidate.get("protocol_fingerprint"):
        raise ValueError("the two sides used different protocol fingerprints")

    reference_checkpoint = reference.get("checkpoint")
    candidate_checkpoint = candidate.get("checkpoint")
    if not isinstance(reference_checkpoint, dict) or not isinstance(
        candidate_checkpoint, dict
    ):
        raise ValueError("both sides must record their checkpoint")
    if _resolved_directory(reference_checkpoint) != _resolved_directory(
        candidate_checkpoint
    ):
        raise ValueError(
            "refusing comparison because the two sides did not use the same checkpoint directory"
        )
    if reference_checkpoint != candidate_checkpoint:
        raise ValueError("the two sides recorded different checkpoint identities")
    if reference_checkpoint.get("kind") != "bf16":
        raise ValueError("engine parity comparison requires the same BF16 checkpoint")

    if reference.get("model_config") != candidate.get("model_config"):
        raise ValueError("the two implementations report different model configurations")
    for key in ("gpu_uuid", "gpu_name", "driver_version"):
        if reference.get("environment", {}).get(key) != candidate.get(
            "environment", {}
        ).get(key):
            raise ValueError(f"the two sides differ on GPU environment field {key}")

    reference_greedy = reference.get("greedy")
    candidate_greedy = candidate.get("greedy")
    if not isinstance(reference_greedy, list) or not isinstance(candidate_greedy, list):
        raise ValueError("both sides must contain greedy records")
    if len(reference_greedy) != len(candidate_greedy):
        raise ValueError("the two sides returned different greedy prompt counts")
    for ref_item, candidate_item in zip(
        reference_greedy, candidate_greedy, strict=True
    ):
        if ref_item.get("prompt_id") != candidate_item.get("prompt_id"):
            raise ValueError("greedy prompt order differs between sides")
        if ref_item.get("prompt_token_ids") != candidate_item.get(
            "prompt_token_ids"
        ):
            raise ValueError(
                f"prompt token IDs differ for {ref_item.get('prompt_id')}"
            )

    reference_first = reference.get("first_token_distribution", {}).get("records")
    candidate_first = candidate.get("first_token_distribution", {}).get("records")
    if not isinstance(reference_first, list) or not isinstance(candidate_first, list):
        raise ValueError("both sides must contain first-token records")
    if len(reference_first) != len(candidate_first):
        raise ValueError("the two sides returned different first-token prompt counts")
    for ref_item, candidate_item in zip(
        reference_first, candidate_first, strict=True
    ):
        if ref_item.get("prompt_id") != candidate_item.get("prompt_id"):
            raise ValueError("first-token prompt order differs between sides")
        if ref_item.get("prompt_token_ids") != candidate_item.get(
            "prompt_token_ids"
        ):
            raise ValueError(
                f"first-token prompt IDs differ for {ref_item.get('prompt_id')}"
            )


def greedy_metrics(
    reference_items: Sequence[dict[str, Any]],
    candidate_items: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    if len(reference_items) != len(candidate_items):
        raise ValueError("greedy sides must contain the same prompt count")
    reference_sequences = [item["output_token_ids"] for item in reference_items]
    candidate_sequences = [item["output_token_ids"] for item in candidate_items]
    identity_rate = greedy_identity_rate(reference_sequences, candidate_sequences)
    raw = []
    divergence_positions = []
    for reference, candidate in zip(
        reference_items, candidate_items, strict=True
    ):
        divergence = first_divergence_index(
            reference["output_token_ids"], candidate["output_token_ids"]
        )
        if divergence is not None:
            divergence_positions.append(divergence)
        raw.append(
            {
                "prompt_id": reference["prompt_id"],
                "category": reference.get("category"),
                "reference_token_ids": list(reference["output_token_ids"]),
                "candidate_token_ids": list(candidate["output_token_ids"]),
                "identical_token_ids": divergence is None,
                "first_divergence_index": divergence,
            }
        )
    token_zero_divergences = sum(
        item["first_divergence_index"] == 0 for item in raw
    )
    return {
        "summary": {
            "prompt_count": len(raw),
            "identical_prompt_count": sum(item["identical_token_ids"] for item in raw),
            "identity_rate": identity_rate,
            "diverged_prompt_count": len(divergence_positions),
            "min_first_divergence_index": (
                min(divergence_positions) if divergence_positions else None
            ),
            "median_first_divergence_index": (
                statistics.median(divergence_positions)
                if divergence_positions
                else None
            ),
            "token_zero_divergence_count": token_zero_divergences,
            "token_zero_divergence_rate": token_zero_divergences / len(raw),
            "identity_basis": "generated token IDs, never decoded strings",
        },
        "raw_per_prompt": raw,
    }


def first_token_metrics(
    reference_logprobs,
    candidate_logprobs,
    prompt_ids: Sequence[str],
) -> dict[str, Any]:
    reference = np.asarray(reference_logprobs, dtype=np.float64)
    candidate = np.asarray(candidate_logprobs, dtype=np.float64)
    if reference.shape != candidate.shape:
        raise ValueError("first-token distribution shapes differ")
    if reference.ndim != 2 or reference.shape[0] != len(prompt_ids):
        raise ValueError("first-token distributions must have one row per prompt")
    kl_values = kl_per_position_from_logprobs(reference, candidate)
    reference_top1 = np.argmax(reference, axis=-1)
    candidate_top1 = np.argmax(candidate, axis=-1)
    matches = reference_top1 == candidate_top1
    agreement = float(np.mean(matches))
    raw = [
        {
            "prompt_id": prompt_id,
            "reference_top1_token_id": int(reference_top1[index]),
            "candidate_top1_token_id": int(candidate_top1[index]),
            "top1_agrees": bool(matches[index]),
            "kl_reference_to_candidate_nats": float(kl_values[index]),
        }
        for index, prompt_id in enumerate(prompt_ids)
    ]
    return {
        "summary": {
            "prompt_count": len(raw),
            "vocab_size": int(reference.shape[1]),
            "top1_agreement": agreement,
            "top1_disagreement_count": int((~matches).sum()),
            "mean_kl_reference_to_candidate_nats": float(np.mean(kl_values)),
            "median_kl_reference_to_candidate_nats": float(np.median(kl_values)),
            "max_kl_reference_to_candidate_nats": float(np.max(kl_values)),
            "kl_direction": "KL(vLLM reference || engine.klinear candidate)",
            "kl_scope": "exact full vocabulary, not truncated top-k",
        },
        "raw_per_prompt": raw,
    }


def _load_distribution(side: dict[str, Any]) -> np.ndarray:
    artifact = side["first_token_distribution"]
    path = Path(artifact["artifact_path"])
    if not path.is_file():
        raise FileNotFoundError(path)
    if sha256_file(path) != artifact["artifact_sha256"]:
        raise ValueError(f"first-token artifact digest mismatch: {path}")
    values = np.load(path, allow_pickle=False)
    if list(values.shape) != artifact["artifact_shape"]:
        raise ValueError(f"first-token artifact shape metadata differs: {path}")
    if str(values.dtype) != artifact["artifact_dtype"]:
        raise ValueError(f"first-token artifact dtype metadata differs: {path}")
    return values


def _validate_sampler_consistency(
    side: dict[str, Any],
    distribution: np.ndarray,
) -> None:
    records = side["first_token_distribution"]["records"]
    greedy = side["greedy"]
    top1 = np.argmax(distribution, axis=-1)
    for index, (first_record, greedy_record) in enumerate(
        zip(records, greedy, strict=True)
    ):
        expected = int(top1[index])
        if first_record["top1_token_id"] != expected:
            raise ValueError(
                f"{side['side']} recorded the wrong top-1 for {first_record['prompt_id']}"
            )
        output_ids = greedy_record["output_token_ids"]
        if not output_ids or output_ids[0] != expected:
            raise ValueError(
                f"{side['side']} greedy token zero disagrees with its logits for "
                f"{first_record['prompt_id']}"
            )


def _validate_side_against_protocol(
    side: dict[str, Any],
    protocol: dict[str, Any],
) -> None:
    prompts = protocol["prompts"]
    expected_tokens = int(protocol["measurement"]["greedy"]["max_tokens"])
    greedy = side["greedy"]
    first = side["first_token_distribution"]["records"]
    if len(greedy) != len(prompts) or len(first) != len(prompts):
        raise ValueError(f"{side['side']} does not cover every protocol prompt")
    for declared, greedy_record, first_record in zip(
        prompts, greedy, first, strict=True
    ):
        for measured in (greedy_record, first_record):
            if measured["prompt_id"] != declared["prompt_id"]:
                raise ValueError(f"{side['side']} prompt order differs from the protocol")
            if measured["prompt_token_ids"] != declared["token_ids"]:
                raise ValueError(
                    f"{side['side']} did not use protocol token IDs for "
                    f"{declared['prompt_id']}"
                )
        if len(greedy_record["output_token_ids"]) != expected_tokens:
            raise ValueError(
                f"{side['side']} did not generate exactly {expected_tokens} tokens for "
                f"{declared['prompt_id']}"
            )


def _threshold_verdict(
    greedy: dict[str, Any],
    first_token: dict[str, Any],
) -> tuple[str, list[dict[str, Any]]]:
    threshold = ENGINE_PARITY_V1
    checks = [
        {
            "name": "prompt_count",
            "actual": greedy["summary"]["prompt_count"],
            "operator": "==",
            "threshold": threshold.prompt_count,
            "pass": greedy["summary"]["prompt_count"] == threshold.prompt_count,
        },
        {
            "name": "first_token_top1_agreement",
            "actual": first_token["summary"]["top1_agreement"],
            "operator": ">=",
            "threshold": threshold.min_first_token_top1_agreement,
            "pass": (
                first_token["summary"]["top1_agreement"]
                >= threshold.min_first_token_top1_agreement
            ),
        },
        {
            "name": "mean_first_token_kl_nats",
            "actual": first_token["summary"][
                "mean_kl_reference_to_candidate_nats"
            ],
            "operator": "<=",
            "threshold": threshold.max_mean_first_token_kl_nats,
            "pass": (
                first_token["summary"]["mean_kl_reference_to_candidate_nats"]
                <= threshold.max_mean_first_token_kl_nats
            ),
        },
        {
            "name": "max_single_prompt_first_token_kl_nats",
            "actual": first_token["summary"][
                "max_kl_reference_to_candidate_nats"
            ],
            "operator": "<=",
            "threshold": threshold.max_single_prompt_first_token_kl_nats,
            "pass": (
                first_token["summary"]["max_kl_reference_to_candidate_nats"]
                <= threshold.max_single_prompt_first_token_kl_nats
            ),
        },
        {
            "name": "token_zero_divergence_rate",
            "actual": greedy["summary"]["token_zero_divergence_rate"],
            "operator": "<=",
            "threshold": threshold.max_token_zero_divergence_rate,
            "pass": (
                greedy["summary"]["token_zero_divergence_rate"]
                <= threshold.max_token_zero_divergence_rate
            ),
        },
    ]
    return ("PASS" if all(check["pass"] for check in checks) else "FAIL", checks)


def build_evidence(
    *,
    protocol_path: Path,
    reference_path: Path,
    candidate_path: Path,
    output_path: Path,
    report_path: Path,
) -> dict[str, Any]:
    protocol = read_json(protocol_path)
    reference = read_json(reference_path)
    candidate = read_json(candidate_path)
    stored_fingerprint = protocol.get("protocol_fingerprint")
    fingerprint_input = dict(protocol)
    fingerprint_input.pop("protocol_fingerprint", None)
    if sha256_json(fingerprint_input) != stored_fingerprint:
        raise ValueError("protocol fingerprint does not match protocol contents")
    if protocol.get("measurement", {}).get("threshold") != ENGINE_PARITY_V1.as_dict():
        raise ValueError("runtime threshold differs from the predeclared source threshold")
    if reference.get("protocol_fingerprint") != stored_fingerprint:
        raise ValueError("reference side did not use the supplied protocol")
    if candidate.get("protocol_fingerprint") != stored_fingerprint:
        raise ValueError("candidate side did not use the supplied protocol")
    if reference.get("checkpoint") != protocol["measurement"]["checkpoint"]:
        raise ValueError("reference checkpoint differs from the protocol checkpoint")
    if candidate.get("checkpoint") != protocol["measurement"]["checkpoint"]:
        raise ValueError("candidate checkpoint differs from the protocol checkpoint")

    _validate_side_against_protocol(reference, protocol)
    _validate_side_against_protocol(candidate, protocol)
    validate_matched_sides(reference, candidate)
    reference_distribution = _load_distribution(reference)
    candidate_distribution = _load_distribution(candidate)
    _validate_sampler_consistency(reference, reference_distribution)
    _validate_sampler_consistency(candidate, candidate_distribution)

    greedy = greedy_metrics(reference["greedy"], candidate["greedy"])
    prompt_ids = [item["prompt_id"] for item in reference["greedy"]]
    first_token = first_token_metrics(
        reference_distribution,
        candidate_distribution,
        prompt_ids,
    )
    verdict, checks = _threshold_verdict(greedy, first_token)
    evidence = {
        "schema_version": "runinfra.kimi_linear.engine_parity_evidence.v1",
        "created_at": utc_now(),
        "verdict": verdict,
        "threshold": ENGINE_PARITY_V1.as_dict(),
        "threshold_checks": checks,
        "interpretation_rules": list(INTERPRETATION_RULES),
        "checkpoint": protocol["measurement"]["checkpoint"],
        "protocol": {
            "path": str(protocol_path),
            "fingerprint": stored_fingerprint,
            "created_at": protocol["created_at"],
            "prompt_set_sha256": protocol["measurement"]["prompt_set_sha256"],
            "prompt_token_ids_sha256": protocol["measurement"][
                "prompt_token_ids_sha256"
            ],
        },
        "implementations": {
            "reference": reference["implementation"],
            "candidate": candidate["implementation"],
        },
        "environment": {
            "reference": reference["environment"],
            "candidate": candidate["environment"],
        },
        "metrics": {
            "greedy_token_identity": greedy,
            "first_token_distribution": first_token,
        },
        "per_layer_hidden_states": {
            "available": False,
            "reference_reason": reference["per_layer_hidden_states"]["reason"],
            "candidate_reason": candidate["per_layer_hidden_states"]["reason"],
        },
        "artifacts": {
            "reference_side_json": str(reference_path),
            "candidate_side_json": str(candidate_path),
            "reference_first_token_logprobs": reference[
                "first_token_distribution"
            ]["artifact_path"],
            "candidate_first_token_logprobs": candidate[
                "first_token_distribution"
            ]["artifact_path"],
        },
    }
    write_json(output_path, evidence)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_report(evidence), encoding="utf-8")
    return evidence


def render_report(evidence: dict[str, Any]) -> str:
    greedy_metric = evidence["metrics"]["greedy_token_identity"]
    greedy = greedy_metric["summary"]
    first = evidence["metrics"]["first_token_distribution"]["summary"]
    checkpoint = evidence["checkpoint"]
    checks = evidence["threshold_checks"]
    lines = [
        "# Kimi Linear BF16 engine validation",
        "",
        f"ENGINE CORRECTNESS VERDICT: **{evidence['verdict']}**",
        "",
        "## Controlled variable",
        "",
        f"Both sides loaded the same BF16 checkpoint directory: `{checkpoint['directory']}`.",
        "The reference was vLLM 0.26.0. The candidate used direct `engine.klinear` prefill, sampling, and decode functions.",
        "The implementation was the only intended variable. Prompt token IDs, checkpoint, GPU, dtype, temperature, and generation length were locked.",
        "",
        "## Quantitative results",
        "",
        f"- Greedy identical token sequences: {greedy['identical_prompt_count']}/{greedy['prompt_count']} ({greedy['identity_rate']:.6f})",
        f"- Token-zero divergence rate: {greedy['token_zero_divergence_rate']:.6f}",
        f"- Minimum first divergence index among divergent prompts: {greedy['min_first_divergence_index']}",
        f"- Median first divergence index among divergent prompts: {greedy['median_first_divergence_index']}",
        f"- First-token top-1 agreement: {first['top1_agreement']:.6f}",
        f"- Mean first-token KL, reference to candidate: {first['mean_kl_reference_to_candidate_nats']:.9f} nats",
        f"- Maximum first-token KL, reference to candidate: {first['max_kl_reference_to_candidate_nats']:.9f} nats",
        "",
        "All greedy comparisons use token IDs, never decoded strings. KL is exact over the full vocabulary in the direction `KL(vLLM reference || engine.klinear candidate)`.",
        "",
        "## First divergence by prompt",
        "",
    ]
    divergent = [
        item
        for item in greedy_metric["raw_per_prompt"]
        if item["first_divergence_index"] is not None
    ]
    if divergent:
        lines.extend(
            [
                "| Prompt | First divergent generated-token index |",
                "| --- | ---: |",
                *[
                    f"| {item['prompt_id']} | {item['first_divergence_index']} |"
                    for item in divergent
                ],
                "",
            ]
        )
    else:
        lines.extend(["No prompt diverged within the measured generation window.", ""])
    lines.extend(
        [
        "## Predeclared threshold checks",
        "",
        "| Check | Actual | Requirement | Result |",
        "| --- | ---: | ---: | --- |",
        ]
    )
    for check in checks:
        lines.append(
            f"| {check['name']} | {check['actual']} | {check['operator']} {check['threshold']} | {'PASS' if check['pass'] else 'FAIL'} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            *[f"- {rule}" for rule in evidence["interpretation_rules"]],
            "",
            "## Per-layer hidden states",
            "",
            "Skipped. vLLM 0.26.0 does not expose matching intermediate states through this unmodified measured path, and adding hooks or a model fork would perturb the comparison.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol-json", required=True, type=Path)
    parser.add_argument("--reference-json", required=True, type=Path)
    parser.add_argument("--candidate-json", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--report-md", required=True, type=Path)
    args = parser.parse_args()
    evidence = build_evidence(
        protocol_path=args.protocol_json,
        reference_path=args.reference_json,
        candidate_path=args.candidate_json,
        output_path=args.output_json,
        report_path=args.report_md,
    )
    print(
        json.dumps(
            {
                "verdict": evidence["verdict"],
                "top1_agreement": evidence["metrics"]["first_token_distribution"]
                ["summary"]["top1_agreement"],
                "mean_kl_nats": evidence["metrics"]["first_token_distribution"]
                ["summary"]["mean_kl_reference_to_candidate_nats"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
