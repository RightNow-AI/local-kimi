"""Fail-closed comparison for two capability evaluation records."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from .prompts import INSTRUCTION_SUITE, TOOL_CALLING_SUITE
from .scoring import summarize_instruction_records, summarize_tool_records


class ComparisonRefusal(ValueError):
    """The records do not describe comparable measurements."""


def _unwrap(record: Mapping[str, Any]) -> Mapping[str, Any]:
    result = record.get("result")
    if "results" not in record and isinstance(result, Mapping):
        return result
    return record


def _measurement(record: Mapping[str, Any], suite: str, side: str) -> Mapping[str, Any]:
    results = record.get("results")
    if not isinstance(results, Mapping):
        raise ComparisonRefusal(f"{side} record has no results object")
    entry = results.get(suite)
    if not isinstance(entry, Mapping):
        raise ComparisonRefusal(f"{side} record has no {suite} result")
    if entry.get("status") != "COMPLETE":
        raise ComparisonRefusal(
            f"{side} {suite} is not COMPLETE: {entry.get('status')!r}"
        )
    measurement = entry.get("measurement")
    if not isinstance(measurement, Mapping):
        raise ComparisonRefusal(f"{side} {suite} has no measurement object")
    if measurement.get("status") != "COMPLETE":
        raise ComparisonRefusal(f"{side} {suite} measurement is not COMPLETE")
    return measurement


def _metric_delta(
    baseline: Mapping[str, Any], optimized: Mapping[str, Any]
) -> dict[str, Any]:
    baseline_value = baseline.get("value")
    optimized_value = optimized.get("value")
    if not isinstance(baseline_value, (int, float)) or isinstance(baseline_value, bool):
        raise ComparisonRefusal("baseline metric has no numeric value")
    if not isinstance(optimized_value, (int, float)) or isinstance(optimized_value, bool):
        raise ComparisonRefusal("optimized metric has no numeric value")
    return {
        "baseline": dict(baseline),
        "optimized": dict(optimized),
        "delta": float(optimized_value) - float(baseline_value),
    }


def _validate_summary(
    measurement: Mapping[str, Any],
    *,
    suite: str,
    side: str,
) -> None:
    records = measurement.get("records")
    if not isinstance(records, list):
        raise ComparisonRefusal(f"{side} {suite} has no raw records")
    if suite == INSTRUCTION_SUITE:
        recomputed = summarize_instruction_records(records)
        fields = (
            "status",
            "verdict",
            "gateEligible",
            "requestedN",
            "completedN",
            "scoreableN",
            "strictItemPassRate",
            "constraintPassRate",
            "invalidRate",
            "constraintKinds",
            "thresholds",
            "thresholdChecks",
            "counts",
        )
    else:
        recomputed = summarize_tool_records(records)
        fields = (
            "status",
            "verdict",
            "gateEligible",
            "requestedN",
            "completedN",
            "scoreableN",
            "invalidRate",
            "metricGroups",
            "thresholds",
            "thresholdChecks",
            "failedChecks",
            "zeroStructuredCallSignal",
        )
    for field in fields:
        if measurement.get(field) != recomputed.get(field):
            raise ComparisonRefusal(
                f"{side} {suite} field {field} does not recompute from raw records"
            )


def _record_map(
    measurement: Mapping[str, Any],
    *,
    id_field: str,
    side: str,
) -> dict[str, Mapping[str, Any]]:
    records = measurement.get("records")
    if not isinstance(records, list):
        raise ComparisonRefusal(f"{side} measurement has no raw records")
    mapped: dict[str, Mapping[str, Any]] = {}
    for record in records:
        if not isinstance(record, Mapping):
            raise ComparisonRefusal(f"{side} measurement has a malformed raw record")
        identity = record.get(id_field)
        if not isinstance(identity, str) or not identity or identity in mapped:
            raise ComparisonRefusal(f"{side} measurement has invalid or duplicate item IDs")
        mapped[identity] = record
    return mapped


def _compare_instruction(
    baseline: Mapping[str, Any], optimized: Mapping[str, Any]
) -> dict[str, Any]:
    baseline_kinds = baseline.get("constraintKinds")
    optimized_kinds = optimized.get("constraintKinds")
    if not isinstance(baseline_kinds, Mapping) or not isinstance(optimized_kinds, Mapping):
        raise ComparisonRefusal("instruction constraint-kind metrics are missing")
    if list(baseline_kinds) != list(optimized_kinds):
        raise ComparisonRefusal("instruction constraint-kind sets differ")
    baseline_records = _record_map(baseline, id_field="itemId", side="baseline instruction")
    optimized_records = _record_map(optimized, id_field="itemId", side="optimized instruction")
    if list(baseline_records) != list(optimized_records):
        raise ComparisonRefusal("instruction item sets or order differ")
    regressions = []
    for item_id in baseline_records:
        before = baseline_records[item_id]
        after = optimized_records[item_id]
        if before.get("strictPass") is True and after.get("strictPass") is False:
            regressions.append({"itemId": item_id, "metric": "strictPass"})
        before_constraints = {
            constraint.get("constraintId"): constraint.get("status")
            for constraint in before.get("constraints", [])
            if isinstance(constraint, Mapping)
        }
        after_constraints = {
            constraint.get("constraintId"): constraint.get("status")
            for constraint in after.get("constraints", [])
            if isinstance(constraint, Mapping)
        }
        if before_constraints.keys() != after_constraints.keys():
            raise ComparisonRefusal(f"instruction constraints differ for {item_id}")
        for constraint_id, status in before_constraints.items():
            if status == "PASS" and after_constraints[constraint_id] == "FAIL":
                regressions.append(
                    {"itemId": item_id, "constraintId": constraint_id, "metric": "constraint"}
                )
    return {
        "strictItemPassRate": _metric_delta(
            baseline["strictItemPassRate"], optimized["strictItemPassRate"]
        ),
        "constraintPassRate": _metric_delta(
            baseline["constraintPassRate"], optimized["constraintPassRate"]
        ),
        "constraintKinds": {
            kind: _metric_delta(baseline_kinds[kind], optimized_kinds[kind])
            for kind in baseline_kinds
        },
        "regressionDetected": bool(regressions),
        "regressions": regressions,
        "baselineVerdict": baseline.get("verdict"),
        "optimizedVerdict": optimized.get("verdict"),
    }


def _compare_tools(
    baseline: Mapping[str, Any], optimized: Mapping[str, Any]
) -> dict[str, Any]:
    baseline_groups = baseline.get("metricGroups")
    optimized_groups = optimized.get("metricGroups")
    if not isinstance(baseline_groups, Mapping) or not isinstance(optimized_groups, Mapping):
        raise ComparisonRefusal("tool-calling metric groups are missing")
    if baseline_groups.keys() != optimized_groups.keys():
        raise ComparisonRefusal("tool-calling metric group sets differ")
    groups = {}
    for group_name in baseline_groups:
        before = baseline_groups[group_name]
        after = optimized_groups[group_name]
        if not isinstance(before, Mapping) or not isinstance(after, Mapping):
            raise ComparisonRefusal(f"tool-calling group {group_name} is malformed")
        if before.get("n") != after.get("n"):
            raise ComparisonRefusal(f"tool-calling group {group_name} has unequal N")
        before_metrics = before.get("metrics")
        after_metrics = after.get("metrics")
        if not isinstance(before_metrics, Mapping) or not isinstance(after_metrics, Mapping):
            raise ComparisonRefusal(f"tool-calling group {group_name} has no metrics")
        if before_metrics.keys() != after_metrics.keys():
            raise ComparisonRefusal(f"tool-calling group {group_name} metric sets differ")
        groups[group_name] = {
            "nPerSide": before.get("n"),
            "metrics": {
                name: _metric_delta(before_metrics[name], after_metrics[name])
                for name in before_metrics
            },
        }
    baseline_records = _record_map(baseline, id_field="caseId", side="baseline tool")
    optimized_records = _record_map(optimized, id_field="caseId", side="optimized tool")
    if list(baseline_records) != list(optimized_records):
        raise ComparisonRefusal("tool-calling case sets or order differ")
    regressions = []
    for case_id in baseline_records:
        before_metrics = baseline_records[case_id].get("metrics", {})
        after_metrics = optimized_records[case_id].get("metrics", {})
        if not isinstance(before_metrics, Mapping) or not isinstance(after_metrics, Mapping):
            raise ComparisonRefusal(f"tool-calling metrics are malformed for {case_id}")
        if before_metrics.keys() != after_metrics.keys():
            raise ComparisonRefusal(f"tool-calling metrics differ for {case_id}")
        for metric, passed in before_metrics.items():
            if passed is True and after_metrics[metric] is False:
                regressions.append({"caseId": case_id, "metric": metric})
    return {
        "metricGroups": groups,
        "regressionDetected": bool(regressions),
        "regressions": regressions,
        "baselineVerdict": baseline.get("verdict"),
        "optimizedVerdict": optimized.get("verdict"),
        "note": "No tool metrics are blended. Every dimension and per-case regression remains separate.",
    }


def compare_capability_records(
    baseline_record: Mapping[str, Any],
    optimized_record: Mapping[str, Any],
) -> dict[str, Any]:
    """Compare two complete records only when prompt and sampling identities match."""
    baseline = _unwrap(baseline_record)
    optimized = _unwrap(optimized_record)
    expected_suites = [INSTRUCTION_SUITE, TOOL_CALLING_SUITE]
    for side, record in (("baseline", baseline), ("optimized", optimized)):
        if record.get("status") != "COMPLETE":
            raise ComparisonRefusal(f"{side} record is not COMPLETE")
        if record.get("selectedSuites") != expected_suites:
            raise ComparisonRefusal(f"{side} record selected a different suite set or order")

    measurements: dict[str, tuple[Mapping[str, Any], Mapping[str, Any]]] = {}
    for suite in expected_suites:
        before = _measurement(baseline, suite, "baseline")
        after = _measurement(optimized, suite, "optimized")
        before_fingerprint = before.get("fingerprint")
        after_fingerprint = after.get("fingerprint")
        if not isinstance(before_fingerprint, Mapping) or not isinstance(
            after_fingerprint, Mapping
        ):
            raise ComparisonRefusal(f"{suite} fingerprint is missing")
        before_prompt = before_fingerprint.get("promptSetSha256")
        after_prompt = after_fingerprint.get("promptSetSha256")
        if not before_prompt or before_prompt != after_prompt:
            raise ComparisonRefusal(
                f"{suite} prompt set mismatch: {before_prompt!r} != {after_prompt!r}"
            )
        sampling_fields = ["temperature", "topP", "maxTokens"]
        if suite == TOOL_CALLING_SUITE:
            sampling_fields.append("toolChoice")
        differing_sampling = [
            field
            for field in sampling_fields
            if before_fingerprint.get(field) != after_fingerprint.get(field)
        ]
        if differing_sampling:
            raise ComparisonRefusal(
                f"{suite} sampling parameters differ on {', '.join(differing_sampling)}"
            )
        if dict(before_fingerprint) != dict(after_fingerprint):
            differing = sorted(
                key
                for key in set(before_fingerprint) | set(after_fingerprint)
                if before_fingerprint.get(key) != after_fingerprint.get(key)
            )
            raise ComparisonRefusal(
                f"{suite} protocol fingerprint differs on {', '.join(differing)}"
            )
        if before.get("thresholds") != after.get("thresholds"):
            raise ComparisonRefusal(f"{suite} threshold policies differ")
        _validate_summary(before, suite=suite, side="baseline")
        _validate_summary(after, suite=suite, side="optimized")
        measurements[suite] = (before, after)

    baseline_ref = baseline.get("modelRef")
    optimized_ref = optimized.get("modelRef")
    if not isinstance(baseline_ref, str) or not baseline_ref:
        raise ComparisonRefusal("baseline record has no modelRef")
    if not isinstance(optimized_ref, str) or not optimized_ref:
        raise ComparisonRefusal("optimized record has no modelRef")
    if baseline_ref == optimized_ref:
        baseline_serving = baseline.get("servingIdentity")
        optimized_serving = optimized.get("servingIdentity")
        if json.dumps(baseline_serving, sort_keys=True) == json.dumps(
            optimized_serving, sort_keys=True
        ):
            raise ComparisonRefusal("baseline and optimized records describe the same served side")

    instruction = _compare_instruction(*measurements[INSTRUCTION_SUITE])
    tools = _compare_tools(*measurements[TOOL_CALLING_SUITE])
    return {
        "status": "COMPLETE",
        "verdict": "MEASURED",
        "suite": expected_suites,
        "baseline": {
            "modelRef": baseline_ref,
            "servingIdentity": baseline.get("servingIdentity"),
        },
        "optimized": {
            "modelRef": optimized_ref,
            "servingIdentity": optimized.get("servingIdentity"),
        },
        "protocolMatched": True,
        "results": {
            INSTRUCTION_SUITE: instruction,
            TOOL_CALLING_SUITE: tools,
        },
        "regressionDetected": bool(
            instruction["regressionDetected"] or tools["regressionDetected"]
        ),
        "note": (
            "No blended capability score is inferred. Read each instruction constraint kind "
            "and each tool-call metric independently."
        ),
    }


__all__ = ["ComparisonRefusal", "compare_capability_records"]
