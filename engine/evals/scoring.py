"""Programmatic constraint and tool-call scoring for authored endpoint evaluations."""

from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from collections.abc import Mapping
from typing import Any

from k3.reasoning import strip_inline_think
from k3.toolcalls import (
    KimiK3ToolParser,
    KimiToolParser,
    ParsedToolCall,
    parse_all,
)

from .thresholds import INSTRUCTION_THRESHOLDS_V1, TOOL_CALLING_THRESHOLDS_V1

_WORD_RE = re.compile(r"[A-Za-z0-9]+(?:['-][A-Za-z0-9]+)*")


def wilson_metric(successes: int, n: int) -> dict[str, Any]:
    """Return a proportion with explicit counts and a Wilson 95 percent interval."""
    if n < 0 or successes < 0 or successes > n:
        raise ValueError(f"invalid metric counts: successes={successes}, n={n}")
    if n == 0:
        return {
            "value": None,
            "numerator": successes,
            "n": 0,
            "ci95": None,
            "ciMethod": "Wilson",
        }
    z = 1.959963984540054
    rate = successes / n
    denominator = 1.0 + z * z / n
    center = (rate + z * z / (2.0 * n)) / denominator
    margin = (
        z
        * math.sqrt((rate * (1.0 - rate) + z * z / (4.0 * n)) / n)
        / denominator
    )
    return {
        "value": rate,
        "numerator": successes,
        "n": n,
        "ci95": {
            "low": max(0.0, center - margin),
            "high": min(1.0, center + margin),
        },
        "ciMethod": "Wilson",
    }


def _json_type_matches(value: Any, expected: str) -> bool:
    if expected == "null":
        return value is None
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value)
        )
    if expected == "string":
        return isinstance(value, str)
    if expected == "array":
        return isinstance(value, list)
    if expected == "object":
        return isinstance(value, dict)
    return False


def evaluate_constraint(constraint: Mapping[str, Any], response: str) -> dict[str, Any]:
    """Evaluate one declared constraint. Checker failure is UNVERIFIABLE, never PASS."""
    kind = constraint.get("kind")
    result = {
        "constraintId": constraint.get("constraintId"),
        "kind": kind,
        "status": "UNVERIFIABLE",
    }
    try:
        if kind == "required_substring":
            value = str(constraint["value"])
            case_sensitive = bool(constraint.get("caseSensitive", True))
            haystack = response if case_sensitive else response.casefold()
            needle = value if case_sensitive else value.casefold()
            passed = needle in haystack
            detail = {"required": value, "caseSensitive": case_sensitive}
        elif kind == "forbidden_substring":
            value = str(constraint["value"])
            case_sensitive = bool(constraint.get("caseSensitive", True))
            haystack = response if case_sensitive else response.casefold()
            needle = value if case_sensitive else value.casefold()
            passed = needle not in haystack
            detail = {"forbidden": value, "caseSensitive": case_sensitive}
        elif kind == "exact_word_count":
            expected = int(constraint["count"])
            actual = len(_WORD_RE.findall(response))
            passed = actual == expected
            detail = {"expected": expected, "actual": actual}
        elif kind == "word_count_range":
            minimum = int(constraint["minimum"])
            maximum = int(constraint["maximum"])
            actual = len(_WORD_RE.findall(response))
            passed = minimum <= actual <= maximum
            detail = {"minimum": minimum, "maximum": maximum, "actual": actual}
        elif kind == "casing":
            mode = str(constraint["mode"])
            cased = [character for character in response if character.isalpha()]
            if mode == "upper":
                passed = bool(cased) and all(character.isupper() for character in cased)
            elif mode == "lower":
                passed = bool(cased) and all(character.islower() for character in cased)
            else:
                raise ValueError(f"unsupported casing mode {mode!r}")
            detail = {"mode": mode, "casedCharacterN": len(cased)}
        elif kind == "json_shape":
            parsed = json.loads(response.strip())
            if not isinstance(parsed, dict):
                passed = False
                detail = {"parseable": True, "reason": "top-level JSON value is not an object"}
            else:
                required = [str(key) for key in constraint.get("requiredKeys", [])]
                missing = [key for key in required if key not in parsed]
                exact_keys = bool(constraint.get("exactKeys", False))
                unexpected = sorted(set(parsed) - set(required)) if exact_keys else []
                type_errors = []
                for key, expected_type in constraint.get("propertyTypes", {}).items():
                    if key in parsed and not _json_type_matches(parsed[key], str(expected_type)):
                        type_errors.append(
                            {
                                "key": key,
                                "expected": expected_type,
                                "actual": type(parsed[key]).__name__,
                            }
                        )
                passed = not missing and not unexpected and not type_errors
                detail = {
                    "parseable": True,
                    "missingKeys": missing,
                    "unexpectedKeys": unexpected,
                    "typeErrors": type_errors,
                }
        elif kind == "exact_line_count":
            expected = int(constraint["count"])
            lines = [line for line in response.splitlines() if line.strip()]
            passed = len(lines) == expected
            detail = {"expected": expected, "actual": len(lines)}
        elif kind == "prefix_suffix":
            prefix = str(constraint["prefix"])
            suffix = str(constraint["suffix"])
            stripped = response.strip()
            passed = stripped.startswith(prefix) and stripped.endswith(suffix)
            detail = {"prefix": prefix, "suffix": suffix}
        else:
            raise ValueError(f"unsupported constraint kind {kind!r}")
    except json.JSONDecodeError as exc:
        passed = False
        detail = {"parseable": False, "error": f"invalid JSON at position {exc.pos}"}
    except Exception as exc:
        result["reason"] = f"{type(exc).__name__}: {str(exc)[:300]}"
        return result

    result["status"] = "PASS" if passed else "FAIL"
    result["detail"] = detail
    return result


def score_instruction_response(item: Mapping[str, Any], response_text: str) -> dict[str, Any]:
    """Score one clean instruction response and retain every constraint outcome."""
    reasoning, visible = strip_inline_think(response_text)
    constraints = [
        evaluate_constraint(constraint, visible)
        for constraint in item.get("constraints", [])
    ]
    statuses = [constraint["status"] for constraint in constraints]
    scoreable = bool(statuses) and all(status in ("PASS", "FAIL") for status in statuses)
    strict_pass = all(status == "PASS" for status in statuses) if scoreable else None
    return {
        "itemId": item.get("itemId"),
        "category": item.get("category"),
        "scoreable": scoreable,
        "strictPass": strict_pass,
        "response": visible,
        "responsePreview": visible[:2000],
        "reasoningPreview": reasoning[:1000],
        "generatedChars": len(visible),
        "constraints": constraints,
    }


def unscoreable_instruction_record(
    item: Mapping[str, Any],
    *,
    reason: str,
    response: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "itemId": item.get("itemId"),
        "category": item.get("category"),
        "scoreable": False,
        "strictPass": None,
        "response": "",
        "responsePreview": "",
        "reasoningPreview": "",
        "generatedChars": 0,
        "constraints": [
            {
                "constraintId": constraint.get("constraintId"),
                "kind": constraint.get("kind"),
                "status": "NOT_SCOREABLE",
                "reason": reason,
            }
            for constraint in item.get("constraints", [])
        ],
        "endpointResponse": dict(response or {}),
        "unscoreableReason": reason,
    }


def summarize_instruction_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    requested_n = len(records)
    scoreable = [record for record in records if record.get("scoreable") is True]
    strict_successes = sum(record.get("strictPass") is True for record in scoreable)
    all_constraints = [
        constraint
        for record in records
        for constraint in record.get("constraints", [])
    ]
    scored_constraints = [
        constraint
        for constraint in all_constraints
        if constraint.get("status") in ("PASS", "FAIL")
    ]
    constraint_successes = sum(
        constraint.get("status") == "PASS" for constraint in scored_constraints
    )
    by_kind: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for constraint in scored_constraints:
        by_kind[str(constraint.get("kind"))].append(constraint)
    kind_metrics = {
        kind: wilson_metric(
            sum(constraint.get("status") == "PASS" for constraint in constraints),
            len(constraints),
        )
        for kind, constraints in sorted(by_kind.items())
    }
    invalid_n = requested_n - len(scoreable)
    thresholds = INSTRUCTION_THRESHOLDS_V1
    item_metric = wilson_metric(strict_successes, len(scoreable))
    constraint_metric = wilson_metric(constraint_successes, len(scored_constraints))
    invalid_metric = wilson_metric(invalid_n, requested_n)
    checks = [
        {
            "name": "strict_item_pass_rate",
            "actual": item_metric["value"],
            "operator": ">=",
            "threshold": thresholds.min_strict_item_pass_rate,
            "pass": item_metric["value"] is not None
            and item_metric["value"] >= thresholds.min_strict_item_pass_rate,
        },
        {
            "name": "constraint_pass_rate",
            "actual": constraint_metric["value"],
            "operator": ">=",
            "threshold": thresholds.min_constraint_pass_rate,
            "pass": constraint_metric["value"] is not None
            and constraint_metric["value"] >= thresholds.min_constraint_pass_rate,
        },
        {
            "name": "invalid_rate",
            "actual": invalid_metric["value"],
            "operator": "<=",
            "threshold": thresholds.max_invalid_rate,
            "pass": invalid_metric["value"] is not None
            and invalid_metric["value"] <= thresholds.max_invalid_rate,
        },
    ]
    for kind, metric in kind_metrics.items():
        checks.append(
            {
                "name": f"constraint_kind:{kind}",
                "actual": metric["value"],
                "operator": ">=",
                "threshold": thresholds.min_per_constraint_kind_rate,
                "pass": metric["value"] is not None
                and metric["value"] >= thresholds.min_per_constraint_kind_rate,
            }
        )
    complete = requested_n > 0 and invalid_n == 0
    verdict = (
        "INCOMPLETE"
        if not complete
        else "PASS"
        if all(check["pass"] for check in checks)
        else "FAIL"
    )
    return {
        "status": "COMPLETE" if complete else "INCOMPLETE",
        "verdict": verdict,
        "gateEligible": complete and all(check["pass"] for check in checks),
        "requestedN": requested_n,
        "completedN": len(scoreable),
        "scoreableN": len(scoreable),
        "strictItemPassRate": item_metric,
        "constraintPassRate": constraint_metric,
        "invalidRate": invalid_metric,
        "constraintKinds": kind_metrics,
        "thresholds": thresholds.as_dict(),
        "thresholdChecks": checks,
        "counts": {
            "constraintObservedN": len(all_constraints),
            "constraintScoreableN": len(scored_constraints),
            "constraintPassN": constraint_successes,
            "constraintFailN": len(scored_constraints) - constraint_successes,
            "constraintNotScoreableN": len(all_constraints) - len(scored_constraints),
        },
    }


def validate_json_schema(
    value: Any,
    schema: Mapping[str, Any],
    *,
    types_only: bool = False,
    path: str = "$",
) -> list[str]:
    """Validate the JSON Schema subset used by the authored tool battery."""
    errors: list[str] = []
    schema_type = schema.get("type")
    if schema_type is not None:
        allowed = schema_type if isinstance(schema_type, list) else [schema_type]
        if not any(_json_type_matches(value, str(expected)) for expected in allowed):
            return [f"{path}: expected {'|'.join(map(str, allowed))}, got {type(value).__name__}"]
    if not types_only and "enum" in schema and value not in schema["enum"]:
        errors.append(f"{path}: value not in enum")
    if isinstance(value, dict):
        properties = schema.get("properties", {})
        if not types_only:
            for required in schema.get("required", []):
                if required not in value:
                    errors.append(f"{path}.{required}: required property missing")
        additional = schema.get("additionalProperties", True)
        for key, child in value.items():
            if key in properties:
                errors.extend(
                    validate_json_schema(
                        child,
                        properties[key],
                        types_only=types_only,
                        path=f"{path}.{key}",
                    )
                )
            elif additional is False:
                errors.append(f"{path}.{key}: additional property not allowed")
    if isinstance(value, list) and isinstance(schema.get("items"), Mapping):
        for index, child in enumerate(value):
            errors.extend(
                validate_json_schema(
                    child,
                    schema["items"],
                    types_only=types_only,
                    path=f"{path}[{index}]",
                )
            )
    return errors


def detect_raw_kimi_tool_signals(content: str) -> dict[str, Any]:
    """Use repository parsers to detect raw Kimi calls without treating them as executable."""
    parsers = (KimiToolParser(), KimiK3ToolParser())
    parsed = []
    syntax_detected = False
    for parser in parsers:
        syntax_detected = syntax_detected or any(token in content for token in parser.TOKENS)
        calls = [
            event
            for event in parse_all(parser, content)
            if isinstance(event, ParsedToolCall)
        ]
        if calls:
            parsed.append(
                {
                    "parser": parser.name,
                    "calls": [
                        {"name": call.name, "arguments": call.arguments, "id": call.id}
                        for call in calls
                    ],
                }
            )
    return {
        "syntaxDetected": syntax_detected,
        "completeCallDetected": bool(parsed),
        "parsers": parsed,
    }


def _declared_tools(case: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        tool["function"]["name"]: tool["function"]["parameters"]
        for tool in case.get("tools", [])
    }


def score_tool_response(case: Mapping[str, Any], response: Mapping[str, Any]) -> dict[str, Any]:
    """Score structured OpenAI calls separately from raw Kimi call-like output."""
    calls = response.get("toolCalls", [])
    if not isinstance(calls, list):
        calls = []
    raw_content = response.get("content", "")
    content = raw_content if isinstance(raw_content, str) else ""
    reasoning, visible = strip_inline_think(content)
    raw_signals = detect_raw_kimi_tool_signals(content)
    schemas = _declared_tools(case)
    expected_names = set(case.get("expectedToolNames", []))
    parsed_calls = []
    for call in calls:
        call = call if isinstance(call, Mapping) else {}
        name = call.get("name")
        arguments = call.get("arguments")
        parsed_arguments = None
        parse_error = None
        if not isinstance(arguments, str):
            parse_error = "arguments payload is not a JSON string"
        else:
            try:
                parsed_arguments = json.loads(arguments)
            except json.JSONDecodeError as exc:
                parse_error = f"invalid JSON at position {exc.pos}"
        tool_exists = isinstance(name, str) and name in schemas
        schema = schemas.get(name) if tool_exists else None
        if parsed_arguments is None or schema is None:
            schema_errors = ["no declared schema available"] if schema is None else []
            type_errors = list(schema_errors)
            required_errors = []
            schema_valid = False
            types_correct = False
            required_keys_present = False
        else:
            schema_errors = validate_json_schema(parsed_arguments, schema)
            type_errors = validate_json_schema(parsed_arguments, schema, types_only=True)
            required_errors = [error for error in schema_errors if "required property missing" in error]
            schema_valid = not schema_errors
            types_correct = not type_errors
            required_keys_present = isinstance(parsed_arguments, dict) and not required_errors
        expected_tool_selected = bool(expected_names) and name in expected_names
        parsed_calls.append(
            {
                "name": name,
                "arguments": arguments,
                "parsedArguments": parsed_arguments,
                "parseError": parse_error,
                "toolExists": tool_exists,
                "requiredArgumentKeysPresent": required_keys_present,
                "schemaValid": schema_valid,
                "argTypesCorrect": types_correct,
                "expectedToolSelected": expected_tool_selected,
                "schemaErrors": schema_errors,
                "typeErrors": type_errors,
                "requiredKeyErrors": required_errors,
            }
        )

    structured_call_n = len(parsed_calls)
    call_signal_detected = structured_call_n > 0 or raw_signals["syntaxDetected"]
    if case.get("expectedCall") is True:
        metrics = {
            "call_signal_detected": call_signal_detected,
            "call_produced": structured_call_n > 0,
            "tool_exists": structured_call_n > 0
            and all(call["toolExists"] for call in parsed_calls),
            "parseable": structured_call_n > 0
            and all(call["parseError"] is None for call in parsed_calls),
            "required_argument_keys_present": structured_call_n > 0
            and all(call["requiredArgumentKeysPresent"] for call in parsed_calls),
            "schema_valid": structured_call_n > 0
            and all(call["schemaValid"] for call in parsed_calls),
            "arg_types_correct": structured_call_n > 0
            and all(call["argTypesCorrect"] for call in parsed_calls),
            "expected_tool_selected": structured_call_n > 0
            and all(call["expectedToolSelected"] for call in parsed_calls),
            "client_executable": structured_call_n > 0
            and all(
                call["toolExists"]
                and call["parseError"] is None
                and call["requiredArgumentKeysPresent"]
                and call["schemaValid"]
                for call in parsed_calls
            ),
            "correct_abstention": None,
        }
    else:
        expected_content = case.get("expectedContent")
        content_matches = expected_content is None or visible.strip() == expected_content
        metrics = {
            "call_signal_detected": call_signal_detected,
            "call_produced": None,
            "tool_exists": None,
            "parseable": None,
            "required_argument_keys_present": None,
            "schema_valid": None,
            "arg_types_correct": None,
            "expected_tool_selected": None,
            "client_executable": None,
            "correct_abstention": not call_signal_detected and content_matches,
        }
    return {
        "caseId": case.get("caseId"),
        "category": case.get("category"),
        "expectedCall": case.get("expectedCall"),
        "expectedToolNames": list(case.get("expectedToolNames", [])),
        "scoreable": True,
        "finishReason": response.get("finishReason"),
        "content": visible,
        "contentPreview": visible[:1000],
        "reasoningPreview": reasoning[:1000],
        "structuredCallCount": structured_call_n,
        "rawToolSignals": raw_signals,
        "toolCalls": parsed_calls,
        "metrics": metrics,
    }


def unscoreable_tool_record(
    case: Mapping[str, Any],
    *,
    reason: str,
    response: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "caseId": case.get("caseId"),
        "category": case.get("category"),
        "expectedCall": case.get("expectedCall"),
        "expectedToolNames": list(case.get("expectedToolNames", [])),
        "scoreable": False,
        "finishReason": None,
        "content": "",
        "contentPreview": "",
        "reasoningPreview": "",
        "structuredCallCount": 0,
        "rawToolSignals": {
            "syntaxDetected": False,
            "completeCallDetected": False,
            "parsers": [],
        },
        "toolCalls": [],
        "metrics": {
            "call_signal_detected": None,
            "call_produced": None,
            "tool_exists": None,
            "parseable": None,
            "required_argument_keys_present": None,
            "schema_valid": None,
            "arg_types_correct": None,
            "expected_tool_selected": None,
            "client_executable": None,
            "correct_abstention": None,
        },
        "endpointResponse": dict(response or {}),
        "unscoreableReason": reason,
    }


def summarize_tool_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    requested_n = len(records)
    scoreable = [record for record in records if record.get("scoreable") is True]
    call_records = [record for record in scoreable if record.get("expectedCall") is True]
    abstention_records = [record for record in scoreable if record.get("expectedCall") is False]
    call_metric_names = (
        "call_signal_detected",
        "call_produced",
        "tool_exists",
        "parseable",
        "required_argument_keys_present",
        "schema_valid",
        "arg_types_correct",
        "expected_tool_selected",
        "client_executable",
    )
    call_metrics = {
        name: wilson_metric(
            sum(record["metrics"].get(name) is True for record in call_records),
            len(call_records),
        )
        for name in call_metric_names
    }
    abstention_metric = wilson_metric(
        sum(record["metrics"].get("correct_abstention") is True for record in abstention_records),
        len(abstention_records),
    )
    invalid_n = requested_n - len(scoreable)
    invalid_metric = wilson_metric(invalid_n, requested_n)
    thresholds = TOOL_CALLING_THRESHOLDS_V1
    threshold_specs = (
        ("call_produced", thresholds.min_call_produced_rate),
        ("tool_exists", thresholds.min_offered_tool_name_rate),
        ("parseable", thresholds.min_arguments_json_parseable_rate),
        (
            "required_argument_keys_present",
            thresholds.min_required_argument_keys_rate,
        ),
        ("schema_valid", thresholds.min_schema_valid_rate),
        ("expected_tool_selected", thresholds.min_expected_tool_selected_rate),
    )
    checks = [
        {
            "name": name,
            "actual": call_metrics[name]["value"],
            "operator": ">=",
            "threshold": threshold,
            "pass": call_metrics[name]["value"] is not None
            and call_metrics[name]["value"] >= threshold,
        }
        for name, threshold in threshold_specs
    ]
    checks.extend(
        [
            {
                "name": "correct_abstention",
                "actual": abstention_metric["value"],
                "operator": ">=",
                "threshold": thresholds.min_correct_abstention_rate,
                "pass": abstention_metric["value"] is not None
                and abstention_metric["value"] >= thresholds.min_correct_abstention_rate,
            },
            {
                "name": "invalid_rate",
                "actual": invalid_metric["value"],
                "operator": "<=",
                "threshold": thresholds.max_invalid_rate,
                "pass": invalid_metric["value"] is not None
                and invalid_metric["value"] <= thresholds.max_invalid_rate,
            },
        ]
    )
    complete = requested_n > 0 and invalid_n == 0
    verdict = (
        "INCOMPLETE"
        if not complete
        else "PASS"
        if all(check["pass"] for check in checks)
        else "FAIL"
    )
    return {
        "status": "COMPLETE" if complete else "INCOMPLETE",
        "verdict": verdict,
        "gateEligible": complete and all(check["pass"] for check in checks),
        "requestedN": requested_n,
        "completedN": len(scoreable),
        "scoreableN": len(scoreable),
        "invalidRate": invalid_metric,
        "metricGroups": {
            "toolCalls": {"n": len(call_records), "metrics": call_metrics},
            "abstention": {
                "n": len(abstention_records),
                "metrics": {"correct_abstention": abstention_metric},
            },
        },
        "thresholds": thresholds.as_dict(),
        "thresholdChecks": checks,
        "failedChecks": [
            {"caseId": record["caseId"], "metric": metric}
            for record in scoreable
            for metric, passed in record["metrics"].items()
            if passed is False
        ],
        "zeroStructuredCallSignal": not any(
            record["metrics"].get("call_produced") is True for record in call_records
        ),
    }


__all__ = [
    "detect_raw_kimi_tool_signals",
    "evaluate_constraint",
    "score_instruction_response",
    "score_tool_response",
    "summarize_instruction_records",
    "summarize_tool_records",
    "unscoreable_instruction_record",
    "unscoreable_tool_record",
    "validate_json_schema",
    "wilson_metric",
]
