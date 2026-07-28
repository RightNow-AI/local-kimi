from __future__ import annotations

import copy

import pytest

from engine.evals.compare import ComparisonRefusal, compare_capability_records
from engine.evals.prompts import (
    INSTRUCTION_SUITE,
    TOOL_CALLING_SUITE,
    build_tool_prompt_set,
)
from engine.evals.scoring import evaluate_constraint, score_tool_response


def _tool_case(case_id: str) -> dict:
    return next(case for case in build_tool_prompt_set() if case["caseId"] == case_id)


def test_constraint_checker_rejects_violating_response() -> None:
    constraint = {
        "constraintId": "required:c1",
        "kind": "required_substring",
        "value": "IDEMPOTENT",
        "caseSensitive": True,
    }

    result = evaluate_constraint(constraint, "Retries should be safe.")

    assert result["status"] == "FAIL"


def test_unparseable_tool_arguments_are_malformed_not_absent() -> None:
    case = _tool_case("tool-single-string-01")
    response = {
        "finishReason": "tool_calls",
        "content": "",
        "toolCalls": [
            {
                "name": "get_weather",
                "arguments": '{"city":',
            }
        ],
    }

    result = score_tool_response(case, response)

    assert result["structuredCallCount"] == 1
    assert result["metrics"]["call_produced"] is True
    assert result["metrics"]["parseable"] is False
    assert result["toolCalls"][0]["parseError"] is not None


def test_unoffered_tool_name_is_wrong() -> None:
    case = _tool_case("tool-single-string-01")
    response = {
        "finishReason": "tool_calls",
        "content": "",
        "toolCalls": [
            {
                "name": "delete_everything",
                "arguments": '{"city":"Amman"}',
            }
        ],
    }

    result = score_tool_response(case, response)

    assert result["metrics"]["call_produced"] is True
    assert result["metrics"]["tool_exists"] is False
    assert result["metrics"]["client_executable"] is False


def test_no_tool_expected_item_fails_when_tool_is_called() -> None:
    case = _tool_case("tool-abstain-21")
    response = {
        "finishReason": "tool_calls",
        "content": "READY",
        "toolCalls": [
            {
                "name": "get_weather",
                "arguments": '{"city":"Amman"}',
            }
        ],
    }

    result = score_tool_response(case, response)

    assert result["metrics"]["correct_abstention"] is False


def _minimal_complete_record(prompt_digest: str, *, model_ref: str) -> dict:
    instruction_fingerprint = {
        "task": INSTRUCTION_SUITE,
        "promptSetSha256": prompt_digest,
        "temperature": 0.0,
        "topP": 1.0,
        "maxTokens": 512,
    }
    tool_fingerprint = {
        "task": TOOL_CALLING_SUITE,
        "promptSetSha256": "same-tool-prompts",
        "temperature": 0.0,
        "topP": 1.0,
        "maxTokens": 512,
        "toolChoice": "auto",
    }
    return {
        "status": "COMPLETE",
        "selectedSuites": [INSTRUCTION_SUITE, TOOL_CALLING_SUITE],
        "modelRef": model_ref,
        "servingIdentity": {"endpointIdentity": model_ref},
        "results": {
            INSTRUCTION_SUITE: {
                "status": "COMPLETE",
                "measurement": {
                    "status": "COMPLETE",
                    "fingerprint": instruction_fingerprint,
                },
            },
            TOOL_CALLING_SUITE: {
                "status": "COMPLETE",
                "measurement": {
                    "status": "COMPLETE",
                    "fingerprint": tool_fingerprint,
                },
            },
        },
    }


def test_comparator_refuses_different_prompt_sets() -> None:
    baseline = _minimal_complete_record("prompt-set-a", model_ref="bf16")
    optimized = copy.deepcopy(baseline)
    optimized["modelRef"] = "int4"
    optimized["servingIdentity"] = {"endpointIdentity": "int4"}
    optimized["results"][INSTRUCTION_SUITE]["measurement"]["fingerprint"][
        "promptSetSha256"
    ] = "prompt-set-b"

    with pytest.raises(ComparisonRefusal, match="prompt set mismatch"):
        compare_capability_records(baseline, optimized)


def test_comparator_refuses_different_sampling_parameters() -> None:
    baseline = _minimal_complete_record("same-prompt-set", model_ref="bf16")
    optimized = copy.deepcopy(baseline)
    optimized["modelRef"] = "int4"
    optimized["servingIdentity"] = {"endpointIdentity": "int4"}
    optimized["results"][INSTRUCTION_SUITE]["measurement"]["fingerprint"][
        "maxTokens"
    ] = 1024

    with pytest.raises(ComparisonRefusal, match="sampling parameters differ"):
        compare_capability_records(baseline, optimized)
