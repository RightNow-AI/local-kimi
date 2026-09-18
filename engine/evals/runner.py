"""Run both authored capability suites against one OpenAI-compatible endpoint."""

from __future__ import annotations

import hashlib
import json
import platform
from datetime import datetime, timezone
from typing import Any

from .client import OpenAIEndpointClient
from .prompts import (
    INSTRUCTION_PROMPT_SET_ID,
    INSTRUCTION_PROMPT_SET_REVISION,
    INSTRUCTION_SUITE,
    TOOL_CALLING_SUITE,
    TOOL_PROMPT_SET_ID,
    TOOL_PROMPT_SET_REVISION,
    build_instruction_prompt_set,
    build_tool_prompt_set,
    prompt_set_metadata,
)
from .scoring import (
    score_instruction_response,
    score_tool_response,
    summarize_instruction_records,
    summarize_tool_records,
    unscoreable_instruction_record,
    unscoreable_tool_record,
)
from .thresholds import INSTRUCTION_THRESHOLDS_V1, TOOL_CALLING_THRESHOLDS_V1

SCHEMA_VERSION = "runinfra.kimi_linear.capability_eval.v1"
EXTRACTION_VERSION = 1
DEFAULT_INSTRUCTION_SAMPLING = {
    "temperature": 0.0,
    "top_p": 1.0,
    "max_tokens": 512,
}
DEFAULT_TOOL_SAMPLING = {
    "temperature": 0.0,
    "top_p": 1.0,
    "max_tokens": 512,
    "tool_choice": "auto",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _instruction_fingerprint(
    prompt_set: dict[str, Any], sampling: dict[str, Any]
) -> dict[str, Any]:
    return {
        "task": INSTRUCTION_SUITE,
        "promptSetId": prompt_set["id"],
        "promptSetRevision": prompt_set["revision"],
        "promptSetSha256": prompt_set["sha256"],
        "nSamples": prompt_set["count"],
        "temperature": sampling["temperature"],
        "topP": sampling["top_p"],
        "maxTokens": sampling["max_tokens"],
        "promptOrder": "authored fixed order",
        "checkerVersion": 1,
        "thresholdPolicy": INSTRUCTION_THRESHOLDS_V1.name,
        "extractionVersion": EXTRACTION_VERSION,
    }


def _tool_fingerprint(
    prompt_set: dict[str, Any], sampling: dict[str, Any]
) -> dict[str, Any]:
    return {
        "task": TOOL_CALLING_SUITE,
        "promptSetId": prompt_set["id"],
        "promptSetRevision": prompt_set["revision"],
        "promptSetSha256": prompt_set["sha256"],
        "nSamples": prompt_set["count"],
        "temperature": sampling["temperature"],
        "topP": sampling["top_p"],
        "maxTokens": sampling["max_tokens"],
        "toolChoice": sampling["tool_choice"],
        "promptOrder": "authored fixed order",
        "structuredCallSource": "OpenAI message.tool_calls",
        "rawSignalParsers": ["kimi", "kimi_k3"],
        "checkerVersion": 1,
        "thresholdPolicy": TOOL_CALLING_THRESHOLDS_V1.name,
        "extractionVersion": EXTRACTION_VERSION,
    }


def _endpoint_evidence(response: dict[str, Any]) -> dict[str, Any]:
    return {
        "scoreable": response.get("scoreable"),
        "cleanTerminal": response.get("cleanTerminal"),
        "finishReason": response.get("finishReason"),
        "httpStatus": response.get("httpStatus"),
        "transportError": response.get("transportError"),
        "malformed": response.get("malformed"),
        "malformedDetail": response.get("malformedDetail"),
        "wallS": response.get("wallS"),
        "rawResponse": response.get("rawResponse"),
        "httpBodyPreview": response.get("httpBodyPreview"),
    }


def _run_instruction_suite(
    client: OpenAIEndpointClient,
    *,
    model_id: str,
    sampling: dict[str, Any],
) -> dict[str, Any]:
    items = build_instruction_prompt_set()
    prompt_set = prompt_set_metadata(
        prompt_set_id=INSTRUCTION_PROMPT_SET_ID,
        revision=INSTRUCTION_PROMPT_SET_REVISION,
        items=items,
    )
    fingerprint = _instruction_fingerprint(prompt_set, sampling)
    records = []
    for item in items:
        try:
            response = client.complete(
                model_id=model_id,
                messages=item["messages"],
                sampling=sampling,
            )
        except Exception as exc:
            response = {
                "scoreable": False,
                "transportError": f"{type(exc).__name__}: {str(exc)[:500]}",
            }
        if response.get("scoreable") is True:
            record = score_instruction_response(item, str(response.get("content", "")))
            record["endpointResponse"] = _endpoint_evidence(response)
        else:
            reason = (
                response.get("transportError")
                or response.get("malformedDetail")
                or f"non-clean terminal: {response.get('finishReason')!r}"
            )
            record = unscoreable_instruction_record(
                item,
                reason=str(reason),
                response=_endpoint_evidence(response),
            )
        records.append(record)
    summary = summarize_instruction_records(records)
    return {
        **summary,
        "evaluation": INSTRUCTION_SUITE,
        "task": "authored instruction following with programmatic constraints",
        "suite": [INSTRUCTION_SUITE] if summary["status"] == "COMPLETE" else [],
        "fingerprint": fingerprint,
        "protocolFingerprint": _canonical_sha256(fingerprint),
        "promptSet": {**prompt_set, "items": list(items)},
        "records": records,
        "protocol": {
            "endpointProtocol": "OpenAI-compatible chat completions",
            "decode": {**sampling, "stream": False, "n": 1},
            "fixedPromptOrder": True,
            "offlineAuthoredPrompts": True,
            "strictItemDefinition": "all declared constraints pass",
            "unscoreableRule": "endpoint or checker failures remain explicit and keep the suite incomplete",
        },
    }


def _run_tool_suite(
    client: OpenAIEndpointClient,
    *,
    model_id: str,
    sampling: dict[str, Any],
) -> dict[str, Any]:
    cases = build_tool_prompt_set()
    prompt_set = prompt_set_metadata(
        prompt_set_id=TOOL_PROMPT_SET_ID,
        revision=TOOL_PROMPT_SET_REVISION,
        items=cases,
    )
    fingerprint = _tool_fingerprint(prompt_set, sampling)
    records = []
    for case in cases:
        try:
            response = client.complete(
                model_id=model_id,
                messages=case["messages"],
                sampling=sampling,
                tools=case["tools"],
            )
        except Exception as exc:
            response = {
                "scoreable": False,
                "transportError": f"{type(exc).__name__}: {str(exc)[:500]}",
            }
        if response.get("scoreable") is True:
            record = score_tool_response(case, response)
            record["endpointResponse"] = _endpoint_evidence(response)
        else:
            reason = (
                response.get("transportError")
                or response.get("malformedDetail")
                or f"non-clean terminal: {response.get('finishReason')!r}"
            )
            record = unscoreable_tool_record(
                case,
                reason=str(reason),
                response=_endpoint_evidence(response),
            )
        records.append(record)
    summary = summarize_tool_records(records)
    return {
        **summary,
        "evaluation": TOOL_CALLING_SUITE,
        "task": TOOL_CALLING_SUITE,
        "suite": [TOOL_CALLING_SUITE] if summary["status"] == "COMPLETE" else [],
        "fingerprint": fingerprint,
        "protocolFingerprint": _canonical_sha256(fingerprint),
        "battery": {**prompt_set, "items": list(cases)},
        "records": records,
        "protocol": {
            "endpointProtocol": "OpenAI-compatible chat completions",
            "decode": {**sampling, "stream": False, "n": 1},
            "fixedPromptOrder": True,
            "offlineAuthoredPrompts": True,
            "clientExecutableDefinition": (
                "a structured message.tool_calls entry whose tool is offered, whose "
                "arguments parse as JSON, and whose required keys are present"
            ),
            "rawKimiSignalRule": (
                "raw Kimi K2 or K3 syntax is detected by k3.toolcalls but does not count "
                "as client-executable OpenAI tool_calls"
            ),
            "unscoreableRule": "endpoint response-shape or transport failures keep the suite incomplete",
        },
    }


def run_capability_evaluation(
    *,
    base_url: str,
    model_id: str,
    endpoint_identity: str | None = None,
    api_key: str | None = None,
    timeout_s: float = 180.0,
    instruction_max_tokens: int = 512,
    tool_max_tokens: int = 512,
) -> dict[str, Any]:
    """Run both suites sequentially and return one factory-shaped evidence record."""
    if not model_id or not model_id.strip():
        raise ValueError("model_id is required")
    if instruction_max_tokens <= 0 or tool_max_tokens <= 0:
        raise ValueError("max token settings must be positive")
    instruction_sampling = {
        **DEFAULT_INSTRUCTION_SAMPLING,
        "max_tokens": int(instruction_max_tokens),
    }
    tool_sampling = {
        **DEFAULT_TOOL_SAMPLING,
        "max_tokens": int(tool_max_tokens),
    }
    client = OpenAIEndpointClient(
        base_url=base_url,
        api_key=api_key,
        timeout_s=timeout_s,
    )
    instruction = _run_instruction_suite(
        client,
        model_id=model_id,
        sampling=instruction_sampling,
    )
    tool_calling = _run_tool_suite(
        client,
        model_id=model_id,
        sampling=tool_sampling,
    )
    results = {
        INSTRUCTION_SUITE: {
            "status": instruction["status"],
            "measurement": instruction,
        },
        TOOL_CALLING_SUITE: {
            "status": tool_calling["status"],
            "measurement": tool_calling,
        },
    }
    selected = [INSTRUCTION_SUITE, TOOL_CALLING_SUITE]
    surfaced = [suite for suite in selected if results[suite]["status"] == "COMPLETE"]
    overall_status = "COMPLETE" if len(surfaced) == len(selected) else "INCOMPLETE"
    identity = endpoint_identity or base_url.rstrip("/")
    return {
        "schemaVersion": SCHEMA_VERSION,
        "createdAt": _utc_now(),
        "status": overall_status,
        "task": "",
        "suite": surfaced,
        "selectedSuites": selected,
        "modelRef": model_id,
        "servingIdentity": {
            "endpointIdentity": identity,
            "baseUrl": base_url.rstrip("/"),
            "modelId": model_id,
        },
        "environment": {
            "runner": "Modal HTTP evaluation job",
            "pythonVersion": platform.python_version(),
            "gpuRequired": False,
        },
        "samplingParameters": {
            INSTRUCTION_SUITE: instruction_sampling,
            TOOL_CALLING_SUITE: tool_sampling,
        },
        "results": results,
    }


__all__ = [
    "DEFAULT_INSTRUCTION_SAMPLING",
    "DEFAULT_TOOL_SAMPLING",
    "EXTRACTION_VERSION",
    "SCHEMA_VERSION",
    "run_capability_evaluation",
]
