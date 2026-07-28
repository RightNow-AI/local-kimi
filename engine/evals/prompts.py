"""Versioned offline prompt batteries for capability evaluation."""

from __future__ import annotations

import hashlib
import json
from typing import Any

INSTRUCTION_SUITE = "instruction_following_constraints"
INSTRUCTION_PROMPT_SET_ID = "runinfra/kimi-linear-instruction-constraints"
INSTRUCTION_PROMPT_SET_REVISION = 1

TOOL_CALLING_SUITE = "tool_calling_json_validity"
TOOL_PROMPT_SET_ID = "runinfra/kimi-linear-tool-calling"
TOOL_PROMPT_SET_REVISION = 1

_INSTRUCTION_SYSTEM = (
    "Follow the user's formatting requirements exactly. Return only the requested answer."
)
_TOOL_SYSTEM = (
    "Use a declared tool when the request requires one. Never invent a tool name. "
    "Arguments must follow the declared JSON schema. If no tool is needed, answer normally."
)


def _constraint(kind: str, **spec: Any) -> dict[str, Any]:
    return {"kind": kind, **spec}


def _instruction_item(
    item_id: str,
    category: str,
    prompt: str,
    *constraints: dict[str, Any],
) -> dict[str, Any]:
    normalized = []
    for index, constraint in enumerate(constraints, start=1):
        normalized.append({"constraintId": f"{item_id}:c{index}", **constraint})
    return {
        "itemId": item_id,
        "category": category,
        "messages": [
            {"role": "system", "content": _INSTRUCTION_SYSTEM},
            {"role": "user", "content": prompt},
        ],
        "constraints": normalized,
    }


def build_instruction_prompt_set() -> tuple[dict[str, Any], ...]:
    """Return 48 fixed items spanning eight independently reported constraint kinds."""
    items = (
        _instruction_item(
            "required-substring-01",
            "required_substring",
            "Explain retry safety in one sentence and include the exact token IDEMPOTENT.",
            _constraint("required_substring", value="IDEMPOTENT", caseSensitive=True),
        ),
        _instruction_item(
            "required-substring-02",
            "required_substring",
            "Write a short checkpoint status that includes the exact phrase CHECKPOINT SAVED.",
            _constraint("required_substring", value="CHECKPOINT SAVED", caseSensitive=True),
        ),
        _instruction_item(
            "required-substring-03",
            "required_substring",
            "Describe expert selection in one sentence and include the lowercase word router.",
            _constraint("required_substring", value="router", caseSensitive=True),
        ),
        _instruction_item(
            "required-substring-04",
            "required_substring",
            "State what the unquantized reference checkpoint is called and include BF16.",
            _constraint("required_substring", value="BF16", caseSensitive=True),
        ),
        _instruction_item(
            "required-substring-05",
            "required_substring",
            "Give a one-line deployment risk assessment containing the exact text risk: low.",
            _constraint("required_substring", value="risk: low", caseSensitive=True),
        ),
        _instruction_item(
            "required-substring-06",
            "required_substring",
            "Confirm that evidence was checked in one short sentence ending with the tag [verified].",
            _constraint("required_substring", value="[verified]", caseSensitive=True),
        ),
        _instruction_item(
            "forbidden-substring-01",
            "forbidden_substring",
            "Explain one benefit of caching without using the word fast.",
            _constraint("forbidden_substring", value="fast", caseSensitive=False),
        ),
        _instruction_item(
            "forbidden-substring-02",
            "forbidden_substring",
            "Explain idempotent retries without using the word again.",
            _constraint("forbidden_substring", value="again", caseSensitive=False),
        ),
        _instruction_item(
            "forbidden-substring-03",
            "forbidden_substring",
            "Describe a database index for a beginner without using the word book.",
            _constraint("forbidden_substring", value="book", caseSensitive=False),
        ),
        _instruction_item(
            "forbidden-substring-04",
            "forbidden_substring",
            "Describe solar panels without using the word sun.",
            _constraint("forbidden_substring", value="sun", caseSensitive=False),
        ),
        _instruction_item(
            "forbidden-substring-05",
            "forbidden_substring",
            "Write a neutral incident update without using the word failure.",
            _constraint("forbidden_substring", value="failure", caseSensitive=False),
        ),
        _instruction_item(
            "forbidden-substring-06",
            "forbidden_substring",
            "Contrast encryption and hashing without using the word secure.",
            _constraint("forbidden_substring", value="secure", caseSensitive=False),
        ),
        _instruction_item(
            "exact-word-count-01",
            "exact_word_count",
            "Describe a checksum in exactly 8 words.",
            _constraint("exact_word_count", count=8),
        ),
        _instruction_item(
            "exact-word-count-02",
            "exact_word_count",
            "Explain why backups matter in exactly 10 words.",
            _constraint("exact_word_count", count=10),
        ),
        _instruction_item(
            "exact-word-count-03",
            "exact_word_count",
            "Describe a load balancer in exactly 12 words.",
            _constraint("exact_word_count", count=12),
        ),
        _instruction_item(
            "exact-word-count-04",
            "exact_word_count",
            "Explain deterministic decoding in exactly 14 words.",
            _constraint("exact_word_count", count=14),
        ),
        _instruction_item(
            "exact-word-count-05",
            "exact_word_count",
            "Describe why raw evidence should be retained in exactly 16 words.",
            _constraint("exact_word_count", count=16),
        ),
        _instruction_item(
            "exact-word-count-06",
            "exact_word_count",
            "Explain why a client needs valid tool arguments in exactly 18 words.",
            _constraint("exact_word_count", count=18),
        ),
        _instruction_item(
            "word-range-01",
            "word_count_range",
            "Summarize the purpose of a health check in 12 to 16 words.",
            _constraint("word_count_range", minimum=12, maximum=16),
        ),
        _instruction_item(
            "word-range-02",
            "word_count_range",
            "Explain rate limiting in 14 to 18 words.",
            _constraint("word_count_range", minimum=14, maximum=18),
        ),
        _instruction_item(
            "word-range-03",
            "word_count_range",
            "Describe a transaction rollback in 16 to 20 words.",
            _constraint("word_count_range", minimum=16, maximum=20),
        ),
        _instruction_item(
            "word-range-04",
            "word_count_range",
            "Explain why prompt order is fixed in 10 to 14 words.",
            _constraint("word_count_range", minimum=10, maximum=14),
        ),
        _instruction_item(
            "word-range-05",
            "word_count_range",
            "Describe a JSON schema in 15 to 19 words.",
            _constraint("word_count_range", minimum=15, maximum=19),
        ),
        _instruction_item(
            "word-range-06",
            "word_count_range",
            "Explain an offline evaluation battery in 18 to 22 words.",
            _constraint("word_count_range", minimum=18, maximum=22),
        ),
        _instruction_item(
            "casing-01",
            "casing",
            "Reply with an uppercase sentence confirming the service is ready.",
            _constraint("casing", mode="upper"),
        ),
        _instruction_item(
            "casing-02",
            "casing",
            "Reply with a lowercase sentence saying the queue is empty.",
            _constraint("casing", mode="lower"),
        ),
        _instruction_item(
            "casing-03",
            "casing",
            "Write an uppercase warning about malformed JSON.",
            _constraint("casing", mode="upper"),
        ),
        _instruction_item(
            "casing-04",
            "casing",
            "Write a lowercase reminder to verify the endpoint.",
            _constraint("casing", mode="lower"),
        ),
        _instruction_item(
            "casing-05",
            "casing",
            "Answer in uppercase letters: what state follows pending when work succeeds?",
            _constraint("casing", mode="upper"),
        ),
        _instruction_item(
            "casing-06",
            "casing",
            "Answer in lowercase letters: what data format uses objects and arrays?",
            _constraint("casing", mode="lower"),
        ),
        _instruction_item(
            "json-shape-01",
            "json_shape",
            "Return only JSON with keys status and retries. status is a string and retries is an integer.",
            _constraint(
                "json_shape",
                requiredKeys=["status", "retries"],
                exactKeys=True,
                propertyTypes={"status": "string", "retries": "integer"},
            ),
        ),
        _instruction_item(
            "json-shape-02",
            "json_shape",
            "Return only JSON with keys ready and reasons. ready is boolean and reasons is an array.",
            _constraint(
                "json_shape",
                requiredKeys=["ready", "reasons"],
                exactKeys=True,
                propertyTypes={"ready": "boolean", "reasons": "array"},
            ),
        ),
        _instruction_item(
            "json-shape-03",
            "json_shape",
            "Return only JSON with keys model, score, and notes. Use string, number, and string types respectively.",
            _constraint(
                "json_shape",
                requiredKeys=["model", "score", "notes"],
                exactKeys=True,
                propertyTypes={"model": "string", "score": "number", "notes": "string"},
            ),
        ),
        _instruction_item(
            "json-shape-04",
            "json_shape",
            "Return only JSON with keys endpoint and headers, both JSON objects.",
            _constraint(
                "json_shape",
                requiredKeys=["endpoint", "headers"],
                exactKeys=True,
                propertyTypes={"endpoint": "object", "headers": "object"},
            ),
        ),
        _instruction_item(
            "json-shape-05",
            "json_shape",
            "Return only JSON with keys id, enabled, and tags using integer, boolean, and array types.",
            _constraint(
                "json_shape",
                requiredKeys=["id", "enabled", "tags"],
                exactKeys=True,
                propertyTypes={"id": "integer", "enabled": "boolean", "tags": "array"},
            ),
        ),
        _instruction_item(
            "json-shape-06",
            "json_shape",
            "Return only JSON with keys answer and confidence. answer is a string and confidence is a number.",
            _constraint(
                "json_shape",
                requiredKeys=["answer", "confidence"],
                exactKeys=True,
                propertyTypes={"answer": "string", "confidence": "number"},
            ),
        ),
        _instruction_item(
            "line-count-01",
            "exact_line_count",
            "Give exactly 3 non-empty lines about reliable deployments. No heading.",
            _constraint("exact_line_count", count=3),
        ),
        _instruction_item(
            "line-count-02",
            "exact_line_count",
            "Give exactly 4 non-empty lines describing a debugging workflow. No heading.",
            _constraint("exact_line_count", count=4),
        ),
        _instruction_item(
            "line-count-03",
            "exact_line_count",
            "List exactly 5 non-empty lines naming software testing levels. No heading.",
            _constraint("exact_line_count", count=5),
        ),
        _instruction_item(
            "line-count-04",
            "exact_line_count",
            "Write exactly 2 non-empty lines contrasting latency and throughput.",
            _constraint("exact_line_count", count=2),
        ),
        _instruction_item(
            "line-count-05",
            "exact_line_count",
            "Write exactly 6 non-empty lines for an incident checklist. No heading.",
            _constraint("exact_line_count", count=6),
        ),
        _instruction_item(
            "line-count-06",
            "exact_line_count",
            "Give exactly 3 non-empty lines explaining why comparisons need matched settings.",
            _constraint("exact_line_count", count=3),
        ),
        _instruction_item(
            "prefix-suffix-01",
            "prefix_suffix",
            "Write one sentence that begins with RESULT: and ends with :END",
            _constraint("prefix_suffix", prefix="RESULT:", suffix=":END"),
        ),
        _instruction_item(
            "prefix-suffix-02",
            "prefix_suffix",
            "Write a short status that begins with CHECK and ends with DONE.",
            _constraint("prefix_suffix", prefix="CHECK", suffix="DONE"),
        ),
        _instruction_item(
            "prefix-suffix-03",
            "prefix_suffix",
            "Write one line that begins with [START] and ends with [STOP].",
            _constraint("prefix_suffix", prefix="[START]", suffix="[STOP]"),
        ),
        _instruction_item(
            "prefix-suffix-04",
            "prefix_suffix",
            "Write a concise note beginning with SAFE: and ending with :SAFE",
            _constraint("prefix_suffix", prefix="SAFE:", suffix=":SAFE"),
        ),
        _instruction_item(
            "prefix-suffix-05",
            "prefix_suffix",
            "Write one sentence beginning with MODEL and ending with VERIFIED.",
            _constraint("prefix_suffix", prefix="MODEL", suffix="VERIFIED"),
        ),
        _instruction_item(
            "prefix-suffix-06",
            "prefix_suffix",
            "Write a brief summary beginning with BEGIN and ending with COMPLETE.",
            _constraint("prefix_suffix", prefix="BEGIN", suffix="COMPLETE"),
        ),
    )
    if len(items) != 48:
        raise AssertionError(f"instruction prompt set must contain 48 items, got {len(items)}")
    return items


def _function_tool(name: str, description: str, parameters: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {"name": name, "description": description, "parameters": parameters},
    }


def _tool_case(
    case_id: str,
    category: str,
    user_text: str,
    tools: list[dict[str, Any]],
    *,
    expected_call: bool = True,
    expected_tool_names: tuple[str, ...] = (),
    expected_content: str | None = None,
) -> dict[str, Any]:
    return {
        "caseId": case_id,
        "category": category,
        "messages": [
            {"role": "system", "content": _TOOL_SYSTEM},
            {"role": "user", "content": user_text},
        ],
        "tools": tools,
        "expectedCall": expected_call,
        "expectedToolNames": list(expected_tool_names),
        "expectedContent": expected_content,
    }


def build_tool_prompt_set() -> tuple[dict[str, Any], ...]:
    """Return 24 fixed tool cases, including four required abstentions."""
    def string_object(
        properties: dict[str, Any], required: list[str]
    ) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": False,
        }
    weather = _function_tool(
        "get_weather",
        "Return current weather for a city.",
        string_object({"city": {"type": "string"}}, ["city"]),
    )
    search_web = _function_tool(
        "search_web",
        "Search the public web.",
        string_object({"query": {"type": "string"}}, ["query"]),
    )
    cases = (
        _tool_case(
            "tool-single-string-01",
            "single_string",
            "What is the weather in Amman right now?",
            [weather],
            expected_tool_names=("get_weather",),
        ),
        _tool_case(
            "tool-file-read-02",
            "single_string",
            "Read the file /workspace/notes.txt.",
            [
                _function_tool(
                    "read_file",
                    "Read one text file.",
                    string_object({"path": {"type": "string"}}, ["path"]),
                )
            ],
            expected_tool_names=("read_file",),
        ),
        _tool_case(
            "tool-required-optional-03",
            "required_optional",
            "Search the codebase for beginV1IdempotentRequest and return at most five matches.",
            [
                _function_tool(
                    "search_code",
                    "Search source text with an optional result limit.",
                    string_object(
                        {"query": {"type": "string"}, "limit": {"type": "integer"}},
                        ["query"],
                    ),
                )
            ],
            expected_tool_names=("search_code",),
        ),
        _tool_case(
            "tool-array-string-04",
            "array_argument",
            "Create an issue titled Parser regression with labels bug and urgent.",
            [
                _function_tool(
                    "create_issue",
                    "Create a tracked issue.",
                    string_object(
                        {
                            "title": {"type": "string"},
                            "labels": {"type": "array", "items": {"type": "string"}},
                        },
                        ["title", "labels"],
                    ),
                )
            ],
            expected_tool_names=("create_issue",),
        ),
        _tool_case(
            "tool-array-object-05",
            "array_argument",
            "Schedule a review with ada@example.com and lin@example.com on 2026-08-01 at 09:00 UTC.",
            [
                _function_tool(
                    "schedule_review",
                    "Schedule a review meeting.",
                    string_object(
                        {
                            "attendees": {"type": "array", "items": {"type": "string"}},
                            "date": {"type": "string"},
                            "time": {"type": "string"},
                            "timezone": {"type": "string"},
                        },
                        ["attendees", "date", "time", "timezone"],
                    ),
                )
            ],
            expected_tool_names=("schedule_review",),
        ),
        _tool_case(
            "tool-enum-06",
            "enum_argument",
            "Deploy service api version 1.4.2 to staging.",
            [
                _function_tool(
                    "deploy_service",
                    "Deploy a service to an allowed environment.",
                    string_object(
                        {
                            "service": {"type": "string"},
                            "version": {"type": "string"},
                            "environment": {
                                "type": "string",
                                "enum": ["development", "staging", "production"],
                            },
                        },
                        ["service", "version", "environment"],
                    ),
                )
            ],
            expected_tool_names=("deploy_service",),
        ),
        _tool_case(
            "tool-number-array-07",
            "array_argument",
            "Calculate summary statistics for 1.5, 2.5, and 9.",
            [
                _function_tool(
                    "summarize_numbers",
                    "Compute summary statistics for numeric values.",
                    string_object(
                        {"values": {"type": "array", "items": {"type": "number"}}},
                        ["values"],
                    ),
                )
            ],
            expected_tool_names=("summarize_numbers",),
        ),
        _tool_case(
            "tool-integer-enum-08",
            "integer_enum",
            "Set ticket 481 to high priority.",
            [
                _function_tool(
                    "update_ticket",
                    "Update ticket priority.",
                    string_object(
                        {
                            "ticket_id": {"type": "integer"},
                            "priority": {"type": "string", "enum": ["low", "medium", "high"]},
                        },
                        ["ticket_id", "priority"],
                    ),
                )
            ],
            expected_tool_names=("update_ticket",),
        ),
        _tool_case(
            "tool-nested-object-09",
            "nested_object",
            "Ship order A19 to 7 Rainbow Street, Amman, postal code 11118.",
            [
                _function_tool(
                    "create_shipment",
                    "Create a shipment to a structured address.",
                    string_object(
                        {
                            "order_id": {"type": "string"},
                            "address": {
                                "type": "object",
                                "properties": {
                                    "street": {"type": "string"},
                                    "city": {"type": "string"},
                                    "postal_code": {"type": "string"},
                                },
                                "required": ["street", "city", "postal_code"],
                                "additionalProperties": False,
                            },
                        },
                        ["order_id", "address"],
                    ),
                )
            ],
            expected_tool_names=("create_shipment",),
        ),
        _tool_case(
            "tool-multiple-choice-10",
            "multiple_tools",
            "Send 'build is green' to the release channel.",
            [
                _function_tool(
                    "send_channel_message",
                    "Send a message to a named channel.",
                    string_object(
                        {"channel": {"type": "string"}, "message": {"type": "string"}},
                        ["channel", "message"],
                    ),
                ),
                _function_tool(
                    "send_email",
                    "Send an email to one recipient.",
                    string_object(
                        {"recipient": {"type": "string"}, "body": {"type": "string"}},
                        ["recipient", "body"],
                    ),
                ),
            ],
            expected_tool_names=("send_channel_message",),
        ),
        _tool_case(
            "tool-boolean-11",
            "boolean_argument",
            "Enable the feature named streaming_tool_calls.",
            [
                _function_tool(
                    "set_feature_flag",
                    "Enable or disable one feature flag.",
                    string_object(
                        {"name": {"type": "string"}, "enabled": {"type": "boolean"}},
                        ["name", "enabled"],
                    ),
                )
            ],
            expected_tool_names=("set_feature_flag",),
        ),
        _tool_case(
            "tool-integer-12",
            "integer_argument",
            "Reserve five seats for event K3Launch.",
            [
                _function_tool(
                    "reserve_seats",
                    "Reserve an integer number of seats.",
                    string_object(
                        {"event": {"type": "string"}, "count": {"type": "integer"}},
                        ["event", "count"],
                    ),
                )
            ],
            expected_tool_names=("reserve_seats",),
        ),
        _tool_case(
            "tool-email-13",
            "single_string",
            "Look up the account for dev@example.com.",
            [
                _function_tool(
                    "lookup_user",
                    "Find a user by email address.",
                    string_object({"email": {"type": "string"}}, ["email"]),
                )
            ],
            expected_tool_names=("lookup_user",),
        ),
        _tool_case(
            "tool-optional-boolean-14",
            "required_optional",
            "Write hello to /tmp/greeting.txt without overwriting an existing file.",
            [
                _function_tool(
                    "write_file",
                    "Write text to a file with an optional overwrite flag.",
                    string_object(
                        {
                            "path": {"type": "string"},
                            "content": {"type": "string"},
                            "overwrite": {"type": "boolean"},
                        },
                        ["path", "content"],
                    ),
                )
            ],
            expected_tool_names=("write_file",),
        ),
        _tool_case(
            "tool-integer-array-15",
            "array_argument",
            "Archive documents 7, 8, and 9.",
            [
                _function_tool(
                    "archive_documents",
                    "Archive documents by integer identifiers.",
                    string_object(
                        {"document_ids": {"type": "array", "items": {"type": "integer"}}},
                        ["document_ids"],
                    ),
                )
            ],
            expected_tool_names=("archive_documents",),
        ),
        _tool_case(
            "tool-query-limit-16",
            "required_optional",
            "Search logs for timeout in the api service and limit results to 20.",
            [
                _function_tool(
                    "search_logs",
                    "Search service logs.",
                    string_object(
                        {
                            "service": {"type": "string"},
                            "query": {"type": "string"},
                            "limit": {"type": "integer"},
                        },
                        ["service", "query"],
                    ),
                )
            ],
            expected_tool_names=("search_logs",),
        ),
        _tool_case(
            "tool-number-enum-17",
            "number_enum",
            "Convert 125.5 USD to EUR.",
            [
                _function_tool(
                    "convert_currency",
                    "Convert an amount between supported currencies.",
                    string_object(
                        {
                            "amount": {"type": "number"},
                            "from_currency": {"type": "string", "enum": ["USD", "EUR", "JOD"]},
                            "to_currency": {"type": "string", "enum": ["USD", "EUR", "JOD"]},
                        },
                        ["amount", "from_currency", "to_currency"],
                    ),
                )
            ],
            expected_tool_names=("convert_currency",),
        ),
        _tool_case(
            "tool-cron-18",
            "two_strings",
            "Create a cron job that runs backup.sh at 02:30 every day.",
            [
                _function_tool(
                    "create_cron_job",
                    "Create a cron job from a schedule and command.",
                    string_object(
                        {"schedule": {"type": "string"}, "command": {"type": "string"}},
                        ["schedule", "command"],
                    ),
                )
            ],
            expected_tool_names=("create_cron_job",),
        ),
        _tool_case(
            "tool-nested-optional-19",
            "nested_object",
            "Add quantity 3 of SKU K3-42 to cart C7 with gift metadata set to true.",
            [
                _function_tool(
                    "add_cart_item",
                    "Add an item with optional metadata to a cart.",
                    string_object(
                        {
                            "cart_id": {"type": "string"},
                            "sku": {"type": "string"},
                            "quantity": {"type": "integer"},
                            "metadata": {
                                "type": "object",
                                "properties": {"gift": {"type": "boolean"}},
                                "required": ["gift"],
                                "additionalProperties": False,
                            },
                        },
                        ["cart_id", "sku", "quantity"],
                    ),
                )
            ],
            expected_tool_names=("add_cart_item",),
        ),
        _tool_case(
            "tool-distractor-20",
            "multiple_tools",
            "Search the web for the latest Python release notes.",
            [weather, search_web],
            expected_tool_names=("search_web",),
        ),
        _tool_case(
            "tool-abstain-21",
            "correct_abstention",
            "Reply with exactly READY. Do not call a tool.",
            [weather],
            expected_call=False,
            expected_content="READY",
        ),
        _tool_case(
            "tool-abstain-22",
            "correct_abstention",
            "What is 2 + 2? Answer with exactly 4 and do not call a tool.",
            [search_web],
            expected_call=False,
            expected_content="4",
        ),
        _tool_case(
            "tool-abstain-23",
            "correct_abstention",
            "Reply with exactly NO ACTION. Do not call a tool.",
            [weather, search_web],
            expected_call=False,
            expected_content="NO ACTION",
        ),
        _tool_case(
            "tool-abstain-24",
            "correct_abstention",
            "State exactly PARIS. Do not call a tool.",
            [weather],
            expected_call=False,
            expected_content="PARIS",
        ),
    )
    positive_n = sum(case["expectedCall"] for case in cases)
    abstention_n = len(cases) - positive_n
    if len(cases) != 24 or positive_n != 20 or abstention_n != 4:
        raise AssertionError(
            "tool prompt set must contain 20 call cases and 4 abstention cases"
        )
    return cases


def prompt_set_sha256(items: tuple[dict[str, Any], ...]) -> str:
    payload = json.dumps(
        list(items),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def prompt_set_metadata(
    *,
    prompt_set_id: str,
    revision: int,
    items: tuple[dict[str, Any], ...],
) -> dict[str, Any]:
    return {
        "id": prompt_set_id,
        "revision": revision,
        "author": "RunInfra",
        "authorship": "RunInfra-authored fixed offline battery",
        "networkRequired": False,
        "count": len(items),
        "sha256": prompt_set_sha256(items),
    }


__all__ = [
    "INSTRUCTION_PROMPT_SET_ID",
    "INSTRUCTION_PROMPT_SET_REVISION",
    "INSTRUCTION_SUITE",
    "TOOL_CALLING_SUITE",
    "TOOL_PROMPT_SET_ID",
    "TOOL_PROMPT_SET_REVISION",
    "build_instruction_prompt_set",
    "build_tool_prompt_set",
    "prompt_set_metadata",
    "prompt_set_sha256",
]
