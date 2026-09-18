"""Predeclared acceptance policy for the endpoint capability evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class InstructionThresholds:
    name: str
    min_strict_item_pass_rate: float
    min_constraint_pass_rate: float
    min_per_constraint_kind_rate: float
    max_invalid_rate: float
    reasoning: tuple[str, ...]

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class ToolCallingThresholds:
    name: str
    min_call_produced_rate: float
    min_offered_tool_name_rate: float
    min_arguments_json_parseable_rate: float
    min_required_argument_keys_rate: float
    min_schema_valid_rate: float
    min_expected_tool_selected_rate: float
    min_correct_abstention_rate: float
    max_invalid_rate: float
    reasoning: tuple[str, ...]

    def as_dict(self) -> dict:
        return asdict(self)


INSTRUCTION_THRESHOLDS_V1 = InstructionThresholds(
    name="kimi-linear-authored-instruction-screen-v1",
    min_strict_item_pass_rate=0.85,
    min_constraint_pass_rate=0.90,
    min_per_constraint_kind_rate=0.80,
    max_invalid_rate=0.0,
    reasoning=(
        "The suite is an authored capability screen, not a public benchmark score.",
        "A strict item passes only when every constraint on that item passes.",
        "Every constraint kind has its own floor so a strong category cannot hide a broken one.",
        "An endpoint or checker failure is invalid evidence and cannot enter a score denominator.",
    ),
)


TOOL_CALLING_THRESHOLDS_V1 = ToolCallingThresholds(
    name="kimi-linear-authored-tool-calling-screen-v1",
    min_call_produced_rate=0.95,
    min_offered_tool_name_rate=1.0,
    min_arguments_json_parseable_rate=1.0,
    min_required_argument_keys_rate=1.0,
    min_schema_valid_rate=0.95,
    min_expected_tool_selected_rate=0.95,
    min_correct_abstention_rate=1.0,
    max_invalid_rate=0.0,
    reasoning=(
        "One missed call is tolerated in the twenty positive cases, but systematic under-calling fails.",
        "Unknown tool names, malformed JSON, and missing required keys are never executable by a client.",
        "Schema and semantic selection allow one miss while still exposing the exact failed case.",
        "Every no-tool case must abstain because over-calling can trigger unintended external actions.",
        "Transport or response-shape failures remain invalid evidence rather than being scored as passes.",
    ),
)


__all__ = [
    "INSTRUCTION_THRESHOLDS_V1",
    "TOOL_CALLING_THRESHOLDS_V1",
    "InstructionThresholds",
    "ToolCallingThresholds",
]
