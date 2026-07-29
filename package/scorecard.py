"""Assemble the Kimi-Linear package scorecard from evidence on disk.

Published measurements come only from committed machine-readable JSON. The
companion Markdown remains the human view and is used only as a consistency
check. A Markdown reflow cannot change a value. A value disagreement refuses
assembly.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
EVIDENCE_SCHEMA_VERSION = 1
DEFAULT_SLUG = "kimi-linear-48b-a3b-int4-runinfra"
PERFORMANCE_DIMENSIONS = ("latency", "throughput", "tokensPerSec")
EVIDENCE_STATUSES = {"MEASURED", "SOURCE_DERIVED", "PROJECTED"}
EVIDENCE_BASIS = {
    "MEASURED": "measured",
    "SOURCE_DERIVED": "source_derived",
    "PROJECTED": "projected",
}


class EvidenceError(ValueError):
    """Raised when a scorecard statement cannot be proved from a file."""


@dataclass(frozen=True)
class EvidencePaths:
    quantization: Path
    serving: Path
    accuracy: Path
    residency: Path
    licence: Path
    shared_accuracy: Path | None = None

    @classmethod
    def from_root(cls, root: Path) -> EvidencePaths:
        return cls(
            quantization=root / "engine" / "quant" / "quantization-results.json",
            serving=root / "engine" / "klinear" / "int4-serving-results.json",
            accuracy=root / "engine" / "accuracy" / "results.json",
            residency=root / "engine" / "residency" / "results.json",
            licence=root / "LICENCE-DECISION.md",
            shared_accuracy=(
                root / "engine" / "accuracy" / "shared-expert-experiment.json"
            ),
        )


@dataclass(frozen=True)
class EvidenceDocument:
    path: Path
    reference: str
    text: str


@dataclass(frozen=True)
class MachineEvidence:
    path: Path
    reference: str
    data: dict[str, Any]
    human_path: Path
    human_reference: str
    human_text: str | None


def _reference(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _read_document(path: Path, root: Path) -> EvidenceDocument:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise EvidenceError(f"required evidence file is missing: {path}")
    try:
        text = resolved.read_text(encoding="utf-8")
    except OSError as exc:
        raise EvidenceError(f"required evidence file is unreadable: {path}: {exc}") from exc
    return EvidenceDocument(
        path=resolved,
        reference=_reference(resolved, root),
        text=text,
    )


def _normalize_human_text(text: str) -> str:
    cleaned = text.replace("**", "").replace("`", "")
    cleaned = cleaned.replace("|", " ").replace("\\", " ").replace("#", " ")
    return " ".join(cleaned.split()).casefold()


def _decimal_from_display(display: str) -> Decimal:
    match = re.search(r"[-+]?\d[\d,]*(?:\.\d+)?", display)
    if match is None:
        raise EvidenceError(f"evidence display is not numeric: {display!r}")
    try:
        return Decimal(match.group(0).replace(",", ""))
    except InvalidOperation as exc:
        raise EvidenceError(f"evidence display is not numeric: {display!r}") from exc


def _assert_display_matches_value(
    *,
    value: Any,
    display: str,
    description: str,
    decimal_places: int | None = None,
) -> None:
    if isinstance(value, bool):
        if display.casefold() != str(value).casefold():
            raise EvidenceError(f"{description} value disagrees with its display")
        return
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        displayed = _decimal_from_display(display)
        expected = Decimal(str(value))
        if decimal_places is not None:
            quantum = Decimal(1).scaleb(-decimal_places)
            expected = expected.quantize(quantum, rounding=ROUND_HALF_UP)
        if displayed != expected:
            raise EvidenceError(f"{description} value disagrees with its display")
        return
    if str(value) != display:
        raise EvidenceError(f"{description} value disagrees with its display")


def _validate_machine_evidence(evidence: MachineEvidence) -> None:
    data = evidence.data
    if data.get("schemaVersion") != EVIDENCE_SCHEMA_VERSION:
        raise EvidenceError(
            f"machine evidence {evidence.reference} schemaVersion must be "
            f"{EVIDENCE_SCHEMA_VERSION}"
        )
    if not isinstance(data.get("evidenceType"), str) or not data["evidenceType"]:
        raise EvidenceError(f"machine evidence {evidence.reference} has no evidenceType")
    if not isinstance(data.get("runIdentity"), dict) or not data["runIdentity"]:
        raise EvidenceError(f"machine evidence {evidence.reference} has no run identity")

    hardware = data.get("hardware")
    if not isinstance(hardware, dict) or not hardware:
        raise EvidenceError(f"machine evidence {evidence.reference} has no hardware")
    for scope, environment in hardware.items():
        if not isinstance(environment, dict):
            raise EvidenceError(
                f"machine evidence {evidence.reference} hardware {scope} is not an object"
            )
        if not isinstance(environment.get("accelerator"), str) or not environment["accelerator"]:
            raise EvidenceError(
                f"machine evidence {evidence.reference} hardware {scope} has no accelerator"
            )
        count = environment.get("acceleratorCount")
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise EvidenceError(
                f"machine evidence {evidence.reference} hardware {scope} has invalid count"
            )
        if environment.get("evidenceStatus") not in EVIDENCE_STATUSES:
            raise EvidenceError(
                f"machine evidence {evidence.reference} hardware {scope} has invalid "
                "evidenceStatus"
            )

    values = data.get("values")
    if not isinstance(values, dict) or not values:
        raise EvidenceError(f"machine evidence {evidence.reference} has no values")
    normalized_human = (
        _normalize_human_text(evidence.human_text)
        if evidence.human_text is not None
        else None
    )
    document_checks = data.get("documentChecks")
    if not isinstance(document_checks, list) or not document_checks:
        raise EvidenceError(
            f"machine evidence {evidence.reference} has no run and hardware "
            "document checks"
        )
    if normalized_human is not None:
        for fragment in document_checks:
            if not isinstance(fragment, str) or not fragment:
                raise EvidenceError(
                    f"machine evidence {evidence.reference} has an invalid document check"
                )
            if _normalize_human_text(fragment) not in normalized_human:
                raise EvidenceError(
                    f"machine evidence {evidence.reference} disagrees with human Markdown "
                    f"for run identity or hardware: {fragment!r} is absent"
                )
    for name, fact in values.items():
        if not isinstance(fact, dict):
            raise EvidenceError(
                f"machine evidence {evidence.reference} value {name} is not an object"
            )
        if "value" not in fact:
            raise EvidenceError(
                f"machine evidence {evidence.reference} value {name} is missing value"
            )
        display = fact.get("displayValue")
        if not isinstance(display, str) or not display:
            raise EvidenceError(
                f"machine evidence {evidence.reference} value {name} has no displayValue"
            )
        if not isinstance(fact.get("unit"), str) or not fact["unit"]:
            raise EvidenceError(
                f"machine evidence {evidence.reference} value {name} has no unit"
            )
        status = fact.get("evidenceStatus")
        if status not in EVIDENCE_STATUSES:
            raise EvidenceError(
                f"machine evidence {evidence.reference} value {name} has invalid "
                "evidenceStatus"
            )
        _assert_display_matches_value(
            value=fact["value"],
            display=display,
            description=f"machine evidence {evidence.reference} value {name}",
        )

        if normalized_human is None:
            continue
        markdown = fact.get("markdown")
        if not isinstance(markdown, dict):
            raise EvidenceError(
                f"machine evidence {evidence.reference} value {name} has no Markdown "
                "cross-check"
            )
        contains = markdown.get("contains")
        if not isinstance(contains, list) or not contains:
            raise EvidenceError(
                f"machine evidence {evidence.reference} value {name} has an invalid "
                "Markdown cross-check"
            )
        for fragment in contains:
            if not isinstance(fragment, str) or not fragment:
                raise EvidenceError(
                    f"machine evidence {evidence.reference} value {name} has an invalid "
                    "Markdown fragment"
                )
            if _normalize_human_text(fragment) not in normalized_human:
                raise EvidenceError(
                    f"machine evidence {evidence.reference} disagrees with human Markdown "
                    f"for {name}: {fragment!r} is absent"
                )
        markdown_display = markdown.get("displayValue")
        if markdown_display is not None:
            if not isinstance(markdown_display, str) or not markdown_display:
                raise EvidenceError(
                    f"machine evidence {evidence.reference} value {name} has invalid "
                    "Markdown displayValue"
                )
            places = markdown.get("decimalPlaces")
            if places is not None and (
                isinstance(places, bool) or not isinstance(places, int) or places < 0
            ):
                raise EvidenceError(
                    f"machine evidence {evidence.reference} value {name} has invalid "
                    "Markdown decimalPlaces"
                )
            _assert_display_matches_value(
                value=fact["value"],
                display=markdown_display,
                description=(
                    f"machine evidence {evidence.reference} value {name} and human Markdown"
                ),
                decimal_places=places,
            )


def _read_machine_evidence(path: Path, root: Path) -> MachineEvidence:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise EvidenceError(f"required evidence file is missing: {path}")
    try:
        loaded = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvidenceError(
            f"required machine evidence is unreadable or invalid JSON: {path}"
        ) from exc
    if not isinstance(loaded, dict):
        raise EvidenceError(f"machine evidence must be an object: {path}")
    human_name = loaded.get("humanDocument")
    if not isinstance(human_name, str) or not human_name.strip():
        raise EvidenceError(f"machine evidence {path} has no humanDocument")
    human_path = (resolved.parent / human_name).resolve()
    human_text: str | None = None
    if human_path.is_file():
        try:
            human_text = human_path.read_text(encoding="utf-8")
        except OSError as exc:
            raise EvidenceError(
                f"human evidence document is unreadable: {human_path}: {exc}"
            ) from exc
    evidence = MachineEvidence(
        path=resolved,
        reference=_reference(resolved, root),
        data=loaded,
        human_path=human_path,
        human_reference=_reference(human_path, root),
        human_text=human_text,
    )
    _validate_machine_evidence(evidence)
    return evidence


def _fact(evidence: MachineEvidence, name: str) -> dict[str, Any]:
    fact = evidence.data["values"].get(name)
    if not isinstance(fact, dict):
        raise EvidenceError(f"machine evidence {evidence.reference} is missing value {name}")
    return fact


def _hardware(evidence: MachineEvidence, scope: str) -> dict[str, Any]:
    environment = evidence.data["hardware"].get(scope)
    if not isinstance(environment, dict):
        raise EvidenceError(
            f"machine evidence {evidence.reference} is missing hardware scope {scope}"
        )
    return {
        "accelerator": environment["accelerator"],
        "acceleratorCount": environment["acceleratorCount"],
    }


def _evidence(evidence: MachineEvidence, selector: str) -> dict[str, str]:
    return {
        "path": evidence.reference,
        "artifact": "machine_readable_evidence_v1",
        "selector": selector,
    }


def _human_crosscheck(evidence: MachineEvidence) -> dict[str, str]:
    return {
        "status": "VERIFIED" if evidence.human_text is not None else "DOCUMENT_MISSING",
        "path": evidence.human_reference,
    }


def _claim(
    *,
    claim_id: str,
    kind: str,
    label: str,
    fact: dict[str, Any],
    hardware: dict[str, Any],
    conditions: str,
    evidence: dict[str, str],
) -> dict[str, Any]:
    status = fact["evidenceStatus"]
    return {
        "id": claim_id,
        "kind": kind,
        "label": label,
        "basis": EVIDENCE_BASIS[status],
        "evidenceStatus": status,
        "value": fact["value"],
        "displayValue": fact["displayValue"],
        "unit": fact["unit"],
        "hardware": hardware,
        "conditions": conditions,
        "evidence": evidence,
    }


def _not_measured(note: str) -> dict[str, str]:
    return {"status": "NOT_MEASURED", "note": note}


def _quantization_claims(
    evidence: MachineEvidence,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    hardware = _hardware(evidence, "artifact")
    conditions = evidence.data["conditions"]
    source = _fact(evidence, "sourceTensorBytes")
    output = _fact(evidence, "outputTensorBytes")
    reduction = _fact(evidence, "weightByteReduction")
    planned = _fact(evidence, "plannedTensorBytes")
    quantized = _fact(evidence, "quantizedTensorCount")
    retained = _fact(evidence, "retainedTensorCount")
    if planned["value"] != output["value"]:
        raise EvidenceError("quantization planned bytes do not equal measured output bytes")

    claims = [
        _claim(
            claim_id="footprint.source_tensor_bytes",
            kind="footprint",
            label="Source checkpoint tensor storage",
            fact=source,
            hardware=hardware,
            conditions=conditions,
            evidence=_evidence(evidence, "$.values.sourceTensorBytes"),
        ),
        _claim(
            claim_id="footprint.int4_tensor_bytes",
            kind="footprint",
            label="Selective INT4 tensor storage",
            fact=output,
            hardware=hardware,
            conditions=conditions,
            evidence=_evidence(evidence, "$.values.outputTensorBytes"),
        ),
        _claim(
            claim_id="footprint.weight_byte_reduction",
            kind="footprint",
            label="Weight byte reduction",
            fact=reduction,
            hardware=hardware,
            conditions=conditions,
            evidence=_evidence(evidence, "$.values.weightByteReduction"),
        ),
        _claim(
            claim_id="coverage.quantized_tensors",
            kind="coverage",
            label="Tensors quantized",
            fact=quantized,
            hardware=hardware,
            conditions=conditions,
            evidence=_evidence(evidence, "$.values.quantizedTensorCount"),
        ),
        _claim(
            claim_id="coverage.retained_tensors",
            kind="coverage",
            label="Tensors retained in source precision",
            fact=retained,
            hardware=hardware,
            conditions=conditions,
            evidence=_evidence(evidence, "$.values.retainedTensorCount"),
        ),
    ]
    controls = []
    for name, decoder in (
        ("wrongGroupAxisDecoder", "wrong_group_axis"),
        ("wrongScaleDecoder", "wrong_scale"),
        ("swappedNibblesDecoder", "swapped_nibbles"),
    ):
        result = _fact(evidence, name)
        if not str(result["value"]).startswith("REJECTED"):
            raise EvidenceError(f"negative control {decoder} was not rejected")
        controls.append(
            {
                "decoder": decoder,
                "result": result["value"],
                "evidenceStatus": result["evidenceStatus"],
                "evidence": _evidence(evidence, f"$.values.{name}"),
            }
        )
    summary = {
        "status": "MEASURED",
        "hardware": hardware,
        "conditions": conditions,
        "plannedBytesEqualActual": True,
        "negativeControls": controls,
        "humanCrossCheck": _human_crosscheck(evidence),
        "evidence": _evidence(evidence, "$.values"),
    }
    return claims, summary


def _serving_claims(
    evidence: MachineEvidence,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    hardware = _hardware(evidence, "serving")
    conditions = evidence.data["conditions"]
    resident = _fact(evidence, "residentWeightBytes")
    checkpoint = _fact(evidence, "checkpointTensorBytes")
    peak = _fact(evidence, "peakReservedBytes")
    if resident["value"] != checkpoint["value"]:
        raise EvidenceError("resident weight bytes do not equal checkpoint tensor storage")
    speed = _fact(evidence, "speedComparisonAgainstVllm")
    if speed["value"] != "NOT_MEASURED":
        raise EvidenceError("serving evidence attempts to introduce a vLLM speed comparison")
    claims = [
        _claim(
            claim_id="serving.resident_weight_bytes",
            kind="footprint",
            label="Resident weight bytes",
            fact=resident,
            hardware=hardware,
            conditions=conditions,
            evidence=_evidence(evidence, "$.values.residentWeightBytes"),
        ),
        _claim(
            claim_id="serving.peak_reserved_bytes",
            kind="footprint",
            label="Peak reserved device memory after generation",
            fact=peak,
            hardware=hardware,
            conditions=conditions,
            evidence=_evidence(evidence, "$.values.peakReservedBytes"),
        ),
    ]
    summary = {
        "status": "MEASURED",
        "hardware": hardware,
        "conditions": conditions,
        "residentMatchesCheckpoint": True,
        "coherentOutputObserved": _fact(evidence, "coherentOutputObserved")["value"],
        "layerCount": _fact(evidence, "layerCount")["value"],
        "expertsPerMoeLayer": _fact(evidence, "expertsPerMoeLayer")["value"],
        "speedComparisonAgainstVllm": speed["value"],
        "humanCrossCheck": _human_crosscheck(evidence),
        "evidence": _evidence(evidence, "$.values"),
    }
    return claims, summary


ACCURACY_METRICS = (
    ("perplexityBaseline", "perplexity_baseline", "Teacher-forced perplexity, BF16"),
    (
        "perplexityCandidate",
        "perplexity_candidate",
        "Teacher-forced perplexity, candidate",
    ),
    ("perplexityIncreasePercent", "perplexity_increase", "Perplexity increase"),
    ("top1AgreementPercent", "top1_agreement", "Next-token top-1 agreement"),
    ("meanKlNats", "mean_kl", "Mean KL(BF16 || candidate), exact full vocabulary"),
    ("greedyIdentityPercent", "greedy_identity", "Greedy output identity"),
    ("routerSetAgreementPercent", "router_set_agreement", "Router set agreement"),
)


def _accuracy_claims(
    evidence: MachineEvidence,
    *,
    profile: str,
    claim_prefix: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    hardware = _hardware(evidence, "accuracy")
    conditions = evidence.data["conditions"]
    claims = []
    for fact_name, suffix, label in ACCURACY_METRICS:
        fact = _fact(evidence, fact_name)
        rendered_label = label if profile == "default-int4" else f"{profile}: {label}"
        claims.append(
            _claim(
                claim_id=f"{claim_prefix}.{suffix}",
                kind="quality",
                label=rendered_label,
                fact=fact,
                hardware=hardware,
                conditions=conditions,
                evidence=_evidence(evidence, f"$.values.{fact_name}"),
            )
        )
    verdict = _fact(evidence, "verdict")
    summary = {
        "status": "MEASURED",
        "profile": profile,
        "verdict": verdict["value"],
        "verdictText": f"Verdict: {verdict['value']}",
        "hardware": hardware,
        "conditions": conditions,
        "claimIds": [claim["id"] for claim in claims],
        "humanCrossCheck": _human_crosscheck(evidence),
        "evidence": _evidence(evidence, "$.values"),
    }
    return claims, summary


def _shared_experiment_claims(
    evidence: MachineEvidence,
    *,
    default_output_bytes: int,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    artifact_hardware = _hardware(evidence, "artifact")
    conditions = evidence.data["conditions"]
    output = _fact(evidence, "outputTensorBytes")
    planned = _fact(evidence, "plannedTensorBytes")
    additional = _fact(evidence, "additionalBytesVersusDefault")
    quantized = _fact(evidence, "quantizedTensorCount")
    retained = _fact(evidence, "retainedTensorCount")
    if output["value"] != planned["value"]:
        raise EvidenceError("shared-expert planned bytes do not equal measured output bytes")
    if output["value"] - default_output_bytes != additional["value"]:
        raise EvidenceError(
            "shared-expert artifact delta disagrees with the default artifact bytes"
        )
    footprint_claims = [
        _claim(
            claim_id="footprint.shared_experts_bf16_tensor_bytes",
            kind="footprint",
            label="Shared-experts-bf16 tensor storage",
            fact=output,
            hardware=artifact_hardware,
            conditions=conditions,
            evidence=_evidence(evidence, "$.values.outputTensorBytes"),
        ),
        _claim(
            claim_id="footprint.shared_experts_bf16_additional_bytes",
            kind="footprint",
            label="Additional bytes versus default selective INT4",
            fact=additional,
            hardware=artifact_hardware,
            conditions=conditions,
            evidence=_evidence(evidence, "$.values.additionalBytesVersusDefault"),
        ),
        _claim(
            claim_id="coverage.shared_experts_bf16_quantized_tensors",
            kind="coverage",
            label="Shared-experts-bf16 tensors quantized",
            fact=quantized,
            hardware=artifact_hardware,
            conditions=conditions,
            evidence=_evidence(evidence, "$.values.quantizedTensorCount"),
        ),
        _claim(
            claim_id="coverage.shared_experts_bf16_retained_tensors",
            kind="coverage",
            label="Shared-experts-bf16 tensors retained in source precision",
            fact=retained,
            hardware=artifact_hardware,
            conditions=conditions,
            evidence=_evidence(evidence, "$.values.retainedTensorCount"),
        ),
    ]
    accuracy_claims, accuracy = _accuracy_claims(
        evidence,
        profile="shared-experts-bf16",
        claim_prefix="quality.shared_experts_bf16",
    )
    summary = {
        "status": "MEASURED",
        "profile": "shared-experts-bf16",
        "hardware": artifact_hardware,
        "conditions": conditions,
        "plannedBytesEqualActual": True,
        "additionalBytesVersusDefault": additional["value"],
        "humanCrossCheck": _human_crosscheck(evidence),
        "evidence": _evidence(evidence, "$.values"),
    }
    return [*footprint_claims, *accuracy_claims], summary, accuracy


def _residency_claims(
    evidence: MachineEvidence,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    validation_hardware = _hardware(evidence, "validation")
    conditions = evidence.data["conditions"]
    validation_points = []
    for index in (1, 2, 3):
        predicted = _fact(evidence, f"point{index}PredictedStateBytes")
        allocated = _fact(evidence, f"point{index}MeasuredAllocatedBytes")
        difference = _fact(evidence, f"point{index}AllocatedMinusPredictedBytes")
        if predicted["value"] != allocated["value"] or difference["value"] != 0:
            raise EvidenceError(f"residency validation point {index} is not an exact match")
        validation_points.append(
            {
                "predictedBytes": predicted["value"],
                "allocatedBytes": allocated["value"],
                "allocatedMinusPredictedBytes": difference["value"],
            }
        )

    capacity = _fact(evidence, "envelopeCapacityGiB")
    max_seqs = _fact(evidence, "envelopeMaxNumSeqs")
    max_len = _fact(evidence, "envelopeMaxModelLen")
    claim = _claim(
        claim_id="residency.max_num_seqs",
        kind="residency",
        label=f"Maximum sequences inside a {capacity['displayValue']} GiB envelope",
        fact=max_seqs,
        hardware={
            "accelerator": f"{capacity['displayValue']} GiB compatible GPU",
            "acceleratorCount": 1,
        },
        conditions=(
            f"{conditions}; max_model_len {max_len['displayValue']}"
        ),
        evidence=_evidence(evidence, "$.values.envelopeMaxNumSeqs"),
    )
    summary = {
        "status": "PROJECTED",
        "validationHardware": validation_hardware,
        "sourceRevision": evidence.data["runIdentity"]["modelRevision"],
        "conditions": conditions,
        "claimIds": [claim["id"]],
        "statePoolValidation": validation_points,
        "weightEvidenceStatus": _fact(evidence, "envelopeWeightBytes")[
            "evidenceStatus"
        ],
        "stateEvidenceStatus": "SOURCE_DERIVED",
        "reserveEvidenceStatus": _fact(evidence, "operationalReserveGiB")[
            "evidenceStatus"
        ],
        "humanCrossCheck": _human_crosscheck(evidence),
        "evidence": _evidence(evidence, "$.values"),
    }
    return [claim], summary


def _clean_cell(value: str) -> str:
    return value.strip().replace("**", "").replace("`", "").replace(r"\|", "|")


def _cells(line: str) -> list[str]:
    return [
        _clean_cell(cell)
        for cell in re.split(r"(?<!\\)\|", line.strip().strip("|"))
    ]


def _is_divider(line: str) -> bool:
    cells = _cells(line)
    return bool(cells) and all(re.fullmatch(r":?-{3,}:?", cell) for cell in cells)


def _tables(text: str) -> list[tuple[list[str], list[list[str]]]]:
    lines = text.splitlines()
    tables: list[tuple[list[str], list[list[str]]]] = []
    index = 0
    while index + 1 < len(lines):
        if not lines[index].lstrip().startswith("|") or not _is_divider(lines[index + 1]):
            index += 1
            continue
        headers = _cells(lines[index])
        rows: list[list[str]] = []
        index += 2
        while index < len(lines) and lines[index].lstrip().startswith("|"):
            rows.append(_cells(lines[index]))
            index += 1
        tables.append((headers, rows))
    return tables


def _table_value(text: str, row_label: str) -> str:
    wanted = row_label.casefold()
    for _, rows in _tables(text):
        for row in rows:
            if row and row[0].casefold() == wanted and len(row) > 1 and row[1]:
                return row[1]
    raise EvidenceError(f"licence evidence table is missing row {row_label!r}")


def _match(text: str, pattern: str, description: str) -> re.Match[str]:
    match = re.search(pattern, text)
    if match is None:
        raise EvidenceError(f"evidence is missing {description}")
    return match


def _licence_summary(document: EvidenceDocument) -> dict[str, Any]:
    location = _table_value(document.text, "Location")
    revision = _match(document.text, r"commit `([0-9a-f]{40})`", "licence commit").group(1)
    spdx = _table_value(document.text, "GitHub API license.spdx_id")
    accepted = "to **accept the\ncompanion-repository instrument as sufficient**" in document.text
    if not accepted:
        raise EvidenceError("licence evidence does not contain the founder acceptance decision")
    return {
        "status": "FOUNDER_ACCEPTED_COMPANION_INSTRUMENT",
        "location": location,
        "revision": revision,
        "spdxId": spdx,
        "evidence": {
            "path": document.reference,
            "artifact": "structural_markdown",
            "selector": "sections:The instrument, The decision",
        },
    }


def _resolve_reference(reference: dict[str, Any], evidence_root: Path) -> Path:
    if not isinstance(reference, dict):
        raise EvidenceError("claim has no evidence reference")
    raw = reference.get("path")
    if not isinstance(raw, str) or not raw.strip():
        raise EvidenceError("claim has no evidence path")
    selector = reference.get("selector")
    if not isinstance(selector, str) or not selector.strip():
        raise EvidenceError("claim has no evidence selector")
    artifact = reference.get("artifact")
    if not isinstance(artifact, str) or not artifact.strip():
        raise EvidenceError("claim has no evidence artifact type")
    path = Path(raw)
    return path if path.is_absolute() else evidence_root / path


def _validate_performance_dimension(
    name: str, dimension: dict[str, Any], evidence_root: Path
) -> None:
    status = dimension.get("status")
    if status == "NOT_MEASURED":
        forbidden = {"baseline", "optimized", "speedup", "value"} & set(dimension)
        if forbidden:
            raise EvidenceError(
                f"{name} is NOT_MEASURED but carries performance figures: {sorted(forbidden)}"
            )
        return
    if status != "MEASURED":
        raise EvidenceError(f"{name} has unsupported status {status!r}")
    for field in ("baseline", "optimized", "unit", "hardware", "concurrency", "conditions"):
        if dimension.get(field) in (None, "", {}):
            raise EvidenceError(f"measured {name} is missing {field}")
    concurrency = dimension["concurrency"]
    if isinstance(concurrency, bool) or not isinstance(concurrency, int) or concurrency < 1:
        raise EvidenceError(f"measured {name} concurrency must be a positive integer")
    reference = dimension.get("evidence")
    if not isinstance(reference, dict):
        raise EvidenceError(f"measured {name} has no evidence reference")
    if reference.get("artifact") not in {
        "factory_scorecard_v1",
        "factory_benchmark_receipt_v1",
    }:
        raise EvidenceError(
            f"measured {name} requires factory performance evidence, not "
            f"{reference.get('artifact')!r}"
        )
    if not _resolve_reference(reference, evidence_root).is_file():
        raise EvidenceError(f"measured {name} evidence file does not exist")


def _validate_accuracy_summary(
    name: str,
    accuracy: dict[str, Any],
    evidence_root: Path,
) -> None:
    if accuracy.get("status") != "MEASURED":
        raise EvidenceError(f"accuracy profile {name} is not MEASURED")
    verdict = accuracy.get("verdict")
    verdict_text = accuracy.get("verdictText")
    if not isinstance(verdict, str) or not verdict.strip():
        if name == "default-int4":
            raise EvidenceError("accuracy verdict cannot be omitted")
        raise EvidenceError(f"accuracy profile {name} verdict cannot be omitted")
    if not isinstance(verdict_text, str) or verdict not in verdict_text:
        raise EvidenceError(
            f"accuracy profile {name} verdict text must carry the verdict verbatim"
        )
    if not _resolve_reference(accuracy.get("evidence"), evidence_root).is_file():
        raise EvidenceError(f"accuracy profile {name} evidence file does not exist")


def validate_scorecard(card: dict[str, Any], evidence_root: Path | None = None) -> None:
    root = (evidence_root or Path(__file__).resolve().parents[1]).resolve()
    if card.get("schemaVersion") != SCHEMA_VERSION:
        raise EvidenceError(f"scorecard schemaVersion must be {SCHEMA_VERSION}")
    for name in PERFORMANCE_DIMENSIONS:
        dimension = card.get(name)
        if not isinstance(dimension, dict):
            raise EvidenceError(f"scorecard is missing {name}")
        _validate_performance_dimension(name, dimension, root)
    measured_concurrency = {
        card[name]["concurrency"]
        for name in PERFORMANCE_DIMENSIONS
        if card[name]["status"] == "MEASURED"
    }
    if len(measured_concurrency) > 1:
        raise EvidenceError("performance dimensions use mismatched concurrency")
    expected_concurrency = next(iter(measured_concurrency), None)
    if card.get("concurrency") != expected_concurrency:
        raise EvidenceError("scorecard concurrency does not match its measured dimensions")

    stability = card.get("stability")
    if not isinstance(stability, dict):
        raise EvidenceError("scorecard is missing stability")
    if stability.get("status") == "NOT_MEASURED":
        if any(key in stability for key in ("baselineFailures", "optimizedFailures")):
            raise EvidenceError("NOT_MEASURED stability carries failure figures")
    elif stability.get("status") == "MEASURED":
        for field in (
            "baselineFailures",
            "optimizedFailures",
            "hardware",
            "concurrency",
            "conditions",
            "evidence",
        ):
            if stability.get(field) in (None, "", {}):
                raise EvidenceError(f"measured stability is missing {field}")
        if not _resolve_reference(stability["evidence"], root).is_file():
            raise EvidenceError("measured stability evidence file does not exist")
    else:
        raise EvidenceError(f"stability has unsupported status {stability.get('status')!r}")

    profiles = card.get("accuracyProfiles")
    if not isinstance(profiles, dict) or set(profiles) != {
        "default-int4",
        "shared-experts-bf16",
    }:
        raise EvidenceError("scorecard must carry both accuracy profiles")
    for name, accuracy in profiles.items():
        if not isinstance(accuracy, dict):
            raise EvidenceError(f"accuracy profile {name} is not an object")
        _validate_accuracy_summary(name, accuracy, root)
        if accuracy["verdict"].upper() != "FAIL":
            raise EvidenceError(
                f"accuracy profile {name} must preserve the measured FAIL verdict"
            )
    if card.get("accuracyRetained") != profiles["default-int4"]:
        raise EvidenceError("accuracyRetained must be the default-int4 profile")

    capability = card.get("capability")
    if not isinstance(capability, dict):
        raise EvidenceError("scorecard is missing the capability result")
    if capability.get("stockVllmBelowBf16") != "REFUSED":
        raise EvidenceError("capability result does not preserve the stock vLLM refusal")
    if capability.get("stockRuntime") != "vLLM 0.26.0":
        raise EvidenceError("capability result does not preserve the measured vLLM version")
    if capability.get("stockFits32GbClassCard") is not False:
        raise EvidenceError("capability result hides the stock BF16 weight-floor limit")
    if capability.get("speedComparisonAgainstVllm") != "NOT_MEASURED":
        raise EvidenceError("capability result attempts to claim a vLLM speed comparison")
    if capability.get("fasterThanVllm") is not None:
        raise EvidenceError("capability result must not imply that this engine is faster")
    references = capability.get("evidence")
    if not isinstance(references, list) or not references:
        raise EvidenceError("capability result has no evidence")
    for reference in references:
        if not _resolve_reference(reference, root).is_file():
            raise EvidenceError("capability evidence file does not exist")

    claims = card.get("claims")
    if not isinstance(claims, list) or not claims:
        raise EvidenceError("scorecard has no evidence-backed claims")
    seen: set[str] = set()
    for claim in claims:
        if not isinstance(claim, dict):
            raise EvidenceError("scorecard claim is not an object")
        claim_id = claim.get("id")
        if not isinstance(claim_id, str) or not claim_id or claim_id in seen:
            raise EvidenceError("scorecard claim ids must be non-empty and unique")
        seen.add(claim_id)
        for field in (
            "kind",
            "label",
            "basis",
            "evidenceStatus",
            "value",
            "displayValue",
            "unit",
            "hardware",
            "conditions",
        ):
            if claim.get(field) in (None, "", {}):
                raise EvidenceError(f"claim {claim_id} is missing {field}")
        if claim["evidenceStatus"] not in EVIDENCE_STATUSES:
            raise EvidenceError(f"claim {claim_id} has invalid evidenceStatus")
        reference = claim.get("evidence")
        path = _resolve_reference(reference, root)
        if not path.is_file():
            raise EvidenceError(f"claim {claim_id} evidence file does not exist: {path}")

    overall = card.get("overall")
    if not isinstance(overall, dict):
        raise EvidenceError("scorecard has no overall verdict")
    accuracy_verdicts = overall.get("accuracyVerdicts")
    if not isinstance(accuracy_verdicts, dict):
        raise EvidenceError("overall verdict omits accuracy profile verdicts")
    for name, profile in profiles.items():
        if profile["verdict"].upper() == "FAIL" and accuracy_verdicts.get(name) != "FAIL":
            raise EvidenceError(f"overall verdict hides the measured {name} accuracy FAIL")


def assemble_scorecard(
    evidence_root: Path | None = None,
    *,
    paths: EvidencePaths | None = None,
    slug: str = DEFAULT_SLUG,
) -> dict[str, Any]:
    root = (evidence_root or Path(__file__).resolve().parents[1]).resolve()
    evidence_paths = paths or EvidencePaths.from_root(root)
    shared_path = evidence_paths.shared_accuracy or (
        root / "engine" / "accuracy" / "shared-expert-experiment.json"
    )
    documents = {
        "quantization": _read_machine_evidence(evidence_paths.quantization, root),
        "serving": _read_machine_evidence(evidence_paths.serving, root),
        "accuracy": _read_machine_evidence(evidence_paths.accuracy, root),
        "shared_accuracy": _read_machine_evidence(shared_path, root),
        "residency": _read_machine_evidence(evidence_paths.residency, root),
        "licence": _read_document(evidence_paths.licence, root),
    }

    quant_claims, quantization = _quantization_claims(documents["quantization"])
    serving_claims, serving = _serving_claims(documents["serving"])
    default_accuracy_claims, default_accuracy = _accuracy_claims(
        documents["accuracy"],
        profile="default-int4",
        claim_prefix="quality",
    )
    default_output = _fact(documents["quantization"], "outputTensorBytes")["value"]
    shared_claims, shared_experiment, shared_accuracy = _shared_experiment_claims(
        documents["shared_accuracy"],
        default_output_bytes=default_output,
    )
    residency_claims, residency = _residency_claims(documents["residency"])
    licence = _licence_summary(documents["licence"])

    source_bytes = _fact(documents["quantization"], "sourceTensorBytes")["value"]
    resident_bytes = _fact(documents["serving"], "residentWeightBytes")["value"]
    checkpoint_bytes = _fact(documents["serving"], "checkpointTensorBytes")["value"]
    stock_floor = _fact(documents["serving"], "stockVllmWeightFloorBytes")["value"]
    if default_output != resident_bytes or resident_bytes != checkpoint_bytes:
        raise EvidenceError("quantization and serving JSON disagree on INT4 weight bytes")
    if source_bytes != stock_floor:
        raise EvidenceError("quantization and serving JSON disagree on the BF16 weight floor")
    default_baseline = _fact(documents["accuracy"], "perplexityBaseline")["value"]
    shared_baseline = _fact(
        documents["shared_accuracy"], "perplexityBaseline"
    )["value"]
    if default_baseline != shared_baseline:
        raise EvidenceError("accuracy JSON files disagree on the BF16 baseline perplexity")

    claims = [
        *quant_claims,
        *serving_claims,
        *default_accuracy_claims,
        *shared_claims,
        *residency_claims,
    ]
    performance_absence = (
        "No speed comparison against vLLM has been measured, and none is claimed. "
        "Footprint and capability evidence must not be converted into throughput."
    )
    stock_version = _fact(documents["serving"], "stockVllmVersion")["value"]
    stock_display = _fact(
        documents["serving"], "stockVllmWeightFloorBytes"
    )["displayValue"]
    optimized_display = _fact(
        documents["quantization"], "outputTensorBytes"
    )["displayValue"]
    capability = {
        "status": "MEASURED_ENABLEMENT",
        "stockRuntime": f"vLLM {stock_version}",
        "stockVllmBelowBf16": _fact(
            documents["serving"], "stockVllmBelowBf16"
        )["value"],
        "stockWeightFloorBytes": stock_floor,
        "optimizedWeightBytes": default_output,
        "optimizedEngineLoadsAndRuns": _fact(
            documents["serving"], "coherentOutputObserved"
        )["value"],
        "stockFits32GbClassCard": stock_floor <= 32 * 1024**3,
        "speedComparisonAgainstVllm": "NOT_MEASURED",
        "fasterThanVllm": None,
        "note": (
            f"Stock vLLM {stock_version} refuses this model below BF16, so its weight "
            f"floor is {stock_display} bytes and a 32 GB-class card cannot hold the "
            f"model at any speed. The {optimized_display}-byte INT4 artifact loads and "
            "runs in this engine. "
            "This is a binary capability result, not a performance comparison."
        ),
        "evidence": [
            _evidence(documents["quantization"], "$.values.sourceTensorBytes"),
            _evidence(documents["serving"], "$.values.stockVllmBelowBf16"),
        ],
    }
    profiles = {
        "default-int4": default_accuracy,
        "shared-experts-bf16": shared_accuracy,
    }
    verdicts = {name: profile["verdict"] for name, profile in profiles.items()}
    card = {
        "schemaVersion": SCHEMA_VERSION,
        "slug": slug,
        "concurrency": None,
        "latency": _not_measured(performance_absence),
        "throughput": _not_measured(performance_absence),
        "tokensPerSec": _not_measured(performance_absence),
        "accuracyRetained": default_accuracy,
        "accuracyProfiles": profiles,
        "stability": _not_measured("No release soak or stability receipt is present on disk."),
        "footprint": quantization,
        "sharedExpertExperiment": shared_experiment,
        "serving": serving,
        "capability": capability,
        "residency": residency,
        "licence": licence,
        "engineCorrectnessAgainstReference": _not_measured(
            "This engine has not been compared against a reference implementation."
        ),
        "claims": claims,
        "overall": {
            "verdict": "BLOCKED_ACCURACY_FAIL",
            "accuracyVerdict": default_accuracy["verdict"],
            "accuracyVerdicts": verdicts,
            "note": (
                "Both measured accuracy profiles say FAIL. Neither artifact may be "
                "described as behaviourally equivalent to the BF16 checkpoint. "
                "Performance, stability, and engine correctness evidence are also "
                "absent. This is not listable."
            ),
        },
    }
    validate_scorecard(card, root)
    return card


def _hardware_inline(hardware: dict[str, Any]) -> str:
    return f"{hardware['acceleratorCount']}x {hardware['accelerator']}"


def _claim_value_inline(claim: dict[str, Any]) -> str:
    display = str(claim["displayValue"])
    unit = str(claim["unit"])
    if unit == "%" and "%" in display:
        return display
    if unit == "x" and re.search(r"x(?:\s|$)", display):
        return display
    if unit.casefold() in display.casefold().split():
        return display
    return f"{display} {unit}"


def render_scorecard(card: dict[str, Any], evidence_root: Path | None = None) -> str:
    validate_scorecard(card, evidence_root)
    lines = [
        f"SCORECARD {card['slug']} (schema v{card['schemaVersion']})",
        "",
    ]
    for name in (*PERFORMANCE_DIMENSIONS, "stability"):
        dimension = card[name]
        if dimension["status"] == "NOT_MEASURED":
            lines.append(f"{name}: NOT_MEASURED. {dimension['note']}")
        elif name == "stability":
            lines.append(
                f"stability: failures {dimension['baselineFailures']} -> "
                f"{dimension['optimizedFailures']} on "
                f"{_hardware_inline(dimension['hardware'])}, concurrency "
                f"{dimension['concurrency']}, conditions {dimension['conditions']}. "
                f"Evidence: {dimension['evidence']['path']}#"
                f"{dimension['evidence']['selector']}."
            )
        else:
            lines.append(
                f"{name}: {dimension['baseline']} -> {dimension['optimized']} "
                f"{dimension['unit']} on {_hardware_inline(dimension['hardware'])}, "
                f"concurrency {dimension['concurrency']}, conditions "
                f"{dimension['conditions']}. Evidence: "
                f"{dimension['evidence']['path']}#{dimension['evidence']['selector']}."
            )

    capability = card["capability"]
    lines.extend(
        [
            "",
            "Capability:",
            capability["note"],
            "No throughput comparison against vLLM has been measured, and none is claimed.",
        ]
    )
    default_accuracy = card["accuracyProfiles"]["default-int4"]
    shared_accuracy = card["accuracyProfiles"]["shared-experts-bf16"]
    lines.extend(
        [
            "",
            f"Accuracy {default_accuracy['verdictText']}",
            f"Profile: {default_accuracy['profile']}.",
            f"Hardware: {_hardware_inline(default_accuracy['hardware'])}.",
            f"Conditions: {default_accuracy['conditions']}.",
            (
                f"Evidence: {default_accuracy['evidence']['path']}#"
                f"{default_accuracy['evidence']['selector']}."
            ),
            "",
            f"Accuracy {shared_accuracy['profile']} {shared_accuracy['verdictText']}",
            f"Hardware: {_hardware_inline(shared_accuracy['hardware'])}.",
            f"Conditions: {shared_accuracy['conditions']}.",
            (
                f"Evidence: {shared_accuracy['evidence']['path']}#"
                f"{shared_accuracy['evidence']['selector']}."
            ),
            "Both accuracy verdicts are FAIL. Behavioural equivalence is not claimed.",
            "",
            "Evidence-backed claims:",
        ]
    )
    for claim in card["claims"]:
        lines.append(
            f"- {claim['label']}: {_claim_value_inline(claim)} on "
            f"{_hardware_inline(claim['hardware'])}. Status: "
            f"{claim['evidenceStatus']}. Conditions: {claim['conditions']}. "
            f"Evidence: {claim['evidence']['path']}#{claim['evidence']['selector']}."
        )
    lines.extend(
        [
            "",
            f"OVERALL: {card['overall']['verdict']}",
            card["overall"]["note"],
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--slug", default=DEFAULT_SLUG)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    try:
        card = assemble_scorecard(args.root, slug=args.slug)
        rendered = render_scorecard(card, args.root)
    except EvidenceError as exc:
        raise SystemExit(f"REFUSED: {exc}") from exc
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(card, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
