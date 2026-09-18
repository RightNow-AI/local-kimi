"""Claims that can only be rendered when their evidence exists on disk."""

from __future__ import annotations

import json
import math
import re
from decimal import Decimal, InvalidOperation
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class ClaimKind(str, Enum):
    PERFORMANCE = "performance"
    FOOTPRINT = "footprint"
    QUALITY = "quality"
    STABILITY = "stability"
    COST = "cost"


class ClaimBasis(str, Enum):
    MEASURED = "measured"
    COMPUTED = "computed"


class Comparison(str, Enum):
    HIGHER_IS_BETTER = "higher_is_better"
    LOWER_IS_BETTER = "lower_is_better"
    EQUAL_IS_REQUIRED = "equal_is_required"


class EvidenceArtifact(str, Enum):
    FACTORY_SCORECARD_V1 = "factory_scorecard_v1"
    FACTORY_BENCHMARK_RECEIPT_V1 = "factory_benchmark_receipt_v1"
    STRUCTURAL_RESULTS = "structural_results"


class RuntimeIdentity(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    name: str = Field(min_length=1)
    version: str = Field(min_length=1)

    def inline(self) -> str:
        return f"{self.name} {self.version}"


class HardwareContext(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    accelerator: str = Field(min_length=1)
    accelerator_count: int = Field(gt=0)
    details: str | None = None

    @field_validator("accelerator_count", mode="before")
    @classmethod
    def validate_accelerator_count(cls, value: Any) -> Any:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("accelerator_count must be an integer")
        return value

    def inline(self) -> str:
        base = f"{self.accelerator_count}x {self.accelerator}"
        return f"{base} ({self.details})" if self.details else base


class EvidenceReference(BaseModel):
    """A selector into an existing factory artifact or structural result file."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    path: Path
    artifact: EvidenceArtifact
    selector: str = Field(min_length=1)
    column: str | None = None

    @model_validator(mode="after")
    def validate_reference(self) -> EvidenceReference:
        resolved = self.path.expanduser().resolve()
        if not resolved.is_file():
            raise ValueError(f"evidence file does not exist: {self.path}")
        if self.artifact is not EvidenceArtifact.STRUCTURAL_RESULTS and self.column:
            raise ValueError("column is only valid for a structural results table")
        object.__setattr__(self, "path", resolved)
        self._validate_artifact_shape()
        return self

    def _load_json(self) -> dict[str, Any]:
        try:
            loaded = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"evidence file is not valid JSON: {self.path}") from exc
        if not isinstance(loaded, dict):
            raise ValueError(f"evidence JSON must be an object: {self.path}")
        return loaded

    def _validate_artifact_shape(self) -> None:
        if self.artifact is EvidenceArtifact.STRUCTURAL_RESULTS:
            return
        document = self._load_json()
        if self.artifact is EvidenceArtifact.FACTORY_SCORECARD_V1:
            if document.get("schemaVersion") != 1:
                raise ValueError("factory scorecard evidence must have schemaVersion 1")
            return
        payload = document.get("payload")
        if not isinstance(payload, dict):
            raise ValueError("benchmark receipt evidence is missing payload")
        if payload.get("receiptVersion") != "catalog-benchmark-receipt.v1":
            raise ValueError("benchmark receipt has an unsupported receiptVersion")
        for field in ("keyId", "signature", "signedBytesB64"):
            value = document.get(field)
            if not isinstance(value, str) or not value or value == "UNSIGNED":
                raise ValueError(f"benchmark receipt evidence has no signed {field}")
        if document.get("algorithm") != "Ed25519":
            raise ValueError("benchmark receipt evidence is not Ed25519 signed")

    @staticmethod
    def _walk(document: Any, selector: str) -> Any:
        node = document
        for part in selector.split("."):
            if isinstance(node, dict) and part in node:
                node = node[part]
                continue
            if isinstance(node, list) and part.isdigit():
                index = int(part)
                if index < len(node):
                    node = node[index]
                    continue
            raise ValueError(f"evidence selector {selector!r} is missing")
        return node

    @staticmethod
    def _as_decimal(value: Any, selector: str) -> Decimal:
        if isinstance(value, bool) or not isinstance(value, (int, float, Decimal, str)):
            raise ValueError(f"evidence selector {selector!r} is not numeric")
        text = str(value).strip().replace(",", "")
        try:
            parsed = Decimal(text)
        except InvalidOperation as exc:
            raise ValueError(f"evidence selector {selector!r} is not numeric") from exc
        if not parsed.is_finite():
            raise ValueError(f"evidence selector {selector!r} is not finite")
        return parsed

    def _read_structural_table_value(self, text: str) -> Decimal:
        if not self.column:
            raise ValueError("structural table evidence requires a column")
        lines = text.splitlines()
        wanted_column = self.column.casefold()
        wanted_row = self.selector.casefold()
        for index, line in enumerate(lines):
            if not line.lstrip().startswith("|"):
                continue
            headers = [cell.strip() for cell in line.strip().strip("|").split("|")]
            normalized_headers = [cell.casefold() for cell in headers]
            if wanted_column not in normalized_headers:
                continue
            column_index = normalized_headers.index(wanted_column)
            for row in lines[index + 2 :]:
                if not row.lstrip().startswith("|"):
                    break
                cells = [cell.strip() for cell in row.strip().strip("|").split("|")]
                if len(cells) <= column_index or not cells:
                    continue
                if cells[0].casefold() != wanted_row:
                    continue
                match = re.fullmatch(r"[-+]?\d[\d,]*(?:\.\d+)?", cells[column_index])
                if not match:
                    raise ValueError(
                        f"structural table cell {self.selector!r}/{self.column!r} is not numeric"
                    )
                return self._as_decimal(match.group(0), self.selector)
        raise ValueError(
            f"structural table row {self.selector!r} and column {self.column!r} are missing"
        )

    def _read_structural_line_value(self, text: str) -> Decimal:
        matches = [
            line
            for line in text.splitlines()
            if self.selector.casefold() in line.casefold()
        ]
        if len(matches) != 1:
            raise ValueError(
                f"structural selector {self.selector!r} must match exactly one line"
            )
        suffix = matches[0].split(":", maxsplit=1)[-1]
        match = re.search(r"[-+]?\d[\d,]*(?:\.\d+)?", suffix)
        if not match:
            raise ValueError(f"structural selector {self.selector!r} has no numeric value")
        return self._as_decimal(match.group(0), self.selector)

    def read_numeric(self) -> Decimal:
        if self.artifact is EvidenceArtifact.STRUCTURAL_RESULTS:
            text = self.path.read_text(encoding="utf-8")
            if self.column:
                return self._read_structural_table_value(text)
            return self._read_structural_line_value(text)
        return self._as_decimal(self._walk(self._load_json(), self.selector), self.selector)

    def read_document(self) -> dict[str, Any]:
        if self.artifact is EvidenceArtifact.STRUCTURAL_RESULTS:
            raise ValueError("structural text evidence has no JSON document")
        return self._load_json()


class Claim(BaseModel):
    """One evidence-backed figure with all conditions required to reproduce it."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    label: str = Field(min_length=1)
    kind: ClaimKind
    basis: ClaimBasis
    value: int | float | Decimal
    unit: str = Field(min_length=1)
    hardware: HardwareContext
    concurrency: int = Field(gt=0)
    request_profile: str = Field(min_length=1)
    candidate_runtime: RuntimeIdentity
    baseline_runtime: RuntimeIdentity | None = None
    comparison: Comparison | None = None
    evidence: EvidenceReference

    @field_validator("value", mode="before")
    @classmethod
    def validate_value(cls, value: Any) -> Any:
        if isinstance(value, bool) or not isinstance(value, (int, float, Decimal)):
            raise ValueError("claim value must be a numeric object")
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("claim value must be finite")
        if isinstance(value, Decimal) and not value.is_finite():
            raise ValueError("claim value must be finite")
        return value

    @field_validator("concurrency", mode="before")
    @classmethod
    def validate_concurrency(cls, value: Any) -> Any:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("concurrency must be an integer")
        return value

    @model_validator(mode="after")
    def validate_claim(self) -> Claim:
        if (self.baseline_runtime is None) != (self.comparison is None):
            raise ValueError("a comparative claim requires both runtime sides and a comparison")
        if (
            self.basis is ClaimBasis.MEASURED
            and self.evidence.artifact is EvidenceArtifact.STRUCTURAL_RESULTS
        ):
            raise ValueError("computed structural evidence cannot back a measured claim")
        if (
            self.basis is ClaimBasis.COMPUTED
            and self.evidence.artifact is not EvidenceArtifact.STRUCTURAL_RESULTS
        ):
            raise ValueError("a computed claim must use the structural results artifact")
        if self.kind is ClaimKind.PERFORMANCE:
            if self.basis is not ClaimBasis.MEASURED:
                raise ValueError("a performance claim must be measured")
            if self.baseline_runtime is None or self.comparison is None:
                raise ValueError("a performance claim requires both runtime sides")
            if self.evidence.artifact is EvidenceArtifact.STRUCTURAL_RESULTS:
                raise ValueError("structural arithmetic cannot back a performance claim")

        claimed = EvidenceReference._as_decimal(self.value, self.evidence.selector)
        evidenced = self.evidence.read_numeric()
        if claimed != evidenced:
            raise ValueError(
                f"claim value {claimed} does not match evidence value {evidenced}"
            )

        if self.evidence.artifact is EvidenceArtifact.FACTORY_SCORECARD_V1:
            document = self.evidence.read_document()
            if document.get("concurrency") != self.concurrency:
                raise ValueError("claim concurrency does not match the factory scorecard")

        if self.evidence.artifact is EvidenceArtifact.FACTORY_BENCHMARK_RECEIPT_V1:
            payload = self.evidence.read_document()["payload"]
            if payload.get("requestProfile") != self.request_profile:
                raise ValueError("claim request profile does not match the benchmark receipt")
            for side in ("baseline", "optimized"):
                record = payload.get(side)
                if not isinstance(record, dict) or record.get("concurrency") != self.concurrency:
                    raise ValueError(
                        f"claim concurrency does not match benchmark receipt {side}"
                    )
        return self


def _format_claim_value(claim: Claim) -> str:
    value = EvidenceReference._as_decimal(claim.value, claim.evidence.selector)
    rendered = format(value, "f")
    if "." in rendered:
        rendered = rendered.rstrip("0").rstrip(".")
    return rendered


def render_claim(claim: Claim) -> str:
    """Render only a validated Claim object, never a bare number."""

    runtime = f"candidate runtime {claim.candidate_runtime.inline()}"
    if claim.baseline_runtime is not None:
        runtime += (
            f", baseline runtime {claim.baseline_runtime.inline()}, "
            f"comparison {claim.comparison.value}"
        )
    return (
        f"{claim.label}: {_format_claim_value(claim)} {claim.unit} on "
        f"{claim.hardware.inline()}, concurrency {claim.concurrency}, "
        f"request profile {claim.request_profile}, {runtime}. "
        f"Evidence: {claim.evidence.path}#{claim.evidence.selector}."
    )
