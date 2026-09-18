"""Strict package definitions for RunInfra catalog hand-off."""

from __future__ import annotations

import re
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .claims import (
    Claim,
    ClaimBasis,
    ClaimKind,
    EvidenceArtifact,
    EvidenceReference,
    HardwareContext,
    RuntimeIdentity,
)


KIMI_LINEAR_MODEL_ID = "moonshotai/Kimi-Linear-48B-A3B-Instruct"
_REVISION_RE = re.compile(r"[0-9a-f]{40}")
_MODEL_ID_RE = re.compile(r"[^/\s]+/[^/\s]+")
_FORBIDDEN_POSITIONING_RE = re.compile(r"\b(?:first|fastest|day[- ]?0)\b", re.IGNORECASE)


class PackageClass(str, Enum):
    RECIPE = "recipe"
    OPTIMIZED_WEIGHTS = "optimized_weights"


class LicenseStatus(str, Enum):
    VERIFIED = "verified"
    HOLD = "hold"
    UNVERIFIED = "unverified"


class BaseModelReference(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    model_id: str = Field(min_length=1)
    revision: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_reference(self) -> BaseModelReference:
        if not _MODEL_ID_RE.fullmatch(self.model_id):
            raise ValueError("base model id must be an owner/repository identifier")
        if not _REVISION_RE.fullmatch(self.revision):
            raise ValueError("base model revision must be an immutable 40-hex commit")
        return self


class ModificationDisclosure(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    weights_modified: bool
    modified: tuple[str, ...] = Field(min_length=1)
    untouched: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_sections(self) -> ModificationDisclosure:
        modified = {item.casefold() for item in self.modified if item}
        untouched = {item.casefold() for item in self.untouched if item}
        if len(modified) != len(self.modified) or len(untouched) != len(self.untouched):
            raise ValueError("modified and untouched entries must be non-empty and unique")
        overlap = modified & untouched
        if overlap:
            raise ValueError(f"modified and untouched entries overlap: {sorted(overlap)}")
        return self


class HardwareRequirement(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    accelerator: str = Field(min_length=1)
    accelerator_count: int = Field(gt=0)
    minimum_memory_value: int | float | Decimal
    minimum_memory_unit: str = Field(min_length=1)
    evidence: EvidenceReference

    @field_validator("accelerator_count", mode="before")
    @classmethod
    def validate_accelerator_count(cls, value: Any) -> Any:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("accelerator_count must be an integer")
        return value

    @field_validator("minimum_memory_value", mode="before")
    @classmethod
    def validate_memory_value(cls, value: Any) -> Any:
        if isinstance(value, bool) or not isinstance(value, (int, float, Decimal)):
            raise ValueError("minimum_memory_value must be a numeric object")
        return value

    @model_validator(mode="after")
    def validate_evidence(self) -> HardwareRequirement:
        stated = EvidenceReference._as_decimal(
            self.minimum_memory_value, self.evidence.selector
        )
        if stated != self.evidence.read_numeric():
            raise ValueError("hardware memory requirement does not match its evidence")
        return self


class DisclosureSet(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    known_limits: tuple[str, ...] = Field(min_length=1)
    calibration: str = Field(min_length=1)
    performance_position: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_copy(self) -> DisclosureSet:
        values = (*self.known_limits, self.calibration, self.performance_position)
        if any(not value.strip() for value in values):
            raise ValueError("disclosures must be non-empty")
        for value in values:
            if _FORBIDDEN_POSITIONING_RE.search(value):
                raise ValueError("authored disclosures cannot assert first, fastest, or day-0")
        return self


class LicensePosition(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    status: LicenseStatus
    spdx_id: str = Field(min_length=1)
    instrument: str = Field(min_length=1)
    derivatives_permitted: bool
    redistribution_permitted: bool
    notes: str = Field(min_length=1)


class PackageDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    schema_version: Literal["runinfra.package.v1"] = "runinfra.package.v1"
    name: str = Field(min_length=1)
    package_class: PackageClass
    base_model: BaseModelReference
    serving_engine: RuntimeIdentity
    summary: str = Field(min_length=1)
    modifications: ModificationDisclosure
    hardware_requirement: HardwareRequirement
    measured_claims: tuple[Claim, ...] = ()
    structural_claims: tuple[Claim, ...] = ()
    disclosures: DisclosureSet
    license_position: LicensePosition

    @field_validator("summary")
    @classmethod
    def validate_summary(cls, value: str) -> str:
        if _FORBIDDEN_POSITIONING_RE.search(value):
            raise ValueError("summary cannot assert first, fastest, or day-0")
        return value

    @model_validator(mode="after")
    def validate_definition(self) -> PackageDefinition:
        if self.package_class is PackageClass.RECIPE and self.modifications.weights_modified:
            raise ValueError("a recipe package cannot modify weight bytes")
        if (
            self.package_class is PackageClass.OPTIMIZED_WEIGHTS
            and not self.modifications.weights_modified
        ):
            raise ValueError("an optimized weights package must disclose modified weight bytes")

        for claim in self.measured_claims:
            if claim.basis is not ClaimBasis.MEASURED:
                raise ValueError("measured_claims can only contain measured evidence")
            baseline_name = (
                re.sub(r"[^a-z0-9]", "", claim.baseline_runtime.name.casefold())
                if claim.baseline_runtime is not None
                else ""
            )
            if (
                claim.kind is ClaimKind.PERFORMANCE
                and baseline_name.startswith("vllm")
            ):
                raise ValueError(
                    "this package cannot carry a speed comparison against vLLM "
                    "without a new contract"
                )
        for claim in self.structural_claims:
            if claim.basis is not ClaimBasis.COMPUTED:
                raise ValueError("structural_claims can only contain computed evidence")
            if claim.kind is ClaimKind.PERFORMANCE:
                raise ValueError("a structural claim cannot be a performance claim")

        labels = [
            claim.label.casefold()
            for claim in (*self.measured_claims, *self.structural_claims)
        ]
        if len(labels) != len(set(labels)):
            raise ValueError("claim labels must be unique")
        return self

    def assert_listable(self) -> PackageDefinition:
        failures: list[str] = []
        if self.license_position.status is not LicenseStatus.VERIFIED:
            failures.append("license instrument is not verified")
        if not self.license_position.derivatives_permitted:
            failures.append("derivative permission is not established")
        if (
            self.package_class is PackageClass.OPTIMIZED_WEIGHTS
            and not self.license_position.redistribution_permitted
        ):
            failures.append("redistribution permission is not established")
        if self.package_class is PackageClass.OPTIMIZED_WEIGHTS and not any(
            claim.kind is ClaimKind.QUALITY
            and claim.baseline_runtime is not None
            and claim.comparison is not None
            for claim in self.measured_claims
        ):
            failures.append("optimized weights have no measured paired quality claim")
        if not self.measured_claims and not self.structural_claims:
            failures.append("the definition has no evidence-backed claim")
        if failures:
            raise ValueError("definition is not listable: " + "; ".join(failures))
        return self


def build_kimi_linear_definition(
    *,
    base_revision: str,
    engine_version: str,
    results_path: Path,
    license_position: LicensePosition,
    measured_claims: tuple[Claim, ...] = (),
) -> PackageDefinition:
    """Build the Kimi definition by reading structural values from RESULTS.md."""

    memory_evidence = EvidenceReference(
        path=results_path,
        artifact=EvidenceArtifact.STRUCTURAL_RESULTS,
        selector="Memory target",
    )
    resident_evidence = EvidenceReference(
        path=results_path,
        artifact=EvidenceArtifact.STRUCTURAL_RESULTS,
        selector="INT4 weight-only",
        column="Resident bytes",
    )
    active_evidence = EvidenceReference(
        path=results_path,
        artifact=EvidenceArtifact.STRUCTURAL_RESULTS,
        selector="INT4 weight-only",
        column="Active bytes/token",
    )
    runtime = RuntimeIdentity(
        name="RunInfra purpose-built Kimi-Linear serving engine",
        version=engine_version,
    )
    hardware = HardwareContext(accelerator="consumer GPU", accelerator_count=1)

    structural_claims = (
        Claim(
            label="INT4 resident weight footprint",
            kind=ClaimKind.FOOTPRINT,
            basis=ClaimBasis.COMPUTED,
            value=resident_evidence.read_numeric(),
            unit="bytes",
            hardware=hardware,
            concurrency=1,
            request_profile="resident weight accounting",
            candidate_runtime=runtime,
            evidence=resident_evidence,
        ),
        Claim(
            label="INT4 active weight traffic per generated token",
            kind=ClaimKind.FOOTPRINT,
            basis=ClaimBasis.COMPUTED,
            value=active_evidence.read_numeric(),
            unit="bytes per token",
            hardware=hardware,
            concurrency=1,
            request_profile="single-token active path accounting",
            candidate_runtime=runtime,
            evidence=active_evidence,
        ),
    )

    return PackageDefinition(
        name="Kimi-Linear purpose-built serving engine package",
        package_class=PackageClass.OPTIMIZED_WEIGHTS,
        base_model=BaseModelReference(
            model_id=KIMI_LINEAR_MODEL_ID,
            revision=base_revision,
        ),
        serving_engine=runtime,
        summary=(
            "A purpose-built serving engine and INT4 weight package for the Kimi-Linear "
            "architecture, positioned on single-card footprint rather than unmeasured speed."
        ),
        modifications=ModificationDisclosure(
            weights_modified=True,
            modified=(
                "weight representation changed to INT4 weight-only",
                "serving runtime implemented for the Kimi-Linear architecture",
            ),
            untouched=(
                "base model identity and pinned source revision",
                "tokenizer contract",
                "model architecture",
            ),
        ),
        hardware_requirement=HardwareRequirement(
            accelerator="consumer GPU",
            accelerator_count=1,
            minimum_memory_value=memory_evidence.read_numeric(),
            minimum_memory_unit="GiB",
            evidence=memory_evidence,
        ),
        measured_claims=measured_claims,
        structural_claims=structural_claims,
        disclosures=DisclosureSet(
            known_limits=(
                "INT4 changes model numerics. Listing requires paired accuracy evidence.",
                (
                    "The existing loader limitations remain those recorded in "
                    "engine/laptop/RESULTS.md."
                ),
                (
                    "Laptop throughput values in engine/laptop/RESULTS.md are projections "
                    "and are not claims."
                ),
            ),
            calibration=(
                "The package definition does not infer a calibration protocol. Record the exact "
                "quantizer and calibration provenance before listing."
            ),
            performance_position=(
                "No matched speed measurement is supplied, so this package makes no speed claim."
            ),
        ),
        license_position=license_position,
    )
