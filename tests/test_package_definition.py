import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from package.card import render_model_card
from package.claims import (
    Claim,
    ClaimBasis,
    ClaimKind,
    Comparison,
    EvidenceArtifact,
    EvidenceReference,
    HardwareContext,
    RuntimeIdentity,
)
from package.definition import (
    BaseModelReference,
    DisclosureSet,
    HardwareRequirement,
    LicensePosition,
    LicenseStatus,
    ModificationDisclosure,
    PackageClass,
    PackageDefinition,
)


def _structural_results(path: Path) -> Path:
    path.write_text("Memory target: 16 GiB minimum\n", encoding="utf-8")
    return path


def _scorecard(path: Path) -> Path:
    path.write_text(
        json.dumps(
            {
                "schemaVersion": 1,
                "slug": "test-package",
                "concurrency": 5,
                "latency": {"speedup": 1.2},
            }
        ),
        encoding="utf-8",
    )
    return path


def _definition(*, tmp_path: Path, claims: tuple[Claim, ...]) -> PackageDefinition:
    structural_path = _structural_results(tmp_path / "RESULTS.md")
    memory_evidence = EvidenceReference(
        path=structural_path,
        artifact=EvidenceArtifact.STRUCTURAL_RESULTS,
        selector="Memory target",
    )
    return PackageDefinition(
        name="Test package",
        package_class=PackageClass.OPTIMIZED_WEIGHTS,
        base_model=BaseModelReference(
            model_id="owner/model",
            revision="a" * 40,
        ),
        serving_engine=RuntimeIdentity(name="candidate engine", version="1.0"),
        summary="A factual test package.",
        modifications=ModificationDisclosure(
            weights_modified=True,
            modified=("weight representation",),
            untouched=("model architecture",),
        ),
        hardware_requirement=HardwareRequirement(
            accelerator="Test GPU",
            accelerator_count=1,
            minimum_memory_value=16,
            minimum_memory_unit="GiB",
            evidence=memory_evidence,
        ),
        measured_claims=claims,
        disclosures=DisclosureSet(
            known_limits=("Safety was not evaluated.",),
            calibration="No calibration corpus was used.",
            performance_position="Only matched measurements are rendered.",
        ),
        license_position=LicensePosition(
            status=LicenseStatus.VERIFIED,
            spdx_id="Apache-2.0",
            instrument="LICENSE",
            derivatives_permitted=True,
            redistribution_permitted=True,
            notes="The license instrument permits this package shape.",
        ),
    )


def _performance_claim(tmp_path: Path) -> Claim:
    return Claim(
        label="Matched latency speedup",
        kind=ClaimKind.PERFORMANCE,
        basis=ClaimBasis.MEASURED,
        value=1.2,
        unit="x",
        hardware=HardwareContext(accelerator="Test GPU", accelerator_count=1),
        concurrency=5,
        request_profile="chat-test",
        candidate_runtime=RuntimeIdentity(name="candidate engine", version="1.0"),
        baseline_runtime=RuntimeIdentity(name="reference engine", version="2.0"),
        comparison=Comparison.LOWER_IS_BETTER,
        evidence=EvidenceReference(
            path=_scorecard(tmp_path / "scorecard.json"),
            artifact=EvidenceArtifact.FACTORY_SCORECARD_V1,
            selector="latency.speedup",
        ),
    )


def test_definition_rejects_performance_claim_without_evidence_reference(
    tmp_path: Path,
) -> None:
    valid = _definition(tmp_path=tmp_path, claims=()).model_dump(mode="python")
    valid["measured_claims"] = (
        {
            "label": "Unbacked speedup",
            "kind": ClaimKind.PERFORMANCE,
            "basis": ClaimBasis.MEASURED,
            "value": 1.2,
            "unit": "x",
            "hardware": {"accelerator": "Test GPU", "accelerator_count": 1},
            "concurrency": 5,
            "request_profile": "chat-test",
            "candidate_runtime": {"name": "candidate engine", "version": "1.0"},
            "baseline_runtime": {"name": "reference engine", "version": "2.0"},
            "comparison": Comparison.LOWER_IS_BETTER,
        },
    )

    with pytest.raises(ValidationError, match="evidence"):
        PackageDefinition.model_validate(valid)


def test_rendered_performance_figure_has_hardware_and_concurrency_inline(
    tmp_path: Path,
) -> None:
    definition = _definition(tmp_path=tmp_path, claims=(_performance_claim(tmp_path),))
    card = render_model_card(definition)
    claim_line = next(line for line in card.splitlines() if "Matched latency speedup" in line)

    assert "1.2 x" in claim_line
    assert "1x Test GPU" in claim_line
    assert "concurrency 5" in claim_line
    assert "candidate engine 1.0" in claim_line
    assert "reference engine 2.0" in claim_line
