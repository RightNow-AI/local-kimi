import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from package.claims import (
    Claim,
    ClaimBasis,
    ClaimKind,
    Comparison,
    EvidenceArtifact,
    HardwareContext,
    RuntimeIdentity,
)


def _scorecard(path: Path, *, value: float = 1.25, concurrency: int = 7) -> Path:
    path.write_text(
        json.dumps(
            {
                "schemaVersion": 1,
                "slug": "test-package",
                "concurrency": concurrency,
                "tokensPerSec": {"speedup": value},
            }
        ),
        encoding="utf-8",
    )
    return path


def _claim_payload(evidence_path: Path) -> dict:
    return {
        "label": "Matched throughput speedup",
        "kind": ClaimKind.PERFORMANCE,
        "basis": ClaimBasis.MEASURED,
        "value": 1.25,
        "unit": "x",
        "hardware": HardwareContext(accelerator="Test GPU", accelerator_count=1),
        "concurrency": 7,
        "request_profile": "chat-test",
        "candidate_runtime": RuntimeIdentity(name="candidate engine", version="1.0"),
        "baseline_runtime": RuntimeIdentity(name="reference engine", version="2.0"),
        "comparison": Comparison.HIGHER_IS_BETTER,
        "evidence": {
            "path": evidence_path,
            "artifact": EvidenceArtifact.FACTORY_SCORECARD_V1,
            "selector": "tokensPerSec.speedup",
        },
    }


def test_claim_with_missing_evidence_file_raises(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="evidence file does not exist"):
        Claim.model_validate(_claim_payload(tmp_path / "missing-scorecard.json"))


def test_comparative_claim_missing_one_side_raises(tmp_path: Path) -> None:
    payload = _claim_payload(_scorecard(tmp_path / "scorecard.json"))
    payload.pop("baseline_runtime")

    with pytest.raises(ValidationError, match="requires both runtime sides"):
        Claim.model_validate(payload)


def test_claim_value_must_match_evidence(tmp_path: Path) -> None:
    payload = _claim_payload(_scorecard(tmp_path / "scorecard.json", value=1.10))

    with pytest.raises(ValidationError, match="does not match evidence value"):
        Claim.model_validate(payload)
