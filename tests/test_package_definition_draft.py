import copy
import json
from pathlib import Path

import pytest

from package.definition_draft import (
    build_definition_draft,
    render_definition_draft,
)
from package.scorecard import EvidenceError, assemble_scorecard

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_definition_refuses_a_speed_claim_without_speed_evidence() -> None:
    card = assemble_scorecard(REPO_ROOT)
    card["claims"].append(
        {
            "id": "performance.unmeasured_speedup",
            "kind": "performance",
            "dimension": "tokensPerSec",
            "label": "Unmeasured speedup",
            "basis": "measured",
            "evidenceStatus": "MEASURED",
            "value": 2,
            "displayValue": "2",
            "unit": "x",
            "hardware": {"accelerator": "Test GPU", "acceleratorCount": 1},
            "concurrency": 1,
            "conditions": "Synthetic claim with no benchmark receipt",
            "evidence": {
                "path": "engine/quant/quantization-results.json",
                "artifact": "machine_readable_evidence_v1",
                "selector": "$.values.weightByteReduction",
            },
        }
    )

    with pytest.raises(EvidenceError, match="has no measured speed evidence"):
        build_definition_draft(card, REPO_ROOT)


def test_definition_refuses_a_claim_without_an_evidence_reference() -> None:
    card = assemble_scorecard(REPO_ROOT)
    card["claims"][0] = copy.deepcopy(card["claims"][0])
    card["claims"][0].pop("evidence")

    with pytest.raises(EvidenceError, match="evidence"):
        build_definition_draft(card, REPO_ROOT)


def test_rendered_performance_figure_has_hardware_and_concurrency_inline(
    tmp_path: Path,
) -> None:
    evidence = tmp_path / "scorecard.json"
    evidence.write_text(json.dumps({"schemaVersion": 1}), encoding="utf-8")
    card = assemble_scorecard(REPO_ROOT)
    card["tokensPerSec"] = {
        "status": "MEASURED",
        "baseline": 10,
        "optimized": 12,
        "unit": "tokens/sec",
        "hardware": {"accelerator": "Test GPU", "acceleratorCount": 1},
        "concurrency": 4,
        "conditions": "Synthetic matched request profile",
        "evidence": {
            "path": evidence.as_posix(),
            "artifact": "factory_scorecard_v1",
            "selector": "tokensPerSec",
        },
    }
    card["concurrency"] = 4

    draft = build_definition_draft(card, REPO_ROOT)
    rendered = render_definition_draft(draft, REPO_ROOT)
    line = next(line for line in rendered.splitlines() if line.startswith("- tokensPerSec:"))

    assert "1x Test GPU" in line
    assert "concurrency 4" in line
    assert "Synthetic matched request profile" in line


def test_definition_states_the_unmeasured_vllm_comparison_plainly() -> None:
    card = assemble_scorecard(REPO_ROOT)
    draft = build_definition_draft(card, REPO_ROOT)
    rendered = render_definition_draft(draft, REPO_ROOT)

    assert "No speed comparison against vLLM has been measured" in rendered
    assert "No throughput comparison against vLLM has been measured" in rendered
    assert "does not claim that this engine is faster than vLLM" in rendered
    assert "Status: BLOCKED. This draft is not ready to list." in rendered
    assert "Verdict: FAIL" in rendered
    assert "default-int4: Verdict: FAIL" in rendered
    assert "shared-experts-bf16: Verdict: FAIL" in rendered
    assert "Behavioural equivalence to BF16 is not claimed" in rendered


def test_definition_frames_vllm_as_binary_capability_not_speed() -> None:
    card = assemble_scorecard(REPO_ROOT)
    draft = build_definition_draft(card, REPO_ROOT)
    capability = draft["capability"]

    assert capability["stockVllmBelowBf16"] == "REFUSED"
    assert capability["stockWeightFloorBytes"] == 98245528576
    assert capability["optimizedWeightBytes"] == 28803304448
    assert capability["optimizedEngineLoadsAndRuns"] is True
    assert capability["speedComparisonAgainstVllm"] == "NOT_MEASURED"
    assert capability["fasterThanVllm"] is None
