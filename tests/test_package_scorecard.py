import json
from dataclasses import replace
from pathlib import Path

import pytest

from package.scorecard import (
    EvidenceError,
    EvidencePaths,
    assemble_scorecard,
    render_scorecard,
    validate_scorecard,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_scorecard_refuses_a_missing_required_evidence_file(tmp_path: Path) -> None:
    paths = EvidencePaths.from_root(REPO_ROOT)
    missing_accuracy = replace(paths, accuracy=tmp_path / "missing-accuracy.json")

    with pytest.raises(EvidenceError, match="required evidence file is missing"):
        assemble_scorecard(REPO_ROOT, paths=missing_accuracy)


def test_accuracy_fail_is_rendered_and_cannot_be_omitted() -> None:
    card = assemble_scorecard(REPO_ROOT)
    rendered = render_scorecard(card, REPO_ROOT)

    assert "Accuracy Verdict: FAIL" in rendered
    assert card["overall"]["accuracyVerdict"] == "FAIL"
    assert card["overall"]["accuracyVerdicts"] == {
        "default-int4": "FAIL",
        "shared-experts-bf16": "FAIL",
    }
    assert "Accuracy shared-experts-bf16 Verdict: FAIL" in rendered

    card["accuracyRetained"].pop("verdict")
    with pytest.raises(EvidenceError, match="accuracy verdict cannot be omitted"):
        validate_scorecard(card, REPO_ROOT)


def test_unmeasured_performance_dimensions_carry_no_figures() -> None:
    card = assemble_scorecard(REPO_ROOT)

    for name in ("latency", "throughput", "tokensPerSec"):
        dimension = card[name]
        assert dimension["status"] == "NOT_MEASURED"
        assert "baseline" not in dimension
        assert "optimized" not in dimension
        assert "speedup" not in dimension


def test_every_claim_has_an_existing_evidence_reference() -> None:
    card = assemble_scorecard(REPO_ROOT)

    for claim in card["claims"]:
        reference = claim["evidence"]
        assert reference["selector"]
        assert reference["artifact"] == "machine_readable_evidence_v1"
        assert (REPO_ROOT / reference["path"]).is_file()


def _copied_quantization_evidence(tmp_path: Path) -> tuple[Path, Path]:
    source_json = REPO_ROOT / "engine" / "quant" / "quantization-results.json"
    source_markdown = REPO_ROOT / "engine" / "quant" / "QUANTIZATION-RESULTS.md"
    copied_json = tmp_path / source_json.name
    copied_markdown = tmp_path / source_markdown.name
    payload = json.loads(source_json.read_text(encoding="utf-8"))
    payload["humanDocument"] = copied_markdown.name
    copied_json.write_text(json.dumps(payload), encoding="utf-8")
    copied_markdown.write_text(source_markdown.read_text(encoding="utf-8"), encoding="utf-8")
    return copied_json, copied_markdown


def test_markdown_reflow_cannot_change_a_published_measurement(tmp_path: Path) -> None:
    copied_json, copied_markdown = _copied_quantization_evidence(tmp_path)
    text = copied_markdown.read_text(encoding="utf-8")
    copied_markdown.write_text(
        text.replace("real BF16\ncheckpoint", "real BF16 checkpoint"),
        encoding="utf-8",
    )
    paths = replace(EvidencePaths.from_root(REPO_ROOT), quantization=copied_json)

    card = assemble_scorecard(REPO_ROOT, paths=paths)

    claim = next(
        item for item in card["claims"] if item["id"] == "footprint.source_tensor_bytes"
    )
    assert claim["value"] == 98245528576
    assert claim["evidence"]["path"] == copied_json.as_posix()


def test_json_markdown_value_disagreement_refuses_assembly(tmp_path: Path) -> None:
    copied_json, copied_markdown = _copied_quantization_evidence(tmp_path)
    text = copied_markdown.read_text(encoding="utf-8")
    copied_markdown.write_text(
        text.replace("98,245,528,576", "98,245,528,577"),
        encoding="utf-8",
    )
    paths = replace(EvidencePaths.from_root(REPO_ROOT), quantization=copied_json)

    with pytest.raises(EvidenceError, match="disagrees with human Markdown"):
        assemble_scorecard(REPO_ROOT, paths=paths)
