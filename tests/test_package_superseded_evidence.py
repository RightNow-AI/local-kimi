"""A package must not be built from a document that says it is out of date.

This exists because it nearly happened. `package/definition.py` reads the
resident-byte figure out of a structural results file by selector. The file it
was pointed at, `engine/laptop/RESULTS.md`, projected 24,561,340,864 bytes from
flat 4.0-bit arithmetic. The artifact that was actually built measures
28,803,304,448, because the codec costs 4.5 bits per quantized parameter once
BF16 group scales are counted and the embedding, LM head, router, norms, KDA
controls and MLA latent down-projection are all deliberately left in source
precision.

Nothing in the packaging layer would have caught that. The definition validates
that a claim HAS evidence, not that the evidence is current, so a 4.2 GB error
would have reached a customer with a perfectly valid evidence reference
attached. The fix is structural rather than a one-off correction: a results
document that has been overtaken declares so in its own text, and the builder
refuses to read past that.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from package.definition import SUPERSEDED_MARKER, _refuse_superseded_evidence


def test_a_superseded_document_is_refused(tmp_path: Path):
    results = tmp_path / "RESULTS.md"
    results.write_text(
        "# Some results\n\n"
        f"## {SUPERSEDED_MARKER} FIGURES, read this first\n\n"
        "- Weight residence: 24,561,340,864 bytes\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="superseded"):
        _refuse_superseded_evidence(results)


def test_a_current_document_is_accepted(tmp_path: Path):
    results = tmp_path / "RESULTS.md"
    results.write_text(
        "# Measured results\n\n- Weight residence: 28,803,304,448 bytes\n",
        encoding="utf-8",
    )
    _refuse_superseded_evidence(results)


def test_a_missing_document_is_refused_rather_than_ignored(tmp_path: Path):
    """Absent evidence must fail loudly, not read as no objection."""
    with pytest.raises(ValueError, match="unreadable"):
        _refuse_superseded_evidence(tmp_path / "does-not-exist.md")


def test_the_real_laptop_results_file_is_currently_refused():
    """The live file, not a fixture. This is the case that motivated the guard.

    If someone removes the superseded banner from engine/laptop/RESULTS.md
    without correcting its figures, this fails and says so.
    """
    repo_root = Path(__file__).resolve().parent.parent
    laptop_results = repo_root / "engine" / "laptop" / "RESULTS.md"
    assert laptop_results.exists(), "the laptop results file moved; update this test"
    with pytest.raises(ValueError, match="superseded"):
        _refuse_superseded_evidence(laptop_results)


def test_the_measured_quantization_results_are_accepted():
    """The document that SHOULD back the footprint claim must pass."""
    repo_root = Path(__file__).resolve().parent.parent
    measured = repo_root / "engine" / "quant" / "QUANTIZATION-RESULTS.md"
    assert measured.exists(), "the measured quantization results moved"
    _refuse_superseded_evidence(measured)
    assert "28,803,304,448" in measured.read_text(encoding="utf-8"), (
        "the measured artifact byte count must appear in the document that backs "
        "the footprint claim"
    )
