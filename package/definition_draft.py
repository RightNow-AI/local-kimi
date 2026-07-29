"""Build a blocked catalog definition draft from an evidence-backed scorecard.

This module creates a structured hand-off and a human-readable rendering. It
does not issue a price, claim listing readiness, or infer missing performance.
"""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

try:
    from .scorecard import (
        DEFAULT_SLUG,
        PERFORMANCE_DIMENSIONS,
        EvidenceError,
        _claim_value_inline,
        _hardware_inline,
        assemble_scorecard,
        validate_scorecard,
    )
except ImportError:  # pragma: no cover - direct script invocation
    from scorecard import (  # type: ignore[no-redef]
        DEFAULT_SLUG,
        PERFORMANCE_DIMENSIONS,
        EvidenceError,
        _claim_value_inline,
        _hardware_inline,
        assemble_scorecard,
        validate_scorecard,
    )


DRAFT_SCHEMA_VERSION = "runinfra.catalog-definition-draft.v1"
MODEL_HF_ID = "moonshotai/Kimi-Linear-48B-A3B-Instruct"


def _resolve_evidence(reference: dict[str, Any], evidence_root: Path) -> Path:
    if not isinstance(reference, dict):
        raise EvidenceError("claim lacks an evidence reference")
    raw_path = reference.get("path")
    selector = reference.get("selector")
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise EvidenceError("claim lacks an evidence path")
    if not isinstance(selector, str) or not selector.strip():
        raise EvidenceError("claim lacks an evidence selector")
    path = Path(raw_path)
    resolved = path if path.is_absolute() else evidence_root / path
    if not resolved.is_file():
        raise EvidenceError(f"claim evidence file does not exist: {resolved}")
    return resolved


def _validate_claim(claim: dict[str, Any], card: dict[str, Any], evidence_root: Path) -> None:
    claim_id = claim.get("id", "<unnamed>")
    _resolve_evidence(claim.get("evidence"), evidence_root)
    if claim.get("kind") != "performance":
        return

    dimension_name = claim.get("dimension")
    if dimension_name not in PERFORMANCE_DIMENSIONS:
        raise EvidenceError(
            f"performance claim {claim_id} does not name a factory scorecard dimension"
        )
    dimension = card[dimension_name]
    if dimension.get("status") != "MEASURED":
        raise EvidenceError(
            f"performance claim {claim_id} is refused because {dimension_name} "
            "has no measured speed evidence"
        )
    if claim.get("evidence") != dimension.get("evidence"):
        raise EvidenceError(
            f"performance claim {claim_id} does not reference its scorecard evidence"
        )
    if claim.get("hardware") != dimension.get("hardware"):
        raise EvidenceError(f"performance claim {claim_id} hardware differs from its evidence")
    if claim.get("concurrency") != dimension.get("concurrency"):
        raise EvidenceError(
            f"performance claim {claim_id} concurrency differs from its evidence"
        )
    if claim.get("conditions") != dimension.get("conditions"):
        raise EvidenceError(
            f"performance claim {claim_id} conditions differ from its evidence"
        )


def _gate_blockers(card: dict[str, Any]) -> list[str]:
    blockers = []
    for name, accuracy in card["accuracyProfiles"].items():
        if accuracy["verdict"].upper() == "FAIL":
            blockers.append(f"The committed {name} accuracy evidence has verdict FAIL.")
    for name in PERFORMANCE_DIMENSIONS:
        if card[name]["status"] != "MEASURED":
            blockers.append(f"{name} has no matched measured comparison.")
    if card["stability"]["status"] != "MEASURED":
        blockers.append("No release soak or stability receipt exists.")
    if card["engineCorrectnessAgainstReference"]["status"] != "MEASURED":
        blockers.append("This engine's correctness against a reference is unmeasured.")
    blockers.extend(
        [
            "The required gsm8k, ifeval, tool-calling, and safety suite is incomplete.",
            "No signed benchmark receipt or signed kit exists.",
            "No weights-tree digest has been recorded for kit assembly.",
            "Founder price and listing approvals remain pending.",
        ]
    )
    return blockers


def build_definition_draft(
    card: dict[str, Any], evidence_root: Path | None = None
) -> dict[str, Any]:
    root = (evidence_root or Path(__file__).resolve().parents[1]).resolve()
    validate_scorecard(card, root)
    claims = deepcopy(card["claims"])
    for claim in claims:
        _validate_claim(claim, card, root)

    accuracy = card["accuracyRetained"]
    accuracy_profiles = deepcopy(card["accuracyProfiles"])
    speed_absence = (
        "No speed comparison against vLLM has been measured, no throughput comparison "
        "against vLLM has been measured, and none is claimed. The capability result "
        "must not be described as a speed result."
    )
    correctness_absence = (
        "This engine's correctness against a reference implementation has not been measured."
    )
    draft = {
        "schemaVersion": DRAFT_SCHEMA_VERSION,
        "status": "BLOCKED",
        "listable": False,
        "slug": card.get("slug") or DEFAULT_SLUG,
        "model": {
            "name": "Kimi-Linear-48B-A3B-Instruct",
            "hfId": MODEL_HF_ID,
            "revision": card["residency"]["sourceRevision"],
        },
        "packageClass": "optimized_weights",
        "servingEngine": {
            "name": "RunInfra purpose-built Kimi-Linear serving engine",
            "version": None,
        },
        "quantMethod": "w4a16",
        "licence": deepcopy(card["licence"]),
        "commercialApproval": {
            "price": "pending",
            "license": "evidence_recorded",
            "listing": "pending",
        },
        "headline": {
            name: deepcopy(card[name]) for name in PERFORMANCE_DIMENSIONS
        },
        "capability": deepcopy(card["capability"]),
        "quality": {
            "status": accuracy["status"],
            "verdict": accuracy["verdict"],
            "verdictText": accuracy["verdictText"],
            "pass": False if accuracy["verdict"].upper() == "FAIL" else None,
            "hardware": deepcopy(accuracy["hardware"]),
            "conditions": accuracy["conditions"],
            "evidence": deepcopy(accuracy["evidence"]),
            "claimIds": list(accuracy["claimIds"]),
            "profiles": accuracy_profiles,
        },
        "residency": deepcopy(card["residency"]),
        "evidenceBackedClaims": claims,
        "whatPackageDoesNotDo": [
            speed_absence,
            correctness_absence,
            "It does not claim behavioural equivalence to the BF16 checkpoint.",
            "It does not claim that this engine is faster than vLLM.",
            "It does not claim a production stability result without a soak receipt.",
        ],
        "gateBlockers": _gate_blockers(card),
        "artifacts": {
            "kitS3Key": None,
            "kitSizeBytes": None,
            "checksumSha256": None,
            "benchmarkReceipt": None,
            "signature": None,
        },
    }
    validate_definition_draft(draft, root)
    return draft


def validate_definition_draft(
    draft: dict[str, Any], evidence_root: Path | None = None
) -> None:
    root = (evidence_root or Path(__file__).resolve().parents[1]).resolve()
    if draft.get("schemaVersion") != DRAFT_SCHEMA_VERSION:
        raise EvidenceError(f"definition draft schemaVersion must be {DRAFT_SCHEMA_VERSION}")
    if draft.get("listable") is not False or draft.get("status") != "BLOCKED":
        raise EvidenceError("a definition draft cannot imply that the package is listable")

    quality = draft.get("quality")
    if not isinstance(quality, dict):
        raise EvidenceError("definition draft is missing quality evidence")
    verdict = quality.get("verdict")
    verdict_text = quality.get("verdictText")
    if not isinstance(verdict, str) or not verdict:
        raise EvidenceError("definition draft cannot omit the accuracy verdict")
    if not isinstance(verdict_text, str) or verdict not in verdict_text:
        raise EvidenceError("definition draft must carry the accuracy verdict verbatim")
    if verdict.upper() == "FAIL" and quality.get("pass") is not False:
        raise EvidenceError("definition draft hides the accuracy FAIL behind a non-false pass")
    _resolve_evidence(quality.get("evidence"), root)
    profiles = quality.get("profiles")
    if not isinstance(profiles, dict) or set(profiles) != {
        "default-int4",
        "shared-experts-bf16",
    }:
        raise EvidenceError("definition draft must carry both accuracy profiles")
    for name, profile in profiles.items():
        if not isinstance(profile, dict):
            raise EvidenceError(f"definition draft accuracy profile {name} is not an object")
        profile_verdict = profile.get("verdict")
        profile_verdict_text = profile.get("verdictText")
        if not isinstance(profile_verdict, str) or not profile_verdict:
            raise EvidenceError(
                f"definition draft accuracy profile {name} omits its verdict"
            )
        if (
            not isinstance(profile_verdict_text, str)
            or profile_verdict not in profile_verdict_text
        ):
            raise EvidenceError(
                f"definition draft accuracy profile {name} must carry its verdict verbatim"
            )
        if profile_verdict.upper() != "FAIL":
            raise EvidenceError(
                f"definition draft accuracy profile {name} must preserve the measured FAIL"
            )
        _resolve_evidence(profile.get("evidence"), root)

    capability = draft.get("capability")
    if not isinstance(capability, dict):
        raise EvidenceError("definition draft is missing the capability result")
    if capability.get("stockVllmBelowBf16") != "REFUSED":
        raise EvidenceError("definition draft omits the stock vLLM BF16-floor refusal")
    if capability.get("speedComparisonAgainstVllm") != "NOT_MEASURED":
        raise EvidenceError("definition draft attempts to claim a vLLM speed comparison")
    if capability.get("fasterThanVllm") is not None:
        raise EvidenceError("definition draft implies that this engine is faster than vLLM")
    capability_evidence = capability.get("evidence")
    if not isinstance(capability_evidence, list) or not capability_evidence:
        raise EvidenceError("definition draft capability result has no evidence")
    for reference in capability_evidence:
        _resolve_evidence(reference, root)

    headline = draft.get("headline")
    if not isinstance(headline, dict):
        raise EvidenceError("definition draft has no headline block")
    for name in PERFORMANCE_DIMENSIONS:
        dimension = headline.get(name)
        if not isinstance(dimension, dict):
            raise EvidenceError(f"definition draft is missing {name}")
        status = dimension.get("status")
        if status == "NOT_MEASURED":
            if any(key in dimension for key in ("baseline", "optimized", "speedup", "value")):
                raise EvidenceError(f"unmeasured {name} carries a performance figure")
            continue
        if status != "MEASURED":
            raise EvidenceError(f"definition draft {name} has unsupported status {status!r}")
        for field in ("hardware", "concurrency", "conditions", "evidence"):
            if dimension.get(field) in (None, "", {}):
                raise EvidenceError(f"measured {name} is missing inline {field}")
        _resolve_evidence(dimension["evidence"], root)

    claims = draft.get("evidenceBackedClaims")
    if not isinstance(claims, list) or not claims:
        raise EvidenceError("definition draft has no evidence-backed claims")
    for claim in claims:
        if not isinstance(claim, dict):
            raise EvidenceError("definition draft claim is not an object")
        _resolve_evidence(claim.get("evidence"), root)
        for field in ("label", "displayValue", "unit", "hardware", "conditions"):
            if claim.get(field) in (None, "", {}):
                raise EvidenceError(f"definition draft claim is missing {field}")
        if claim.get("kind") == "performance":
            dimension_name = claim.get("dimension")
            if dimension_name not in PERFORMANCE_DIMENSIONS:
                raise EvidenceError("performance claim has no factory scorecard dimension")
            dimension = headline[dimension_name]
            if dimension.get("status") != "MEASURED":
                raise EvidenceError(
                    f"performance claim is refused because {dimension_name} has no "
                    "measured speed evidence"
                )
            for field in ("concurrency", "hardware", "conditions"):
                if claim.get(field) in (None, "", {}):
                    raise EvidenceError(f"performance claim is missing inline {field}")

    exclusions = draft.get("whatPackageDoesNotDo")
    if not isinstance(exclusions, list) or not any(
        "no speed comparison against vllm has been measured" in str(item).casefold()
        for item in exclusions
    ):
        raise EvidenceError("definition draft omits the unmeasured vLLM speed disclosure")
    if not any(
        "does not claim behavioural equivalence" in str(item).casefold()
        for item in exclusions
    ):
        raise EvidenceError("definition draft omits the behavioural-equivalence refusal")


def render_definition_draft(
    draft: dict[str, Any], evidence_root: Path | None = None
) -> str:
    validate_definition_draft(draft, evidence_root)
    lines = [
        f"# Catalog definition draft: {draft['slug']}",
        "",
        "Status: BLOCKED. This draft is not ready to list.",
        "",
        "## Identity",
        "",
        f"- Package class: `{draft['packageClass']}`",
        f"- Model: `{draft['model']['hfId']}`",
        f"- Pinned revision: `{draft['model']['revision']}`",
        f"- Quantization: `{draft['quantMethod']}`",
        f"- Licence: `{draft['licence']['spdxId']}`",
        "",
        "## Performance",
        "",
    ]
    for name in PERFORMANCE_DIMENSIONS:
        dimension = draft["headline"][name]
        if dimension["status"] == "NOT_MEASURED":
            lines.append(f"- {name}: NOT_MEASURED. {dimension['note']}")
            continue
        lines.append(
            f"- {name}: {dimension['baseline']} -> {dimension['optimized']} "
            f"{dimension['unit']} on {_hardware_inline(dimension['hardware'])}, "
            f"concurrency {dimension['concurrency']}, conditions "
            f"{dimension['conditions']}. Evidence: "
            f"{dimension['evidence']['path']}#{dimension['evidence']['selector']}."
        )

    capability = draft["capability"]
    lines.extend(
        [
            "",
            "## Capability",
            "",
            f"- {capability['note']}",
            "- No throughput comparison against vLLM has been measured, and none is claimed.",
            "- This definition does not claim that this engine is faster than vLLM.",
        ]
    )

    quality = draft["quality"]
    lines.extend(
        [
            "",
            "## Quality gate",
            "",
            f"- {quality['verdictText']}",
            f"- Pass: {str(quality['pass']).lower()}",
            f"- Hardware: {_hardware_inline(quality['hardware'])}",
            f"- Conditions: {quality['conditions']}",
            (
                f"- Evidence: {quality['evidence']['path']}#"
                f"{quality['evidence']['selector']}"
            ),
            "",
            "### Accuracy profiles",
            "",
        ]
    )
    for name, profile in quality["profiles"].items():
        lines.extend(
            [
                f"- {name}: {profile['verdictText']}",
                f"  Hardware: {_hardware_inline(profile['hardware'])}",
                f"  Conditions: {profile['conditions']}",
                (
                    f"  Evidence: {profile['evidence']['path']}#"
                    f"{profile['evidence']['selector']}"
                ),
            ]
        )
    lines.extend(
        [
            "- Both profiles are FAIL. Behavioural equivalence to BF16 is not claimed.",
            "",
            "## Evidence-backed figures",
            "",
        ]
    )
    for claim in draft["evidenceBackedClaims"]:
        inline = (
            f"- {claim['label']}: {_claim_value_inline(claim)} on "
            f"{_hardware_inline(claim['hardware'])}. Conditions: {claim['conditions']}."
        )
        if claim.get("kind") == "performance":
            inline += f" Concurrency: {claim['concurrency']}."
        inline += f" Evidence: {claim['evidence']['path']}#{claim['evidence']['selector']}."
        lines.append(inline)

    lines.extend(["", "## What this package does not do", ""])
    lines.extend(f"- {item}" for item in draft["whatPackageDoesNotDo"])
    lines.extend(["", "## Unsatisfied pre-listing gates", ""])
    lines.extend(f"- {item}" for item in draft["gateBlockers"])
    lines.append("")
    return "\n".join(lines)


def _load_scorecard(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise EvidenceError(f"scorecard file does not exist: {path}")
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvidenceError(f"scorecard file is unreadable or invalid JSON: {path}") from exc
    if not isinstance(loaded, dict):
        raise EvidenceError("scorecard JSON must be an object")
    return loaded


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--scorecard", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--json", action="store_true", help="render structured JSON")
    args = parser.parse_args()

    try:
        card = _load_scorecard(args.scorecard) if args.scorecard else assemble_scorecard(args.root)
        draft = build_definition_draft(card, args.root)
        rendered = (
            json.dumps(draft, indent=2)
            if args.json
            else render_definition_draft(draft, args.root)
        )
    except EvidenceError as exc:
        raise SystemExit(f"REFUSED: {exc}") from exc

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered + "\n", encoding="utf-8", newline="\n")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
