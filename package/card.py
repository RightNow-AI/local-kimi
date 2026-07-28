"""Render a factual model card from a validated package definition."""

from __future__ import annotations

from .claims import render_claim
from .definition import PackageDefinition


def render_model_card(definition: PackageDefinition) -> str:
    """Render the definition without creating or inferring any new claim."""

    hardware = definition.hardware_requirement
    lines = [
        f"# {definition.name}",
        "",
        "## Package identity",
        "",
        f"- Package class: `{definition.package_class.value}`",
        f"- Base model: `{definition.base_model.model_id}`",
        f"- Base revision: `{definition.base_model.revision}`",
        f"- Serving runtime: {definition.serving_engine.inline()}",
        "",
        "## Engine implementation",
        "",
        (
            "This serving engine was written from scratch for the Kimi-Linear architecture. "
            "It is not a configuration, plugin, or tuning profile for an existing serving engine. "
            "vLLM is an existing, separate runtime that supports this architecture."
        ),
        "",
        definition.summary,
        "",
        "## Hardware requirement",
        "",
        (
            f"{hardware.accelerator_count}x {hardware.accelerator} with at least "
            f"{hardware.minimum_memory_value} {hardware.minimum_memory_unit}. "
            f"Evidence: {hardware.evidence.path}#{hardware.evidence.selector}."
        ),
        "",
        "## What changed",
        "",
    ]
    lines.extend(f"- {item}" for item in definition.modifications.modified)
    lines.extend(["", "## What stayed unchanged", ""])
    lines.extend(f"- {item}" for item in definition.modifications.untouched)

    lines.extend(["", "## Evidence-backed claims", ""])
    claims = (*definition.structural_claims, *definition.measured_claims)
    if claims:
        lines.extend(f"- {render_claim(claim)}" for claim in claims)
    else:
        lines.append("No evidence-backed claims are present.")

    lines.extend(["", "## What this package does not claim", ""])
    lines.extend(
        [
            "- This is not a speed comparison against vLLM.",
            (
                "- This package does not claim first architecture support, fastest serving, "
                "or day-0 support."
            ),
            "- Projected laptop throughput is not rendered as measured performance.",
        ]
    )

    lines.extend(["", "## Known limits", ""])
    lines.extend(f"- {item}" for item in definition.disclosures.known_limits)
    lines.extend(
        [
            "",
            "## Disclosure",
            "",
            f"- Calibration: {definition.disclosures.calibration}",
            f"- Performance: {definition.disclosures.performance_position}",
            "",
            "## Licence position",
            "",
            f"- Status: `{definition.license_position.status.value}`",
            f"- SPDX identifier: `{definition.license_position.spdx_id}`",
            f"- Instrument: {definition.license_position.instrument}",
            f"- Position: {definition.license_position.notes}",
            "",
        ]
    )
    return "\n".join(lines)
