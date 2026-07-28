"""Typed, fail-closed ledger for every quality-affecting transformation."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum


class Transformation(str, Enum):
    SKELETON_QUANTIZATION = "skeleton quantization"
    EXPERT_REQUANTIZATION = "expert requantization below 4.25 bpw"
    REDUCED_TOP_K = "reduced top-k"
    KERNEL_NUMERICS = "kernel numerics"
    SPECULATIVE_DECODING = "speculative decoding"


class MeasurementKind(str, Enum):
    MEASURED = "MEASURED"
    MODELLED = "MODELLED"
    UNMEASURED = "UNMEASURED"


@dataclass(frozen=True)
class LossEntry:
    transformation: Transformation
    what_changed: str
    metric: str
    reference: str
    kind: MeasurementKind = MeasurementKind.UNMEASURED
    value: float | None = None
    arithmetic: str | None = None
    contributes_to_total: bool = True

    def __post_init__(self) -> None:
        if not self.what_changed.strip():
            raise ValueError("what_changed must not be empty")
        if not self.metric.strip():
            raise ValueError("metric must not be empty")
        if not self.reference.strip():
            raise ValueError("reference must not be empty")
        if self.kind is MeasurementKind.UNMEASURED:
            if self.value is not None:
                raise ValueError("an UNMEASURED entry cannot carry a numeric value")
            if self.arithmetic is not None:
                raise ValueError("an UNMEASURED entry cannot claim measurement arithmetic")
            return
        if self.value is None or not math.isfinite(self.value):
            raise ValueError("MEASURED and MODELLED entries require a finite value")
        if self.arithmetic is None or not self.arithmetic.strip():
            raise ValueError("every numeric value must carry the arithmetic that produced it")


@dataclass(frozen=True)
class LedgerTotal:
    metric: str
    value: float
    kind: MeasurementKind
    arithmetic: str


@dataclass
class LossLedger:
    entries: list[LossEntry] = field(default_factory=list)

    def add(self, entry: LossEntry) -> None:
        self.entries.append(entry)

    def total_loss(self, metric: str = "quality_loss") -> LedgerTotal:
        contributing = [
            entry
            for entry in self.entries
            if entry.contributes_to_total and entry.metric == metric
        ]
        if not contributing:
            raise ValueError(f"no contributing entries use metric {metric!r}")
        missing = [
            entry.transformation.value
            for entry in contributing
            if entry.kind is MeasurementKind.UNMEASURED
        ]
        if missing:
            joined = ", ".join(missing)
            raise ValueError(f"total loss is unavailable because these entries are UNMEASURED: {joined}")
        value = math.fsum(entry.value for entry in contributing if entry.value is not None)
        kind = (
            MeasurementKind.MODELLED
            if any(entry.kind is MeasurementKind.MODELLED for entry in contributing)
            else MeasurementKind.MEASURED
        )
        terms = " + ".join(f"{entry.value:g}" for entry in contributing if entry.value is not None)
        return LedgerTotal(metric=metric, value=value, kind=kind, arithmetic=f"{terms} = {value:g}")

    def render(self) -> str:
        headers = ("TRANSFORMATION", "STATUS", "WHAT CHANGED", "METRIC", "VALUE", "REFERENCE", "ARITHMETIC")
        rows = []
        for entry in self.entries:
            if entry.kind is MeasurementKind.UNMEASURED:
                value = "UNMEASURED"
                arithmetic = "UNMEASURED"
            else:
                value = f"{entry.value:g}"
                arithmetic = entry.arithmetic or ""
            rows.append(
                (
                    entry.transformation.value,
                    entry.kind.value,
                    entry.what_changed,
                    entry.metric,
                    value,
                    entry.reference,
                    arithmetic,
                )
            )
        widths = [len(header) for header in headers]
        for row in rows:
            widths = [max(width, len(cell)) for width, cell in zip(widths, row, strict=True)]

        def format_row(row: tuple[str, ...]) -> str:
            return " | ".join(cell.ljust(width) for cell, width in zip(row, widths, strict=True))

        divider = "-+-".join("-" * width for width in widths)
        return "\n".join([format_row(headers), divider, *(format_row(row) for row in rows)])


def standard_loss_ledger() -> LossLedger:
    """Return the required transformation inventory with no invented zeroes."""
    reference = "Moonshot Kimi release or HuggingFace Kimi-Linear reference, as applicable"
    return LossLedger(
        [
            LossEntry(
                Transformation.SKELETON_QUANTIZATION,
                "Quantize BF16 attention, router, embeddings, or shared-expert weights",
                "quality_loss",
                reference,
            ),
            LossEntry(
                Transformation.EXPERT_REQUANTIZATION,
                "Requantize published 4.25 bpw expert tensors below their released precision",
                "quality_loss",
                reference,
            ),
            LossEntry(
                Transformation.REDUCED_TOP_K,
                "Select fewer routed experts per token than the published model",
                "quality_loss",
                reference,
            ),
            LossEntry(
                Transformation.KERNEL_NUMERICS,
                "Replace HuggingFace operators with engine kernels or lower-precision accumulation",
                "quality_loss",
                reference,
            ),
            LossEntry(
                Transformation.SPECULATIVE_DECODING,
                "Use draft-and-verify decoding instead of direct reference decoding",
                "quality_loss",
                reference,
            ),
        ]
    )


def comparison_loss_ledger(
    metrics: dict[str, float | str],
    *,
    candidate_label: str,
    reference: str,
) -> LossLedger:
    """Record the four observed comparison metrics without inventing totals."""
    required = {
        "mean_token_kl_nats",
        "mean_token_kl_arithmetic",
        "top1_agreement",
        "top1_arithmetic",
        "routing_agreement",
        "routing_arithmetic",
        "perplexity_relative_delta",
        "perplexity_arithmetic",
    }
    missing = sorted(required - set(metrics))
    if missing:
        raise ValueError(f"comparison metrics are missing required fields: {missing}")

    ledger = standard_loss_ledger()
    ledger.entries = [
        entry
        for entry in ledger.entries
        if entry.transformation is not Transformation.KERNEL_NUMERICS
    ]
    description = f"Run {candidate_label} in place of the covered HuggingFace components"
    observed = (
        ("mean_token_kl_nats", "mean_token_kl_arithmetic"),
        ("top1_agreement", "top1_arithmetic"),
        ("routing_agreement", "routing_arithmetic"),
        ("perplexity_relative_delta", "perplexity_arithmetic"),
    )
    for metric, arithmetic in observed:
        value = metrics[metric]
        formula = metrics[arithmetic]
        if not isinstance(value, (int, float)):
            raise TypeError(f"comparison metric {metric} must be numeric")
        if not isinstance(formula, str):
            raise TypeError(f"comparison arithmetic {arithmetic} must be text")
        ledger.add(
            LossEntry(
                Transformation.KERNEL_NUMERICS,
                description,
                metric,
                reference,
                MeasurementKind.MEASURED,
                float(value),
                formula,
                contributes_to_total=False,
            )
        )
    return ledger


def render_ledger(ledger: LossLedger) -> str:
    """Render a ledger as an audit-friendly plain-text table."""
    return ledger.render()
