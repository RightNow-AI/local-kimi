"""Generate the Kimi K3 batching decision tables from the analytic model."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

try:
    from engine.batching.union_model import (
        DEFAULT_CONCURRENCIES,
        ExpertUnionModel,
        HardwareConfig,
        RoutingPrior,
        default_hardware_configs,
        dirichlet_prior,
        zipf_prior,
    )
except ModuleNotFoundError:  # Direct execution as engine/batching/report.py.
    from union_model import (
        DEFAULT_CONCURRENCIES,
        ExpertUnionModel,
        HardwareConfig,
        RoutingPrior,
        default_hardware_configs,
        dirichlet_prior,
        zipf_prior,
    )


@dataclass(frozen=True)
class DecisionRow:
    concurrency: int
    expected_union: float
    routed_traffic_gb_per_token: float
    aggregate_tokens_per_second: float
    per_agent_tokens_per_second: float


def build_decision_rows(
    *,
    model: ExpertUnionModel,
    hardware: HardwareConfig,
    concurrencies: Iterable[int] = DEFAULT_CONCURRENCIES,
    prior: RoutingPrior | None = None,
) -> list[DecisionRow]:
    """Build rows entirely from the supplied model and hardware calibration."""
    rows = []
    for concurrency in concurrencies:
        prediction = hardware.predict(model, concurrency, prior)
        rows.append(
            DecisionRow(
                concurrency=concurrency,
                expected_union=prediction.expected_union,
                routed_traffic_gb_per_token=(
                    prediction.routed_traffic_gb_per_token
                ),
                aggregate_tokens_per_second=(
                    prediction.aggregate_tokens_per_second
                ),
                per_agent_tokens_per_second=prediction.per_agent_tokens_per_second,
            )
        )
    return rows


def _uniform_union_concurrency(model: ExpertUnionModel, fraction: float) -> int:
    if not 0.0 < fraction < 1.0:
        raise ValueError("fraction must lie in (0, 1)")
    miss = 1.0 - model.experts_per_token / model.total_experts
    return math.ceil(math.log1p(-fraction) / math.log(miss))


def _max_agent_count(
    rows: Iterable[DecisionRow], minimum_per_agent_tokens_per_second: float
) -> int | None:
    qualifying = [
        row.concurrency
        for row in rows
        if row.per_agent_tokens_per_second >= minimum_per_agent_tokens_per_second
    ]
    return max(qualifying, default=None)


def _fmt(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}f}"


def _table(rows: Iterable[DecisionRow]) -> list[str]:
    lines = [
        "| Agents | Expected union | Routed GB/token | Aggregate tok/s | Per-agent tok/s |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                (
                    str(row.concurrency),
                    _fmt(row.expected_union, 1),
                    _fmt(row.routed_traffic_gb_per_token, 2),
                    _fmt(row.aggregate_tokens_per_second, 2),
                    _fmt(row.per_agent_tokens_per_second, 2),
                )
            )
            + " |"
        )
    return lines


def render_markdown(
    *,
    model: ExpertUnionModel | None = None,
    concurrencies: tuple[int, ...] = DEFAULT_CONCURRENCIES,
) -> str:
    """Render the founder-facing decision report with modelled labels."""
    model = model or ExpertUnionModel()
    hardware_configs = default_hardware_configs(model)
    epyc_12 = hardware_configs[0]
    epyc_8 = hardware_configs[1]
    pcie = hardware_configs[2]
    zipf_mild = zipf_prior(exponent=0.5)
    zipf_strong = zipf_prior(exponent=1.0)
    dirichlet = dirichlet_prior(alpha=0.3)

    batch1_bytes = model.batch1_routed_traffic_bytes
    union_95 = _uniform_union_concurrency(model, 0.95)
    union_99 = _uniform_union_concurrency(model, 0.99)

    lines = [
        "# Kimi K3 parallel-agent batching decision report",
        "",
        "> Every throughput number below is modelled, not measured, except the stated batch-1 calibration inputs. The Modal harness is the measurement gate.",
        "",
        "## Arithmetic",
        "",
        f"- Batch-1 routed expert traffic = {model.experts_per_token} experts/token x {model.expert_bytes:,} bytes/expert x {model.moe_layers} MoE layers = {batch1_bytes:,} bytes = {batch1_bytes / 1e9:.9f} GB/token.",
        f"- Uniform union = {model.total_experts} x (1 - (1 - {model.experts_per_token}/{model.total_experts})^B).",
        "- Routed traffic/token = expected union x bytes/expert x MoE layers / B.",
        "- Batch time = routed union GB / routed bandwidth + B x calibrated non-union seconds/token.",
        "- Aggregate tok/s = B / batch time. Per-agent tok/s = aggregate tok/s / B.",
        f"- 12-channel residual = 1/9.0 - {batch1_bytes / 1e9:.9f}/450 = {epyc_12.non_union_seconds_per_token:.9f} seconds/token. Its Amdahl ceiling is 1/residual = {epyc_12.asymptotic_aggregate_tokens_per_second:.6f} tok/s.",
        f"- 8-channel batch-1 prediction = 1 / ({batch1_bytes / 1e9:.9f}/350 + {epyc_8.non_union_seconds_per_token:.9f}) = {epyc_8.predict(model, 1).aggregate_tokens_per_second:.6f} tok/s.",
        f"- PCIe residual = 1/2.1 - {batch1_bytes / 1e9:.9f}/55 = {pcie.non_union_seconds_per_token:.9f} seconds/token.",
        f"- Under uniform routing, the union reaches 95% of all experts at B={union_95} and 99% at B={union_99}.",
        "",
        "The non-union term is an Amdahl-style calibration residual. It includes compute, router, attention, shared-expert, synchronization, and implementation costs that the union-only equation does not explain. Treating all of it as per-token work gives a finite ceiling. Real batching may amortize some of it, while contention may make other parts worse.",
        "",
        "## Uniform-routing decision tables",
        "",
    ]

    all_rows: dict[str, list[DecisionRow]] = {}
    for hardware in hardware_configs:
        rows = build_decision_rows(
            model=model,
            hardware=hardware,
            concurrencies=concurrencies,
        )
        all_rows[hardware.key] = rows
        feasibility = "feasible" if hardware.feasible else "capacity-infeasible"
        ceiling = hardware.asymptotic_aggregate_tokens_per_second
        lines.extend(
            [
                f"### {hardware.label}",
                "",
                f"Status: {feasibility}. {hardware.calibration}.",
                f"Modelled non-union ceiling: {ceiling:.2f} aggregate tok/s.",
                f"Caveat: {hardware.caveat}",
                "",
                *_table(rows),
                "",
            ]
        )

    primary = hardware_configs[0]
    lines.extend(
        [
            "## Routing-skew sensitivity on the primary workstation",
            "",
            "Skew reduces the union because the same experts recur across agents. These are priors only. Gate tensors can constrain the prior, but only generation traces can measure it.",
            "",
            "| Agents | Uniform union | Zipf 0.5 union | Zipf 1.0 union | Dirichlet alpha 0.3 union | Uniform agg tok/s | Zipf 1.0 agg tok/s |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for concurrency in concurrencies:
        uniform_prediction = primary.predict(model, concurrency)
        strong_prediction = primary.predict(model, concurrency, zipf_strong)
        lines.append(
            "| "
            + " | ".join(
                (
                    str(concurrency),
                    _fmt(model.expected_union(concurrency), 1),
                    _fmt(model.expected_union(concurrency, zipf_mild), 1),
                    _fmt(model.expected_union(concurrency, zipf_strong), 1),
                    _fmt(model.expected_union(concurrency, dirichlet), 1),
                    _fmt(uniform_prediction.aggregate_tokens_per_second, 2),
                    _fmt(strong_prediction.aggregate_tokens_per_second, 2),
                )
            )
            + " |"
        )

    primary_rows = all_rows["epyc-12ch-5090"]
    secondary_rows = all_rows["epyc-8ch-5090"]
    primary_usable = _max_agent_count(primary_rows, 1.0)
    primary_comfortable = _max_agent_count(primary_rows, 2.0)
    secondary_usable = _max_agent_count(secondary_rows, 1.0)
    secondary_comfortable = _max_agent_count(secondary_rows, 2.0)
    primary_first_unusable = next(
        (
            row.concurrency
            for row in primary_rows
            if row.per_agent_tokens_per_second < 1.0
        ),
        None,
    )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            "This report defines 1 tok/s/agent as the minimum usable interactive decode rate and 2 tok/s/agent as comfortable. Those thresholds are product assumptions, not benchmark facts.",
            "",
            f"On the central 12-channel model, {primary_comfortable} agents are comfortable and {primary_usable} agents remain minimally usable among the requested concurrency points. At {primary_first_unusable} agents the uniform model is already below 1 tok/s/agent.",
            "",
            f"On the central 8-channel model, {secondary_comfortable} agents are comfortable and {secondary_usable} agents remain minimally usable. The 8-agent point is close enough to the threshold that the stated 300-400 GB/s bandwidth spread can move it across the line.",
            "",
            f"The honest planning number is therefore {primary_comfortable} parallel coding agents without qualification, or {primary_usable} on a well-tuned 12-channel workstation if roughly 1 tok/s/agent is acceptable. Do not plan around {primary_first_unusable} or more interactive agents from this model. Aggregate throughput keeps rising after that, but per-agent output becomes too slow, and the modelled aggregate ceiling is {primary.asymptotic_aggregate_tokens_per_second:.2f} tok/s before any real-router correction.",
            "",
            f"The expert union itself is effectively saturated around {union_95}-{union_99} concurrent tokens. The product becomes per-agent-rate-limited much earlier, between {primary_usable} and {primary_first_unusable} agents. The real measurement harness must replace the skew prior and the calibrated residual before this becomes a measured capacity claim.",
        ]
    )
    return "\n".join(lines) + "\n"


def build_json_report(
    *,
    model: ExpertUnionModel | None = None,
    concurrencies: tuple[int, ...] = DEFAULT_CONCURRENCIES,
) -> dict:
    model = model or ExpertUnionModel()
    scenarios: list[tuple[str, RoutingPrior | None]] = [
        ("uniform", None),
        ("zipf-0.5", zipf_prior(exponent=0.5)),
        ("zipf-1.0", zipf_prior(exponent=1.0)),
        ("dirichlet-alpha-0.3", dirichlet_prior(alpha=0.3)),
    ]
    return {
        "status": "modelled, not measured",
        "model": asdict(model),
        "scenarios": {
            scenario: {
                hardware.key: {
                    "label": hardware.label,
                    "feasible": hardware.feasible,
                    "calibration": hardware.calibration,
                    "caveat": hardware.caveat,
                    "rows": [
                        asdict(row)
                        for row in build_decision_rows(
                            model=model,
                            hardware=hardware,
                            concurrencies=concurrencies,
                            prior=prior,
                        )
                    ],
                }
                for hardware in default_hardware_configs(model)
            }
            for scenario, prior in scenarios
        },
    }


def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--format", choices=("markdown", "json"), default="markdown")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    if args.format == "json":
        rendered = json.dumps(build_json_report(), indent=2) + "\n"
    else:
        rendered = render_markdown()
    if args.output is None:
        print(rendered, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
