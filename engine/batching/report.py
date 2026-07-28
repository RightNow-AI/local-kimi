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
        DEFAULT_DENSE_BYTES,
        DEFAULT_DENSE_PARAMETERS,
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
        DEFAULT_DENSE_BYTES,
        DEFAULT_DENSE_PARAMETERS,
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
    dense_milliseconds_per_token: float
    aggregate_tokens_per_second: float
    per_agent_tokens_per_second: float


def build_decision_rows(
    *,
    model: ExpertUnionModel,
    hardware: HardwareConfig,
    concurrencies: Iterable[int] = DEFAULT_CONCURRENCIES,
    prior: RoutingPrior | None = None,
    compute_scale: float = 1.0,
) -> list[DecisionRow]:
    """Build rows entirely from the supplied model and hardware calibration."""
    rows = []
    for concurrency in concurrencies:
        prediction = hardware.predict(
            model, concurrency, prior, compute_scale=compute_scale
        )
        rows.append(
            DecisionRow(
                concurrency=concurrency,
                expected_union=prediction.expected_union,
                routed_traffic_gb_per_token=(
                    prediction.routed_traffic_gb_per_token
                ),
                dense_milliseconds_per_token=(
                    prediction.dense_seconds_per_token * 1_000.0
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
        "| Agents | Expected union | Routed GB/token | Dense ms/token | Aggregate tok/s | Per-agent tok/s |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                (
                    str(row.concurrency),
                    _fmt(row.expected_union, 1),
                    _fmt(row.routed_traffic_gb_per_token, 2),
                    _fmt(row.dense_milliseconds_per_token, 3),
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
    zipf_mild = zipf_prior(exponent=0.5)
    zipf_strong = zipf_prior(exponent=1.0)
    dirichlet = dirichlet_prior(alpha=0.3)

    batch1_bytes = model.batch1_routed_traffic_bytes
    routed_gb = batch1_bytes / 1e9
    dense_gb = DEFAULT_DENSE_BYTES / 1e9
    total_batch1_weight_gb = routed_gb + dense_gb
    batch1_weight_seconds = total_batch1_weight_gb / epyc_12.routed_bandwidth_gb_s
    old_remaining_seconds = 1.0 / 9.0 - routed_gb / 450.0
    old_implied_dense_bandwidth = dense_gb / old_remaining_seconds
    union_95 = _uniform_union_concurrency(model, 0.95)
    union_99 = _uniform_union_concurrency(model, 0.99)

    lines = [
        "# Kimi K3 parallel-agent batching decision report",
        "",
        "> Every throughput number below is modelled, not measured. The 9.0 and 2.1 tok/s batch-1 anchors are retained only as refuted historical comparisons. The Modal harness is the measurement gate.",
        "",
        "## Arithmetic",
        "",
        f"- Batch-1 routed expert traffic = {model.experts_per_token} experts/token x {model.expert_bytes:,} bytes/expert x {model.moe_layers} MoE layers = {batch1_bytes:,} bytes = {batch1_bytes / 1e9:.9f} GB/token.",
        "- K3 activates 18 experts per token: 16 routed 4-bit experts plus 2 shared BF16 experts. The 2 shared experts are included in the dense mass, not in routed expert traffic.",
        f"- Uniform union = {model.total_experts} x (1 - (1 - {model.experts_per_token}/{model.total_experts})^B).",
        "- Routed traffic/token = expected union x bytes/expert x MoE layers / B.",
        f"- Dense mass = {DEFAULT_DENSE_PARAMETERS:,} BF16 parameters x 2 bytes = {DEFAULT_DENSE_BYTES:,} bytes = {dense_gb:.3f} GB per forward pass.",
        "- Corrected batch time = (routed union bytes + dense bytes) / bandwidth + B x per-token compute.",
        "- Aggregate tok/s = B / batch time. Per-agent tok/s = aggregate tok/s / B.",
        f"- True 12-channel batch-1 weight time = ({routed_gb:.9f} + {dense_gb:.3f})/450 = {batch1_weight_seconds:.9f} seconds, so the optimistic weight-only roofline is {1.0 / batch1_weight_seconds:.6f} tok/s.",
        f"- The refuted 9.0 tok/s anchor leaves only {old_remaining_seconds * 1_000:.6f} ms after routed traffic and therefore implies {old_implied_dense_bandwidth:,.1f} GB/s for the dense mass against a 450 GB/s bus.",
        "- Per-token compute defaults to 0 seconds because no physically valid measurement remains to calibrate it. This makes every corrected table an optimistic upper bound; real compute can only reduce throughput.",
        f"- Dense cost/token at B=32 is ({dense_gb:.3f}/450)/32 = {dense_gb / 450.0 / 32.0 * 1_000:.6f} ms, exactly 1/32 of B=1.",
        f"- Under uniform routing, the union reaches 95% of all experts at B={union_95} and 99% at B={union_99}.",
        "",
        "The uniform independent-routing prior is the upper-bound union within the fixed-marginal skew family used here. Real token correlations are not measured by this prior. Only instrumented generation traces establish the real union curve.",
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
        lines.extend(
            [
                f"### {hardware.label}",
                "",
                f"Status: {feasibility}. {hardware.calibration}.",
                f"Old batch-1 anchor or inherited estimate: {hardware.old_batch1_tokens_per_second:.4f} tok/s.",
                f"Corrected optimistic batch-1 roofline: {hardware.predict(model, 1).aggregate_tokens_per_second:.4f} tok/s.",
                f"Dense pass cost: {hardware.dense_bytes / 1e9 / hardware.routed_bandwidth_gb_s * 1_000:.3f} ms for {hardware.dense_bytes / 1e9:.3f} GB at {hardware.routed_bandwidth_gb_s:,.0f} GB/s.",
                f"Per-token compute input: {hardware.per_token_compute_seconds * 1_000:.3f} ms/token.",
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

    compute_scales = (1.0, 0.5, 0.33)
    sensitivity_rows = {
        scale: build_decision_rows(
            model=model,
            hardware=primary,
            concurrencies=concurrencies,
            compute_scale=scale,
        )
        for scale in compute_scales
    }
    lines.extend(
        [
            "",
            "## Kernel-optimization sensitivity on the primary workstation",
            "",
            "Everything except the per-token compute term is held fixed. Its baseline is deliberately 0 ms because both prior anchors are physically impossible. Therefore 1.0x, 0.5x, and 0.33x are identical optimistic weight-bandwidth ceilings.",
            "",
            "Kernel work cannot improve these modeled numbers because the model already assumes free token compute. Measured nonzero compute would lower the 1.0x result, and optimization could only recover toward this table.",
            "",
            "| Agents | 1.0x agg | 1.0x per agent | 0.5x agg | 0.5x per agent | 0.33x agg | 0.33x per agent |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for index, concurrency in enumerate(concurrencies):
        base = sensitivity_rows[1.0][index]
        half = sensitivity_rows[0.5][index]
        third = sensitivity_rows[0.33][index]
        lines.append(
            "| "
            + " | ".join(
                (
                    str(concurrency),
                    _fmt(base.aggregate_tokens_per_second, 2),
                    _fmt(base.per_agent_tokens_per_second, 2),
                    _fmt(half.aggregate_tokens_per_second, 2),
                    _fmt(half.per_agent_tokens_per_second, 2),
                    _fmt(third.aggregate_tokens_per_second, 2),
                    _fmt(third.per_agent_tokens_per_second, 2),
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
    optimized_usable = {
        scale: _max_agent_count(rows, 1.0)
        for scale, rows in sensitivity_rows.items()
    }
    optimized_comfortable = {
        scale: _max_agent_count(rows, 2.0)
        for scale, rows in sensitivity_rows.items()
    }
    lines.extend(
        [
            "",
            "## Decision",
            "",
            "This report defines 1 tok/s/agent as the minimum usable interactive decode rate and 2 tok/s/agent as comfortable. Those thresholds are product assumptions, not benchmark facts.",
            "",
            f"On the central 12-channel model, {primary_comfortable} agents are comfortable and {primary_usable} agents remain minimally usable among the requested concurrency points. At {primary_first_unusable} agents the uniform model is already below 1 tok/s/agent.",
            "",
            f"On the central 8-channel model, {secondary_comfortable} agents are comfortable and {secondary_usable} agents remain minimally usable. The corrected 8-agent midpoint is {secondary_rows[concurrencies.index(8)].per_agent_tokens_per_second:.2f} tok/s/agent.",
            "",
            f"The optimistic planning ceiling is {primary_comfortable} parallel coding agents at or above 2 tok/s/agent, or {primary_usable} at or above 1 tok/s/agent under the uniform upper-bound union. At {primary_first_unusable} agents even the zero-compute roofline is only {primary_rows[concurrencies.index(primary_first_unusable)].per_agent_tokens_per_second:.2f} tok/s/agent.",
            "",
            f"Kernel sensitivity does not move the result because the baseline already assumes zero token-compute cost: 1.0x, 0.5x, and 0.33x each retain {optimized_usable[1.0]}, {optimized_usable[0.5]}, and {optimized_usable[0.33]} agents at or above 1 tok/s/agent. Comfortable counts are {optimized_comfortable[1.0]}, {optimized_comfortable[0.5]}, and {optimized_comfortable[0.33]}. At every scale, 16 uniform-routed agents reach only {sensitivity_rows[0.33][concurrencies.index(16)].per_agent_tokens_per_second:.2f} tok/s/agent.",
            "",
            f"The hard purchase-planning answer is therefore no more than {primary_usable} minimally usable agents on the central 12-channel roofline, with {primary_comfortable} the safer interactive count. Because compute is assumed free, measured capacity can only be worse. The real measurement harness must replace the routing prior and zero-compute assumption before this becomes a measured capacity claim.",
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
        "timing_decomposition": {
            "formula": (
                "(union_bytes+dense_bytes)/bandwidth + "
                "B*per_token_compute"
            ),
            "dense_parameters_bf16": DEFAULT_DENSE_PARAMETERS,
            "dense_bytes": DEFAULT_DENSE_BYTES,
            "default_per_token_compute_seconds": 0.0,
        },
        "scenarios": {
            scenario: {
                hardware.key: {
                    "label": hardware.label,
                    "feasible": hardware.feasible,
                    "calibration": hardware.calibration,
                    "caveat": hardware.caveat,
                    "old_batch1_tokens_per_second": (
                        hardware.old_batch1_tokens_per_second
                    ),
                    "corrected_batch1_tokens_per_second": (
                        hardware.predict(model, 1, prior).aggregate_tokens_per_second
                    ),
                    "dense_seconds_per_batch": (
                        hardware.dense_bytes / 1e9 / hardware.routed_bandwidth_gb_s
                    ),
                    "per_token_compute_seconds": (
                        hardware.per_token_compute_seconds
                    ),
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
        "kernel_sensitivity_primary_uniform": {
            str(scale): {
                "rows": [
                    asdict(row)
                    for row in build_decision_rows(
                        model=model,
                        hardware=default_hardware_configs(model)[0],
                        concurrencies=concurrencies,
                        compute_scale=scale,
                    )
                ],
            }
            for scale in (1.0, 0.5, 0.33)
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
