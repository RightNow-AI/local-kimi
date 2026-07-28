"""Render measured benchmark reports and their loss ledger."""

from __future__ import annotations

from typing import Any

from .ledger import comparison_loss_ledger


def _number(value: float) -> str:
    return f"{value:.12g}"


def _layer_summary(values: list[int]) -> str:
    if not values:
        return "none"
    return f"{len(values)} layers: " + ", ".join(str(value) for value in values)


def render_measured_results(report: dict[str, Any]) -> tuple[str, str]:
    """Return RESULTS.md and LOSS-LEDGER.md text for one completed run."""
    metrics = report["metrics"]
    coverage = report["coverage"]
    reference = f"HuggingFace {report['model_id']}@{report['resolved_revision']}"
    ledger = comparison_loss_ledger(
        metrics,
        candidate_label=coverage["candidate_label"],
        reference=reference,
    )
    gpu = report["requested_gpu"]
    revision = report["resolved_revision"]
    factory = report["engine_factory"]
    status = "PASS" if report["passed"] else "FAIL"
    gates = report["gates"]
    hardware_names = ", ".join(report["hardware"]["names"]) or "not reported"
    uncovered = coverage.get("uncovered", [])

    lines = [
        "# Kimi-Linear engine benchmark results",
        "",
        f"Status: **MEASURED {status}**",
        "",
        f"Reference: `{reference}`",
        "",
        f"Candidate: `{coverage['candidate_label']}`",
        "",
        f"GPU request: `{gpu}`. Observed devices: {hardware_names}.",
        "",
        "## Measured metrics",
        "",
        "| Metric | Status | Measured value | Gate |",
        "|---|---|---:|---:|",
        (
            f"| Mean token KL, nats | MEASURED | {_number(metrics['mean_token_kl_nats'])} | "
            f"at most {_number(report['policy_limits']['max_mean_kl_nats'])} "
            f"({'PASS' if gates['mean_kl'] else 'FAIL'}) |"
        ),
        (
            f"| Top-1 agreement | MEASURED | {_number(metrics['top1_agreement'])} | "
            f"at least {_number(report['policy_limits']['min_top1_agreement'])} "
            f"({'PASS' if gates['top1'] else 'FAIL'}) |"
        ),
        (
            f"| Routing agreement | MEASURED | {_number(metrics['routing_agreement'])} | "
            f"at least {_number(report['policy_limits']['min_routing_agreement'])} "
            f"({'PASS' if gates['routing'] else 'FAIL'}) |"
        ),
        (
            f"| Reference perplexity | MEASURED | {_number(metrics['reference_perplexity'])} | n/a |"
        ),
        (
            f"| Candidate perplexity | MEASURED | {_number(metrics['candidate_perplexity'])} | n/a |"
        ),
        (
            f"| Perplexity relative delta | MEASURED | "
            f"{_number(metrics['perplexity_relative_delta'])} | absolute value at most "
            f"{_number(report['policy_limits']['max_abs_perplexity_relative_delta'])} "
            f"({'PASS' if gates['perplexity'] else 'FAIL'}) |"
        ),
        "",
        "Routing agreement covers only router keys installed by the candidate adapter. "
        f"The measured set contains {len(coverage['measured_router_keys'])} router layers.",
        "",
        "## Candidate coverage",
        "",
        f"- KDA replacements: {_layer_summary(coverage['kda_layers_replaced'])}.",
        f"- Full latent-MoE replacements: {_layer_summary(coverage['latent_moe_layers_replaced'])}.",
        f"- Router-only replacements: {_layer_summary(coverage['router_only_layers_replaced'])}.",
        "- HuggingFace components retained: "
        + ", ".join(coverage["huggingface_components_retained"])
        + ".",
        f"- Logit interpretation: {coverage['logit_interpretation']}.",
        "",
    ]
    if uncovered:
        lines.extend(["### Uncovered adapter components", ""])
        for item in uncovered:
            measured_instead = item.get("measured_instead")
            suffix = f" Measured instead: {measured_instead}." if measured_instead else ""
            lines.append(
                f"- Layer {item['layer']} {item['component']}: {item['reason']}.{suffix}"
            )
        lines.append("")

    lines.extend(
        [
            "## What this establishes",
            "",
            "This run establishes the recorded agreement metrics for the fixed Kimi-Linear "
            "corpus, exact checkpoint revision, observed GPU shape, and disclosed mixed candidate path.",
            "",
            "## What this does not establish",
            "",
            "- It does not establish full Kimi K3 quality.",
            "- It does not establish a full custom engine because the listed HuggingFace components remain.",
            "- It does not establish benchmark-task accuracy, customer-workload quality, speed, cost, or reliability.",
            "- It does not establish any transformation still marked UNMEASURED in the loss ledger.",
            "",
            "## Reproduce the measured run",
            "",
            "Run these commands in order from the repository root:",
            "",
            "```powershell",
            f"modal run engine/modal_bench.py --action download --revision {revision}",
            f"modal run engine/modal_bench.py --action reference --gpu {gpu} --revision {revision}",
            (
                f"modal run engine/modal_bench.py --action compare --gpu {gpu} "
                f"--engine-factory {factory}"
            ),
            "```",
            "",
            "The compare command rewrites this file and `engine/bench/LOSS-LEDGER.md` "
            "from the returned measured report.",
            "",
            "## Observed runtime",
            "",
            f"- Download: {_number(report['download_seconds'])} seconds.",
            f"- HuggingFace reference capture: {_number(report['reference_seconds'])} seconds.",
            f"- Candidate comparison: {_number(report['comparison_seconds'])} seconds.",
            "",
        ]
    )
    ledger_text = "# Kimi-Linear quality loss ledger\n\n" + ledger.render() + "\n"
    return "\n".join(lines), ledger_text
