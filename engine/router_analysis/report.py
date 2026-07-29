"""Human-readable rendering for the measured router comparison."""

from __future__ import annotations

from typing import Any


def render_markdown(report: dict[str, Any]) -> str:
    overall = report["overall"]
    overlap = overall["router_probability_mass_overlap"]
    disagreeing = overall["disagreeing_expert_weight"]
    histogram = overall["disagreement_rank_histogram"]
    lines = [
        "# Weighted router comparison",
        "",
        f"- Reference checkpoint: `{report['reference_checkpoint']}`",
        f"- Candidate checkpoint: `{report['candidate_checkpoint']}`",
        f"- Prompt count: {report['prompt_count']}",
        f"- Prompt tokens: {report['prompt_token_count']}",
        f"- Token-layer observations: {overall['routing_observation_count']}",
        "",
        "## Overall metrics",
        "",
        f"- Top-1 expert agreement: {overall['top1_expert_agreement']:.6f}",
        f"- Probability mass overlap mean: {overlap['mean']:.6f}",
        f"- Probability mass overlap median: {overlap['median']:.6f}",
        f"- Probability mass overlap 10th percentile: {overlap['p10']:.6f}",
        "- Mean disagreeing expert weight: "
        + _format_optional(disagreeing["mean"]),
        f"- Disagreeing expert occurrences: {disagreeing['occurrence_count']}",
        "",
        "## Disagreement rank histogram",
        "",
        "| Rank | Count |",
        "| ---: | ---: |",
    ]
    for rank, count in histogram.items():
        lines.append(f"| {rank} | {count} |")
    hypothesis = report["rank_7_8_hypothesis"]
    lines.extend(
        [
            "",
            "## Rank 7 and 8 hypothesis",
            "",
            hypothesis["assessment"].capitalize() + ".",
            "",
            "## Per-layer metrics",
            "",
            "| Layer | Tokens | Top-1 | Mass mean | Mass median | Mass p10 | "
            "Mean disagreeing weight | R1 | R2 | R3 | R4 | R5 | R6 | R7 | R8 |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | "
            "---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for layer in report["per_layer"]:
        layer_overlap = layer["router_probability_mass_overlap"]
        layer_histogram = layer["disagreement_rank_histogram"]
        layer_disagreeing = layer["disagreeing_expert_weight"]
        rank_counts = " | ".join(str(layer_histogram[rank]) for rank in range(1, 9))
        lines.append(
            f"| {layer['layer_index']} | {layer['routing_observation_count']} | "
            f"{layer['top1_expert_agreement']:.6f} | {layer_overlap['mean']:.6f} | "
            f"{layer_overlap['median']:.6f} | {layer_overlap['p10']:.6f} | "
            f"{_format_optional(layer_disagreeing['mean'])} | {rank_counts} |"
        )
    lines.extend(
        [
            "",
            "No unordered set-agreement score is emitted. Each component above is "
            "reported separately.",
        ]
    )
    return "\n".join(lines)


def _format_optional(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.6f}"
