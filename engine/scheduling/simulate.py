"""Simulate routing-aware composition under round-based arrival processes.

All routes and performance numbers produced here are simulated. Uniform routes
use exact uniform top-k sampling. Skewed routes use randomized systematic
fixed-size sampling with the marginal inclusion probabilities supplied by
``engine.batching.union_model``. Neither process is a trace from the real K3
router.

The real grouped-top-k constraint is not reconstructed here. That correlation
requires real router traces or a faithful router implementation.
"""

from __future__ import annotations

import argparse
import bisect
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Iterator, Literal, Sequence

from engine.batching.union_model import (
    ExpertUnionModel,
    HardwareConfig,
    RoutingPrior,
    default_hardware_configs,
    dirichlet_prior,
    zipf_prior,
)

from .compose import PendingToken, compare_composers, compose_batch
from .metrics import (
    FairnessMetrics,
    measure_fairness,
    measure_union,
    predict_observed_union,
)

ArrivalProcess = Literal["saturated", "bursty", "sparse"]


@dataclass
class _WaitState:
    arrival_round: int
    arrival_seconds: float
    deferred_rounds: int = 0


@dataclass(frozen=True)
class PolicySimulation:
    routing_prior: str
    strategy: str
    arrival_process: ArrivalProcess
    pool_size: int
    max_batch_size: int
    rounds: int
    batches_run: int
    tokens_served: int
    mean_batch_size: float
    total_expert_layer_pairs: int
    mean_union_per_layer: float
    analytic_expected_union_per_layer: float
    aggregate_tokens_per_second: float
    total_service_seconds: float
    fairness: FairnessMetrics
    guard_fallbacks: int
    paired_random_expert_layer_pairs: int


@dataclass(frozen=True)
class SimulationComparison:
    routing_prior: str
    arrival_process: ArrivalProcess
    pool_size: int
    max_batch_size: int
    rounds: int
    random: PolicySimulation
    greedy: PolicySimulation
    same_pool_union_reduction_fraction: float
    end_to_end_union_reduction_fraction: float
    bytes_saved_per_token: float
    throughput_gain_fraction: float
    added_worst_deferred_rounds: int
    added_worst_wait_seconds: float


@dataclass(frozen=True)
class DiminishingReturnPoint:
    routing_prior: str
    arrival_process: ArrivalProcess
    max_batch_size: int
    pool_size: int
    previous_pool_size: int
    incremental_throughput_gain_fraction: float
    added_worst_deferred_rounds: int
    criterion: str


class _RouteSampler:
    """Produce fixed-size routes with model-supplied first-order marginals."""

    def __init__(
        self,
        model: ExpertUnionModel,
        prior: RoutingPrior | None,
        *,
        seed: int,
        skew_orderings: int = 32,
    ) -> None:
        self._model = model
        self._prior = prior
        self._rng = random.Random(seed)
        self._expert_range = range(model.total_experts)
        self._skew_tables: tuple[tuple[tuple[int, ...], tuple[float, ...]], ...] = ()
        if prior is not None:
            prior.validate(model.total_experts, model.experts_per_token)
            if skew_orderings <= 0:
                raise ValueError("skew_orderings must be positive")
            tables = []
            for _ in range(skew_orderings):
                order = list(self._expert_range)
                self._rng.shuffle(order)
                cumulative = []
                running = 0.0
                for expert in order:
                    running += prior.inclusion_probabilities[expert]
                    cumulative.append(running)
                cumulative[-1] = float(model.experts_per_token)
                tables.append((tuple(order), tuple(cumulative)))
            self._skew_tables = tuple(tables)

    def sample(self) -> frozenset[int]:
        if self._prior is None:
            return frozenset(
                self._rng.sample(self._expert_range, self._model.experts_per_token)
            )

        order, cumulative = self._rng.choice(self._skew_tables)
        start = self._rng.random()
        selected = {
            order[min(bisect.bisect_right(cumulative, start + offset), len(order) - 1)]
            for offset in range(self._model.experts_per_token)
        }
        if len(selected) != self._model.experts_per_token:
            raise RuntimeError("fixed-marginal route sampler produced a duplicate expert")
        return frozenset(selected)


def _token_stream(
    model: ExpertUnionModel,
    prior: RoutingPrior | None,
    *,
    seed: int,
) -> Iterator[PendingToken]:
    sampler = _RouteSampler(model, prior, seed=seed)
    token_id = 0
    while True:
        routes = tuple(sampler.sample() for _ in range(model.moe_layers))
        yield PendingToken(token_id=token_id, layer_experts=routes)
        token_id += 1


def _arrival_count(
    process: ArrivalProcess, round_index: int, max_batch_size: int
) -> int:
    if round_index == 0:
        return 0
    if process == "saturated":
        return max_batch_size
    if process == "bursty":
        return 2 * max_batch_size if round_index % 2 == 1 else 0
    if process == "sparse":
        return max(1, max_batch_size // 2)
    raise ValueError(f"unknown arrival process: {process}")


def _initial_arrival_count(
    process: ArrivalProcess, pool_size: int, max_batch_size: int
) -> int:
    if process in ("saturated", "bursty"):
        return pool_size
    if process == "sparse":
        return min(pool_size, max(1, max_batch_size // 2))
    raise ValueError(f"unknown arrival process: {process}")


def _validate_positive_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def simulate_policy(
    *,
    model: ExpertUnionModel,
    hardware: HardwareConfig,
    prior: RoutingPrior | None,
    routing_prior_name: str,
    strategy: Literal["random", "greedy"],
    arrival_process: ArrivalProcess,
    pool_size: int,
    max_batch_size: int,
    rounds: int,
    route_seed: int,
    selection_seed: int,
) -> PolicySimulation:
    """Run one policy against a reproducible route and arrival stream."""

    pool_size = _validate_positive_integer(pool_size, "pool_size")
    max_batch_size = _validate_positive_integer(max_batch_size, "max_batch_size")
    rounds = _validate_positive_integer(rounds, "rounds")
    stream = _token_stream(model, prior, seed=route_seed)
    pending: list[PendingToken] = []
    waits: dict[object, _WaitState] = {}
    for _ in range(_initial_arrival_count(arrival_process, pool_size, max_batch_size)):
        token = next(stream)
        pending.append(token)
        waits[token.token_id] = _WaitState(arrival_round=0, arrival_seconds=0.0)

    clock = 0.0
    total_union = 0
    paired_random_total = 0
    analytic_union_total = 0.0
    served_round_waits: list[int] = []
    served_second_waits: list[float] = []
    batch_sizes: list[int] = []
    guard_fallbacks = 0

    for round_index in range(rounds):
        for _ in range(_arrival_count(arrival_process, round_index, max_batch_size)):
            token = next(stream)
            pending.append(token)
            waits[token.token_id] = _WaitState(
                arrival_round=round_index,
                arrival_seconds=clock,
            )

        candidate_pool = pending[:pool_size]
        if not candidate_pool:
            continue
        round_seed = selection_seed + round_index
        if strategy == "greedy":
            decision = compare_composers(
                candidate_pool,
                max_batch_size,
                seed=round_seed,
            )
            selected = decision.selected
            paired_random_total += decision.random_total_union
            guard_fallbacks += int(decision.used_random_guard)
        else:
            selected = compose_batch(
                candidate_pool,
                max_batch_size,
                strategy="random",
                seed=round_seed,
            )

        union = measure_union(selected)
        batch_size = len(selected)
        prediction = predict_observed_union(
            hardware,
            model,
            batch_size,
            union.mean_union_per_layer,
            prior=prior,
        )
        batch_sizes.append(batch_size)
        total_union += union.total_union
        if strategy == "random":
            paired_random_total += union.total_union
        analytic_union_total += model.expected_union(batch_size, prior)

        selected_ids = {token.token_id for token in selected}
        for token in selected:
            state = waits.pop(token.token_id)
            served_round_waits.append(state.deferred_rounds)
            served_second_waits.append(clock - state.arrival_seconds)
        pending = [token for token in pending if token.token_id not in selected_ids]
        for token in pending:
            waits[token.token_id].deferred_rounds += 1
        clock += prediction.total_seconds_per_batch

    outstanding_rounds = [waits[token.token_id].deferred_rounds for token in pending]
    outstanding_seconds = [clock - waits[token.token_id].arrival_seconds for token in pending]
    fairness = measure_fairness(
        served_round_waits,
        outstanding_deferred_rounds=outstanding_rounds,
        served_wait_seconds=served_second_waits,
        outstanding_wait_seconds=outstanding_seconds,
    )
    tokens_served = sum(batch_sizes)
    batches_run = len(batch_sizes)
    return PolicySimulation(
        routing_prior=routing_prior_name,
        strategy=strategy,
        arrival_process=arrival_process,
        pool_size=pool_size,
        max_batch_size=max_batch_size,
        rounds=rounds,
        batches_run=batches_run,
        tokens_served=tokens_served,
        mean_batch_size=(math.fsum(batch_sizes) / batches_run if batches_run else 0.0),
        total_expert_layer_pairs=total_union,
        mean_union_per_layer=(
            total_union / batches_run / model.moe_layers if batches_run else 0.0
        ),
        analytic_expected_union_per_layer=(
            analytic_union_total / batches_run if batches_run else 0.0
        ),
        aggregate_tokens_per_second=(tokens_served / clock if clock else 0.0),
        total_service_seconds=clock,
        fairness=fairness,
        guard_fallbacks=guard_fallbacks,
        paired_random_expert_layer_pairs=paired_random_total,
    )


def simulate_comparison(
    *,
    model: ExpertUnionModel,
    hardware: HardwareConfig,
    prior: RoutingPrior | None,
    routing_prior_name: str,
    arrival_process: ArrivalProcess,
    pool_size: int,
    max_batch_size: int,
    rounds: int,
    seed: int,
) -> SimulationComparison:
    common = {
        "model": model,
        "hardware": hardware,
        "prior": prior,
        "routing_prior_name": routing_prior_name,
        "arrival_process": arrival_process,
        "pool_size": pool_size,
        "max_batch_size": max_batch_size,
        "rounds": rounds,
        "route_seed": seed,
        "selection_seed": seed + 10_000_019,
    }
    random_result = simulate_policy(strategy="random", **common)
    greedy_result = simulate_policy(strategy="greedy", **common)
    if random_result.tokens_served != greedy_result.tokens_served:
        raise RuntimeError("policies served different token counts under the same arrivals")

    random_union = random_result.total_expert_layer_pairs
    greedy_union = greedy_result.total_expert_layer_pairs
    local_random_union = greedy_result.paired_random_expert_layer_pairs
    tokens_served = random_result.tokens_served
    return SimulationComparison(
        routing_prior=routing_prior_name,
        arrival_process=arrival_process,
        pool_size=pool_size,
        max_batch_size=max_batch_size,
        rounds=rounds,
        random=random_result,
        greedy=greedy_result,
        same_pool_union_reduction_fraction=(
            (local_random_union - greedy_union) / local_random_union
            if local_random_union
            else 0.0
        ),
        end_to_end_union_reduction_fraction=(
            (random_union - greedy_union) / random_union if random_union else 0.0
        ),
        bytes_saved_per_token=(
            (random_union - greedy_union) * model.expert_bytes / tokens_served
            if tokens_served
            else 0.0
        ),
        throughput_gain_fraction=(
            greedy_result.aggregate_tokens_per_second
            / random_result.aggregate_tokens_per_second
            - 1.0
            if random_result.aggregate_tokens_per_second
            else 0.0
        ),
        added_worst_deferred_rounds=(
            greedy_result.fairness.worst_case_deferred_rounds
            - random_result.fairness.worst_case_deferred_rounds
        ),
        added_worst_wait_seconds=(
            greedy_result.fairness.worst_case_wait_seconds
            - random_result.fairness.worst_case_wait_seconds
        ),
    )


def run_sweep(
    *,
    model: ExpertUnionModel,
    hardware: HardwareConfig,
    routing_priors: Sequence[tuple[str, RoutingPrior | None]],
    arrival_processes: Sequence[ArrivalProcess],
    batch_sizes: Sequence[int],
    pool_multipliers: Sequence[int],
    rounds: int,
    seed: int,
) -> list[SimulationComparison]:
    results = []
    for prior_index, (prior_name, prior) in enumerate(routing_priors):
        for process_index, arrival_process in enumerate(arrival_processes):
            for max_batch_size in batch_sizes:
                for multiplier in pool_multipliers:
                    pool_size = max_batch_size * multiplier
                    scenario_seed = (
                        seed
                        + prior_index * 100_000_007
                        + process_index * 1_000_003
                        + max_batch_size * 10_007
                    )
                    results.append(
                        simulate_comparison(
                            model=model,
                            hardware=hardware,
                            prior=prior,
                            routing_prior_name=prior_name,
                            arrival_process=arrival_process,
                            pool_size=pool_size,
                            max_batch_size=max_batch_size,
                            rounds=rounds,
                            seed=scenario_seed,
                        )
                    )
    return results


def find_diminishing_returns(
    results: Iterable[SimulationComparison],
    *,
    minimum_incremental_gain_fraction: float = 0.01,
) -> list[DiminishingReturnPoint]:
    """Find the first larger pool adding under 1% throughput while wait worsens."""

    grouped: dict[tuple[str, ArrivalProcess, int], list[SimulationComparison]] = {}
    for result in results:
        key = (result.routing_prior, result.arrival_process, result.max_batch_size)
        grouped.setdefault(key, []).append(result)

    points = []
    for (prior_name, process, batch_size), group in grouped.items():
        ordered = sorted(group, key=lambda result: result.pool_size)
        for previous, current in zip(ordered, ordered[1:]):
            incremental_gain = (
                current.greedy.aggregate_tokens_per_second
                / previous.greedy.aggregate_tokens_per_second
                - 1.0
            )
            added_wait = (
                current.greedy.fairness.worst_case_deferred_rounds
                - previous.greedy.fairness.worst_case_deferred_rounds
            )
            if incremental_gain < minimum_incremental_gain_fraction and added_wait > 0:
                points.append(
                    DiminishingReturnPoint(
                        routing_prior=prior_name,
                        arrival_process=process,
                        max_batch_size=batch_size,
                        pool_size=current.pool_size,
                        previous_pool_size=previous.pool_size,
                        incremental_throughput_gain_fraction=incremental_gain,
                        added_worst_deferred_rounds=added_wait,
                        criterion=(
                            "incremental modeled throughput below 1% while worst-case "
                            "deferral increased"
                        ),
                    )
                )
                break
    return points


def render_markdown(
    results: Sequence[SimulationComparison],
    *,
    model: ExpertUnionModel,
    hardware: HardwareConfig,
    rounds: int,
) -> str:
    """Render a complete simulated-results report for RESULTS.md."""

    lines = [
        "# Routing-aware batch composition results",
        "",
        "> Status: SIMULATED, not measured on real K3 router traces or a real scheduler.",
        "",
        "## Fixed arithmetic",
        "",
        (
            f"- Batch-1 routed traffic is `{model.experts_per_token} * "
            f"{model.expert_bytes:,} * {model.moe_layers} = "
            f"{model.batch1_routed_traffic_bytes:,}` bytes, or "
            f"`{model.batch1_routed_traffic_bytes / 1e9:.9f}` GB per token."
        ),
        (
            "- Simulated bytes saved per token are "
            "`(random expert-layer pairs - greedy expert-layer pairs) * "
            f"{model.expert_bytes:,} / tokens served`."
        ),
        (
            "- The throughput feedback uses `HardwareConfig.predict` from "
            f"`engine/batching/union_model.py` with `{hardware.key}` and replaces only "
            "the observed union-derived routed-byte and dequant terms."
        ),
        "",
        "## Simulation method",
        "",
        f"- Each row runs `{rounds}` scheduling rounds with seed-controlled routes.",
        "- Uniform routing is exact uniform top-k sampling.",
        (
            "- Skewed routing uses the union model's inclusion probabilities with "
            "randomized systematic fixed-size sampling. It is not a measured K3 trace."
        ),
        "- Real grouped-top-k routing correlations are not modeled.",
        (
            "- The optimization objective is the sum of per-layer unions, which equals "
            "average union times layer count and directly tracks routed bytes."
        ),
        (
            "- Fairness includes served and still-outstanding tokens. Worst deferral is "
            "the maximum number of completed scheduling rounds a token waited."
        ),
        "",
        "## Simulated sweep",
        "",
        (
            "| prior | arrivals | B | P | analytic union/layer | random union/layer | "
            "greedy union/layer | local/end-to-end reduction | saved MB/token | "
            "throughput gain | worst rounds random/greedy | worst wait ms random/greedy | "
            "guard fallbacks |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        lines.append(
            "| "
            f"{result.routing_prior} | {result.arrival_process} | "
            f"{result.max_batch_size} | {result.pool_size} | "
            f"{result.random.analytic_expected_union_per_layer:.3f} | "
            f"{result.random.mean_union_per_layer:.3f} | "
            f"{result.greedy.mean_union_per_layer:.3f} | "
            f"{result.same_pool_union_reduction_fraction * 100:.2f}%/"
            f"{result.end_to_end_union_reduction_fraction * 100:.2f}% | "
            f"{result.bytes_saved_per_token / 1e6:.3f} | "
            f"{result.throughput_gain_fraction * 100:.2f}% | "
            f"{result.random.fairness.worst_case_deferred_rounds}/"
            f"{result.greedy.fairness.worst_case_deferred_rounds} | "
            f"{result.random.fairness.worst_case_wait_seconds * 1_000:.3f}/"
            f"{result.greedy.fairness.worst_case_wait_seconds * 1_000:.3f} | "
            f"{result.greedy.guard_fallbacks} |"
        )

    points = find_diminishing_returns(results)
    lines.extend(["", "## Where a larger composition pool stops paying", ""])
    no_choice = [result for result in results if result.pool_size <= result.max_batch_size]
    if no_choice:
        lines.append(
            "- At `P <= B`, every pending token is selected, so composition has no choice and "
            "must produce exactly the random union."
        )
    if points:
        for point in points:
            lines.append(
                f"- `{point.routing_prior}`, `{point.arrival_process}`, B={point.max_batch_size}: "
                f"stop at P={point.previous_pool_size} before P={point.pool_size} under the "
                f"declared 1% rule. Incremental throughput was "
                f"{point.incremental_throughput_gain_fraction * 100:.2f}% and worst deferral "
                f"increased by {point.added_worst_deferred_rounds} rounds."
            )
    else:
        lines.append(
            "- No row met the declared stop rule: less than 1% incremental modeled throughput "
            "with a higher worst-case deferral. This does not prove larger pools are free."
        )

    best = max(results, key=lambda result: result.throughput_gain_fraction, default=None)
    worst_fairness = max(results, key=lambda result: result.added_worst_deferred_rounds, default=None)
    lines.extend(["", "## Honest read", ""])
    if best is None or best.throughput_gain_fraction <= 0.01:
        lines.append(
            "- This simulation does not show enough modeled throughput gain to justify a real "
            "scheduler implementation yet."
        )
    else:
        lines.append(
            f"- The best simulated modeled throughput gain is "
            f"{best.throughput_gain_fraction * 100:.2f}% for `{best.routing_prior}`, "
            f"`{best.arrival_process}`, B={best.max_batch_size}, P={best.pool_size}. This is "
            "enough to justify a bounded trace-driven prototype, not production integration."
        )
    if worst_fairness is not None:
        lines.append(
            f"- The largest simulated fairness cost is "
            f"{worst_fairness.added_worst_deferred_rounds:+d} worst-case deferral rounds "
            f"relative to random. A real scheduler needs an explicit age or deadline cap."
        )
    lines.append(
        "- Real-model routing traces, router lookahead latency, request cancellation, and "
        "continuous-time arrivals remain unmeasured."
    )
    lines.append(
        "- Composer CPU time and real grouped-top-k routing correlations are also unmeasured, "
        "so a small modeled throughput gain may disappear in implementation overhead."
    )
    return "\n".join(lines) + "\n"


def _parse_ints(value: str) -> tuple[int, ...]:
    parsed = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not parsed or any(item <= 0 for item in parsed):
        raise argparse.ArgumentTypeError("expected comma-separated positive integers")
    return parsed


def _build_priors(names: Sequence[str]) -> tuple[tuple[str, RoutingPrior | None], ...]:
    if not names:
        raise ValueError("at least one routing prior is required")
    priors = []
    for name in names:
        if name == "uniform":
            priors.append(("uniform", None))
        elif name == "zipf":
            priors.append(("zipf-1", zipf_prior(exponent=1.0)))
        elif name == "dirichlet":
            priors.append(("dirichlet-0.3", dirichlet_prior(alpha=0.3)))
        else:
            raise ValueError(f"unknown prior: {name}")
    return tuple(priors)


def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-sizes", type=_parse_ints, default=(8, 16, 32))
    parser.add_argument("--pool-multipliers", type=_parse_ints, default=(1, 2, 4))
    parser.add_argument("--rounds", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument(
        "--arrival-processes",
        default="saturated,bursty,sparse",
        help="comma-separated saturated, bursty, sparse",
    )
    parser.add_argument(
        "--priors",
        default="uniform,zipf,dirichlet",
        help="comma-separated uniform, zipf, dirichlet",
    )
    parser.add_argument("--hardware", default="epyc-12ch-5090")
    parser.add_argument("--format", choices=("json", "markdown"), default="markdown")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    model = ExpertUnionModel()
    hardware_by_key = {item.key: item for item in default_hardware_configs(model)}
    if args.hardware not in hardware_by_key:
        parser.error(f"unknown hardware key: {args.hardware}")
    try:
        arrival_processes = tuple(
            part.strip() for part in args.arrival_processes.split(",") if part.strip()
        )
        if not arrival_processes:
            raise ValueError("at least one arrival process is required")
        if any(item not in ("saturated", "bursty", "sparse") for item in arrival_processes):
            raise ValueError("unknown arrival process")
        priors = _build_priors(
            tuple(part.strip() for part in args.priors.split(",") if part.strip())
        )
    except ValueError as exc:
        parser.error(str(exc))

    hardware = hardware_by_key[args.hardware]
    results = run_sweep(
        model=model,
        hardware=hardware,
        routing_priors=priors,
        arrival_processes=arrival_processes,
        batch_sizes=args.batch_sizes,
        pool_multipliers=args.pool_multipliers,
        rounds=args.rounds,
        seed=args.seed,
    )
    if args.format == "json":
        output = json.dumps(
            {
                "status": "simulated, not measured on a real model",
                "model": asdict(model),
                "hardware": asdict(hardware),
                "results": [asdict(result) for result in results],
                "diminishing_returns": [
                    asdict(point) for point in find_diminishing_returns(results)
                ],
            },
            indent=2,
        )
    else:
        output = render_markdown(results, model=model, hardware=hardware, rounds=args.rounds)

    if args.output is None:
        print(output, end="")
    else:
        args.output.write_text(output, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
