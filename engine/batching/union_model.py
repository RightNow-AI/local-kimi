"""Analytic Kimi K3 expert-union and physics-constrained throughput model.

For one expert and one token, exact top-k routing selects that expert with
probability k / n under independent uniform routing. Across B independently
routed tokens, the probability that the expert is never selected is

    (1 - k / n) ** B.

Linearity of expectation then gives

    E[distinct experts] = n * (1 - (1 - k / n) ** B).

No independence between experts within one token is required. Only routing
decisions for different tokens are assumed independent.

Skewed priors are represented by per-token expert inclusion probabilities
q_i, with 0 <= q_i <= 1 and sum(q_i) = k. The corresponding expectation is

    sum_i(1 - (1 - q_i) ** B).

Zipf and Dirichlet propensities are converted to valid inclusion probabilities
with a Poissonized weighted-without-replacement approximation. This is a prior,
not a claim about K3's measured router traffic. Within this independent
fixed-marginal family, concavity makes the uniform prior the upper bound on the
expected union. Correlated real routing still requires generation traces.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict, dataclass
from typing import Iterable, Sequence


DEFAULT_CONCURRENCIES = (1, 2, 4, 8, 16, 32, 64, 128)
DEFAULT_DENSE_PARAMETERS = 57_222_000_000
DEFAULT_DENSE_BYTES = DEFAULT_DENSE_PARAMETERS * 2


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


@dataclass(frozen=True)
class RoutingPrior:
    """Marginal probability that each expert is selected by one token."""

    name: str
    inclusion_probabilities: tuple[float, ...]
    description: str

    def validate(self, total_experts: int, experts_per_token: int) -> None:
        if len(self.inclusion_probabilities) != total_experts:
            raise ValueError(
                f"prior has {len(self.inclusion_probabilities)} experts, "
                f"expected {total_experts}"
            )
        if any(q < 0.0 or q > 1.0 for q in self.inclusion_probabilities):
            raise ValueError("inclusion probabilities must lie in [0, 1]")
        total = math.fsum(self.inclusion_probabilities)
        if not math.isclose(total, experts_per_token, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError(
                f"inclusion probabilities sum to {total}, expected {experts_per_token}"
            )


@dataclass(frozen=True)
class ThroughputPrediction:
    concurrency: int
    expected_union: float
    batch_routed_traffic_gb: float
    routed_traffic_gb_per_token: float
    aggregate_tokens_per_second: float
    per_agent_tokens_per_second: float
    union_seconds_per_batch: float
    dense_seconds_per_batch: float
    dense_seconds_per_token: float
    weight_seconds_per_batch: float
    token_compute_seconds_per_batch: float
    token_compute_scale: float
    total_seconds_per_batch: float


@dataclass(frozen=True)
class ExpertUnionModel:
    total_experts: int = 896
    experts_per_token: int = 16
    moe_layers: int = 92
    expert_bytes: int = 17_547_264

    def __post_init__(self) -> None:
        _positive_integer(self.total_experts, "total_experts")
        _positive_integer(self.experts_per_token, "experts_per_token")
        _positive_integer(self.moe_layers, "moe_layers")
        _positive_integer(self.expert_bytes, "expert_bytes")
        if self.experts_per_token > self.total_experts:
            raise ValueError("experts_per_token cannot exceed total_experts")

    @property
    def batch1_routed_traffic_bytes(self) -> int:
        return self.experts_per_token * self.expert_bytes * self.moe_layers

    def expected_uniform_union(self, concurrency: int) -> float:
        """Closed-form expected union under independent uniform top-k routing."""
        concurrency = _positive_integer(concurrency, "concurrency")
        if concurrency == 1:
            return float(self.experts_per_token)
        log_miss = math.log1p(-self.experts_per_token / self.total_experts)
        return self.total_experts * -math.expm1(concurrency * log_miss)

    def expected_union(
        self, concurrency: int, prior: RoutingPrior | None = None
    ) -> float:
        concurrency = _positive_integer(concurrency, "concurrency")
        if prior is None:
            return self.expected_uniform_union(concurrency)
        prior.validate(self.total_experts, self.experts_per_token)
        if concurrency == 1:
            return float(self.experts_per_token)
        return math.fsum(
            -math.expm1(concurrency * math.log1p(-q)) if q < 1.0 else 1.0
            for q in prior.inclusion_probabilities
        )

    def batch_routed_traffic_bytes(
        self, concurrency: int, prior: RoutingPrior | None = None
    ) -> float:
        return self.expected_union(concurrency, prior) * self.expert_bytes * self.moe_layers

    def routed_traffic_bytes_per_token(
        self, concurrency: int, prior: RoutingPrior | None = None
    ) -> float:
        return self.batch_routed_traffic_bytes(concurrency, prior) / concurrency

    def curve(
        self,
        concurrencies: Iterable[int] = DEFAULT_CONCURRENCIES,
        prior: RoutingPrior | None = None,
    ) -> list[dict[str, float | int]]:
        rows = []
        for concurrency in concurrencies:
            union = self.expected_union(concurrency, prior)
            rows.append(
                {
                    "concurrency": concurrency,
                    "expected_union": union,
                    "union_fraction": union / self.total_experts,
                    "batch_routed_traffic_gb": (
                        union * self.expert_bytes * self.moe_layers / 1e9
                    ),
                    "routed_traffic_gb_per_token": (
                        self.routed_traffic_bytes_per_token(concurrency, prior) / 1e9
                    ),
                }
            )
        return rows


@dataclass(frozen=True)
class HardwareConfig:
    """A routed transport plus explicit per-pass and per-token costs."""

    key: str
    label: str
    routed_bandwidth_gb_s: float
    dense_bytes: int
    per_token_compute_seconds: float
    old_batch1_tokens_per_second: float
    calibration: str
    feasible: bool = True
    caveat: str = ""

    def __post_init__(self) -> None:
        if self.routed_bandwidth_gb_s <= 0.0:
            raise ValueError("routed_bandwidth_gb_s must be positive")
        if self.dense_bytes <= 0:
            raise ValueError("dense_bytes must be positive")
        if self.per_token_compute_seconds < 0.0:
            raise ValueError("per_token_compute_seconds cannot be negative")
        if self.old_batch1_tokens_per_second <= 0.0:
            raise ValueError("old_batch1_tokens_per_second must be positive")

    @classmethod
    def calibrated_batch1(
        cls,
        *,
        key: str,
        label: str,
        routed_bandwidth_gb_s: float,
        batch1_tokens_per_second: float,
        model: ExpertUnionModel,
        dense_bytes: int = DEFAULT_DENSE_BYTES,
        calibration: str,
        feasible: bool = True,
        caveat: str = "",
    ) -> "HardwareConfig":
        if batch1_tokens_per_second <= 0.0:
            raise ValueError("batch1_tokens_per_second must be positive")
        routed_gb = model.batch1_routed_traffic_bytes / 1e9
        dense_gb = dense_bytes / 1e9
        weight_seconds = (routed_gb + dense_gb) / routed_bandwidth_gb_s
        roofline_tokens_per_second = 1.0 / weight_seconds
        requested_seconds = 1.0 / batch1_tokens_per_second
        per_token_compute = requested_seconds - weight_seconds
        if per_token_compute < -1e-12:
            raise ValueError(
                "physically impossible batch-1 calibration: "
                f"requested={batch1_tokens_per_second:.6f} tok/s "
                f"({requested_seconds:.6f} s/pass), true_weight_roofline="
                f"{roofline_tokens_per_second:.6f} tok/s "
                f"({weight_seconds:.6f} s/pass), routed={routed_gb:.9f} GB, "
                f"dense={dense_gb:.9f} GB, bandwidth="
                f"{routed_bandwidth_gb_s:.6f} GB/s"
            )
        return cls(
            key=key,
            label=label,
            routed_bandwidth_gb_s=routed_bandwidth_gb_s,
            dense_bytes=dense_bytes,
            per_token_compute_seconds=max(0.0, per_token_compute),
            old_batch1_tokens_per_second=batch1_tokens_per_second,
            calibration=calibration,
            feasible=feasible,
            caveat=caveat,
        )

    def batch1_weight_roofline_tokens_per_second(
        self, model: ExpertUnionModel
    ) -> float:
        total_gb = (model.batch1_routed_traffic_bytes + self.dense_bytes) / 1e9
        return self.routed_bandwidth_gb_s / total_gb

    def predict(
        self,
        model: ExpertUnionModel,
        concurrency: int,
        prior: RoutingPrior | None = None,
        compute_scale: float = 1.0,
    ) -> ThroughputPrediction:
        concurrency = _positive_integer(concurrency, "concurrency")
        if compute_scale <= 0.0 or not math.isfinite(compute_scale):
            raise ValueError("compute_scale must be finite and positive")
        union = model.expected_union(concurrency, prior)
        batch_bytes = union * model.expert_bytes * model.moe_layers
        union_seconds = batch_bytes / 1e9 / self.routed_bandwidth_gb_s
        dense_seconds = self.dense_bytes / 1e9 / self.routed_bandwidth_gb_s
        weight_seconds = union_seconds + dense_seconds
        token_compute_seconds = (
            concurrency * self.per_token_compute_seconds * compute_scale
        )
        total_seconds = weight_seconds + token_compute_seconds
        aggregate = concurrency / total_seconds
        return ThroughputPrediction(
            concurrency=concurrency,
            expected_union=union,
            batch_routed_traffic_gb=batch_bytes / 1e9,
            routed_traffic_gb_per_token=batch_bytes / concurrency / 1e9,
            aggregate_tokens_per_second=aggregate,
            per_agent_tokens_per_second=aggregate / concurrency,
            union_seconds_per_batch=union_seconds,
            dense_seconds_per_batch=dense_seconds,
            dense_seconds_per_token=dense_seconds / concurrency,
            weight_seconds_per_batch=weight_seconds,
            token_compute_seconds_per_batch=token_compute_seconds,
            token_compute_scale=compute_scale,
            total_seconds_per_batch=total_seconds,
        )


def _propensities_to_inclusions(
    propensities: Sequence[float], experts_per_token: int
) -> tuple[float, ...]:
    """Map positive routing propensities to marginals that sum to top-k.

    q_i = 1 - exp(-lambda * p_i) is the Poissonized inclusion probability.
    Lambda is solved so the expected number of unique selections is exactly k.
    """
    if not propensities or any(p <= 0.0 or not math.isfinite(p) for p in propensities):
        raise ValueError("propensities must be finite and strictly positive")
    if experts_per_token <= 0 or experts_per_token > len(propensities):
        raise ValueError("experts_per_token must lie in [1, number of propensities]")
    total = math.fsum(propensities)
    probabilities = tuple(p / total for p in propensities)

    def selected(scale: float) -> float:
        return math.fsum(-math.expm1(-scale * p) for p in probabilities)

    low, high = 0.0, float(experts_per_token)
    while selected(high) < experts_per_token:
        high *= 2.0
    for _ in range(100):
        mid = (low + high) / 2.0
        if selected(mid) < experts_per_token:
            low = mid
        else:
            high = mid
    scale = (low + high) / 2.0
    values = [-math.expm1(-scale * p) for p in probabilities]
    # Remove the last few ulps so strict validation and B=1 stay exact enough.
    correction = experts_per_token - math.fsum(values)
    target = max(range(len(values)), key=values.__getitem__)
    values[target] += correction
    return tuple(values)


def zipf_prior(
    *, total_experts: int = 896, experts_per_token: int = 16, exponent: float = 1.0
) -> RoutingPrior:
    """Create a deterministic Zipf propensity prior over ranked experts."""
    _positive_integer(total_experts, "total_experts")
    _positive_integer(experts_per_token, "experts_per_token")
    if exponent <= 0.0 or not math.isfinite(exponent):
        raise ValueError("exponent must be finite and positive")
    propensities = [1.0 / (rank**exponent) for rank in range(1, total_experts + 1)]
    inclusion = _propensities_to_inclusions(propensities, experts_per_token)
    return RoutingPrior(
        name=f"zipf-{exponent:g}",
        inclusion_probabilities=inclusion,
        description=(
            f"Modelled Zipf routing propensity with exponent {exponent:g}; "
            "not measured K3 routing"
        ),
    )


def dirichlet_prior(
    *,
    total_experts: int = 896,
    experts_per_token: int = 16,
    alpha: float = 0.3,
    seed: int = 20260728,
) -> RoutingPrior:
    """Create one reproducible symmetric-Dirichlet routing prior."""
    _positive_integer(total_experts, "total_experts")
    _positive_integer(experts_per_token, "experts_per_token")
    if alpha <= 0.0 or not math.isfinite(alpha):
        raise ValueError("alpha must be finite and positive")
    rng = random.Random(seed)
    propensities = [rng.gammavariate(alpha, 1.0) for _ in range(total_experts)]
    inclusion = _propensities_to_inclusions(propensities, experts_per_token)
    return RoutingPrior(
        name=f"dirichlet-alpha-{alpha:g}-seed-{seed}",
        inclusion_probabilities=inclusion,
        description=(
            f"Modelled symmetric Dirichlet routing propensity with alpha={alpha:g}; "
            "not measured K3 routing"
        ),
    )


def default_hardware_configs(
    model: ExpertUnionModel | None = None,
) -> tuple[HardwareConfig, ...]:
    """Central estimates for the hardware envelopes in the lane brief.

    These are optimistic bandwidth-only rooflines. Every forward pass moves the
    expected routed union plus 57.222B BF16 non-routed parameters on the stated
    bandwidth path. Per-token compute is set to zero because the prior 9.0 and
    2.1 tok/s anchors are physically impossible and cannot calibrate it.
    """
    model = model or ExpertUnionModel()
    epyc_12 = HardwareConfig(
        key="epyc-12ch-5090",
        label="12-channel DDR5-6000 EPYC + RTX 5090",
        routed_bandwidth_gb_s=450.0,
        dense_bytes=DEFAULT_DENSE_BYTES,
        per_token_compute_seconds=0.0,
        old_batch1_tokens_per_second=9.0,
        calibration="Physics-derived bandwidth-only upper bound",
        caveat=(
            "450 GB/s is the midpoint of the stated 400-500 GB/s envelope; "
            "real compute can only reduce throughput"
        ),
    )
    epyc_8 = HardwareConfig(
        key="epyc-8ch-5090",
        label="8-channel DDR5-6000 EPYC + RTX 5090",
        routed_bandwidth_gb_s=350.0,
        dense_bytes=DEFAULT_DENSE_BYTES,
        per_token_compute_seconds=0.0,
        old_batch1_tokens_per_second=7.8425,
        calibration="Physics-derived bandwidth-only upper bound",
        caveat=(
            "350 GB/s is the midpoint of the stated 300-400 GB/s envelope; "
            "real compute can only reduce throughput"
        ),
    )
    pcie = HardwareConfig(
        key="pcie5-stream-5090",
        label="PCIe 5.0 x16 expert streaming to RTX 5090",
        routed_bandwidth_gb_s=55.0,
        dense_bytes=DEFAULT_DENSE_BYTES,
        per_token_compute_seconds=0.0,
        old_batch1_tokens_per_second=2.1,
        calibration="Physics-derived bandwidth-only upper bound",
        caveat=(
            "Treats routed and dense bytes on one 55 GB/s path as requested; "
            "real topology and compute can only reduce this ceiling"
        ),
    )
    nvme = HardwareConfig(
        key="nvme-gen5-stream",
        label="NVMe Gen5 expert streaming",
        routed_bandwidth_gb_s=14.0,
        dense_bytes=DEFAULT_DENSE_BYTES,
        per_token_compute_seconds=0.0,
        old_batch1_tokens_per_second=0.5267,
        calibration="Physics-derived bandwidth-only upper bound",
        caveat="Optimistic ceiling before filesystem and page-cache overhead",
    )
    vram = HardwareConfig(
        key="rtx5090-resident-hypothetical",
        label="RTX 5090 VRAM-resident expert bank, hypothetical",
        routed_bandwidth_gb_s=1_790.0,
        dense_bytes=DEFAULT_DENSE_BYTES,
        per_token_compute_seconds=0.0,
        old_batch1_tokens_per_second=14.6752,
        calibration="Physics-derived bandwidth-only upper bound",
        feasible=False,
        caveat="Not buildable: 1,446.46 GB of experts cannot fit in 32 GiB VRAM",
    )
    return epyc_12, epyc_8, pcie, nvme, vram


def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zipf", type=float, default=None, help="Zipf exponent")
    parser.add_argument("--dirichlet-alpha", type=float, default=None)
    parser.add_argument("--seed", type=int, default=20260728)
    args = parser.parse_args()

    model = ExpertUnionModel()
    prior = None
    if args.zipf is not None and args.dirichlet_alpha is not None:
        parser.error("choose at most one skew prior")
    if args.zipf is not None:
        prior = zipf_prior(exponent=args.zipf)
    elif args.dirichlet_alpha is not None:
        prior = dirichlet_prior(alpha=args.dirichlet_alpha, seed=args.seed)

    output = {
        "status": "modelled, not measured",
        "routing_prior": prior.description if prior else "independent uniform top-16",
        "model": asdict(model),
        "curve": model.curve(prior=prior),
        "hardware": {
            hardware.key: {
                "label": hardware.label,
                "calibration": hardware.calibration,
                "feasible": hardware.feasible,
                "caveat": hardware.caveat,
                "dense_bytes": hardware.dense_bytes,
                "per_token_compute_seconds": hardware.per_token_compute_seconds,
                "old_batch1_tokens_per_second": (
                    hardware.old_batch1_tokens_per_second
                ),
                "corrected_batch1_tokens_per_second": (
                    hardware.predict(model, 1, prior).aggregate_tokens_per_second
                ),
                "batch1_weight_roofline_tokens_per_second": (
                    hardware.batch1_weight_roofline_tokens_per_second(model)
                ),
                "curve": [
                    asdict(hardware.predict(model, b, prior))
                    for b in DEFAULT_CONCURRENCIES
                ],
            }
            for hardware in default_hardware_configs(model)
        },
    }
    print(json.dumps(output, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
