"""Analytic Kimi K3 expert-union and calibrated throughput model.

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
not a claim about K3's measured router traffic.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict, dataclass
from typing import Iterable, Sequence


DEFAULT_CONCURRENCIES = (1, 2, 4, 8, 16, 32, 64, 128)


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
    non_union_seconds_per_batch: float


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
    """A bandwidth path plus a calibrated non-union token-time floor."""

    key: str
    label: str
    routed_bandwidth_gb_s: float
    non_union_seconds_per_token: float
    calibration: str
    feasible: bool = True
    caveat: str = ""

    def __post_init__(self) -> None:
        if self.routed_bandwidth_gb_s <= 0.0:
            raise ValueError("routed_bandwidth_gb_s must be positive")
        if self.non_union_seconds_per_token < 0.0:
            raise ValueError("non_union_seconds_per_token cannot be negative")

    @classmethod
    def calibrated_batch1(
        cls,
        *,
        key: str,
        label: str,
        routed_bandwidth_gb_s: float,
        batch1_tokens_per_second: float,
        model: ExpertUnionModel,
        calibration: str,
        feasible: bool = True,
        caveat: str = "",
    ) -> "HardwareConfig":
        if batch1_tokens_per_second <= 0.0:
            raise ValueError("batch1_tokens_per_second must be positive")
        union_seconds = (
            model.batch1_routed_traffic_bytes / 1e9 / routed_bandwidth_gb_s
        )
        residual = 1.0 / batch1_tokens_per_second - union_seconds
        if residual < -1e-12:
            raise ValueError(
                "batch-1 calibration is faster than the routed-bandwidth roofline"
            )
        return cls(
            key=key,
            label=label,
            routed_bandwidth_gb_s=routed_bandwidth_gb_s,
            non_union_seconds_per_token=max(0.0, residual),
            calibration=calibration,
            feasible=feasible,
            caveat=caveat,
        )

    @property
    def asymptotic_aggregate_tokens_per_second(self) -> float:
        if self.non_union_seconds_per_token == 0.0:
            return math.inf
        return 1.0 / self.non_union_seconds_per_token

    def predict(
        self,
        model: ExpertUnionModel,
        concurrency: int,
        prior: RoutingPrior | None = None,
    ) -> ThroughputPrediction:
        concurrency = _positive_integer(concurrency, "concurrency")
        union = model.expected_union(concurrency, prior)
        batch_bytes = union * model.expert_bytes * model.moe_layers
        union_seconds = batch_bytes / 1e9 / self.routed_bandwidth_gb_s
        non_union_seconds = concurrency * self.non_union_seconds_per_token
        aggregate = concurrency / (union_seconds + non_union_seconds)
        return ThroughputPrediction(
            concurrency=concurrency,
            expected_union=union,
            batch_routed_traffic_gb=batch_bytes / 1e9,
            routed_traffic_gb_per_token=batch_bytes / concurrency / 1e9,
            aggregate_tokens_per_second=aggregate,
            per_agent_tokens_per_second=aggregate / concurrency,
            union_seconds_per_batch=union_seconds,
            non_union_seconds_per_batch=non_union_seconds,
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

    The 12-channel path is calibrated to the midpoint of the established
    8.4-9.6 tok/s batch-1 range. The 8-channel path inherits the same 5090 and
    non-union residual. PCIe is calibrated to the established 2.1 tok/s cap.
    NVMe and impossible all-resident VRAM rows inherit the PCIe/GPU residual.
    """
    model = model or ExpertUnionModel()
    epyc_12 = HardwareConfig.calibrated_batch1(
        key="epyc-12ch-5090",
        label="12-channel DDR5-6000 EPYC + RTX 5090",
        routed_bandwidth_gb_s=450.0,
        batch1_tokens_per_second=9.0,
        model=model,
        calibration="Modelled midpoint of measured 8.4-9.6 tok/s batch-1 v1",
        caveat="450 GB/s is the midpoint of the stated 400-500 GB/s envelope",
    )
    epyc_8 = HardwareConfig(
        key="epyc-8ch-5090",
        label="8-channel DDR5-6000 EPYC + RTX 5090",
        routed_bandwidth_gb_s=350.0,
        non_union_seconds_per_token=epyc_12.non_union_seconds_per_token,
        calibration="Modelled with 12-channel residual; no batch measurement yet",
        caveat="350 GB/s is the midpoint of the stated 300-400 GB/s envelope",
    )
    pcie = HardwareConfig.calibrated_batch1(
        key="pcie5-stream-5090",
        label="PCIe 5.0 x16 expert streaming to RTX 5090",
        routed_bandwidth_gb_s=55.0,
        batch1_tokens_per_second=2.1,
        model=model,
        calibration="Modelled curve calibrated to established near-2.1 tok/s batch-1 cap",
        caveat="Streaming avoids VRAM capacity limits but is transfer-bound",
    )
    nvme = HardwareConfig(
        key="nvme-gen5-stream",
        label="NVMe Gen5 expert streaming",
        routed_bandwidth_gb_s=14.0,
        non_union_seconds_per_token=pcie.non_union_seconds_per_token,
        calibration="Modelled from 14 GB/s roofline with PCIe/GPU residual",
        caveat="Optimistic ceiling before filesystem and page-cache overhead",
    )
    vram = HardwareConfig(
        key="rtx5090-resident-hypothetical",
        label="RTX 5090 VRAM-resident expert bank, hypothetical",
        routed_bandwidth_gb_s=1_790.0,
        non_union_seconds_per_token=pcie.non_union_seconds_per_token,
        calibration="Modelled 1.79 TB/s roofline with PCIe/GPU residual",
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
                "asymptotic_aggregate_tokens_per_second": (
                    hardware.asymptotic_aggregate_tokens_per_second
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

