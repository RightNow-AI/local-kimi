"""GPU residency budget for Kimi-Linear-48B-A3B-Instruct.

The persistent state shapes are derived from the model repository at revision
e1df551a447157d4658b573f9a695d57658590e9 and from the unpinned ``fla-core``
dependency at revision 9c8e42e762fce087c27b673af4922795d9edb85e. The
compressed MLA policy is derived from vLLM 0.26.0 at revision
568afb3a13806beb53bb2e6bd518269357b237c0.

The shipped BF16 and selective INT4 weight profiles are measured tensor-storage
inputs. Persistent state rates are source-derived, and operational headroom is
an explicit projected policy. This module performs byte arithmetic only. It
does not claim that a full live-server envelope was measured or can be served
until the Modal harness and an engine boot produce matching allocation results.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Iterable

GIB = 1 << 30
MIB = 1 << 20
MODEL_ID = "moonshotai/Kimi-Linear-48B-A3B-Instruct"
MODEL_REVISION = "e1df551a447157d4658b573f9a695d57658590e9"
FLA_REVISION = "9c8e42e762fce087c27b673af4922795d9edb85e"
VLLM_REVISION = "568afb3a13806beb53bb2e6bd518269357b237c0"
MODEL_MAX_LENGTH = 1_048_576


@dataclass(frozen=True, slots=True)
class ScalarDType:
    """A storage dtype used by one persistent state pool."""

    key: str
    bytes_per_element: int

    def __post_init__(self) -> None:
        if not self.key:
            raise ValueError("dtype key cannot be empty")
        if self.bytes_per_element <= 0:
            raise ValueError("dtype bytes_per_element must be positive")


BF16 = ScalarDType("bfloat16", 2)
FP16 = ScalarDType("float16", 2)
FP32 = ScalarDType("float32", 4)
DTYPES = {dtype.key: dtype for dtype in (BF16, FP16, FP32)}


@dataclass(frozen=True, slots=True)
class StateDTypes:
    """Persistent state dtypes.

    ``FP32/BF16/BF16`` matches the inspected model and FLA allocation paths.
    Other values are projections for a changed runtime, not the shipped path.
    """

    recurrent_state: ScalarDType = FP32
    short_conv_state: ScalarDType = BF16
    mla_kv_cache: ScalarDType = BF16

    @property
    def matches_inspected_runtime(self) -> bool:
        return (
            self.recurrent_state == FP32
            and self.short_conv_state == BF16
            and self.mla_kv_cache == BF16
        )


INSPECTED_STATE_DTYPES = StateDTypes()


class MLACachePolicy(str, Enum):
    """Persistent MLA cache layouts used by real model implementations."""

    EXPANDED = "expanded"
    COMPRESSED_LATENT = "compressed_latent"


HF_REFERENCE_MLA_CACHE_POLICY = MLACachePolicy.EXPANDED
VLLM_MLA_CACHE_POLICY = MLACachePolicy.COMPRESSED_LATENT


@dataclass(frozen=True, slots=True)
class QuantizationProfile:
    """Resident weight bytes supplied by a named weight format profile."""

    key: str
    label: str
    weight_bytes: int
    evidence_status: str = "PROJECTED"

    def __post_init__(self) -> None:
        if not self.key or not self.label:
            raise ValueError("quantization profile key and label cannot be empty")
        if self.weight_bytes <= 0:
            raise ValueError("weight_bytes must be positive")
        if self.evidence_status not in {"MEASURED", "PROJECTED"}:
            raise ValueError("weight evidence_status must be MEASURED or PROJECTED")


MEASURED_BF16_WEIGHTS = QuantizationProfile(
    key="bf16-as-shipped-measured",
    label="BF16 source tensor storage as shipped",
    weight_bytes=98_245_528_576,
    evidence_status="MEASURED",
)
MEASURED_INT4_SELECTIVE_WEIGHTS = QuantizationProfile(
    key="int4-selective-measured",
    label="Selective INT4 output tensor storage",
    weight_bytes=28_803_304_448,
    evidence_status="MEASURED",
)
SUPERSEDED_FLAT_INT4_WEIGHTS = QuantizationProfile(
    key="int4-flat-4bit-superseded",
    label="Superseded flat 4.0-bit whole-model arithmetic",
    weight_bytes=24_561_340_864,
    evidence_status="PROJECTED",
)

# Compatibility aliases now resolve to the measured tensor-storage profiles.
BF16_WEIGHTS = MEASURED_BF16_WEIGHTS
INT4_WEIGHTS = MEASURED_INT4_SELECTIVE_WEIGHTS
QUANTIZATION_PROFILES = {
    "bf16": MEASURED_BF16_WEIGHTS,
    MEASURED_BF16_WEIGHTS.key: MEASURED_BF16_WEIGHTS,
    "int4": MEASURED_INT4_SELECTIVE_WEIGHTS,
    MEASURED_INT4_SELECTIVE_WEIGHTS.key: MEASURED_INT4_SELECTIVE_WEIGHTS,
    "int4-weight-only": INT4_WEIGHTS,
}


@dataclass(frozen=True, slots=True)
class KimiLinearResidencyShape:
    """Architecture fields that determine persistent GPU state."""

    kda_layers: int = 20
    mla_layers: int = 7
    kda_num_heads: int = 32
    kda_key_head_dim: int = 128
    kda_value_head_dim: int = 128
    short_conv_kernel_size: int = 4
    mla_num_heads: int = 32
    mla_kv_lora_rank: int = 512
    mla_qk_nope_head_dim: int = 128
    mla_qk_rope_head_dim: int = 64
    mla_value_head_dim: int = 128

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")

    @property
    def kda_q_width(self) -> int:
        return self.kda_num_heads * self.kda_key_head_dim

    @property
    def kda_k_width(self) -> int:
        return self.kda_num_heads * self.kda_key_head_dim

    @property
    def kda_v_width(self) -> int:
        return self.kda_num_heads * self.kda_value_head_dim

    @property
    def mla_key_elements_per_token_per_layer(self) -> int:
        return self.mla_num_heads * (
            self.mla_qk_nope_head_dim + self.mla_qk_rope_head_dim
        )

    @property
    def mla_value_elements_per_token_per_layer(self) -> int:
        return self.mla_num_heads * self.mla_value_head_dim

    @property
    def mla_compressed_elements_per_token_per_layer(self) -> int:
        return self.mla_kv_lora_rank + self.mla_qk_rope_head_dim


KIMI_LINEAR_SHAPE = KimiLinearResidencyShape()


@dataclass(frozen=True, slots=True)
class RuntimeHeadroom:
    """Explicit policy reserves for allocations not represented by state tensors."""

    activation_bytes: int = 2 * GIB
    workspace_bytes: int = 1 * GIB

    def __post_init__(self) -> None:
        if self.activation_bytes < 0 or self.workspace_bytes < 0:
            raise ValueError("headroom bytes cannot be negative")

    @property
    def total_bytes(self) -> int:
        return self.activation_bytes + self.workspace_bytes


DEFAULT_HEADROOM = RuntimeHeadroom()


@dataclass(frozen=True, slots=True)
class ResidencyBreakdown:
    """A complete mixed-evidence byte breakdown for one server envelope."""

    quantization_profile: str
    mla_cache_policy: str
    max_num_seqs: int
    max_model_len: int
    state_dtypes: StateDTypes
    weights_bytes: int
    weights_evidence_status: str
    kda_recurrent_state_bytes: int
    short_conv_state_bytes: int
    mla_kv_cache_bytes: int
    activation_headroom_bytes: int
    workspace_headroom_bytes: int
    total_bytes: int
    evidence_status: str = "PROJECTED"

    @property
    def state_pool_bytes(self) -> int:
        return (
            self.kda_recurrent_state_bytes
            + self.short_conv_state_bytes
            + self.mla_kv_cache_bytes
        )

    @property
    def operational_headroom_bytes(self) -> int:
        return self.activation_headroom_bytes + self.workspace_headroom_bytes

    def fits_in(self, vram_bytes: int) -> bool:
        if vram_bytes < 0:
            raise ValueError("vram_bytes cannot be negative")
        return self.total_bytes <= vram_bytes

    def as_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["state_pool_bytes"] = self.state_pool_bytes
        data["operational_headroom_bytes"] = self.operational_headroom_bytes
        data["state_dtypes_match_inspected_runtime"] = (
            self.state_dtypes.matches_inspected_runtime
        )
        return data


@dataclass(frozen=True, slots=True)
class FrontierPoint:
    """One non-dominated server envelope under a fixed VRAM budget."""

    max_num_seqs: int
    max_model_len: int
    budget: ResidencyBreakdown
    vram_bytes: int
    slack_bytes: int
    evidence_status: str = "PROJECTED"

    def __post_init__(self) -> None:
        if self.budget.total_bytes > self.vram_bytes:
            raise ValueError("frontier point exceeds its VRAM budget")
        if self.slack_bytes != self.vram_bytes - self.budget.total_bytes:
            raise ValueError("frontier point slack does not match its budget")

    def as_dict(self) -> dict[str, object]:
        return {
            "max_num_seqs": self.max_num_seqs,
            "max_model_len": self.max_model_len,
            "vram_bytes": self.vram_bytes,
            "slack_bytes": self.slack_bytes,
            "evidence_status": self.evidence_status,
            "budget": self.budget.as_dict(),
        }


class ResidencyBudgetExceeded(ValueError):
    """Raised when a requested envelope does not fit the supplied capacity."""


def resolve_quantization_profile(
    profile: QuantizationProfile | str,
) -> QuantizationProfile:
    if isinstance(profile, QuantizationProfile):
        return profile
    try:
        return QUANTIZATION_PROFILES[profile]
    except KeyError as exc:
        known = ", ".join(sorted(QUANTIZATION_PROFILES))
        raise ValueError(f"unknown quantization profile {profile!r}; choose {known}") from exc


def resolve_mla_cache_policy(
    policy: MLACachePolicy | str,
) -> MLACachePolicy:
    if isinstance(policy, MLACachePolicy):
        return policy
    if not isinstance(policy, str):
        raise ValueError("MLA cache policy must be a string or MLACachePolicy")
    aliases = {
        "expanded": MLACachePolicy.EXPANDED,
        "compressed": MLACachePolicy.COMPRESSED_LATENT,
        "compressed-latent": MLACachePolicy.COMPRESSED_LATENT,
        "compressed latent": MLACachePolicy.COMPRESSED_LATENT,
        "compressed_latent": MLACachePolicy.COMPRESSED_LATENT,
    }
    try:
        return aliases[policy.strip().lower()]
    except KeyError as exc:
        known = ", ".join(item.value for item in MLACachePolicy)
        raise ValueError(f"unknown MLA cache policy {policy!r}; choose {known}") from exc


def kda_recurrent_bytes_per_sequence(
    *,
    shape: KimiLinearResidencyShape = KIMI_LINEAR_SHAPE,
    dtype: ScalarDType = FP32,
) -> int:
    elements = (
        shape.kda_layers
        * shape.kda_num_heads
        * shape.kda_key_head_dim
        * shape.kda_value_head_dim
    )
    return elements * dtype.bytes_per_element


def short_conv_bytes_per_sequence(
    *,
    shape: KimiLinearResidencyShape = KIMI_LINEAR_SHAPE,
    dtype: ScalarDType = BF16,
) -> int:
    channels = shape.kda_q_width + shape.kda_k_width + shape.kda_v_width
    elements = shape.kda_layers * channels * shape.short_conv_kernel_size
    return elements * dtype.bytes_per_element


def mla_kv_bytes_per_token_per_sequence(
    *,
    cache_policy: MLACachePolicy | str = HF_REFERENCE_MLA_CACHE_POLICY,
    shape: KimiLinearResidencyShape = KIMI_LINEAR_SHAPE,
    dtype: ScalarDType = BF16,
) -> int:
    policy = resolve_mla_cache_policy(cache_policy)
    if policy is MLACachePolicy.EXPANDED:
        elements_per_layer = (
            shape.mla_key_elements_per_token_per_layer
            + shape.mla_value_elements_per_token_per_layer
        )
    else:
        elements_per_layer = shape.mla_compressed_elements_per_token_per_layer
    return shape.mla_layers * elements_per_layer * dtype.bytes_per_element


def build_residency_budget(
    quantization_profile: QuantizationProfile | str,
    max_num_seqs: int,
    max_model_len: int,
    *,
    mla_cache_policy: MLACachePolicy | str = HF_REFERENCE_MLA_CACHE_POLICY,
    state_dtypes: StateDTypes = INSPECTED_STATE_DTYPES,
    headroom: RuntimeHeadroom = DEFAULT_HEADROOM,
    shape: KimiLinearResidencyShape = KIMI_LINEAR_SHAPE,
) -> ResidencyBreakdown:
    """Return every persistent pool and explicit operational reserve in bytes."""

    if max_num_seqs <= 0:
        raise ValueError("max_num_seqs must be positive")
    if max_model_len <= 0:
        raise ValueError("max_model_len must be positive")
    if max_model_len > MODEL_MAX_LENGTH:
        raise ValueError(
            f"max_model_len cannot exceed the advertised model limit {MODEL_MAX_LENGTH}"
        )

    profile = resolve_quantization_profile(quantization_profile)
    resolved_mla_cache_policy = resolve_mla_cache_policy(mla_cache_policy)
    recurrent = max_num_seqs * kda_recurrent_bytes_per_sequence(
        shape=shape,
        dtype=state_dtypes.recurrent_state,
    )
    conv = max_num_seqs * short_conv_bytes_per_sequence(
        shape=shape,
        dtype=state_dtypes.short_conv_state,
    )
    mla = (
        max_num_seqs
        * max_model_len
        * mla_kv_bytes_per_token_per_sequence(
            cache_policy=resolved_mla_cache_policy,
            shape=shape,
            dtype=state_dtypes.mla_kv_cache,
        )
    )
    total = (
        profile.weight_bytes
        + recurrent
        + conv
        + mla
        + headroom.activation_bytes
        + headroom.workspace_bytes
    )
    return ResidencyBreakdown(
        quantization_profile=profile.key,
        mla_cache_policy=resolved_mla_cache_policy.value,
        max_num_seqs=max_num_seqs,
        max_model_len=max_model_len,
        state_dtypes=state_dtypes,
        weights_bytes=profile.weight_bytes,
        weights_evidence_status=profile.evidence_status,
        kda_recurrent_state_bytes=recurrent,
        short_conv_state_bytes=conv,
        mla_kv_cache_bytes=mla,
        activation_headroom_bytes=headroom.activation_bytes,
        workspace_headroom_bytes=headroom.workspace_bytes,
        total_bytes=total,
    )


def require_envelope_fits(
    vram_bytes: int,
    quantization_profile: QuantizationProfile | str,
    max_num_seqs: int,
    max_model_len: int,
    *,
    mla_cache_policy: MLACachePolicy | str = HF_REFERENCE_MLA_CACHE_POLICY,
    state_dtypes: StateDTypes = INSPECTED_STATE_DTYPES,
    headroom: RuntimeHeadroom = DEFAULT_HEADROOM,
    shape: KimiLinearResidencyShape = KIMI_LINEAR_SHAPE,
) -> ResidencyBreakdown:
    """Return a requested envelope or refuse it if any byte exceeds capacity."""

    if vram_bytes <= 0:
        raise ValueError("vram_bytes must be positive")
    budget = build_residency_budget(
        quantization_profile,
        max_num_seqs,
        max_model_len,
        mla_cache_policy=mla_cache_policy,
        state_dtypes=state_dtypes,
        headroom=headroom,
        shape=shape,
    )
    if not budget.fits_in(vram_bytes):
        excess = budget.total_bytes - vram_bytes
        raise ResidencyBudgetExceeded(
            "requested envelope exceeds VRAM: "
            f"required={budget.total_bytes}, available={vram_bytes}, excess={excess}"
        )
    return budget


def solve_residency_frontier(
    vram_bytes: int,
    quantization_profile: QuantizationProfile | str,
    *,
    max_num_seqs_values: Iterable[int] = (1, 2, 4, 8, 16, 32, 64, 128, 256),
    max_model_len_cap: int = MODEL_MAX_LENGTH,
    mla_cache_policy: MLACachePolicy | str = HF_REFERENCE_MLA_CACHE_POLICY,
    state_dtypes: StateDTypes = INSPECTED_STATE_DTYPES,
    headroom: RuntimeHeadroom = DEFAULT_HEADROOM,
    shape: KimiLinearResidencyShape = KIMI_LINEAR_SHAPE,
) -> tuple[FrontierPoint, ...]:
    """Return the exact non-dominated integer envelope frontier.

    For each supplied sequence-pool size, the solver computes the largest
    integer ``max_model_len`` that fits. It never rounds an infeasible point
    into the result.
    """

    if vram_bytes <= 0:
        raise ValueError("vram_bytes must be positive")
    if max_model_len_cap <= 0 or max_model_len_cap > MODEL_MAX_LENGTH:
        raise ValueError(
            f"max_model_len_cap must be in [1, {MODEL_MAX_LENGTH}]"
        )

    profile = resolve_quantization_profile(quantization_profile)
    resolved_mla_cache_policy = resolve_mla_cache_policy(mla_cache_policy)
    sequence_values = tuple(sorted(set(max_num_seqs_values)))
    if not sequence_values or any(value <= 0 for value in sequence_values):
        raise ValueError("max_num_seqs_values must contain positive integers")

    recurrent_per_seq = kda_recurrent_bytes_per_sequence(
        shape=shape,
        dtype=state_dtypes.recurrent_state,
    )
    conv_per_seq = short_conv_bytes_per_sequence(
        shape=shape,
        dtype=state_dtypes.short_conv_state,
    )
    mla_per_token_per_seq = mla_kv_bytes_per_token_per_sequence(
        cache_policy=resolved_mla_cache_policy,
        shape=shape,
        dtype=state_dtypes.mla_kv_cache,
    )

    candidates: list[FrontierPoint] = []
    for max_num_seqs in sequence_values:
        fixed = (
            profile.weight_bytes
            + headroom.total_bytes
            + max_num_seqs * (recurrent_per_seq + conv_per_seq)
        )
        bytes_per_model_token = max_num_seqs * mla_per_token_per_seq
        available_for_mla = vram_bytes - fixed
        if available_for_mla < bytes_per_model_token:
            continue
        max_model_len = min(
            max_model_len_cap,
            available_for_mla // bytes_per_model_token,
        )
        budget = require_envelope_fits(
            vram_bytes,
            profile,
            max_num_seqs,
            max_model_len,
            mla_cache_policy=resolved_mla_cache_policy,
            state_dtypes=state_dtypes,
            headroom=headroom,
            shape=shape,
        )
        candidates.append(
            FrontierPoint(
                max_num_seqs=max_num_seqs,
                max_model_len=max_model_len,
                budget=budget,
                vram_bytes=vram_bytes,
                slack_bytes=vram_bytes - budget.total_bytes,
            )
        )

    frontier = []
    for candidate in candidates:
        dominated = any(
            other.max_num_seqs >= candidate.max_num_seqs
            and other.max_model_len >= candidate.max_model_len
            and (
                other.max_num_seqs > candidate.max_num_seqs
                or other.max_model_len > candidate.max_model_len
            )
            for other in candidates
        )
        if not dominated:
            frontier.append(candidate)
    return tuple(frontier)
