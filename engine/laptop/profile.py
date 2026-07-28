"""Offline byte and bandwidth model for Kimi-Linear on laptop memory paths."""

from __future__ import annotations

from dataclasses import dataclass

MODEL_ID = "moonshotai/Kimi-Linear-48B-A3B-Instruct"
DECIMAL_GB = 1_000_000_000
GIB = 1 << 30
MEMORY_CAPACITIES_GIB = (16, 24, 32, 64)
K3_MEASURED_BANDWIDTH_ATTAINMENT = 0.60


@dataclass(frozen=True)
class KimiLinearArchitecture:
    """Config fields that determine resident and per-token parameter traffic."""

    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    q_lora_rank: int | None
    kv_lora_rank: int
    qk_nope_head_dim: int
    qk_rope_head_dim: int
    v_head_dim: int
    kda_num_heads: int
    kda_head_dim: int
    short_conv_kernel_size: int
    kda_use_full_rank_gate: bool
    full_attention_layers: tuple[int, ...]
    num_experts: int
    num_experts_per_token: int
    num_shared_experts: int
    moe_intermediate_size: int
    first_k_dense_replace: int
    moe_layer_freq: int
    routed_expert_hidden_size: int | None = None
    latent_moe_use_norm: bool = False
    mla_use_output_gate: bool = False
    attn_res_block_size: int | None = None


# Pinned from the model config. Layer numbers in full_attention_layers are one-based.
KIMI_LINEAR_48B = KimiLinearArchitecture(
    vocab_size=163_840,
    hidden_size=2_304,
    intermediate_size=9_216,
    num_hidden_layers=27,
    num_attention_heads=32,
    q_lora_rank=None,
    kv_lora_rank=512,
    qk_nope_head_dim=128,
    qk_rope_head_dim=64,
    v_head_dim=128,
    kda_num_heads=32,
    kda_head_dim=128,
    short_conv_kernel_size=4,
    kda_use_full_rank_gate=False,
    full_attention_layers=(4, 8, 12, 16, 20, 24, 27),
    num_experts=256,
    num_experts_per_token=8,
    num_shared_experts=1,
    moe_intermediate_size=1_024,
    first_k_dense_replace=1,
    moe_layer_freq=1,
)


@dataclass(frozen=True)
class ParameterCounts:
    total_parameters: int
    active_parameters_per_token: int
    resident_embedding_parameters: int
    active_embedding_parameters: int
    lm_head_parameters: int
    decoder_parameters: int
    active_decoder_parameters: int


def _dense_mlp_parameters(config: KimiLinearArchitecture) -> int:
    # Gate, up, and down matrices.
    return 3 * config.hidden_size * config.intermediate_size


def _moe_parameters(config: KimiLinearArchitecture, routed_experts: int) -> int:
    expert_hidden = config.routed_expert_hidden_size or config.hidden_size
    routed = routed_experts * 3 * expert_hidden * config.moe_intermediate_size
    router = config.num_experts * config.hidden_size + config.num_experts
    shared = (
        config.num_shared_experts
        * 3
        * config.hidden_size
        * config.moe_intermediate_size
    )
    latent = 0
    if config.routed_expert_hidden_size is not None:
        latent = 2 * config.hidden_size * expert_hidden
        if config.latent_moe_use_norm:
            latent += expert_hidden
    return routed + router + shared + latent


def _kda_parameters(config: KimiLinearArchitecture) -> int:
    hidden = config.hidden_size
    projection = config.kda_num_heads * config.kda_head_dim
    qkv = 3 * projection * hidden
    short_convs = 3 * projection * config.short_conv_kernel_size
    decay = (
        config.kda_num_heads
        + config.kda_head_dim * hidden
        + projection * config.kda_head_dim
        + projection
        + config.kda_num_heads * hidden
    )
    if config.kda_use_full_rank_gate:
        output_gate = projection * hidden
    else:
        output_gate = config.kda_head_dim * hidden + projection * config.kda_head_dim
    output = config.kda_head_dim + hidden * projection
    return qkv + short_convs + decay + output_gate + output


def _mla_parameters(config: KimiLinearArchitecture) -> int:
    hidden = config.hidden_size
    q_head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim
    q_projection = config.num_attention_heads * q_head_dim
    if config.q_lora_rank is None:
        query = q_projection * hidden
    else:
        query = (
            config.q_lora_rank * hidden
            + config.q_lora_rank
            + q_projection * config.q_lora_rank
        )
    compressed_kv = (
        (config.kv_lora_rank + config.qk_rope_head_dim) * hidden
        + config.kv_lora_rank
        + config.num_attention_heads
        * (config.qk_nope_head_dim + config.v_head_dim)
        * config.kv_lora_rank
    )
    output = hidden * config.num_attention_heads * config.v_head_dim
    if config.mla_use_output_gate:
        output += config.num_attention_heads * config.v_head_dim * hidden
    return query + compressed_kv + output


def _common_layer_parameters(config: KimiLinearArchitecture) -> int:
    parameters = 2 * config.hidden_size
    if config.attn_res_block_size is not None:
        # Two learned residual norms and two hidden-to-scalar projections.
        parameters += 4 * config.hidden_size
    return parameters


def parameter_counts(config: KimiLinearArchitecture) -> ParameterCounts:
    """Recompute exact resident and active weight counts from architecture fields."""
    decoder = 0
    active_decoder = 0
    for layer_idx in range(config.num_hidden_layers):
        attention = (
            _mla_parameters(config)
            if layer_idx + 1 in config.full_attention_layers
            else _kda_parameters(config)
        )
        is_moe = (
            layer_idx >= config.first_k_dense_replace
            and layer_idx % config.moe_layer_freq == 0
        )
        if is_moe:
            feed_forward = _moe_parameters(config, config.num_experts)
            active_feed_forward = _moe_parameters(
                config,
                config.num_experts_per_token,
            )
        else:
            feed_forward = _dense_mlp_parameters(config)
            active_feed_forward = feed_forward
        common = _common_layer_parameters(config)
        decoder += attention + feed_forward + common
        active_decoder += attention + active_feed_forward + common

    embedding = config.vocab_size * config.hidden_size
    lm_head = config.vocab_size * config.hidden_size
    final_norm = config.hidden_size
    total = embedding + decoder + final_norm + lm_head
    # One embedding row is fetched for the input token. The full untied LM head
    # is read to score the vocabulary for the output token.
    active_embedding = config.hidden_size
    active = active_embedding + active_decoder + final_norm + lm_head
    return ParameterCounts(
        total_parameters=total,
        active_parameters_per_token=active,
        resident_embedding_parameters=embedding,
        active_embedding_parameters=active_embedding,
        lm_head_parameters=lm_head,
        decoder_parameters=decoder,
        active_decoder_parameters=active_decoder,
    )


KIMI_LINEAR_PARAMETER_COUNTS = parameter_counts(KIMI_LINEAR_48B)


@dataclass(frozen=True)
class WeightFormat:
    key: str
    label: str
    bits_per_parameter: int

    def bytes_for(self, parameters: int) -> int:
        return (parameters * self.bits_per_parameter + 7) // 8


BF16 = WeightFormat("bf16", "BF16", 16)
FP8 = WeightFormat("fp8", "FP8", 8)
INT4_WEIGHT_ONLY = WeightFormat("int4-weight-only", "INT4 weight-only", 4)
WEIGHT_FORMATS = (BF16, FP8, INT4_WEIGHT_ONLY)


@dataclass(frozen=True)
class MemoryFit:
    capacity_gib: int
    capacity_bytes: int
    resident_bytes: int
    fits: bool
    headroom_bytes: int


@dataclass(frozen=True)
class HardwareProfile:
    key: str
    label: str
    bandwidth_gb_s: float
    attainment: float = K3_MEASURED_BANDWIDTH_ATTAINMENT

    @property
    def bandwidth_bytes_s(self) -> float:
        return self.bandwidth_gb_s * DECIMAL_GB


LAPTOP_DGPU = HardwareProfile(
    key="laptop-dgpu-900gb-s",
    label="Laptop dGPU resident-memory path",
    bandwidth_gb_s=900.0,
)
LAPTOP_DDR5 = HardwareProfile(
    key="laptop-ddr5-100gb-s",
    label="Laptop DDR5 system-memory path",
    bandwidth_gb_s=100.0,
)
HARDWARE_PROFILES = (LAPTOP_DGPU, LAPTOP_DDR5)


@dataclass(frozen=True)
class ThroughputProjection:
    hardware_key: str
    physical_ceiling_tokens_per_second: float
    projected_tokens_per_second_at_attainment: float
    attainment: float


@dataclass(frozen=True)
class LaptopProfile:
    format_key: str
    resident_bytes: int
    active_bytes_per_token: int
    memory_fits: tuple[MemoryFit, ...]
    throughput: tuple[ThroughputProjection, ...]


def _memory_fits(resident_bytes: int) -> tuple[MemoryFit, ...]:
    rows = []
    for capacity_gib in MEMORY_CAPACITIES_GIB:
        capacity_bytes = capacity_gib * GIB
        rows.append(
            MemoryFit(
                capacity_gib=capacity_gib,
                capacity_bytes=capacity_bytes,
                resident_bytes=resident_bytes,
                fits=resident_bytes <= capacity_bytes,
                headroom_bytes=capacity_bytes - resident_bytes,
            )
        )
    return tuple(rows)


def build_profiles(
    counts: ParameterCounts = KIMI_LINEAR_PARAMETER_COUNTS,
) -> tuple[LaptopProfile, ...]:
    """Return deterministic byte and bandwidth rows without hardware access."""
    rows = []
    for weight_format in WEIGHT_FORMATS:
        resident = weight_format.bytes_for(counts.total_parameters)
        active = weight_format.bytes_for(counts.active_parameters_per_token)
        projections = []
        for hardware in HARDWARE_PROFILES:
            ceiling = hardware.bandwidth_bytes_s / active
            projections.append(
                ThroughputProjection(
                    hardware_key=hardware.key,
                    physical_ceiling_tokens_per_second=ceiling,
                    projected_tokens_per_second_at_attainment=(
                        ceiling * hardware.attainment
                    ),
                    attainment=hardware.attainment,
                )
            )
        rows.append(
            LaptopProfile(
                format_key=weight_format.key,
                resident_bytes=resident,
                active_bytes_per_token=active,
                memory_fits=_memory_fits(resident),
                throughput=tuple(projections),
            )
        )
    return tuple(rows)
