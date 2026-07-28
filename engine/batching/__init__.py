"""Analytic and measured batching tools for Kimi K3."""

from .union_model import (
    DEFAULT_DENSE_BYTES,
    DEFAULT_DENSE_PARAMETERS,
    DEFAULT_DEQUANT_MODES,
    FUSED_DEQUANT,
    MEASURED_DEQUANT_SECONDS_PER_TENSOR,
    MEASURED_EXPERT_GEMM_RESIDUAL_SECONDS_PER_TOKEN,
    MEASURED_H100_B1_BANDWIDTH_EFFICIENCY,
    UNFUSED_DEQUANT,
    DequantMode,
    ExpertUnionModel,
    HardwareConfig,
    RoutingPrior,
    default_hardware_configs,
    dirichlet_prior,
    zipf_prior,
)

__all__ = [
    "DEFAULT_DENSE_BYTES",
    "DEFAULT_DENSE_PARAMETERS",
    "DEFAULT_DEQUANT_MODES",
    "FUSED_DEQUANT",
    "MEASURED_DEQUANT_SECONDS_PER_TENSOR",
    "MEASURED_EXPERT_GEMM_RESIDUAL_SECONDS_PER_TOKEN",
    "MEASURED_H100_B1_BANDWIDTH_EFFICIENCY",
    "UNFUSED_DEQUANT",
    "DequantMode",
    "ExpertUnionModel",
    "HardwareConfig",
    "RoutingPrior",
    "default_hardware_configs",
    "dirichlet_prior",
    "zipf_prior",
]
