"""Plain PyTorch numerical reference for one Kimi K3 decoder layer."""

from .attention import KDAAttention, KDAState, MLAAttention, MLAState
from .config import K3LayerConfig
from .dequant import dequantize_mxfp4
from .layer import K3LayerOutput, K3ReferenceLayer
from .moe import K3ExpertMLP, K3SharedMLP, LatentMoE
from .router import K3Router

__all__ = [
    "K3ExpertMLP",
    "K3LayerConfig",
    "K3LayerOutput",
    "K3ReferenceLayer",
    "K3Router",
    "K3SharedMLP",
    "KDAAttention",
    "KDAState",
    "LatentMoE",
    "MLAAttention",
    "MLAState",
    "dequantize_mxfp4",
]
