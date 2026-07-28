"""Plain PyTorch numerical reference for one Kimi K3 decoder layer."""

from .attention import KDAAttention, KDAState, MLAAttention, MLAState
from .config import K3LayerConfig
from .dequant import dequantize_mxfp4
from .layer import K3LayerOutput, K3ReferenceLayer
from .manifest import K3_LAYER_TENSOR_MANIFEST, TensorSpec
from .moe import K3ExpertMLP, K3SharedMLP, LatentMoE
from .router import K3Router

__all__ = [
    "K3ExpertMLP",
    "K3LayerConfig",
    "K3LayerOutput",
    "K3_LAYER_TENSOR_MANIFEST",
    "K3ReferenceLayer",
    "K3Router",
    "K3SharedMLP",
    "KDAAttention",
    "KDAState",
    "LatentMoE",
    "MLAAttention",
    "MLAState",
    "TensorSpec",
    "dequantize_mxfp4",
]
