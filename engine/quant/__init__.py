"""Weight-only quantization primitives for the K3 dense path."""

from .plan import build_quantization_plan
from .w4a16 import GROUP_SIZE, W4A16Tensor, dequantise, quantise

__all__ = [
    "GROUP_SIZE",
    "W4A16Tensor",
    "build_quantization_plan",
    "dequantise",
    "quantise",
]
