"""Symmetric group-wise INT4 weights with BF16 activations and scales.

The reduction axis is the final axis of a dense weight matrix shaped
``[out_features, in_features]``. Values at even reduction-axis positions are
stored in the low nibble and odd positions in the high nibble.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

GROUP_SIZE = 32
QMIN = -7
QMAX = 7
SCALE_DTYPE = torch.bfloat16


@dataclass(frozen=True)
class W4A16Tensor:
    """Packed INT4 weights and one BF16 scale per reduction-axis group."""

    packed: torch.Tensor
    scales: torch.Tensor
    original_shape: tuple[int, int]
    original_dtype: torch.dtype = torch.bfloat16
    group_size: int = GROUP_SIZE

    def __post_init__(self) -> None:
        if len(self.original_shape) != 2:
            raise ValueError("W4A16 weights must have shape [out_features, in_features]")
        out_features, in_features = self.original_shape
        if out_features <= 0 or in_features <= 0:
            raise ValueError("W4A16 dimensions must be positive")
        if self.group_size != GROUP_SIZE:
            raise ValueError(f"K3 W4A16 group size must be {GROUP_SIZE}")
        if in_features % self.group_size:
            raise ValueError("the reduction axis must be divisible by group size 32")
        if self.packed.dtype != torch.uint8:
            raise TypeError("packed weights must use torch.uint8 storage")
        if self.scales.dtype != SCALE_DTYPE:
            raise TypeError("W4A16 scales must use torch.bfloat16 storage")
        if self.original_dtype != torch.bfloat16:
            raise TypeError("W4A16 currently accepts and restores BF16 weights only")
        expected_packed = (out_features, in_features // 2)
        expected_scales = (out_features, in_features // self.group_size)
        if tuple(self.packed.shape) != expected_packed:
            raise ValueError(
                f"packed shape must be {expected_packed}, got {tuple(self.packed.shape)}"
            )
        if tuple(self.scales.shape) != expected_scales:
            raise ValueError(
                f"scale shape must be {expected_scales}, got {tuple(self.scales.shape)}"
            )
        if self.packed.device != self.scales.device:
            raise ValueError("packed weights and scales must be on the same device")

    @property
    def storage_bytes(self) -> int:
        """Physical tensor storage, excluding the small Python metadata object."""
        return (
            self.packed.numel() * self.packed.element_size()
            + self.scales.numel() * self.scales.element_size()
        )


def quantise(weight: torch.Tensor) -> W4A16Tensor:
    """Quantise a BF16 dense matrix along its reduction axis.

    Each group uses ``scale = max(abs(group)) / 7`` and nearest-integer
    rounding. All-zero groups use scale 1 so the division is always finite.
    """
    if weight.ndim != 2:
        raise ValueError("weight must have shape [out_features, in_features]")
    if weight.dtype != torch.bfloat16:
        raise TypeError("weight must have dtype torch.bfloat16")
    if weight.shape[1] % GROUP_SIZE:
        raise ValueError("the reduction axis must be divisible by group size 32")
    if not bool(torch.isfinite(weight).all()):
        raise ValueError("weight must contain only finite values")

    out_features, in_features = weight.shape
    groups = weight.detach().reshape(out_features, in_features // GROUP_SIZE, GROUP_SIZE)
    group_absmax = groups.float().abs().amax(dim=-1)
    raw_scales = group_absmax / float(QMAX)
    stored_scales = raw_scales.to(dtype=SCALE_DTYPE)
    scales = torch.where(
        group_absmax == 0,
        torch.ones_like(raw_scales),
        stored_scales.float().clamp_min(torch.finfo(SCALE_DTYPE).tiny),
    ).to(dtype=SCALE_DTYPE)

    # Quantise against the stored scale so the decoder and encoder share the
    # exact same grid, including BF16 scale rounding.
    codes = torch.round(groups.float() / scales.float().unsqueeze(-1))
    codes = codes.clamp(QMIN, QMAX).to(torch.int16).reshape(out_features, in_features)
    nibbles = torch.bitwise_and(codes, 0xF).to(torch.uint8)
    packed = nibbles[:, 0::2] | (nibbles[:, 1::2] << 4)

    return W4A16Tensor(
        packed=packed.contiguous(),
        scales=scales.contiguous(),
        original_shape=(out_features, in_features),
    )


def dequantise(
    quantized: W4A16Tensor,
    *,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Restore a packed W4A16 matrix to BF16 by default."""
    output_dtype = quantized.original_dtype if dtype is None else dtype
    if output_dtype not in (torch.bfloat16, torch.float32):
        raise TypeError("dequantise output dtype must be torch.bfloat16 or torch.float32")

    low = quantized.packed & 0xF
    high = (quantized.packed >> 4) & 0xF
    nibbles = torch.stack((low, high), dim=-1).reshape(quantized.original_shape)
    signed = nibbles.to(torch.int16)
    signed = torch.where(signed >= 8, signed - 16, signed)
    grouped = signed.reshape(
        quantized.original_shape[0],
        quantized.original_shape[1] // quantized.group_size,
        quantized.group_size,
    )
    restored = grouped.float() * quantized.scales.float().unsqueeze(-1)
    return restored.reshape(quantized.original_shape).to(dtype=output_dtype)
