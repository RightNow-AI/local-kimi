"""Symmetric group-wise INT3 weights with BF16 activations and scales.

The reduction axis is the final axis of a dense weight matrix shaped
``[out_features, in_features]``. Eight consecutive signed 3-bit values occupy
three bytes with no padding.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

GROUP_SIZE = 32
QMIN = -4
QMAX = 3
SCALE_DTYPE = torch.bfloat16


@dataclass(frozen=True)
class W3A16Tensor:
    """Packed INT3 weights and one BF16 scale per reduction-axis group."""

    packed: torch.Tensor
    scales: torch.Tensor
    original_shape: tuple[int, int]
    group_size: int = GROUP_SIZE

    def __post_init__(self) -> None:
        if len(self.original_shape) != 2:
            raise ValueError("W3A16 weights must have shape [out_features, in_features]")
        out_features, in_features = self.original_shape
        if out_features <= 0 or in_features <= 0:
            raise ValueError("W3A16 dimensions must be positive")
        if not isinstance(self.group_size, int) or self.group_size <= 0:
            raise ValueError("group_size must be a positive integer")
        if in_features % 8:
            raise ValueError("the reduction axis must be divisible by 8 for INT3 packing")
        if in_features % self.group_size:
            raise ValueError(
                f"the reduction axis must be divisible by group_size={self.group_size}"
            )
        if self.packed.dtype != torch.uint8:
            raise TypeError("packed weights must use torch.uint8 storage")
        if self.scales.dtype != SCALE_DTYPE:
            raise TypeError("W3A16 scales must use torch.bfloat16 storage")
        expected_packed = (out_features, in_features // 8 * 3)
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


def _pack_int3_codes(codes: torch.Tensor) -> torch.Tensor:
    """Pack a signed integer matrix into the canonical W3A16 byte layout."""
    if codes.ndim != 2:
        raise ValueError("codes must be a two-dimensional matrix")
    if codes.shape[1] % 8:
        raise ValueError("the reduction axis must be divisible by 8 for INT3 packing")
    if bool(((codes < QMIN) | (codes > QMAX)).any()):
        raise ValueError("INT3 codes must be in the range -4 to 3")

    rows, reduction = codes.shape
    unsigned = torch.bitwise_and(codes.to(torch.int16), 0x7)
    values = unsigned.reshape(rows, reduction // 8, 8)

    # Eight values c0 through c7 form one little-endian 24-bit word where ci
    # occupies word bits [3*i, 3*i+2]. The exact byte layout is:
    #   byte 0 bits 0..2 = c0, bits 3..5 = c1, bits 6..7 = c2 bits 0..1
    #   byte 1 bit 0 = c2 bit 2, bits 1..3 = c3, bits 4..6 = c4,
    #          bit 7 = c5 bit 0
    #   byte 2 bits 0..1 = c5 bits 1..2, bits 2..4 = c6,
    #          bits 5..7 = c7
    # Each code is its low three two's complement bits. This layout must stay
    # identical to the Triton decoder because a mismatch still yields values
    # in range and therefore produces plausible-looking garbage.
    byte0 = values[..., 0] | (values[..., 1] << 3) | ((values[..., 2] & 0x3) << 6)
    byte1 = (
        (values[..., 2] >> 2)
        | (values[..., 3] << 1)
        | (values[..., 4] << 4)
        | ((values[..., 5] & 0x1) << 7)
    )
    byte2 = (values[..., 5] >> 1) | (values[..., 6] << 2) | (values[..., 7] << 5)
    return torch.stack((byte0, byte1, byte2), dim=-1).reshape(rows, -1).to(torch.uint8)


def _unpack_int3_codes(
    packed: torch.Tensor,
    original_shape: tuple[int, int],
) -> torch.Tensor:
    """Unpack the canonical byte layout to signed INT3 codes."""
    rows, reduction = original_shape
    if reduction % 8:
        raise ValueError("the reduction axis must be divisible by 8 for INT3 packing")
    expected_shape = (rows, reduction // 8 * 3)
    if packed.dtype != torch.uint8 or tuple(packed.shape) != expected_shape:
        raise ValueError(
            f"packed must be a uint8 tensor with shape {expected_shape}"
        )

    groups = packed.reshape(rows, reduction // 8, 3).to(torch.int16)
    byte0 = groups[..., 0]
    byte1 = groups[..., 1]
    byte2 = groups[..., 2]
    unsigned = torch.stack(
        (
            byte0 & 0x7,
            (byte0 >> 3) & 0x7,
            ((byte0 >> 6) | (byte1 << 2)) & 0x7,
            (byte1 >> 1) & 0x7,
            (byte1 >> 4) & 0x7,
            ((byte1 >> 7) | (byte2 << 1)) & 0x7,
            (byte2 >> 2) & 0x7,
            (byte2 >> 5) & 0x7,
        ),
        dim=-1,
    ).reshape(rows, reduction)
    return torch.where(unsigned >= 4, unsigned - 8, unsigned).to(torch.int16)


def quantise(weight: torch.Tensor, group_size: int = GROUP_SIZE) -> W3A16Tensor:
    """Quantise a BF16 dense matrix along its reduction axis.

    Each group uses ``scale = max(abs(group)) / 3`` and nearest-integer
    rounding followed by a clamp to ``[-4, 3]``. The divisor is 3 rather than
    4 because the positive side saturates at 3. Dividing by 4 would waste a
    positive code point. Dividing by 3 keeps the useful grid symmetric around
    zero, at the cost of clipping the single most-negative excess to -4.
    All-zero groups use scale 1 so division is always finite.
    """
    if weight.ndim != 2:
        raise ValueError("weight must have shape [out_features, in_features]")
    if weight.dtype != torch.bfloat16:
        raise TypeError("weight must have dtype torch.bfloat16")
    if not isinstance(group_size, int) or group_size <= 0:
        raise ValueError("group_size must be a positive integer")
    if weight.shape[1] % 8:
        raise ValueError("the reduction axis must be divisible by 8 for INT3 packing")
    if weight.shape[1] % group_size:
        raise ValueError(
            f"the reduction axis must be divisible by group_size={group_size}"
        )
    if not bool(torch.isfinite(weight).all()):
        raise ValueError("weight must contain only finite values")

    out_features, in_features = weight.shape
    groups = weight.detach().reshape(out_features, in_features // group_size, group_size)
    group_absmax = groups.float().abs().amax(dim=-1)
    raw_scales = group_absmax / float(QMAX)
    stored_scales = raw_scales.to(dtype=SCALE_DTYPE)
    scales = torch.where(
        group_absmax == 0,
        torch.ones_like(raw_scales),
        stored_scales.float().clamp_min(torch.finfo(SCALE_DTYPE).tiny),
    ).to(dtype=SCALE_DTYPE)

    # Quantise against the stored BF16 scale so the encoder and both decoders
    # use the same grid, including scale-storage rounding.
    codes = torch.round(groups.float() / scales.float().unsqueeze(-1))
    codes = codes.clamp(QMIN, QMAX).to(torch.int16).reshape(out_features, in_features)
    packed = _pack_int3_codes(codes)

    return W3A16Tensor(
        packed=packed.contiguous(),
        scales=scales.contiguous(),
        original_shape=(out_features, in_features),
        group_size=group_size,
    )


def dequantise(
    quantized: W3A16Tensor,
    *,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Restore a packed W3A16 matrix in pure PyTorch."""
    if dtype not in (torch.bfloat16, torch.float32):
        raise TypeError("dequantise output dtype must be torch.bfloat16 or torch.float32")

    signed = _unpack_int3_codes(quantized.packed, quantized.original_shape)
    grouped = signed.reshape(
        quantized.original_shape[0],
        quantized.original_shape[1] // quantized.group_size,
        quantized.group_size,
    )
    restored = grouped.float() * quantized.scales.float().unsqueeze(-1)
    return restored.reshape(quantized.original_shape).to(dtype=dtype)
