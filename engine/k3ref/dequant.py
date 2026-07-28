"""Canonical MXFP4 codec for the Kimi K3 routed-expert tensors."""

from __future__ import annotations

import torch

from .manifest import MXFP4_GROUP_SIZE

MXFP4_EXPONENT_BIAS = 127
# MXFP4_GROUP_SIZE is imported from .manifest, which holds the checkpoint-derived
# facts. Defining it here as well would recreate the duplication this module
# exists to remove.

_E2M1_POSITIVE = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
_E2M1_CODEBOOK = _E2M1_POSITIVE + tuple(-value for value in _E2M1_POSITIVE)


def unpack_mxfp4(
    packed: torch.Tensor,
    *,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Unpack low-nibble-first E2M1 values while preserving negative zero."""
    if packed.dtype != torch.uint8:
        raise TypeError("packed tensor must use torch.uint8")
    if packed.ndim != 2:
        raise ValueError("packed tensor must be a matrix")
    if not torch.empty((), dtype=dtype).is_floating_point():
        raise TypeError("MXFP4 output dtype must be floating point")

    rows, packed_columns = packed.shape
    value_columns = packed_columns * 2
    codebook = torch.tensor(_E2M1_CODEBOOK, dtype=torch.float32, device=packed.device)
    nibbles = torch.empty(
        (rows, value_columns), dtype=torch.long, device=packed.device
    )
    # The storage contract places the first logical value in the low nibble.
    nibbles[:, 0::2] = (packed & 0x0F).long()
    nibbles[:, 1::2] = (packed >> 4).long()
    return codebook[nibbles].to(dtype)


def dequantize_mxfp4(
    packed: torch.Tensor,
    scale: torch.Tensor,
    *,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Decode E2M1 nibbles with one E8M0 exponent per group of 32 values."""
    if packed.dtype != torch.uint8 or scale.dtype != torch.uint8:
        raise TypeError("packed and scale tensors must both use torch.uint8")
    if packed.ndim != 2 or scale.ndim != 2:
        raise ValueError("packed and scale tensors must both be matrices")
    if packed.device != scale.device:
        raise ValueError("packed and scale tensors must be on the same device")
    if not torch.empty((), dtype=dtype).is_floating_point():
        raise TypeError("MXFP4 output dtype must be floating point")
    rows, packed_columns = packed.shape
    value_columns = packed_columns * 2
    if (
        scale.shape[0] != rows
        or value_columns != scale.shape[1] * MXFP4_GROUP_SIZE
    ):
        raise ValueError(
            "MXFP4 scale shape must provide one exponent for every 32 decoded values"
        )

    values = unpack_mxfp4(packed)
    exponents = torch.exp2(
        scale.to(torch.int16).float() - float(MXFP4_EXPONENT_BIAS)
    )
    values = values * exponents.repeat_interleave(MXFP4_GROUP_SIZE, dim=1)
    return values.to(dtype)
