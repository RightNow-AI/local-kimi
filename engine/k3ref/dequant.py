"""MXFP4 decoding used by the Kimi K3 routed experts."""

from __future__ import annotations

import torch

from .manifest import MXFP4_GROUP_SIZE


_E2M1_POSITIVE = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


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
    rows, packed_columns = packed.shape
    value_columns = packed_columns * 2
    if (
        scale.shape[0] != rows
        or value_columns != scale.shape[1] * MXFP4_GROUP_SIZE
    ):
        raise ValueError(
            "MXFP4 scale shape must provide one exponent for every 32 decoded values"
        )

    codebook = torch.tensor(
        _E2M1_POSITIVE + tuple(-value for value in _E2M1_POSITIVE),
        dtype=torch.float32,
        device=packed.device,
    )
    nibbles = torch.empty(
        (rows, value_columns), dtype=torch.long, device=packed.device
    )
    # The low nibble is the first logical value in each packed byte.
    nibbles[:, 0::2] = (packed & 0x0F).long()
    nibbles[:, 1::2] = (packed >> 4).long()
    values = codebook[nibbles]
    exponents = torch.exp2(scale.to(torch.int16).float() - 127.0)
    values = values * exponents.repeat_interleave(MXFP4_GROUP_SIZE, dim=1)
    return values.to(dtype)
