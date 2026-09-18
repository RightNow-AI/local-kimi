"""Correctness gates and negative controls for the K3 W4A16 codec."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable

import torch

from .w4a16 import GROUP_SIZE, W4A16Tensor, dequantise, quantise

BF16_EPSILON = torch.finfo(torch.bfloat16).eps


class VerificationError(AssertionError):
    """Raised when a codec or candidate decoder violates the W4A16 contract."""


@dataclass(frozen=True)
class RoundTripStats:
    tensor_class: str
    elements: int
    max_abs_error: float
    mean_abs_error: float
    rmse: float
    max_rel_error: float
    max_allowed_abs_error: float | None

    def as_dict(self) -> dict:
        return asdict(self)


def error_statistics(
    tensor_class: str,
    original: torch.Tensor,
    restored: torch.Tensor,
    *,
    max_allowed_abs_error: float | None,
) -> RoundTripStats:
    if original.shape != restored.shape:
        raise VerificationError(
            f"shape mismatch: expected {tuple(original.shape)}, got {tuple(restored.shape)}"
        )
    error = restored.float() - original.float()
    absolute = error.abs()
    relative = absolute / original.float().abs().clamp_min(1e-12)
    return RoundTripStats(
        tensor_class=tensor_class,
        elements=original.numel(),
        max_abs_error=float(absolute.max()),
        mean_abs_error=float(absolute.mean()),
        rmse=float(error.square().mean().sqrt()),
        max_rel_error=float(relative.max()),
        max_allowed_abs_error=max_allowed_abs_error,
    )


def verify_round_trip(
    tensor_class: str,
    original: torch.Tensor,
    quantized: W4A16Tensor | None = None,
) -> RoundTripStats:
    """Verify the scale-aware nearest-grid error bound and return statistics.

    The claimed element-wise tolerance is half a stored quantization step plus
    ``2 * BF16 epsilon * group_absmax``. The first term is nearest-grid error;
    the second conservatively covers BF16 scale storage and BF16 output rounding.
    """
    if original.dtype != torch.bfloat16:
        raise TypeError("round-trip verification requires a BF16 original")
    encoded = quantise(original) if quantized is None else quantized
    restored = dequantise(encoded)
    grouped_original = original.float().reshape(
        original.shape[0], original.shape[1] // GROUP_SIZE, GROUP_SIZE
    )
    grouped_error = (restored.float() - original.float()).abs().reshape_as(
        grouped_original
    )
    group_absmax = grouped_original.abs().amax(dim=-1)
    allowed = (
        0.5 * encoded.scales.float().abs()
        + 2.0 * BF16_EPSILON * group_absmax
    ).unsqueeze(-1)
    excess = grouped_error - allowed
    if bool((excess > 0).any()):
        raise VerificationError(
            "round-trip error exceeded the scale-aware W4A16 tolerance: "
            f"max excess {float(excess.max())}"
        )
    return error_statistics(
        tensor_class,
        original,
        restored,
        max_allowed_abs_error=float(allowed.max()),
    )


Decoder = Callable[[W4A16Tensor], torch.Tensor]


def verify_dequantizer(
    quantized: W4A16Tensor,
    decoder: Decoder,
    *,
    decoder_name: str,
) -> None:
    """Require a candidate decoder to reproduce the canonical BF16 grid exactly."""
    expected = dequantise(quantized)
    actual = decoder(quantized)
    if actual.shape != expected.shape:
        raise VerificationError(
            f"{decoder_name} returned shape {tuple(actual.shape)}, expected {tuple(expected.shape)}"
        )
    if actual.dtype != expected.dtype:
        raise VerificationError(
            f"{decoder_name} returned dtype {actual.dtype}, expected {expected.dtype}"
        )
    if not torch.equal(actual, expected):
        difference = (actual.float() - expected.float()).abs()
        raise VerificationError(
            f"{decoder_name} violated the W4A16 decode contract; "
            f"max abs difference {float(difference.max())}"
        )


def _signed_groups(quantized: W4A16Tensor, *, swapped: bool = False) -> torch.Tensor:
    low = quantized.packed & 0xF
    high = (quantized.packed >> 4) & 0xF
    first, second = (high, low) if swapped else (low, high)
    nibbles = torch.stack((first, second), dim=-1).reshape(quantized.original_shape)
    signed = nibbles.to(torch.int16)
    signed = torch.where(signed >= 8, signed - 16, signed)
    return signed.reshape(
        quantized.original_shape[0],
        quantized.original_shape[1] // GROUP_SIZE,
        GROUP_SIZE,
    )


def wrong_group_axis_dequantise(quantized: W4A16Tensor) -> torch.Tensor:
    """Negative control that associates scale groups with the output axis."""
    scales = quantized.scales
    if scales.shape[0] == scales.shape[1]:
        wrong_scales = scales.transpose(0, 1)
    else:
        wrong_scales = scales.roll(shifts=1, dims=0)
    restored = _signed_groups(quantized).float() * wrong_scales.float().unsqueeze(-1)
    return restored.reshape(quantized.original_shape).to(torch.bfloat16)


def wrong_scale_dequantise(quantized: W4A16Tensor) -> torch.Tensor:
    """Negative control that applies a factor-of-two scale error."""
    restored = _signed_groups(quantized).float()
    restored = restored * (quantized.scales.float() * 2.0).unsqueeze(-1)
    return restored.reshape(quantized.original_shape).to(torch.bfloat16)


def swapped_nibble_dequantise(quantized: W4A16Tensor) -> torch.Tensor:
    """Negative control that decodes odd reduction positions before even ones."""
    restored = _signed_groups(quantized, swapped=True).float()
    restored = restored * quantized.scales.float().unsqueeze(-1)
    return restored.reshape(quantized.original_shape).to(torch.bfloat16)


def negative_control_quantized_tensor() -> W4A16Tensor:
    """Build a non-degenerate exact-grid tensor for decoder rejection tests."""
    code_pattern = torch.tensor(
        [index % 15 - 7 for index in range(GROUP_SIZE)], dtype=torch.bfloat16
    )
    codes = code_pattern.reshape(1, 1, GROUP_SIZE).expand(2, 2, GROUP_SIZE)
    scales = torch.tensor([[0.25, 2.0], [8.0, 0.5]], dtype=torch.bfloat16)
    weight = (codes * scales.unsqueeze(-1)).reshape(2, 2 * GROUP_SIZE)
    encoded = quantise(weight)
    if not torch.equal(dequantise(encoded), weight):
        raise VerificationError("the negative-control probe must be exact on its grid")
    return encoded


def run_negative_control_checks() -> dict[str, bool]:
    """Prove that three independently broken decoders are rejected."""
    encoded = negative_control_quantized_tensor()
    decoders = {
        "wrong_group_axis": wrong_group_axis_dequantise,
        "wrong_scale": wrong_scale_dequantise,
        "swapped_nibbles": swapped_nibble_dequantise,
    }
    rejected = {}
    for name, decoder in decoders.items():
        try:
            verify_dequantizer(encoded, decoder, decoder_name=name)
        except VerificationError:
            rejected[name] = True
        else:
            raise VerificationError(f"negative control {name} was incorrectly accepted")
    return rejected
