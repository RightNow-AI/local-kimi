"""Verify the canonical MXFP4 decoder against an independent implementation.

Only K3's routed-expert matrices are MXFP4. Shared experts, attention, latent
projections, and embeddings remain BF16 and total 114.4 GB; two shared experts
participate alongside 16 routed experts per token.

This script reads raw ``weight_packed`` and ``weight_scale`` arrays cached by
``research/expert_spectrum*.py``. It compares the canonical codec in
``engine/k3ref/dequant.py`` with ``compressed-tensors`` using float32 bit
patterns, so the sign of negative zero is part of the assertion. It also runs
three deliberately wrong decoders and refuses to pass unless all are rejected.

If ``compressed-tensors`` is unavailable, the script exits 77 with a loud SKIP
instead of substituting its own logic or reporting success.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import re
import sys
from functools import lru_cache
from pathlib import Path
from typing import Callable

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
dequantize_mxfp4 = importlib.import_module(
    "engine.k3ref.dequant"
).dequantize_mxfp4

Decoder = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


class ReferenceUnavailable(RuntimeError):
    """The independent compressed-tensors reference cannot be loaded."""


class DecoderMismatch(AssertionError):
    """A decoder disagrees with the independent reference at the bit level."""


@lru_cache(maxsize=1)
def _load_reference_api() -> tuple[str, object, object, object, object, object, object]:
    try:
        version = importlib.metadata.version("compressed-tensors")
        helpers = importlib.import_module(
            "compressed_tensors.compressors.nvfp4.helpers"
        )
        mx_utils = importlib.import_module("compressed_tensors.compressors.mx_utils")
        forward_helpers = importlib.import_module(
            "compressed_tensors.quantization.lifecycle.forward_helpers"
        )
        quantization = importlib.import_module("compressed_tensors.quantization")
        return (
            version,
            helpers.unpack_fp4_from_uint8,
            mx_utils.decompress_mx_scale,
            forward_helpers._process_group,
            quantization.QuantizationArgs,
            quantization.QuantizationStrategy,
            quantization.QuantizationType,
        )
    except (ImportError, AttributeError, importlib.metadata.PackageNotFoundError) as exc:
        raise ReferenceUnavailable(
            "compressed-tensors with the MXFP4 reference API is required"
        ) from exc


def compressed_tensors_dequantize(
    packed: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    """Decode through the compressed-tensors reference API used by version 0.17.1."""
    (
        _version,
        unpack_fp4_from_uint8,
        decompress_mx_scale,
        process_group,
        QuantizationArgs,
        QuantizationStrategy,
        QuantizationType,
    ) = _load_reference_api()

    rows, packed_columns = packed.shape
    unpacked = unpack_fp4_from_uint8(
        packed,
        rows,
        packed_columns * 2,
        dtype=torch.float32,
    )
    scale_float = decompress_mx_scale(scale).to(torch.float32)
    args = QuantizationArgs(
        num_bits=4,
        type=QuantizationType.FLOAT,
        symmetric=True,
        strategy=QuantizationStrategy.GROUP,
        group_size=32,
    )
    return process_group(
        x=unpacked,
        scale=scale_float,
        zero_point=None,
        args=args,
        q_min=torch.tensor(-6.0, device=packed.device),
        q_max=torch.tensor(6.0, device=packed.device),
        dtype=torch.float32,
        do_quantize=False,
        do_dequantize=True,
        g_idx=None,
        global_scale=None,
    ).to(torch.float32)


def _float32_bits(values: torch.Tensor) -> torch.Tensor:
    if values.dtype != torch.float32:
        raise TypeError(f"bit-exact verification requires float32, got {values.dtype}")
    return values.detach().contiguous().cpu().view(torch.int32)


def assert_decoder_matches_reference(
    packed: torch.Tensor,
    scale: torch.Tensor,
    reference: torch.Tensor,
    *,
    decoder: Decoder = dequantize_mxfp4,
    label: str = "canonical decoder",
) -> torch.Tensor:
    """Assert exact float32 bits, including the sign bit on zero values."""
    actual = decoder(packed, scale)
    if actual.shape != reference.shape:
        raise DecoderMismatch(
            f"{label} returned shape {tuple(actual.shape)}, expected {tuple(reference.shape)}"
        )

    actual_bits = _float32_bits(actual)
    reference_bits = _float32_bits(reference)
    mismatches = actual_bits != reference_bits
    if bool(mismatches.any()):
        flat_index = int(torch.nonzero(mismatches.flatten(), as_tuple=False)[0].item())
        columns = reference.shape[1]
        row, column = divmod(flat_index, columns)
        actual_value = float(actual[row, column].item())
        reference_value = float(reference[row, column].item())
        actual_bit_pattern = int(actual_bits[row, column].item()) & 0xFFFFFFFF
        reference_bit_pattern = int(reference_bits[row, column].item()) & 0xFFFFFFFF
        raise DecoderMismatch(
            f"{label} differs at [{row}, {column}]: "
            f"actual={actual_value} bits=0x{actual_bit_pattern:08x}, "
            f"reference={reference_value} bits=0x{reference_bit_pattern:08x}"
        )
    return actual


def _swapped_nibble_decoder(
    packed: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    swapped = ((packed & 0x0F) << 4) | (packed >> 4)
    return dequantize_mxfp4(swapped, scale)


def _shuffled_codebook_decoder(
    packed: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    mapping = torch.tensor(
        [1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14],
        dtype=torch.long,
        device=packed.device,
    )
    low = mapping[(packed & 0x0F).long()]
    high = mapping[(packed >> 4).long()]
    remapped = (low | (high << 4)).to(torch.uint8)
    return dequantize_mxfp4(remapped, scale)


def _bias_120_decoder(
    packed: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    # Bias 120 makes every nonzero decoded value 2^(127-120) times too large.
    return dequantize_mxfp4(packed, scale) * 128.0


NEGATIVE_CONTROLS: tuple[tuple[str, Decoder], ...] = (
    ("swapped nibble order", _swapped_nibble_decoder),
    ("shuffled code table", _shuffled_codebook_decoder),
    ("exponent bias 120", _bias_120_decoder),
)


def assert_negative_controls_rejected(
    packed: torch.Tensor,
    scale: torch.Tensor,
    reference: torch.Tensor,
) -> tuple[str, ...]:
    """Fail if any deliberately corrupted decoder escapes detection."""
    rejected = []
    for label, decoder in NEGATIVE_CONTROLS:
        try:
            assert_decoder_matches_reference(
                packed,
                scale,
                reference,
                decoder=decoder,
                label=label,
            )
        except DecoderMismatch:
            rejected.append(label)
        else:
            raise AssertionError(f"verification accepted wrong decoder: {label}")
    return tuple(rejected)


def cached_tensor_pairs(
    cache: Path,
    *,
    layer: int,
    experts: int,
) -> list[tuple[Path, Path]]:
    """Find raw packed/scale cache pairs written by the spectrum scripts."""
    layer_marker = f"_layers_{layer}_"
    pairs = []
    for packed_path in sorted(cache.glob("*_weight_packed.npy")):
        if layer_marker not in packed_path.name:
            continue
        match = re.search(r"_experts_(\d+)_", packed_path.name)
        if match is None or (experts > 0 and int(match.group(1)) >= experts):
            continue
        scale_path = packed_path.with_name(
            packed_path.name.replace("_weight_packed.npy", "_weight_scale.npy")
        )
        if not scale_path.is_file():
            raise FileNotFoundError(f"missing scale cache for {packed_path.name}")
        pairs.append((packed_path, scale_path))
    return pairs


def _load_uint8(path: Path) -> torch.Tensor:
    array = np.load(path, allow_pickle=False)
    if array.dtype != np.uint8:
        raise TypeError(f"{path} must contain uint8 data, got {array.dtype}")
    return torch.from_numpy(np.array(array, copy=True))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layer", type=int, default=12)
    parser.add_argument("--experts", type=int, default=32)
    parser.add_argument("--cache", type=Path, default=Path("research/.cache"))
    args = parser.parse_args()

    try:
        reference_version = _load_reference_api()[0]
    except ReferenceUnavailable as exc:
        print(f"SKIP: {exc}. Install compressed-tensors; no verification was run.")
        return 77

    try:
        pairs = cached_tensor_pairs(
            args.cache,
            layer=args.layer,
            experts=args.experts,
        )
    except (FileNotFoundError, TypeError) as exc:
        print(f"FAIL: invalid MXFP4 cache: {exc}")
        return 2
    if not pairs:
        print(
            "FAIL: no raw weight_packed/weight_scale cache pairs found. "
            "Run a spectrum script to populate research/.cache first."
        )
        return 2

    checked = 0
    total_values = 0
    negative_zeros = 0
    rejected_controls: tuple[str, ...] | None = None
    for packed_path, scale_path in pairs:
        try:
            packed = _load_uint8(packed_path)
            scale = _load_uint8(scale_path)
            reference = compressed_tensors_dequantize(packed, scale)
            assert_decoder_matches_reference(packed, scale, reference)
            if rejected_controls is None:
                rejected_controls = assert_negative_controls_rejected(
                    packed,
                    scale,
                    reference,
                )
        except (AssertionError, RuntimeError, TypeError, ValueError) as exc:
            print(f"FAIL: {packed_path.name}: {exc}")
            return 1

        negative_zeros += int(((reference == 0) & torch.signbit(reference)).sum().item())
        total_values += reference.numel()
        checked += 1

    if negative_zeros == 0:
        print("FAIL: cached sample contained no negative-zero codes to verify")
        return 1

    print(f"VERIFIED: {checked} cached routed-expert tensors, {total_values} values")
    print(f"  canonical decoder matches compressed-tensors {reference_version} bit for bit")
    print(f"  negative-zero sign bits compared: {negative_zeros}")
    print(f"  wrong decoders rejected: {', '.join(rejected_controls or ())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
