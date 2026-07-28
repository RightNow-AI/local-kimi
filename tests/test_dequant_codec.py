import pytest
import torch

from engine.k3ref.dequant import dequantize_mxfp4, unpack_mxfp4
from research.verify_lossless import (
    NEGATIVE_CONTROLS,
    DecoderMismatch,
    assert_decoder_matches_reference,
)

PACKED = torch.tensor(
    [
        [
            0x80,
            0x91,
            0xA2,
            0xB3,
            0xC4,
            0xD5,
            0xE6,
            0xF7,
            0x08,
            0x19,
            0x2A,
            0x3B,
            0x4C,
            0x5D,
            0x6E,
            0x7F,
        ]
    ],
    dtype=torch.uint8,
)
SCALE = torch.tensor([[128]], dtype=torch.uint8)
CODES = torch.tensor(
    [
        0,
        8,
        1,
        9,
        2,
        10,
        3,
        11,
        4,
        12,
        5,
        13,
        6,
        14,
        7,
        15,
        8,
        0,
        9,
        1,
        10,
        2,
        11,
        3,
        12,
        4,
        13,
        5,
        14,
        6,
        15,
        7,
    ]
)
CODEBOOK = torch.tensor(
    [
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        3.0,
        4.0,
        6.0,
        -0.0,
        -0.5,
        -1.0,
        -1.5,
        -2.0,
        -3.0,
        -4.0,
        -6.0,
    ],
    dtype=torch.float32,
)
EXPECTED_UNSCALED = CODEBOOK[CODES].reshape(1, 32)
EXPECTED = EXPECTED_UNSCALED * 2.0


def _bits(values: torch.Tensor) -> torch.Tensor:
    return values.contiguous().view(torch.int32)


def _negative_control(label: str):
    return dict(NEGATIVE_CONTROLS)[label]


def test_canonical_codec_decodes_hand_built_buffer_bit_exactly():
    unpacked = unpack_mxfp4(PACKED)
    actual = dequantize_mxfp4(PACKED, SCALE)

    assert torch.equal(_bits(unpacked), _bits(EXPECTED_UNSCALED))
    assert torch.equal(_bits(actual), _bits(EXPECTED))
    assert_decoder_matches_reference(PACKED, SCALE, EXPECTED)
    assert torch.signbit(actual[0, 1])
    assert torch.signbit(actual[0, 16])
    assert not torch.signbit(actual[0, 0])


def test_verification_rejects_swapped_nibble_order():
    with pytest.raises(DecoderMismatch):
        assert_decoder_matches_reference(
            PACKED,
            SCALE,
            EXPECTED,
            decoder=_negative_control("swapped nibble order"),
        )


def test_verification_rejects_shuffled_code_table():
    with pytest.raises(DecoderMismatch):
        assert_decoder_matches_reference(
            PACKED,
            SCALE,
            EXPECTED,
            decoder=_negative_control("shuffled code table"),
        )


def test_verification_rejects_exponent_bias_120():
    with pytest.raises(DecoderMismatch):
        assert_decoder_matches_reference(
            PACKED,
            SCALE,
            EXPECTED,
            decoder=_negative_control("exponent bias 120"),
        )


def test_verification_rejects_lost_negative_zero_sign():
    def loses_negative_zero(packed: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        decoded = dequantize_mxfp4(packed, scale)
        decoded[(decoded == 0) & torch.signbit(decoded)] = 0.0
        return decoded

    with pytest.raises(DecoderMismatch):
        assert_decoder_matches_reference(
            PACKED,
            SCALE,
            EXPECTED,
            decoder=loses_negative_zero,
        )
