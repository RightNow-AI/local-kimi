import torch

from engine.k3ref.dequant import dequantize_mxfp4
from engine.k3ref.manifest import MXFP4_GROUP_SIZE


def _pack(codes: list[int]) -> list[int]:
    return [
        codes[index] | (codes[index + 1] << 4)
        for index in range(0, MXFP4_GROUP_SIZE, 2)
    ]


def test_mxfp4_decodes_both_nibbles_and_e8m0_exponents_exactly():
    repeats = MXFP4_GROUP_SIZE // 16
    first_codes = list(range(16)) * repeats
    second_codes = list(reversed(range(16))) * repeats
    packed = torch.tensor(
        [_pack(first_codes), _pack(second_codes)], dtype=torch.uint8
    )
    scale = torch.tensor([[128], [126]], dtype=torch.uint8)

    positive = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
    codebook = torch.tensor(positive + [-value for value in positive])
    expected = torch.stack(
        (
            codebook[torch.tensor(first_codes)] * 2.0,
            codebook[torch.tensor(second_codes)] * 0.5,
        )
    )

    actual = dequantize_mxfp4(packed, scale)

    assert torch.equal(actual, expected)
    assert actual[0, 0].item() == 0.0
    assert actual[0, 1].item() == 1.0
    assert actual[0, 8].item() == 0.0
    assert actual[0, 9].item() == -1.0
