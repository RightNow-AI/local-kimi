"""Batch-1 dense W3A16 GEMV with a prefill-safe linear fallback.

For every eight weights, INT4 extraction uses four masks and four shifts.
This INT3 layout uses eight masks, nine shifts, and two OR operations. That is
11 extra source-level integer operations per eight weights, or 1.375 per
weight, before compiler folding. At group size 32 the packed weights and scales
still use 22.22 percent fewer bytes than W4A16. Batch-1 GEMV has very low
arithmetic intensity, so it is expected to remain memory bound. The benchmark
must decide whether the byte reduction outweighs the extra integer work.
"""

from __future__ import annotations

from typing import NamedTuple

import torch
import torch.nn.functional as F

from engine.quant.w3a16 import W3A16Tensor, dequantise

try:
    import triton
    import triton.language as tl
except ModuleNotFoundError:  # CPU-only environments do not install Triton.
    triton = None
    tl = None


class DenseGemvConfig(NamedTuple):
    """One explicit launch candidate for the batch-1 dense GEMV."""

    name: str
    block_n: int
    block_k: int
    split_k: int
    num_warps: int
    num_stages: int


# These are the measured W4A16 dense candidates and selector choices. They are
# inherited as the W3A16 starting table because this lane may not run kernels.
# BENCH-W3A16.py measures the shipped choice against W4A16 at the expert shapes.
W3A16_DENSE_GEMV_CONFIGS = (
    DenseGemvConfig("n32_k64_s2_w4_st3", 32, 64, 2, 4, 3),
    DenseGemvConfig("n32_k64_s4_w4_st3", 32, 64, 4, 4, 3),
    DenseGemvConfig("n64_k64_s4_w4_st4", 64, 64, 4, 4, 4),
    DenseGemvConfig("n32_k128_s2_w8_st2", 32, 128, 2, 8, 2),
    DenseGemvConfig("n64_k64_s1_w4_st3", 64, 64, 1, 4, 3),
    DenseGemvConfig("n128_k32_s1_w8_st3", 128, 32, 1, 8, 3),
    DenseGemvConfig("n32_k64_s1_w4_st3", 32, 64, 1, 4, 3),
    DenseGemvConfig("n16_k64_s1_w4_st3", 16, 64, 1, 4, 3),
)


if triton is not None:

    @triton.jit
    def _w3a16_dense_gemv_kernel(
        activation_ptr,
        packed_ptr,
        scale_ptr,
        result_ptr,
        N: tl.constexpr,
        K: tl.constexpr,
        stride_ak,
        stride_pn,
        stride_pk,
        stride_sn,
        stride_sk,
        stride_on,
        GROUP: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        SPLIT_K: tl.constexpr,
    ):
        block_n = tl.program_id(0)
        split = tl.program_id(1)
        offsets_n = block_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offsets_pack_group = tl.arange(0, BLOCK_K // 8)
        accumulator = tl.zeros((BLOCK_N,), dtype=tl.float32)

        # Each three-byte group supplies eight consecutive K values. Packed
        # weights are [N, K / 8 * 3], so K remains the lane-contiguous axis.
        for k_group_start in range(0, K, BLOCK_K * SPLIT_K):
            current_k0 = (
                k_group_start
                + split * BLOCK_K
                + offsets_pack_group * 8
            )
            packed_byte0 = (current_k0 // 8) * 3
            weight_mask = (
                (offsets_n[:, None] < N)
                & (current_k0[None, :] < K)
            )
            byte0 = tl.load(
                packed_ptr
                + offsets_n[:, None] * stride_pn
                + packed_byte0[None, :] * stride_pk,
                mask=weight_mask,
                other=0,
            )
            byte1 = tl.load(
                packed_ptr
                + offsets_n[:, None] * stride_pn
                + (packed_byte0[None, :] + 1) * stride_pk,
                mask=weight_mask,
                other=0,
            )
            byte2 = tl.load(
                packed_ptr
                + offsets_n[:, None] * stride_pn
                + (packed_byte0[None, :] + 2) * stride_pk,
                mask=weight_mask,
                other=0,
            )

            # This exactly mirrors engine.quant.w3a16. Codes c2 and c5 cross
            # byte boundaries. Changing either side independently can produce
            # in-range values with entirely wrong signs and positions.
            scale = tl.load(
                scale_ptr
                + offsets_n[:, None] * stride_sn
                + (current_k0[None, :] // GROUP) * stride_sk,
                mask=weight_mask,
                other=1.0,
            ).to(tl.float32)

            code0 = byte0 & 0x7
            signed0 = code0.to(tl.int32)
            signed0 = tl.where(signed0 >= 4, signed0 - 8, signed0)
            activation0 = tl.load(
                activation_ptr + current_k0 * stride_ak,
                mask=current_k0 < K,
                other=0.0,
            ).to(tl.float32)
            decoded0 = (signed0.to(tl.float32) * scale).to(tl.bfloat16)
            products = activation0[None, :] * decoded0.to(tl.float32)

            code1 = (byte0 >> 3) & 0x7
            signed1 = code1.to(tl.int32)
            signed1 = tl.where(signed1 >= 4, signed1 - 8, signed1)
            activation1 = tl.load(
                activation_ptr + (current_k0 + 1) * stride_ak,
                mask=current_k0 + 1 < K,
                other=0.0,
            ).to(tl.float32)
            decoded1 = (signed1.to(tl.float32) * scale).to(tl.bfloat16)
            products += activation1[None, :] * decoded1.to(tl.float32)

            code2 = ((byte0 >> 6) | (byte1 << 2)) & 0x7
            signed2 = code2.to(tl.int32)
            signed2 = tl.where(signed2 >= 4, signed2 - 8, signed2)
            activation2 = tl.load(
                activation_ptr + (current_k0 + 2) * stride_ak,
                mask=current_k0 + 2 < K,
                other=0.0,
            ).to(tl.float32)
            decoded2 = (signed2.to(tl.float32) * scale).to(tl.bfloat16)
            products += activation2[None, :] * decoded2.to(tl.float32)

            code3 = (byte1 >> 1) & 0x7
            signed3 = code3.to(tl.int32)
            signed3 = tl.where(signed3 >= 4, signed3 - 8, signed3)
            activation3 = tl.load(
                activation_ptr + (current_k0 + 3) * stride_ak,
                mask=current_k0 + 3 < K,
                other=0.0,
            ).to(tl.float32)
            decoded3 = (signed3.to(tl.float32) * scale).to(tl.bfloat16)
            products += activation3[None, :] * decoded3.to(tl.float32)

            code4 = (byte1 >> 4) & 0x7
            signed4 = code4.to(tl.int32)
            signed4 = tl.where(signed4 >= 4, signed4 - 8, signed4)
            activation4 = tl.load(
                activation_ptr + (current_k0 + 4) * stride_ak,
                mask=current_k0 + 4 < K,
                other=0.0,
            ).to(tl.float32)
            decoded4 = (signed4.to(tl.float32) * scale).to(tl.bfloat16)
            products += activation4[None, :] * decoded4.to(tl.float32)

            code5 = ((byte1 >> 7) | (byte2 << 1)) & 0x7
            signed5 = code5.to(tl.int32)
            signed5 = tl.where(signed5 >= 4, signed5 - 8, signed5)
            activation5 = tl.load(
                activation_ptr + (current_k0 + 5) * stride_ak,
                mask=current_k0 + 5 < K,
                other=0.0,
            ).to(tl.float32)
            decoded5 = (signed5.to(tl.float32) * scale).to(tl.bfloat16)
            products += activation5[None, :] * decoded5.to(tl.float32)

            code6 = (byte2 >> 2) & 0x7
            signed6 = code6.to(tl.int32)
            signed6 = tl.where(signed6 >= 4, signed6 - 8, signed6)
            activation6 = tl.load(
                activation_ptr + (current_k0 + 6) * stride_ak,
                mask=current_k0 + 6 < K,
                other=0.0,
            ).to(tl.float32)
            decoded6 = (signed6.to(tl.float32) * scale).to(tl.bfloat16)
            products += activation6[None, :] * decoded6.to(tl.float32)

            code7 = (byte2 >> 5) & 0x7
            signed7 = code7.to(tl.int32)
            signed7 = tl.where(signed7 >= 4, signed7 - 8, signed7)
            activation7 = tl.load(
                activation_ptr + (current_k0 + 7) * stride_ak,
                mask=current_k0 + 7 < K,
                other=0.0,
            ).to(tl.float32)
            decoded7 = (signed7.to(tl.float32) * scale).to(tl.bfloat16)
            products += activation7[None, :] * decoded7.to(tl.float32)
            accumulator += tl.sum(products, axis=1)

        if SPLIT_K == 1:
            output_offsets = result_ptr + offsets_n * stride_on
        else:
            output_offsets = result_ptr + split * N + offsets_n
        tl.store(output_offsets, accumulator, mask=offsets_n < N)


    @triton.jit
    def _reduce_w3a16_dense_partials_kernel(
        partial_ptr,
        output_ptr,
        N: tl.constexpr,
        stride_on,
        BLOCK_N: tl.constexpr,
        SPLIT_K: tl.constexpr,
    ):
        block_n = tl.program_id(0)
        offsets_n = block_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offsets_split = tl.arange(0, SPLIT_K)
        partials = tl.load(
            partial_ptr
            + offsets_split[:, None] * N
            + offsets_n[None, :],
            mask=offsets_n[None, :] < N,
            other=0.0,
        )
        result = tl.sum(partials, axis=0)
        tl.store(
            output_ptr + offsets_n * stride_on,
            result,
            mask=offsets_n < N,
        )

else:
    _w3a16_dense_gemv_kernel = None
    _reduce_w3a16_dense_partials_kernel = None


def _validate_inputs(
    activations: torch.Tensor,
    packed_weights: torch.Tensor,
    scales: torch.Tensor,
    group_size: int,
) -> tuple[int, int, int]:
    if activations.ndim != 2 or activations.dtype != torch.bfloat16:
        raise TypeError("activations must be a BF16 [tokens, reduction] matrix")
    if packed_weights.ndim != 2 or packed_weights.dtype != torch.uint8:
        raise TypeError("packed weights must be a uint8 [output, reduction / 8 * 3] matrix")
    if scales.ndim != 2 or scales.dtype != torch.bfloat16:
        raise TypeError("scales must be a BF16 [output, reduction / group_size] matrix")
    if not isinstance(group_size, int) or group_size <= 0:
        raise ValueError("group_size must be a positive integer")
    if group_size % 8:
        raise ValueError("W3A16 GEMV requires group_size to be divisible by 8")
    if activations.device.type != "cuda":
        raise ValueError("W3A16 dense GEMV requires CUDA")
    if not (activations.device == packed_weights.device == scales.device):
        raise ValueError("activations, packed weights, and scales must share a device")

    m, reduction = activations.shape
    output_size, packed_reduction = packed_weights.shape
    if output_size == 0 or reduction == 0:
        raise ValueError("W3A16 dense GEMV requires nonzero dimensions")
    if reduction % 8:
        raise ValueError("the reduction dimension must be divisible by 8")
    if reduction % group_size:
        raise ValueError(
            f"the reduction dimension must be divisible by group_size={group_size}"
        )
    if packed_reduction != reduction // 8 * 3:
        raise ValueError("packed weights have the wrong reduction width")
    if scales.shape != (output_size, reduction // group_size):
        raise ValueError("W3A16 scales have the wrong shape")
    return m, output_size, reduction


def _select_dense_config(output_size: int) -> DenseGemvConfig:
    """Use the W4A16 end-to-end winners until W3A16 is measured."""
    if output_size <= 3072:
        return W3A16_DENSE_GEMV_CONFIGS[1]
    if output_size <= 8192:
        return W3A16_DENSE_GEMV_CONFIGS[0]
    return W3A16_DENSE_GEMV_CONFIGS[4]


def _launch_w3a16_dense_gemv(
    activations: torch.Tensor,
    packed_weights: torch.Tensor,
    scales: torch.Tensor,
    config: DenseGemvConfig,
    *,
    group_size: int,
) -> torch.Tensor:
    """Launch one explicit batch-1 W3A16 candidate."""
    if triton is None or _w3a16_dense_gemv_kernel is None:
        raise RuntimeError("Triton is required for W3A16 dense GEMV")
    m, output_size, reduction = _validate_inputs(
        activations, packed_weights, scales, group_size
    )
    if m != 1:
        raise ValueError("the explicit dense GEMV launcher requires exactly one token")

    output = torch.empty(
        (1, output_size), dtype=torch.bfloat16, device=activations.device
    )
    result = output
    if config.split_k > 1:
        result = torch.empty(
            (config.split_k, output_size),
            dtype=torch.float32,
            device=activations.device,
        )

    grid = (triton.cdiv(output_size, config.block_n), config.split_k)
    with torch.cuda.device(activations.device):
        _w3a16_dense_gemv_kernel[grid](
            activations,
            packed_weights,
            scales,
            result,
            N=output_size,
            K=reduction,
            stride_ak=activations.stride(1),
            stride_pn=packed_weights.stride(0),
            stride_pk=packed_weights.stride(1),
            stride_sn=scales.stride(0),
            stride_sk=scales.stride(1),
            stride_on=output.stride(1),
            GROUP=group_size,
            BLOCK_N=config.block_n,
            BLOCK_K=config.block_k,
            SPLIT_K=config.split_k,
            num_warps=config.num_warps,
            num_stages=config.num_stages,
        )
        if config.split_k > 1:
            reduction_grid = (triton.cdiv(output_size, config.block_n),)
            _reduce_w3a16_dense_partials_kernel[reduction_grid](
                result,
                output,
                N=output_size,
                stride_on=output.stride(1),
                BLOCK_N=config.block_n,
                SPLIT_K=config.split_k,
                num_warps=4,
                num_stages=1,
            )
    return output


def _as_encoded(
    packed_weights: torch.Tensor,
    scales: torch.Tensor,
    output_size: int,
    reduction: int,
    group_size: int,
) -> W3A16Tensor:
    return W3A16Tensor(
        packed=packed_weights,
        scales=scales,
        original_shape=(output_size, reduction),
        group_size=group_size,
    )


def w3a16_dense_gemv(
    activations: torch.Tensor,
    packed_weights: torch.Tensor,
    scales: torch.Tensor,
    *,
    group_size: int,
) -> torch.Tensor:
    """Use the W3A16 GEMV for one token and F.linear for prefill."""
    m, output_size, reduction = _validate_inputs(
        activations, packed_weights, scales, group_size
    )
    if m == 0:
        return torch.empty(
            (0, output_size), dtype=torch.bfloat16, device=activations.device
        )

    encoded = _as_encoded(
        packed_weights,
        scales,
        output_size,
        reduction,
        group_size,
    )
    if m > 1:
        return F.linear(activations, dequantise(encoded))
    if triton is None:
        raise RuntimeError("Triton is required for batch-1 W3A16 dense GEMV")

    config = _select_dense_config(output_size)
    return _launch_w3a16_dense_gemv(
        activations,
        packed_weights,
        scales,
        config,
        group_size=group_size,
    )
