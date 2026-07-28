"""Triton GEMM that consumes K3 MXFP4 weights without materializing them."""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except ModuleNotFoundError:  # CPU-only test environments do not install Triton.
    triton = None
    tl = None

MXFP4_GROUP_SIZE = 32


if triton is not None:

    @triton.jit
    def _decode_e2m1(code):
        magnitude_code = code & 0x7
        magnitude = tl.where(
            magnitude_code == 0,
            0.0,
            tl.where(
                magnitude_code == 1,
                0.5,
                tl.where(
                    magnitude_code == 2,
                    1.0,
                    tl.where(
                        magnitude_code == 3,
                        1.5,
                        tl.where(
                            magnitude_code == 4,
                            2.0,
                            tl.where(
                                magnitude_code == 5,
                                3.0,
                                tl.where(magnitude_code == 6, 4.0, 6.0),
                            ),
                        ),
                    ),
                ),
            ),
        )
        sign = tl.where((code & 0x8) == 0, 1.0, -1.0)
        return magnitude * sign


    @triton.autotune(
        configs=[
            triton.Config(
                {"BLOCK_M": 16, "BLOCK_N": 32, "BLOCK_K": 32},
                num_stages=4,
                num_warps=4,
            ),
            triton.Config(
                {"BLOCK_M": 16, "BLOCK_N": 64, "BLOCK_K": 64},
                num_stages=4,
                num_warps=4,
            ),
            triton.Config(
                {"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 64},
                num_stages=4,
                num_warps=4,
            ),
            triton.Config(
                {"BLOCK_M": 32, "BLOCK_N": 128, "BLOCK_K": 64},
                num_stages=3,
                num_warps=8,
            ),
        ],
        key=["M", "N", "K"],
    )
    @triton.jit
    def _mxfp4_gemm_kernel(
        activation_ptr,
        packed_ptr,
        scale_ptr,
        output_ptr,
        M,
        N,
        K,
        stride_am,
        stride_ak,
        stride_pn,
        stride_pk,
        stride_sn,
        stride_sk,
        stride_om,
        stride_on,
        GROUP_SIZE: tl.constexpr,
        IS_BF16: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offsets_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offsets_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offsets_k = tl.arange(0, BLOCK_K)
        accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        # Each reduction tile expands only the values consumed by this dot.
        for k_start in range(0, K, BLOCK_K):
            current_k = k_start + offsets_k
            activation = tl.load(
                activation_ptr
                + offsets_m[:, None] * stride_am
                + current_k[None, :] * stride_ak,
                mask=(offsets_m[:, None] < M) & (current_k[None, :] < K),
                other=0.0,
            )

            packed_byte = tl.load(
                packed_ptr
                + offsets_n[None, :] * stride_pn
                + (current_k[:, None] // 2) * stride_pk,
                mask=(current_k[:, None] < K) & (offsets_n[None, :] < N),
                other=0,
            )
            low_nibble = packed_byte & 0x0F
            high_nibble = packed_byte >> 4
            code = tl.where((current_k[:, None] & 1) == 0, low_nibble, high_nibble)

            scale_byte = tl.load(
                scale_ptr
                + offsets_n[None, :] * stride_sn
                + (current_k[:, None] // GROUP_SIZE) * stride_sk,
                mask=(current_k[:, None] < K) & (offsets_n[None, :] < N),
                other=127,
            )
            decoded = _decode_e2m1(code)
            weight = decoded * tl.exp2(scale_byte.to(tl.float32) - 127.0)
            if IS_BF16:
                weight = weight.to(tl.bfloat16)
            else:
                weight = weight.to(tl.float16)
            accumulator += tl.dot(activation, weight, out_dtype=tl.float32)

        output_offsets = (
            output_ptr
            + offsets_m[:, None] * stride_om
            + offsets_n[None, :] * stride_on
        )
        tl.store(
            output_offsets,
            accumulator,
            mask=(offsets_m[:, None] < M) & (offsets_n[None, :] < N),
        )

else:
    _mxfp4_gemm_kernel = None


def _validate_inputs(
    activations: torch.Tensor,
    weight_packed: torch.Tensor,
    weight_scale: torch.Tensor,
) -> tuple[int, int, int]:
    if activations.ndim != 2:
        raise ValueError("activations must have shape [tokens, reduction]")
    if weight_packed.ndim != 2 or weight_scale.ndim != 2:
        raise ValueError("packed weights and scales must both be matrices")
    if activations.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError("activations must use torch.float16 or torch.bfloat16")
    if weight_packed.dtype != torch.uint8 or weight_scale.dtype != torch.uint8:
        raise TypeError("packed weights and scales must both use torch.uint8")
    if activations.device.type != "cuda":
        raise ValueError("the fused MXFP4 GEMM requires CUDA activations")
    if not (
        activations.device == weight_packed.device == weight_scale.device
    ):
        raise ValueError("activations, packed weights, and scales must share a device")

    m, k = activations.shape
    n, packed_k = weight_packed.shape
    if k == 0 or n == 0:
        raise ValueError("reduction and output dimensions must be nonzero")
    if k % MXFP4_GROUP_SIZE != 0:
        raise ValueError("MXFP4 reduction dimension must be divisible by 32")
    if packed_k * 2 != k:
        raise ValueError("packed weight shape does not match the reduction dimension")
    if weight_scale.shape != (n, k // MXFP4_GROUP_SIZE):
        raise ValueError("scale shape must provide one byte per 32 reduction values")
    return m, n, k


def mxfp4_gemm(
    activations: torch.Tensor,
    weight_packed: torch.Tensor,
    weight_scale: torch.Tensor,
) -> torch.Tensor:
    """Compute ``activations @ dequant(weight).T`` without a decoded tensor."""
    if triton is None or _mxfp4_gemm_kernel is None:
        raise RuntimeError("Triton is required for the fused MXFP4 GEMM")
    m, n, k = _validate_inputs(activations, weight_packed, weight_scale)
    output = torch.empty((m, n), dtype=activations.dtype, device=activations.device)
    if m == 0:
        return output

    def grid(meta):
        return (triton.cdiv(m, meta["BLOCK_M"]), triton.cdiv(n, meta["BLOCK_N"]))
    with torch.cuda.device(activations.device):
        _mxfp4_gemm_kernel[grid](
            activations,
            weight_packed,
            weight_scale,
            output,
            m,
            n,
            k,
            activations.stride(0),
            activations.stride(1),
            weight_packed.stride(0),
            weight_packed.stride(1),
            weight_scale.stride(0),
            weight_scale.stride(1),
            output.stride(0),
            output.stride(1),
            GROUP_SIZE=MXFP4_GROUP_SIZE,
            IS_BF16=activations.dtype == torch.bfloat16,
        )
    return output


def mxfp4_linear(
    activations: torch.Tensor,
    weight_packed: torch.Tensor,
    weight_scale: torch.Tensor,
) -> torch.Tensor:
    """Linear-style name for callers that store weights as output-by-input."""
    return mxfp4_gemm(activations, weight_packed, weight_scale)
