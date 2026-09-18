"""Triton GEMM that consumes W4A16 weights without materializing BF16."""

from __future__ import annotations

import torch

from .w4a16 import GROUP_SIZE, W4A16Tensor

try:
    import triton
    import triton.language as tl
except ModuleNotFoundError:  # CPU-only environments do not install Triton.
    triton = None
    tl = None


if triton is not None:

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
    def _w4a16_gemm_kernel(
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
                mask=(offsets_n[None, :] < N) & (current_k[:, None] < K),
                other=0,
            )
            low_nibble = packed_byte & 0xF
            high_nibble = packed_byte >> 4
            nibble = tl.where(
                (current_k[:, None] & 1) == 0,
                low_nibble,
                high_nibble,
            )
            signed = nibble.to(tl.int32)
            signed = tl.where(signed >= 8, signed - 16, signed)
            scale = tl.load(
                scale_ptr
                + offsets_n[None, :] * stride_sn
                + (current_k[:, None] // GROUP_SIZE) * stride_sk,
                mask=(offsets_n[None, :] < N) & (current_k[:, None] < K),
                other=1.0,
            )
            # BF16 before tl.dot matches the explicit dequantized reference.
            decoded_weight = (
                signed.to(tl.float32) * scale.to(tl.float32)
            ).to(tl.bfloat16)
            accumulator += tl.dot(
                activation,
                decoded_weight,
                out_dtype=tl.float32,
            )

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
    _w4a16_gemm_kernel = None


def _validate_inputs(
    activations: torch.Tensor,
    encoded: W4A16Tensor,
) -> tuple[int, int, int]:
    if activations.ndim != 2 or activations.dtype != torch.bfloat16:
        raise TypeError("activations must be a BF16 [tokens, reduction] matrix")
    if activations.device.type != "cuda":
        raise ValueError("the fused W4A16 GEMM requires CUDA")
    if not (
        activations.device == encoded.packed.device == encoded.scales.device
    ):
        raise ValueError("activations, packed weights, and scales must share a device")
    m, k = activations.shape
    n, expected_k = encoded.original_shape
    if k != expected_k:
        raise ValueError(f"activation reduction {k} does not match weight {expected_k}")
    if k % GROUP_SIZE:
        raise ValueError("the reduction dimension must be divisible by group size 32")
    return m, n, k


def w4a16_linear(
    activations: torch.Tensor,
    encoded: W4A16Tensor,
) -> torch.Tensor:
    """Compute ``activations @ dequantise(encoded).T`` in one Triton kernel."""
    if triton is None or _w4a16_gemm_kernel is None:
        raise RuntimeError("Triton is required for the fused W4A16 GEMM")
    m, n, k = _validate_inputs(activations, encoded)
    output = torch.empty((m, n), dtype=torch.bfloat16, device=activations.device)
    if m == 0:
        return output

    def grid(meta):
        return (
            triton.cdiv(m, meta["BLOCK_M"]),
            triton.cdiv(n, meta["BLOCK_N"]),
        )

    with torch.cuda.device(activations.device):
        _w4a16_gemm_kernel[grid](
            activations,
            encoded.packed,
            encoded.scales,
            output,
            m,
            n,
            k,
            activations.stride(0),
            activations.stride(1),
            encoded.packed.stride(0),
            encoded.packed.stride(1),
            encoded.scales.stride(0),
            encoded.scales.stride(1),
            output.stride(0),
            output.stride(1),
            GROUP_SIZE=encoded.group_size,
        )
    return output
