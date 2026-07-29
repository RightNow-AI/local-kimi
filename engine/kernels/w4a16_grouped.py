"""Shape-stable grouped W4A16 matrix-vector products for decode."""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except ModuleNotFoundError:  # CPU-only environments do not install Triton.
    triton = None
    tl = None

GROUP_SIZE = 32


if triton is not None:

    @triton.jit
    def _grouped_w4a16_kernel(
        activation_ptr,
        expert_index_ptr,
        packed_ptr,
        scale_ptr,
        output_ptr,
        TOKENS,
        ROUTES,
        N,
        K,
        stride_at,
        stride_ak,
        stride_it,
        stride_ir,
        stride_we,
        stride_wn,
        stride_wk,
        stride_se,
        stride_sn,
        stride_sk,
        stride_ot,
        stride_or,
        stride_on,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        # Passed in rather than read from the module. Triton refuses a plain
        # global inside a jitted function, and this failed at compile time on
        # the GPU with "Cannot access global variable GROUP_SIZE from within
        # @jit'ed function".
        GROUP: tl.constexpr,
    ):
        assignment = tl.program_id(0)
        block_n = tl.program_id(1)
        token = assignment // ROUTES
        route = assignment - token * ROUTES
        expert = tl.load(
            expert_index_ptr + token * stride_it + route * stride_ir
        ).to(tl.int64)
        offsets_m = tl.arange(0, BLOCK_M)
        offsets_n = block_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offsets_k = tl.arange(0, BLOCK_K)
        accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for k_start in range(0, K, BLOCK_K):
            current_k = k_start + offsets_k
            activation = tl.load(
                activation_ptr
                + token * stride_at
                + offsets_m[:, None] * 0
                + current_k[None, :] * stride_ak,
                mask=current_k[None, :] < K,
                other=0.0,
            )
            packed_byte = tl.load(
                packed_ptr
                + expert * stride_we
                + offsets_n[None, :] * stride_wn
                + (current_k[:, None] // 2) * stride_wk,
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
                + expert * stride_se
                + offsets_n[None, :] * stride_sn
                + (current_k[:, None] // GROUP) * stride_sk,
                mask=(offsets_n[None, :] < N) & (current_k[:, None] < K),
                other=1.0,
            )
            decoded_weight = (
                signed.to(tl.float32) * scale.to(tl.float32)
            ).to(tl.bfloat16)
            accumulator += tl.dot(
                activation,
                decoded_weight,
                out_dtype=tl.float32,
            )

        tl.store(
            output_ptr
            + token * stride_ot
            + route * stride_or
            + offsets_n * stride_on,
            tl.max(accumulator, axis=0),
            mask=offsets_n < N,
        )

else:
    _grouped_w4a16_kernel = None


def grouped_w4a16_linear(
    activations: torch.Tensor,
    expert_indices: torch.Tensor,
    packed_weights: torch.Tensor,
    scales: torch.Tensor,
) -> torch.Tensor:
    """Run one packed linear for every fixed token-route assignment in one launch."""
    if triton is None or _grouped_w4a16_kernel is None:
        raise RuntimeError("Triton is required for grouped W4A16 decode")
    if activations.ndim != 2 or activations.dtype != torch.bfloat16:
        raise TypeError("activations must be a BF16 [tokens, reduction] matrix")
    if expert_indices.ndim != 2 or expert_indices.dtype != torch.long:
        raise TypeError("expert_indices must be an int64 [tokens, routes] matrix")
    if packed_weights.ndim != 3 or scales.ndim != 3:
        raise ValueError("grouped packed weights and scales must be rank three")
    if activations.device.type != "cuda":
        raise ValueError("grouped W4A16 decode requires CUDA")
    if not (
        activations.device
        == expert_indices.device
        == packed_weights.device
        == scales.device
    ):
        raise ValueError("grouped W4A16 inputs must share one device")
    tokens, reduction = activations.shape
    route_tokens, routes = expert_indices.shape
    experts, output_size, packed_reduction = packed_weights.shape
    if route_tokens != tokens:
        raise ValueError("expert routing must have one row per token")
    if routes == 0 or experts == 0:
        raise ValueError("grouped W4A16 decode requires routes and experts")
    if packed_reduction * 2 != reduction:
        raise ValueError("packed grouped weights have the wrong reduction width")
    if scales.shape != (experts, output_size, reduction // GROUP_SIZE):
        raise ValueError("grouped W4A16 scales have the wrong shape")

    output = torch.empty(
        tokens,
        routes,
        output_size,
        dtype=torch.bfloat16,
        device=activations.device,
    )
    grid = (tokens * routes, triton.cdiv(output_size, 64))
    with torch.cuda.device(activations.device):
        _grouped_w4a16_kernel[grid](
            activations,
            expert_indices,
            packed_weights,
            scales,
            output,
            tokens,
            routes,
            output_size,
            reduction,
            activations.stride(0),
            activations.stride(1),
            expert_indices.stride(0),
            expert_indices.stride(1),
            packed_weights.stride(0),
            packed_weights.stride(1),
            packed_weights.stride(2),
            scales.stride(0),
            scales.stride(1),
            scales.stride(2),
            output.stride(0),
            output.stride(1),
            output.stride(2),
            BLOCK_M=16,
            BLOCK_N=64,
            BLOCK_K=64,
            GROUP=GROUP_SIZE,
        )
    return output
