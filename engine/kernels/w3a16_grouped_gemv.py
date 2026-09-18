"""Batch-1 grouped W3A16 GEMV with K-contiguous packed reads."""

from __future__ import annotations

from typing import NamedTuple

import torch

from engine.quant.w3a16 import W3A16Tensor, dequantise

try:
    import triton
    import triton.language as tl
except ModuleNotFoundError:  # CPU-only environments do not install Triton.
    triton = None
    tl = None


class GroupedGemvConfig(NamedTuple):
    """The single conservative launch shape for grouped W3A16 decode."""

    name: str
    block_n: int
    block_k: int
    num_warps: int
    num_stages: int


# Start from the original fast grouped W4A16 tile. The W3 codec performs more
# integer work per byte, so isolated timing and end-to-end decode must decide
# whether a different tile is justified later.
GROUPED_W3A16_CONFIG = GroupedGemvConfig(
    "n32_k64_w4_st3",
    block_n=32,
    block_k=64,
    num_warps=4,
    num_stages=3,
)


if triton is not None:

    @triton.jit
    def _grouped_w3a16_gemv_kernel(
        activation_ptr,
        expert_index_ptr,
        packed_ptr,
        scale_ptr,
        result_ptr,
        ROUTES: tl.constexpr,
        N: tl.constexpr,
        K: tl.constexpr,
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
        GROUP: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        assignment = tl.program_id(0)
        block_n = tl.program_id(1)
        token = assignment // ROUTES
        route = assignment - token * ROUTES
        expert = tl.load(
            expert_index_ptr + token * stride_it + route * stride_ir
        ).to(tl.int64)

        offsets_n = block_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offsets_pack_group = tl.arange(0, BLOCK_K // 8)
        accumulator = tl.zeros((BLOCK_N,), dtype=tl.float32)

        # Each inner lane advances through consecutive three-byte groups on K.
        # The accumulator stays one-dimensional because decode has one input
        # row per assignment. There is no split-K buffer or atomic reduction.
        for k_group_start in range(0, K, BLOCK_K):
            current_k0 = k_group_start + offsets_pack_group * 8
            packed_byte0 = (current_k0 // 8) * 3
            weight_mask = (
                (offsets_n[:, None] < N)
                & (current_k0[None, :] < K)
            )
            byte0 = tl.load(
                packed_ptr
                + expert * stride_we
                + offsets_n[:, None] * stride_wn
                + packed_byte0[None, :] * stride_wk,
                mask=weight_mask,
                other=0,
            )
            byte1 = tl.load(
                packed_ptr
                + expert * stride_we
                + offsets_n[:, None] * stride_wn
                + (packed_byte0[None, :] + 1) * stride_wk,
                mask=weight_mask,
                other=0,
            )
            byte2 = tl.load(
                packed_ptr
                + expert * stride_we
                + offsets_n[:, None] * stride_wn
                + (packed_byte0[None, :] + 2) * stride_wk,
                mask=weight_mask,
                other=0,
            )
            scale = tl.load(
                scale_ptr
                + expert * stride_se
                + offsets_n[:, None] * stride_sn
                + (current_k0[None, :] // GROUP) * stride_sk,
                mask=weight_mask,
                other=1.0,
            ).to(tl.float32)

            # This is the canonical little-endian 24-bit layout from
            # engine.quant.w3a16. Codes 2 and 5 cross byte boundaries.
            code0 = byte0 & 0x7
            signed0 = code0.to(tl.int32)
            signed0 = tl.where(signed0 >= 4, signed0 - 8, signed0)
            activation0 = tl.load(
                activation_ptr
                + token * stride_at
                + current_k0 * stride_ak,
                mask=current_k0 < K,
                other=0.0,
            ).to(tl.float32)
            decoded0 = (signed0.to(tl.float32) * scale).to(tl.bfloat16)
            products = activation0[None, :] * decoded0.to(tl.float32)

            code1 = (byte0 >> 3) & 0x7
            signed1 = code1.to(tl.int32)
            signed1 = tl.where(signed1 >= 4, signed1 - 8, signed1)
            activation1 = tl.load(
                activation_ptr
                + token * stride_at
                + (current_k0 + 1) * stride_ak,
                mask=current_k0 + 1 < K,
                other=0.0,
            ).to(tl.float32)
            decoded1 = (signed1.to(tl.float32) * scale).to(tl.bfloat16)
            products += activation1[None, :] * decoded1.to(tl.float32)

            code2 = ((byte0 >> 6) | (byte1 << 2)) & 0x7
            signed2 = code2.to(tl.int32)
            signed2 = tl.where(signed2 >= 4, signed2 - 8, signed2)
            activation2 = tl.load(
                activation_ptr
                + token * stride_at
                + (current_k0 + 2) * stride_ak,
                mask=current_k0 + 2 < K,
                other=0.0,
            ).to(tl.float32)
            decoded2 = (signed2.to(tl.float32) * scale).to(tl.bfloat16)
            products += activation2[None, :] * decoded2.to(tl.float32)

            code3 = (byte1 >> 1) & 0x7
            signed3 = code3.to(tl.int32)
            signed3 = tl.where(signed3 >= 4, signed3 - 8, signed3)
            activation3 = tl.load(
                activation_ptr
                + token * stride_at
                + (current_k0 + 3) * stride_ak,
                mask=current_k0 + 3 < K,
                other=0.0,
            ).to(tl.float32)
            decoded3 = (signed3.to(tl.float32) * scale).to(tl.bfloat16)
            products += activation3[None, :] * decoded3.to(tl.float32)

            code4 = (byte1 >> 4) & 0x7
            signed4 = code4.to(tl.int32)
            signed4 = tl.where(signed4 >= 4, signed4 - 8, signed4)
            activation4 = tl.load(
                activation_ptr
                + token * stride_at
                + (current_k0 + 4) * stride_ak,
                mask=current_k0 + 4 < K,
                other=0.0,
            ).to(tl.float32)
            decoded4 = (signed4.to(tl.float32) * scale).to(tl.bfloat16)
            products += activation4[None, :] * decoded4.to(tl.float32)

            code5 = ((byte1 >> 7) | (byte2 << 1)) & 0x7
            signed5 = code5.to(tl.int32)
            signed5 = tl.where(signed5 >= 4, signed5 - 8, signed5)
            activation5 = tl.load(
                activation_ptr
                + token * stride_at
                + (current_k0 + 5) * stride_ak,
                mask=current_k0 + 5 < K,
                other=0.0,
            ).to(tl.float32)
            decoded5 = (signed5.to(tl.float32) * scale).to(tl.bfloat16)
            products += activation5[None, :] * decoded5.to(tl.float32)

            code6 = (byte2 >> 2) & 0x7
            signed6 = code6.to(tl.int32)
            signed6 = tl.where(signed6 >= 4, signed6 - 8, signed6)
            activation6 = tl.load(
                activation_ptr
                + token * stride_at
                + (current_k0 + 6) * stride_ak,
                mask=current_k0 + 6 < K,
                other=0.0,
            ).to(tl.float32)
            decoded6 = (signed6.to(tl.float32) * scale).to(tl.bfloat16)
            products += activation6[None, :] * decoded6.to(tl.float32)

            code7 = (byte2 >> 5) & 0x7
            signed7 = code7.to(tl.int32)
            signed7 = tl.where(signed7 >= 4, signed7 - 8, signed7)
            activation7 = tl.load(
                activation_ptr
                + token * stride_at
                + (current_k0 + 7) * stride_ak,
                mask=current_k0 + 7 < K,
                other=0.0,
            ).to(tl.float32)
            decoded7 = (signed7.to(tl.float32) * scale).to(tl.bfloat16)
            products += activation7[None, :] * decoded7.to(tl.float32)

            accumulator += tl.sum(products, axis=1)

        tl.store(
            result_ptr
            + token * stride_ot
            + route * stride_or
            + offsets_n * stride_on,
            accumulator,
            mask=offsets_n < N,
        )

else:
    _grouped_w3a16_gemv_kernel = None


def _validate_inputs(
    activations: torch.Tensor,
    expert_indices: torch.Tensor,
    packed_weights: torch.Tensor,
    scales: torch.Tensor,
    group_size: int,
    *,
    require_cuda: bool,
) -> tuple[int, int, int, int]:
    if activations.ndim != 2 or activations.dtype != torch.bfloat16:
        raise TypeError("activations must be a BF16 [tokens, reduction] matrix")
    if expert_indices.ndim != 2 or expert_indices.dtype != torch.long:
        raise TypeError("expert_indices must be an int64 [tokens, routes] matrix")
    if packed_weights.ndim != 3 or packed_weights.dtype != torch.uint8:
        raise TypeError(
            "packed weights must be a uint8 [experts, output, reduction / 8 * 3] tensor"
        )
    if scales.ndim != 3 or scales.dtype != torch.bfloat16:
        raise TypeError(
            "scales must be a BF16 [experts, output, reduction / group_size] tensor"
        )
    if not isinstance(group_size, int) or group_size <= 0:
        raise ValueError("group_size must be a positive integer")
    if group_size % 8:
        raise ValueError("grouped W3A16 GEMV requires group_size to be divisible by 8")
    if require_cuda and activations.device.type != "cuda":
        raise ValueError("grouped W3A16 GEMV requires CUDA")
    if not (
        activations.device
        == expert_indices.device
        == packed_weights.device
        == scales.device
    ):
        raise ValueError("grouped W3A16 inputs must share one device")

    tokens, reduction = activations.shape
    route_tokens, routes = expert_indices.shape
    experts, output_size, packed_reduction = packed_weights.shape
    if route_tokens != tokens:
        raise ValueError("expert routing must have one row per token")
    if routes == 0 or experts == 0 or output_size == 0 or reduction == 0:
        raise ValueError("grouped W3A16 GEMV requires nonzero dimensions")
    if reduction % 8:
        raise ValueError("the reduction dimension must be divisible by 8")
    if reduction % group_size:
        raise ValueError(
            f"the reduction dimension must be divisible by group_size={group_size}"
        )
    if packed_reduction != reduction // 8 * 3:
        raise ValueError("packed grouped weights have the wrong reduction width")
    if scales.shape != (experts, output_size, reduction // group_size):
        raise ValueError("grouped W3A16 scales have the wrong shape")
    return tokens, routes, output_size, reduction


def grouped_w3a16_gemv_reference(
    activations: torch.Tensor,
    expert_indices: torch.Tensor,
    packed_weights: torch.Tensor,
    scales: torch.Tensor,
    *,
    group_size: int,
) -> torch.Tensor:
    """Dequantise selected expert matrices and run their GEMVs in PyTorch."""
    tokens, routes, output_size, reduction = _validate_inputs(
        activations,
        expert_indices,
        packed_weights,
        scales,
        group_size,
        require_cuda=False,
    )
    if tokens == 0:
        return torch.empty(
            (0, routes, output_size),
            dtype=torch.bfloat16,
            device=activations.device,
        )

    assignments = tokens * routes
    selected_packed = packed_weights.index_select(0, expert_indices.reshape(-1))
    selected_scales = scales.index_select(0, expert_indices.reshape(-1))
    encoded = W3A16Tensor(
        packed=selected_packed.reshape(assignments * output_size, -1),
        scales=selected_scales.reshape(assignments * output_size, -1),
        original_shape=(assignments * output_size, reduction),
        group_size=group_size,
    )
    restored = dequantise(encoded).reshape(
        assignments, output_size, reduction
    )
    assignment_activations = (
        activations[:, None, :]
        .expand(tokens, routes, reduction)
        .reshape(assignments, reduction)
    )
    output = torch.bmm(
        restored.float(), assignment_activations.float().unsqueeze(-1)
    ).squeeze(-1)
    return output.to(torch.bfloat16).reshape(tokens, routes, output_size)


def _launch_grouped_w3a16_gemv(
    activations: torch.Tensor,
    expert_indices: torch.Tensor,
    packed_weights: torch.Tensor,
    scales: torch.Tensor,
    config: GroupedGemvConfig,
    *,
    group_size: int,
) -> torch.Tensor:
    """Launch one fixed-tree grouped W3A16 GEMV candidate."""
    if triton is None or _grouped_w3a16_gemv_kernel is None:
        raise RuntimeError("Triton is required for grouped W3A16 GEMV")
    tokens, routes, output_size, reduction = _validate_inputs(
        activations,
        expert_indices,
        packed_weights,
        scales,
        group_size,
        require_cuda=True,
    )
    if config.block_k % 8:
        raise ValueError("grouped W3A16 BLOCK_K must be divisible by 8")

    output = torch.empty(
        (tokens, routes, output_size),
        dtype=torch.bfloat16,
        device=activations.device,
    )
    if tokens == 0:
        return output

    grid = (
        tokens * routes,
        triton.cdiv(output_size, config.block_n),
    )
    with torch.cuda.device(activations.device):
        _grouped_w3a16_gemv_kernel[grid](
            activations,
            expert_indices,
            packed_weights,
            scales,
            output,
            ROUTES=routes,
            N=output_size,
            K=reduction,
            stride_at=activations.stride(0),
            stride_ak=activations.stride(1),
            stride_it=expert_indices.stride(0),
            stride_ir=expert_indices.stride(1),
            stride_we=packed_weights.stride(0),
            stride_wn=packed_weights.stride(1),
            stride_wk=packed_weights.stride(2),
            stride_se=scales.stride(0),
            stride_sn=scales.stride(1),
            stride_sk=scales.stride(2),
            stride_ot=output.stride(0),
            stride_or=output.stride(1),
            stride_on=output.stride(2),
            GROUP=group_size,
            BLOCK_N=config.block_n,
            BLOCK_K=config.block_k,
            num_warps=config.num_warps,
            num_stages=config.num_stages,
        )
    return output


def grouped_w3a16_gemv(
    activations: torch.Tensor,
    expert_indices: torch.Tensor,
    packed_weights: torch.Tensor,
    scales: torch.Tensor,
    *,
    group_size: int,
) -> torch.Tensor:
    """Run one packed W3A16 GEMV for every fixed token-route assignment."""
    return _launch_grouped_w3a16_gemv(
        activations,
        expert_indices,
        packed_weights,
        scales,
        GROUPED_W3A16_CONFIG,
        group_size=group_size,
    )
