"""Fused grouped W4A16 down projection and MoE route combination."""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except ModuleNotFoundError:  # CPU-only environments do not install Triton.
    triton = None
    tl = None

GROUP_SIZE = 32

# Keep the measured production W2 tile shape. The shipped GEMV uses split-K=4,
# but this fusion must have one program per token and N block. Four consecutive
# K tiles are therefore unrolled inside that one program instead of producing a
# global partial buffer and launching a second reduction kernel.
LAUNCH_CONFIG = "n128_k32_s4_w8_st3"
BLOCK_N = 128
BLOCK_K = 32
K_TILES_PER_ITERATION = 4
NUM_WARPS = 8
NUM_STAGES = 3


if triton is not None:

    @triton.jit
    def _fused_w2_combine_kernel(
        activation_ptr,
        expert_index_ptr,
        combine_weight_ptr,
        packed_ptr,
        scale_ptr,
        output_ptr,
        ROUTES: tl.constexpr,
        N: tl.constexpr,
        K: tl.constexpr,
        stride_at,
        stride_ar,
        stride_ak,
        stride_it,
        stride_ir,
        stride_ct,
        stride_cr,
        stride_we,
        stride_wn,
        stride_wk,
        stride_se,
        stride_sn,
        stride_sk,
        stride_ot,
        stride_on,
        GROUP: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        K_TILES: tl.constexpr,
    ):
        token = tl.program_id(0)
        block_n = tl.program_id(1)
        offsets_n = block_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offsets_byte = tl.arange(0, BLOCK_K // 2)
        n_mask = offsets_n < N
        accumulator = tl.zeros((BLOCK_N,), dtype=tl.float32)

        # The route order is fixed and the shared expert is always last. Keep a
        # single FP32 output tile live across all routes, then round once when it
        # is stored. The last combine weight is deliberately not loaded.
        for route in tl.static_range(0, ROUTES):
            expert = tl.load(
                expert_index_ptr + token * stride_it + route * stride_ir
            ).to(tl.int64)
            if route == ROUTES - 1:
                route_weight = 1.0
            else:
                route_weight = tl.load(
                    combine_weight_ptr + token * stride_ct + route * stride_cr
                ).to(tl.float32)

            for k_group_start in range(0, K, BLOCK_K * K_TILES):
                for tile in tl.static_range(0, K_TILES):
                    current_even_k = (
                        k_group_start + tile * BLOCK_K + offsets_byte * 2
                    )
                    current_odd_k = current_even_k + 1
                    k_mask = current_even_k < K
                    weight_mask = n_mask[:, None] & k_mask[None, :]

                    activation_even = tl.load(
                        activation_ptr
                        + token * stride_at
                        + route * stride_ar
                        + current_even_k * stride_ak,
                        mask=k_mask,
                        other=0.0,
                    )
                    activation_odd = tl.load(
                        activation_ptr
                        + token * stride_at
                        + route * stride_ar
                        + current_odd_k * stride_ak,
                        mask=current_odd_k < K,
                        other=0.0,
                    )
                    packed_byte = tl.load(
                        packed_ptr
                        + expert * stride_we
                        + offsets_n[:, None] * stride_wn
                        + (current_even_k[None, :] // 2) * stride_wk,
                        mask=weight_mask,
                        other=0,
                    )
                    low_nibble = packed_byte & 0xF
                    high_nibble = packed_byte >> 4
                    low_signed = low_nibble.to(tl.int32)
                    low_signed = tl.where(
                        low_signed >= 8, low_signed - 16, low_signed
                    )
                    high_signed = high_nibble.to(tl.int32)
                    high_signed = tl.where(
                        high_signed >= 8, high_signed - 16, high_signed
                    )
                    scale = tl.load(
                        scale_ptr
                        + expert * stride_se
                        + offsets_n[:, None] * stride_sn
                        + (current_even_k[None, :] // GROUP) * stride_sk,
                        mask=weight_mask,
                        other=1.0,
                    )
                    decoded_low = (
                        low_signed.to(tl.float32) * scale.to(tl.float32)
                    ).to(tl.bfloat16)
                    decoded_high = (
                        high_signed.to(tl.float32) * scale.to(tl.float32)
                    ).to(tl.bfloat16)
                    products = (
                        activation_even[None, :].to(tl.float32)
                        * decoded_low.to(tl.float32)
                        + activation_odd[None, :].to(tl.float32)
                        * decoded_high.to(tl.float32)
                    )
                    accumulator += route_weight * tl.sum(products, axis=1)

        tl.store(
            output_ptr + token * stride_ot + offsets_n * stride_on,
            accumulator.to(tl.bfloat16),
            mask=n_mask,
        )

else:
    _fused_w2_combine_kernel = None


def _validate_inputs(
    activated: torch.Tensor,
    expert_indices: torch.Tensor,
    combine_weights: torch.Tensor,
    w2_packed: torch.Tensor,
    w2_scales: torch.Tensor,
    *,
    require_cuda: bool,
) -> tuple[int, int, int, int]:
    if activated.ndim != 3 or activated.dtype != torch.bfloat16:
        raise TypeError(
            "activated must be a BF16 [tokens, routes, reduction] tensor"
        )
    if expert_indices.ndim != 2 or expert_indices.dtype != torch.long:
        raise TypeError("expert_indices must be an int64 [tokens, routes] matrix")
    if combine_weights.ndim != 2 or combine_weights.dtype != torch.float32:
        raise TypeError("combine_weights must be a float32 [tokens, routes] matrix")
    if w2_packed.ndim != 3 or w2_scales.ndim != 3:
        raise ValueError("grouped W2 packed weights and scales must be rank three")
    if w2_packed.dtype != torch.uint8:
        raise TypeError("grouped W2 packed weights must use torch.uint8 storage")
    if w2_scales.dtype != torch.bfloat16:
        raise TypeError("grouped W2 scales must use torch.bfloat16 storage")
    if require_cuda and activated.device.type != "cuda":
        raise ValueError("fused W2 combine requires CUDA")
    if not (
        activated.device
        == expert_indices.device
        == combine_weights.device
        == w2_packed.device
        == w2_scales.device
    ):
        raise ValueError("fused W2 combine inputs must share one device")

    tokens, routes, reduction = activated.shape
    experts, output_size, packed_reduction = w2_packed.shape
    if expert_indices.shape != (tokens, routes):
        raise ValueError("expert_indices must match the activated token-route shape")
    if combine_weights.shape != (tokens, routes):
        raise ValueError("combine_weights must match the activated token-route shape")
    if routes == 0 or experts == 0 or output_size == 0 or reduction == 0:
        raise ValueError("fused W2 combine requires nonzero route and weight dimensions")
    if reduction % GROUP_SIZE:
        raise ValueError("the grouped W2 reduction width must be divisible by 32")
    if packed_reduction * 2 != reduction:
        raise ValueError("grouped W2 packed weights have the wrong reduction width")
    expected_scale_shape = (experts, output_size, reduction // GROUP_SIZE)
    if w2_scales.shape != expected_scale_shape:
        raise ValueError("grouped W2 scales have the wrong shape")
    return tokens, routes, output_size, reduction


def fused_w2_combine_reference(
    activated: torch.Tensor,
    expert_indices: torch.Tensor,
    combine_weights: torch.Tensor,
    w2_packed: torch.Tensor,
    w2_scales: torch.Tensor,
) -> torch.Tensor:
    """Pure PyTorch transcription of the current W2 then combine sequence.

    The last route is added as the shared expert with weight exactly 1.0. Its
    value in ``combine_weights`` is ignored so this invariant cannot silently
    become a router weight.
    """
    tokens, routes, output_size, reduction = _validate_inputs(
        activated,
        expert_indices,
        combine_weights,
        w2_packed,
        w2_scales,
        require_cuda=False,
    )
    group_indices = torch.arange(
        reduction, device=activated.device, dtype=torch.long
    ) // GROUP_SIZE
    expert_outputs = []
    for route in range(routes):
        route_experts = expert_indices[:, route]
        packed = w2_packed.index_select(0, route_experts)
        scales = w2_scales.index_select(0, route_experts)
        low = packed & 0xF
        high = packed >> 4
        nibbles = torch.stack((low, high), dim=-1).reshape(
            tokens, output_size, reduction
        )
        signed = nibbles.to(torch.int16)
        signed = torch.where(signed >= 8, signed - 16, signed)
        decoded = (
            signed.float() * scales.float()[..., group_indices]
        ).to(torch.bfloat16)
        route_output = (
            activated[:, route].float().unsqueeze(1) * decoded.float()
        ).sum(dim=-1).to(torch.bfloat16)
        expert_outputs.append(route_output)

    outputs = torch.stack(expert_outputs, dim=1)
    routed = (
        outputs[:, :-1]
        .float()
        .mul(combine_weights[:, :-1].float().unsqueeze(-1))
        .sum(dim=1)
        .to(activated.dtype)
    )
    return routed + outputs[:, -1]


def fused_w2_combine(
    activated: torch.Tensor,
    expert_indices: torch.Tensor,
    combine_weights: torch.Tensor,
    w2_packed: torch.Tensor,
    w2_scales: torch.Tensor,
) -> torch.Tensor:
    """Project every route through W2 and combine it in one Triton launch."""
    if triton is None or _fused_w2_combine_kernel is None:
        raise RuntimeError("Triton is required for fused W2 combine")
    tokens, routes, output_size, reduction = _validate_inputs(
        activated,
        expert_indices,
        combine_weights,
        w2_packed,
        w2_scales,
        require_cuda=True,
    )
    output = torch.empty(
        (tokens, output_size),
        dtype=torch.bfloat16,
        device=activated.device,
    )
    if tokens == 0:
        return output

    grid = (tokens, triton.cdiv(output_size, BLOCK_N))
    with torch.cuda.device(activated.device):
        _fused_w2_combine_kernel[grid](
            activated,
            expert_indices,
            combine_weights,
            w2_packed,
            w2_scales,
            output,
            ROUTES=routes,
            N=output_size,
            K=reduction,
            stride_at=activated.stride(0),
            stride_ar=activated.stride(1),
            stride_ak=activated.stride(2),
            stride_it=expert_indices.stride(0),
            stride_ir=expert_indices.stride(1),
            stride_ct=combine_weights.stride(0),
            stride_cr=combine_weights.stride(1),
            stride_we=w2_packed.stride(0),
            stride_wn=w2_packed.stride(1),
            stride_wk=w2_packed.stride(2),
            stride_se=w2_scales.stride(0),
            stride_sn=w2_scales.stride(1),
            stride_sk=w2_scales.stride(2),
            stride_ot=output.stride(0),
            stride_on=output.stride(1),
            GROUP=GROUP_SIZE,
            BLOCK_N=BLOCK_N,
            BLOCK_K=BLOCK_K,
            K_TILES=K_TILES_PER_ITERATION,
            num_warps=NUM_WARPS,
            num_stages=NUM_STAGES,
        )
    return output
