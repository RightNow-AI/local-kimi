"""One-launch batch-1 W4A16 QKV GEMV with a prefill-safe fallback."""

from __future__ import annotations

from typing import NamedTuple

import torch

from engine.kernels.w4a16_dense_gemv import w4a16_dense_gemv
from engine.quant.w4a16 import GROUP_SIZE

try:
    import triton
    import triton.language as tl
except ModuleNotFoundError:  # CPU-only environments do not install Triton.
    triton = None
    tl = None


class FusedQkvConfig(NamedTuple):
    """One explicit launch candidate for the batch-1 fused QKV GEMV."""

    name: str
    block_n: int
    block_k: int
    num_warps: int
    num_stages: int


# The first entry is the 64-wide starting point required by this lane's brief.
# The current dense selector in this checkout actually picks n32_k64_s2 at
# N=4096, so BENCH-QKV.py reports that source truth separately. The other
# entries isolate BLOCK_N for the required end-to-end sweep. No candidate uses
# split-K because this path must produce all three projections in one launch.
FUSED_QKV_CONFIGS = (
    FusedQkvConfig("n64_k64_w4_st3", 64, 64, 4, 3),
    FusedQkvConfig("n32_k64_w4_st3", 32, 64, 4, 3),
    FusedQkvConfig("n16_k64_w4_st3", 16, 64, 4, 3),
    FusedQkvConfig("n128_k64_w8_st3", 128, 64, 8, 3),
)


def _fused_qkv_grid_shape(output_size: int, block_n: int) -> tuple[int, int]:
    """Return the pure-Python `(projections, N tiles)` launch shape."""
    if output_size <= 0:
        raise ValueError("output_size must be positive")
    if block_n <= 0:
        raise ValueError("block_n must be positive")
    return 3, (output_size + block_n - 1) // block_n


if triton is not None:

    @triton.jit
    def _w4a16_fused_qkv_kernel(
        activation_ptr,
        q_packed_ptr,
        q_scale_ptr,
        k_packed_ptr,
        k_scale_ptr,
        v_packed_ptr,
        v_scale_ptr,
        q_result_ptr,
        k_result_ptr,
        v_result_ptr,
        N: tl.constexpr,
        K: tl.constexpr,
        stride_ak,
        stride_qpn,
        stride_qpk,
        stride_qsn,
        stride_qsk,
        stride_kpn,
        stride_kpk,
        stride_ksn,
        stride_ksk,
        stride_vpn,
        stride_vpk,
        stride_vsn,
        stride_vsk,
        stride_on,
        GROUP: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        projection = tl.program_id(0)
        block_n = tl.program_id(1)

        # The branch is uniform across the whole program. It selects one weight,
        # scale, and output pointer before the reduction loop. Every program then
        # has the same accumulator shape and lifetime as the dense GEMV.
        packed_ptr = q_packed_ptr
        scale_ptr = q_scale_ptr
        result_ptr = q_result_ptr
        stride_pn = stride_qpn
        stride_pk = stride_qpk
        stride_sn = stride_qsn
        stride_sk = stride_qsk
        if projection == 1:
            packed_ptr = k_packed_ptr
            scale_ptr = k_scale_ptr
            result_ptr = k_result_ptr
            stride_pn = stride_kpn
            stride_pk = stride_kpk
            stride_sn = stride_ksn
            stride_sk = stride_ksk
        elif projection == 2:
            packed_ptr = v_packed_ptr
            scale_ptr = v_scale_ptr
            result_ptr = v_result_ptr
            stride_pn = stride_vpn
            stride_pk = stride_vpk
            stride_sn = stride_vsn
            stride_sk = stride_vsk

        offsets_n = block_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offsets_byte = tl.arange(0, BLOCK_K // 2)
        accumulator = tl.zeros((BLOCK_N,), dtype=tl.float32)

        # K is lane-contiguous for packed weights shaped [N, K / 2]. Each byte
        # supplies the even low nibble and odd high nibble for one output row.
        for k_group_start in range(0, K, BLOCK_K):
            current_even_k = k_group_start + offsets_byte * 2
            current_odd_k = current_even_k + 1
            activation_even = tl.load(
                activation_ptr + current_even_k * stride_ak,
                mask=current_even_k < K,
                other=0.0,
            )
            activation_odd = tl.load(
                activation_ptr + current_odd_k * stride_ak,
                mask=current_odd_k < K,
                other=0.0,
            )
            packed_byte = tl.load(
                packed_ptr
                + offsets_n[:, None] * stride_pn
                + (current_even_k[None, :] // 2) * stride_pk,
                mask=(offsets_n[:, None] < N)
                & (current_even_k[None, :] < K),
                other=0,
            )
            low_nibble = packed_byte & 0xF
            high_nibble = packed_byte >> 4
            low_signed = low_nibble.to(tl.int32)
            low_signed = tl.where(low_signed >= 8, low_signed - 16, low_signed)
            high_signed = high_nibble.to(tl.int32)
            high_signed = tl.where(
                high_signed >= 8, high_signed - 16, high_signed
            )
            scale = tl.load(
                scale_ptr
                + offsets_n[:, None] * stride_sn
                + (current_even_k[None, :] // GROUP) * stride_sk,
                mask=(offsets_n[:, None] < N)
                & (current_even_k[None, :] < K),
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
            accumulator += tl.sum(products, axis=1)

        tl.store(
            result_ptr + offsets_n * stride_on,
            accumulator,
            mask=offsets_n < N,
        )


else:
    _w4a16_fused_qkv_kernel = None


def _validate_projection(
    name: str,
    packed_weights: torch.Tensor,
    scales: torch.Tensor,
    *,
    device: torch.device,
    output_size: int,
    reduction: int,
) -> None:
    if packed_weights.ndim != 2 or packed_weights.dtype != torch.uint8:
        raise TypeError(
            f"{name} packed weights must be a uint8 [output, reduction / 2] matrix"
        )
    if scales.ndim != 2 or scales.dtype != torch.bfloat16:
        raise TypeError(
            f"{name} scales must be a BF16 [output, reduction / 32] matrix"
        )
    if packed_weights.device != device or scales.device != device:
        raise ValueError("all fused QKV inputs must share a device")
    if packed_weights.shape != (output_size, reduction // 2):
        raise ValueError(f"{name} packed weights have the wrong shape")
    if scales.shape != (output_size, reduction // GROUP_SIZE):
        raise ValueError(f"{name} scales have the wrong shape")


def _validate_inputs(
    activations: torch.Tensor,
    q_packed: torch.Tensor,
    q_scales: torch.Tensor,
    k_packed: torch.Tensor,
    k_scales: torch.Tensor,
    v_packed: torch.Tensor,
    v_scales: torch.Tensor,
) -> tuple[int, int, int]:
    if activations.ndim != 2 or activations.dtype != torch.bfloat16:
        raise TypeError("activations must be a BF16 [tokens, reduction] matrix")
    if activations.device.type != "cuda":
        raise ValueError("fused W4A16 QKV requires CUDA")

    m, reduction = activations.shape
    if q_packed.ndim != 2:
        raise TypeError(
            "q packed weights must be a uint8 [output, reduction / 2] matrix"
        )
    output_size = q_packed.shape[0]
    if output_size == 0 or reduction == 0:
        raise ValueError("fused W4A16 QKV requires nonzero dimensions")
    if reduction % GROUP_SIZE:
        raise ValueError("the reduction dimension must be divisible by group size 32")

    projections = (
        ("q", q_packed, q_scales),
        ("k", k_packed, k_scales),
        ("v", v_packed, v_scales),
    )
    for name, packed_weights, scales in projections:
        _validate_projection(
            name,
            packed_weights,
            scales,
            device=activations.device,
            output_size=output_size,
            reduction=reduction,
        )
    return m, output_size, reduction


def _select_fused_qkv_config(output_size: int) -> FusedQkvConfig:
    """Keep the lane-specified 64-wide default until an end-to-end sweep wins."""
    del output_size
    return FUSED_QKV_CONFIGS[0]


def fused_qkv_w4a16_reference(
    activations: torch.Tensor,
    q_packed: torch.Tensor,
    q_scales: torch.Tensor,
    k_packed: torch.Tensor,
    k_scales: torch.Tensor,
    v_packed: torch.Tensor,
    v_scales: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Call the shipped dense W4A16 path once for each QKV projection."""
    return (
        w4a16_dense_gemv(activations, q_packed, q_scales),
        w4a16_dense_gemv(activations, k_packed, k_scales),
        w4a16_dense_gemv(activations, v_packed, v_scales),
    )


def _launch_fused_qkv_w4a16(
    activations: torch.Tensor,
    q_packed: torch.Tensor,
    q_scales: torch.Tensor,
    k_packed: torch.Tensor,
    k_scales: torch.Tensor,
    v_packed: torch.Tensor,
    v_scales: torch.Tensor,
    config: FusedQkvConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Launch one explicit batch-1 candidate for the benchmark harness."""
    if triton is None or _w4a16_fused_qkv_kernel is None:
        raise RuntimeError("Triton is required for fused W4A16 QKV")
    m, output_size, reduction = _validate_inputs(
        activations,
        q_packed,
        q_scales,
        k_packed,
        k_scales,
        v_packed,
        v_scales,
    )
    if m != 1:
        raise ValueError("the explicit fused QKV launcher requires exactly one token")

    q_output = torch.empty(
        (1, output_size), dtype=torch.bfloat16, device=activations.device
    )
    k_output = torch.empty_like(q_output)
    v_output = torch.empty_like(q_output)
    grid = _fused_qkv_grid_shape(output_size, config.block_n)
    with torch.cuda.device(activations.device):
        _w4a16_fused_qkv_kernel[grid](
            activations,
            q_packed,
            q_scales,
            k_packed,
            k_scales,
            v_packed,
            v_scales,
            q_output,
            k_output,
            v_output,
            N=output_size,
            K=reduction,
            stride_ak=activations.stride(1),
            stride_qpn=q_packed.stride(0),
            stride_qpk=q_packed.stride(1),
            stride_qsn=q_scales.stride(0),
            stride_qsk=q_scales.stride(1),
            stride_kpn=k_packed.stride(0),
            stride_kpk=k_packed.stride(1),
            stride_ksn=k_scales.stride(0),
            stride_ksk=k_scales.stride(1),
            stride_vpn=v_packed.stride(0),
            stride_vpk=v_packed.stride(1),
            stride_vsn=v_scales.stride(0),
            stride_vsk=v_scales.stride(1),
            stride_on=q_output.stride(1),
            GROUP=GROUP_SIZE,
            BLOCK_N=config.block_n,
            BLOCK_K=config.block_k,
            num_warps=config.num_warps,
            num_stages=config.num_stages,
        )
    return q_output, k_output, v_output


def fused_qkv_w4a16(
    activations: torch.Tensor,
    q_packed: torch.Tensor,
    q_scales: torch.Tensor,
    k_packed: torch.Tensor,
    k_scales: torch.Tensor,
    v_packed: torch.Tensor,
    v_scales: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fuse Q, K, and V for one token and preserve the dense fallback otherwise."""
    if triton is None:
        raise RuntimeError("Triton is required for fused W4A16 QKV")
    m, output_size, _ = _validate_inputs(
        activations,
        q_packed,
        q_scales,
        k_packed,
        k_scales,
        v_packed,
        v_scales,
    )
    if m == 0:
        q_output = torch.empty(
            (0, output_size), dtype=torch.bfloat16, device=activations.device
        )
        return q_output, torch.empty_like(q_output), torch.empty_like(q_output)
    if m > 1:
        return fused_qkv_w4a16_reference(
            activations,
            q_packed,
            q_scales,
            k_packed,
            k_scales,
            v_packed,
            v_scales,
        )

    config = _select_fused_qkv_config(output_size)
    return _launch_fused_qkv_w4a16(
        activations,
        q_packed,
        q_scales,
        k_packed,
        k_scales,
        v_packed,
        v_scales,
        config,
    )
