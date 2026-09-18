"""Linear layers and grouped banks backed by packed W3A16 tensors."""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import nn

from engine.quant.w3a16 import GROUP_SIZE, W3A16Tensor


class W3A16Linear(nn.Module):
    """A weight-only INT3 linear that never materializes a BF16 matrix."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        group_size: int = GROUP_SIZE,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        if in_features <= 0 or out_features <= 0:
            raise ValueError("linear dimensions must be positive")
        if not isinstance(group_size, int) or group_size <= 0:
            raise ValueError("group_size must be a positive integer")
        if group_size % 8:
            raise ValueError("W3A16 group_size must be divisible by 8")
        if in_features % 8:
            raise ValueError("W3A16 in_features must be divisible by 8")
        if in_features % group_size:
            raise ValueError(
                f"W3A16 in_features must be divisible by group_size={group_size}"
            )
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size
        self.register_buffer(
            "packed_weight",
            torch.empty(
                out_features,
                in_features // 8 * 3,
                dtype=torch.uint8,
                device=device,
            ),
        )
        self.register_buffer(
            "scales",
            torch.empty(
                out_features,
                in_features // group_size,
                dtype=torch.bfloat16,
                device=device,
            ),
        )
        self.register_parameter("weight", None)
        self._retained_bf16 = False
        self._dense_kernel: Callable[..., torch.Tensor] | None = None

    @classmethod
    def from_encoded(cls, encoded: W3A16Tensor) -> "W3A16Linear":
        out_features, in_features = encoded.original_shape
        module = cls(
            in_features,
            out_features,
            group_size=encoded.group_size,
            device="meta",
        )
        module.load_encoded(encoded)
        return module

    @classmethod
    def from_bf16_retained(cls, weight: torch.Tensor) -> "W3A16Linear":
        """Create the explicit BF16 path reserved for plan-retained tensors."""
        if weight.ndim != 2:
            raise ValueError("retained linear weight must be two-dimensional")
        if weight.dtype != torch.bfloat16:
            raise TypeError("retained linear weight must use torch.bfloat16")
        out_features, in_features = weight.shape
        module = cls.__new__(cls)
        nn.Module.__init__(module)
        module.in_features = in_features
        module.out_features = out_features
        module.group_size = GROUP_SIZE
        module.register_buffer("packed_weight", None)
        module.register_buffer("scales", None)
        module.register_parameter(
            "weight", nn.Parameter(weight, requires_grad=False)
        )
        module._retained_bf16 = True
        module._dense_kernel = None
        return module

    @property
    def is_quantized(self) -> bool:
        return not self._retained_bf16

    @property
    def resident_bytes(self) -> int:
        """Physical bytes held by this module's weight tensors."""
        tensors = (
            (self.weight,) if self._retained_bf16 else (self.packed_weight, self.scales)
        )
        return sum(
            tensor.numel() * tensor.element_size()
            for tensor in tensors
            if tensor is not None and tensor.device.type != "meta"
        )

    @property
    def encoded(self) -> W3A16Tensor:
        if self._retained_bf16:
            raise RuntimeError("a retained BF16 linear has no W3A16 encoding")
        if (
            self.packed_weight.device.type == "meta"
            or self.scales.device.type == "meta"
        ):
            raise RuntimeError("W3A16 weight payload has not been loaded")
        return W3A16Tensor(
            packed=self.packed_weight,
            scales=self.scales,
            original_shape=(self.out_features, self.in_features),
            group_size=self.group_size,
        )

    def load_encoded(self, encoded: W3A16Tensor) -> None:
        if self._retained_bf16:
            raise RuntimeError("cannot load W3A16 payload into a retained BF16 linear")
        expected = (self.out_features, self.in_features)
        if encoded.original_shape != expected:
            raise ValueError(
                f"W3A16 shape mismatch: expected {expected}, "
                f"got {encoded.original_shape}"
            )
        if encoded.group_size != self.group_size:
            raise ValueError(
                f"W3A16 group size mismatch: expected {self.group_size}, "
                f"got {encoded.group_size}"
            )
        self._buffers["packed_weight"] = encoded.packed
        self._buffers["scales"] = encoded.scales

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if hidden_states.shape[-1] != self.in_features:
            raise ValueError(
                f"linear input width {hidden_states.shape[-1]} does not match "
                f"{self.in_features}"
            )
        if self._retained_bf16:
            return F.linear(hidden_states, self.weight)
        original_shape = hidden_states.shape[:-1]
        flattened = hidden_states.reshape(-1, self.in_features)
        kernel = self._dense_kernel
        if kernel is None:
            # The existing W3A16 kernel is not a registered operation. Import
            # and cache it once so module lookup stays outside the decode loop.
            from engine.kernels.w3a16_gemv import w3a16_dense_gemv

            kernel = w3a16_dense_gemv
            self._dense_kernel = kernel
        output = kernel(
            flattened,
            self.packed_weight,
            self.scales,
            group_size=self.group_size,
        )
        return output.reshape(*original_shape, self.out_features)

    def extra_repr(self) -> str:
        storage = "W3A16" if self.is_quantized else "BF16 retained"
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"group_size={self.group_size}, storage={storage}"
        )


def _stack_and_release(
    modules: list[W3A16Linear],
    buffer_name: str,
) -> torch.Tensor:
    payloads: list[torch.Tensor] = []
    for module in modules:
        payload = getattr(module, buffer_name)
        if payload is None:
            raise ValueError("cannot group an already released W3A16 payload")
        payloads.append(payload)
    grouped = torch.stack(payloads, dim=0).contiguous()
    for module in modules:
        module._buffers[buffer_name] = None
    return grouped


def _set_grouped_buffer(module: nn.Module, name: str, value: torch.Tensor) -> None:
    if name in module._buffers:
        module._buffers[name] = value
    else:
        module.register_buffer(name, value, persistent=False)


def prepare_grouped_w3a16(module: nn.Module) -> None:
    """Transfer resident W3A16 experts into contiguous per-layer banks."""
    if getattr(module, "grouped_w1_packed", None) is not None:
        return
    expert_provider = getattr(module, "expert_provider", None)
    if expert_provider is None:
        return
    shared = getattr(module, "shared_experts", None)
    if shared is None:
        raise ValueError("grouped W3A16 decode requires the shared expert")
    shared_linears = (
        shared.gate_proj,
        shared.down_proj,
        shared.up_proj,
    )
    if not all(isinstance(linear, W3A16Linear) for linear in shared_linears):
        return

    gate = getattr(module, "gate")
    device = gate.weight.device
    dtype = gate.weight.dtype
    layer_idx = getattr(module, "layer_idx")
    num_experts = getattr(module, "num_experts")
    routed = [
        expert_provider(layer_idx, expert_id, device, dtype)
        for expert_id in range(num_experts)
    ]
    if not all(
        isinstance(linear, W3A16Linear)
        for weights in routed
        for linear in weights
    ):
        return

    w1_modules = [weights[0] for weights in routed] + [shared.gate_proj]
    w2_modules = [weights[1] for weights in routed] + [shared.down_proj]
    w3_modules = [weights[2] for weights in routed] + [shared.up_proj]
    group_sizes = {
        linear.group_size
        for linears in (w1_modules, w2_modules, w3_modules)
        for linear in linears
    }
    if len(group_sizes) != 1:
        raise ValueError("grouped W3A16 banks require one shared group size")

    _set_grouped_buffer(
        module,
        "grouped_w1_packed",
        _stack_and_release(w1_modules, "packed_weight"),
    )
    _set_grouped_buffer(
        module,
        "grouped_w1_scales",
        _stack_and_release(w1_modules, "scales"),
    )
    _set_grouped_buffer(
        module,
        "grouped_w2_packed",
        _stack_and_release(w2_modules, "packed_weight"),
    )
    _set_grouped_buffer(
        module,
        "grouped_w2_scales",
        _stack_and_release(w2_modules, "scales"),
    )
    _set_grouped_buffer(
        module,
        "grouped_w3_packed",
        _stack_and_release(w3_modules, "packed_weight"),
    )
    _set_grouped_buffer(
        module,
        "grouped_w3_scales",
        _stack_and_release(w3_modules, "scales"),
    )
    setattr(module, "grouped_w3a16_group_size", group_sizes.pop())

    provider_weights = getattr(expert_provider, "_weights", None)
    if not isinstance(provider_weights, dict):
        raise TypeError("resident W3A16 provider does not expose releasable weights")
    for expert_id in range(num_experts):
        provider_weights.pop((layer_idx, expert_id))
