"""Linear layers backed by packed W4A16 checkpoint tensors."""

from __future__ import annotations

from typing import Protocol

import torch
import torch.nn.functional as F
from torch import nn

from engine.quant.triton_w4a16 import w4a16_linear
from engine.quant.w4a16 import GROUP_SIZE, W4A16Tensor


class LinearFactory(Protocol):
    def __call__(
        self,
        tensor_name: str,
        in_features: int,
        out_features: int,
        *,
        device: torch.device | str | None,
        dtype: torch.dtype | None,
    ) -> nn.Module: ...


def make_linear(
    tensor_name: str,
    in_features: int,
    out_features: int,
    *,
    linear_factory: LinearFactory | None,
    device: torch.device | str | None,
    dtype: torch.dtype | None,
) -> nn.Module:
    """Construct a normal linear unless an index-derived factory replaces it."""
    if linear_factory is None:
        return nn.Linear(
            in_features,
            out_features,
            bias=False,
            device=device,
            dtype=dtype,
        )
    return linear_factory(
        tensor_name,
        in_features,
        out_features,
        device=device,
        dtype=dtype,
    )


class W4A16Linear(nn.Module):
    """A weight-only INT4 linear that never materializes a BF16 matrix."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        if in_features <= 0 or out_features <= 0:
            raise ValueError("linear dimensions must be positive")
        if in_features % GROUP_SIZE:
            raise ValueError(
                f"W4A16 in_features must be divisible by group size {GROUP_SIZE}"
            )
        self.in_features = in_features
        self.out_features = out_features
        self.register_buffer(
            "packed_weight",
            torch.empty(
                out_features,
                in_features // 2,
                dtype=torch.uint8,
                device=device,
            ),
        )
        self.register_buffer(
            "scales",
            torch.empty(
                out_features,
                in_features // GROUP_SIZE,
                dtype=torch.bfloat16,
                device=device,
            ),
        )
        self.register_parameter("weight", None)
        self._retained_bf16 = False

    @classmethod
    def from_encoded(cls, encoded: W4A16Tensor) -> "W4A16Linear":
        out_features, in_features = encoded.original_shape
        module = cls(in_features, out_features, device="meta")
        module.load_encoded(encoded)
        return module

    @classmethod
    def from_bf16_retained(cls, weight: torch.Tensor) -> "W4A16Linear":
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
        module.register_buffer("packed_weight", None)
        module.register_buffer("scales", None)
        module.register_parameter(
            "weight", nn.Parameter(weight, requires_grad=False)
        )
        module._retained_bf16 = True
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
    def encoded(self) -> W4A16Tensor:
        if self._retained_bf16:
            raise RuntimeError("a retained BF16 linear has no W4A16 encoding")
        if self.packed_weight.device.type == "meta" or self.scales.device.type == "meta":
            raise RuntimeError("W4A16 weight payload has not been loaded")
        return W4A16Tensor(
            packed=self.packed_weight,
            scales=self.scales,
            original_shape=(self.out_features, self.in_features),
        )

    def load_encoded(self, encoded: W4A16Tensor) -> None:
        if self._retained_bf16:
            raise RuntimeError("cannot load W4A16 payload into a retained BF16 linear")
        expected = (self.out_features, self.in_features)
        if encoded.original_shape != expected:
            raise ValueError(
                f"W4A16 shape mismatch: expected {expected}, got {encoded.original_shape}"
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
        output = w4a16_linear(flattened, self.encoded)
        return output.reshape(*original_shape, self.out_features)

    def extra_repr(self) -> str:
        storage = "W4A16" if self.is_quantized else "BF16 retained"
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"storage={storage}"
        )
