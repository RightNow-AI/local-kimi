"""Auditable K3 dense-tensor W4A16 plan derived from the real manifest."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import prod

from engine.k3ref.manifest import BF16, F32, K3_LAYER_TENSOR_MANIFEST, TensorSpec

from .w4a16 import GROUP_SIZE

LM_HEAD_NAME = "language_model.lm_head.weight"
LM_HEAD_SPEC = TensorSpec((163840, 7168), BF16)

_ATTENTION_MAIN = {
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.g_proj.weight",
    "self_attn.o_proj.weight",
}
_ATTENTION_LATENT = {
    "self_attn.b_proj.weight",
    "self_attn.f_a_proj.weight",
    "self_attn.f_b_proj.weight",
}
_SHARED_EXPERT = {
    "block_sparse_moe.shared_experts.gate_proj.weight",
    "block_sparse_moe.shared_experts.up_proj.weight",
    "block_sparse_moe.shared_experts.down_proj.weight",
}
_ROUTED_DENSE = {
    "block_sparse_moe.routed_expert_up_proj.weight",
    "block_sparse_moe.routed_expert_down_proj.weight",
}
_RESIDUAL_MIXERS = {
    "mlp_res_proj.weight",
    "self_attention_res_proj.weight",
}

_CLASS_ORDER = (
    "attention projections",
    "attention latent projections",
    "shared expert projections",
    "routed dense projections",
    "lm head",
    "router",
    "residual mixers",
    "normalization vectors",
    "F32 state and convolutions",
)

_CLASS_POLICY = {
    "attention projections": (
        True,
        "Large BF16 GEMMs are streamed every decode step and have reduction axes divisible by 32.",
    ),
    "attention latent projections": (
        True,
        "These are BF16 dense GEMMs on the token path; the same group format applies even though they are smaller.",
    ),
    "shared expert projections": (
        True,
        "All tokens execute the shared experts, so their BF16 projection bytes are paid every step.",
    ),
    "routed dense projections": (
        True,
        "These shared routed-path projections are BF16 matrices, not the already-compressed per-expert weights.",
    ),
    "lm head": (
        True,
        "K3 declares an untied 163840 by 7168 output head. Quantize it because it is streamed for every decoded token; the input embedding lookup can remain BF16.",
    ),
    "router": (
        False,
        "Keep BF16 initially because small logit changes can alter discrete top-k expert selection, while its byte share is modest.",
    ),
    "residual mixers": (
        False,
        "The one-row projections are tiny and sit on a numerically sensitive residual path.",
    ),
    "normalization vectors": (
        False,
        "Vectors are tiny and quantizing normalization weights risks numerics without meaningful bandwidth savings.",
    ),
    "F32 state and convolutions": (
        False,
        "A_log, dt_bias, output norm, correction bias, and short-convolution kernels are tiny F32 state that should retain precision.",
    ),
}


@dataclass(frozen=True)
class TensorDecision:
    name: str
    tensor_class: str
    source: str
    shape: tuple[int, ...]
    dtype: str
    quantize: bool
    reason: str
    original_bytes: int
    planned_bytes: int

    @property
    def saved_bytes(self) -> int:
        return self.original_bytes - self.planned_bytes


@dataclass(frozen=True)
class ClassSaving:
    tensor_class: str
    quantize: bool
    reason: str
    tensor_count: int
    original_bytes: int
    planned_bytes: int
    saved_bytes: int
    saving_fraction: float


@dataclass(frozen=True)
class QuantizationPlan:
    tensors: tuple[TensorDecision, ...]

    @property
    def original_bytes(self) -> int:
        return sum(item.original_bytes for item in self.tensors)

    @property
    def planned_bytes(self) -> int:
        return sum(item.planned_bytes for item in self.tensors)

    @property
    def saved_bytes(self) -> int:
        return self.original_bytes - self.planned_bytes

    def by_class(self) -> tuple[ClassSaving, ...]:
        reports = []
        for tensor_class in _CLASS_ORDER:
            items = [item for item in self.tensors if item.tensor_class == tensor_class]
            if not items:
                continue
            original = sum(item.original_bytes for item in items)
            planned = sum(item.planned_bytes for item in items)
            quantize, reason = _CLASS_POLICY[tensor_class]
            reports.append(
                ClassSaving(
                    tensor_class=tensor_class,
                    quantize=quantize,
                    reason=reason,
                    tensor_count=len(items),
                    original_bytes=original,
                    planned_bytes=planned,
                    saved_bytes=original - planned,
                    saving_fraction=(original - planned) / original,
                )
            )
        return tuple(reports)

    def as_dict(self) -> dict:
        return {
            "format": {
                "weight_bits": 4,
                "scale_dtype": "BF16",
                "group_size": GROUP_SIZE,
                "bits_per_parameter_including_scales": 4.5,
            },
            "totals": {
                "original_bytes": self.original_bytes,
                "planned_bytes": self.planned_bytes,
                "saved_bytes": self.saved_bytes,
                "saving_fraction": self.saved_bytes / self.original_bytes,
            },
            "classes": [asdict(item) for item in self.by_class()],
            "tensors": [asdict(item) | {"saved_bytes": item.saved_bytes} for item in self.tensors],
        }


def original_storage_bytes(spec: TensorSpec) -> int:
    bytes_per_element = {BF16: 2, F32: 4}
    try:
        element_size = bytes_per_element[spec.dtype]
    except KeyError as exc:
        raise ValueError(f"unsupported manifest dtype: {spec.dtype}") from exc
    return prod(spec.shape) * element_size


def w4a16_storage_bytes(shape: tuple[int, ...]) -> int:
    if len(shape) != 2:
        raise ValueError("W4A16 storage applies only to dense matrices")
    elements = prod(shape)
    if shape[-1] % GROUP_SIZE:
        raise ValueError("the reduction axis must be divisible by group size 32")
    packed_bytes = elements // 2
    bf16_scale_bytes = (elements // GROUP_SIZE) * 2
    return packed_bytes + bf16_scale_bytes


def _tensor_class(name: str, spec: TensorSpec) -> str:
    if name == LM_HEAD_NAME:
        return "lm head"
    if name in _ATTENTION_MAIN:
        return "attention projections"
    if name in _ATTENTION_LATENT:
        return "attention latent projections"
    if name in _SHARED_EXPERT:
        return "shared expert projections"
    if name in _ROUTED_DENSE:
        return "routed dense projections"
    if name == "block_sparse_moe.gate.weight":
        return "router"
    if name in _RESIDUAL_MIXERS:
        return "residual mixers"
    if spec.dtype == F32:
        return "F32 state and convolutions"
    if spec.dtype == BF16 and len(spec.shape) == 1:
        return "normalization vectors"
    raise KeyError(f"no quantization policy for manifest tensor {name}")


def _decision(name: str, spec: TensorSpec, *, source: str) -> TensorDecision:
    tensor_class = _tensor_class(name, spec)
    should_quantize, reason = _CLASS_POLICY[tensor_class]
    original = original_storage_bytes(spec)
    planned = w4a16_storage_bytes(spec.shape) if should_quantize else original
    return TensorDecision(
        name=name,
        tensor_class=tensor_class,
        source=source,
        shape=spec.shape,
        dtype=spec.dtype,
        quantize=should_quantize,
        reason=reason,
        original_bytes=original,
        planned_bytes=planned,
    )


def build_quantization_plan(*, include_lm_head: bool = True) -> QuantizationPlan:
    """Build the layer-12 manifest plan, plus the explicit model-level LM head."""
    decisions = [
        _decision(name, spec, source="layer")
        for name, spec in K3_LAYER_TENSOR_MANIFEST.items()
    ]
    if include_lm_head:
        decisions.append(_decision(LM_HEAD_NAME, LM_HEAD_SPEC, source="model"))
    return QuantizationPlan(tuple(decisions))
