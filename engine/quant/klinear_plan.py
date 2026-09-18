"""Index-derived W4A16 plan for Kimi-Linear checkpoint tensors.

The safetensors headers are the authority for names, shapes, dtypes, and byte
counts. This module contains policy only. It does not embed a model-wide byte
total and it never pads a reduction dimension that the codec cannot represent.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from math import prod
from typing import Iterable

from .w4a16 import GROUP_SIZE

RESULTS_PROJECTION_BYTES = 24_561_340_864

_DTYPE_BYTES = {
    "BOOL": 1,
    "BF16": 2,
    "F16": 2,
    "F32": 4,
    "F64": 8,
    "I8": 1,
    "I16": 2,
    "I32": 4,
    "I64": 8,
    "U8": 1,
    "U16": 2,
    "U32": 4,
    "U64": 8,
}

_LAYER_PATTERN = re.compile(r"(?:^|\.)(?:layers|h)\.(\d+)\.")
_KDA_MARKERS = (
    ".self_attn.A_log",
    ".self_attn.b_proj.",
    ".self_attn.dt_bias",
    ".self_attn.f_a_proj.",
    ".self_attn.f_b_proj.",
    ".self_attn.g_a_proj.",
    ".self_attn.g_b_proj.",
    ".self_attn.q_conv1d.",
)
_KDA_SENSITIVE_MARKERS = (
    ".self_attn.A_log",
    ".self_attn.b_proj.",
    ".self_attn.dt_bias",
    ".self_attn.f_a_proj.",
    ".self_attn.f_b_proj.",
    ".self_attn.g_a_proj.",
    ".self_attn.g_b_proj.",
    ".self_attn.g_proj.",
    ".self_attn.k_conv1d.",
    ".self_attn.o_norm.",
    ".self_attn.q_conv1d.",
    ".self_attn.v_conv1d.",
)
_DENSE_MLP_PROJECTIONS = (
    ".mlp.down_proj.weight",
    ".mlp.gate_proj.weight",
    ".mlp.up_proj.weight",
    ".mlp.w1.weight",
    ".mlp.w2.weight",
    ".mlp.w3.weight",
)

DEFAULT_PROFILE_NAME = "default"
SHARED_EXPERTS_BF16_PROFILE_NAME = "shared-experts-bf16"

_ROUTED_EXPERT_CLASS = "routed expert projections"
_SHARED_EXPERT_CLASS = "shared expert projections"
_ATTENTION_CLASS = "attention projections"
_DENSE_LAYER_0_MLP_CLASS = "dense layer 0 MLP projections"


@dataclass(frozen=True)
class QuantizationProfile:
    name: str
    description: str
    quantized_tensor_classes: frozenset[str]

    def quantizes(self, tensor_class: str) -> bool:
        return tensor_class in self.quantized_tensor_classes


_DEFAULT_QUANTIZED_CLASSES = frozenset(
    {
        _ROUTED_EXPERT_CLASS,
        _SHARED_EXPERT_CLASS,
        _ATTENTION_CLASS,
        _DENSE_LAYER_0_MLP_CLASS,
    }
)

_PROFILES = {
    DEFAULT_PROFILE_NAME: QuantizationProfile(
        name=DEFAULT_PROFILE_NAME,
        description=(
            "The measured selective W4A16 policy, unchanged from the original "
            "Kimi-Linear quantization artifact."
        ),
        quantized_tensor_classes=_DEFAULT_QUANTIZED_CLASSES,
    ),
    SHARED_EXPERTS_BF16_PROFILE_NAME: QuantizationProfile(
        name=SHARED_EXPERTS_BF16_PROFILE_NAME,
        description=(
            "The selective W4A16 policy with all shared expert projections "
            "retained in BF16 for a controlled accuracy experiment."
        ),
        quantized_tensor_classes=_DEFAULT_QUANTIZED_CLASSES - {_SHARED_EXPERT_CLASS},
    ),
}


def get_klinear_quantization_profile(profile: str) -> QuantizationProfile:
    """Resolve a named profile, refusing unknown names rather than falling back."""
    if not isinstance(profile, str):
        raise TypeError("quantization profile name must be a string")
    try:
        return _PROFILES[profile]
    except KeyError as exc:
        supported = ", ".join(sorted(_PROFILES))
        raise ValueError(
            f"unknown Kimi-Linear quantization profile {profile!r}; "
            f"supported profiles: {supported}"
        ) from exc


@dataclass(frozen=True)
class TensorMetadata:
    name: str
    shape: tuple[int, ...]
    dtype: str
    source_file: str | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("tensor name cannot be empty")
        if not self.shape or any(dimension <= 0 for dimension in self.shape):
            raise ValueError(f"tensor {self.name} must have positive dimensions")
        if self.dtype not in _DTYPE_BYTES:
            raise ValueError(f"tensor {self.name} has unsupported dtype {self.dtype}")

    @property
    def original_bytes(self) -> int:
        return prod(self.shape) * _DTYPE_BYTES[self.dtype]


@dataclass(frozen=True)
class TensorDecision:
    name: str
    shape: tuple[int, ...]
    dtype: str
    source_file: str | None
    tensor_class: str
    quantize: bool
    reason: str
    original_bytes: int
    planned_bytes: int

    @property
    def saved_bytes(self) -> int:
        return self.original_bytes - self.planned_bytes

    @property
    def packed_name(self) -> str | None:
        return f"{self.name}.w4a16_packed" if self.quantize else None

    @property
    def scales_name(self) -> str | None:
        return f"{self.name}.w4a16_scales" if self.quantize else None

    def as_dict(self) -> dict:
        return asdict(self) | {
            "saved_bytes": self.saved_bytes,
            "packed_name": self.packed_name,
            "scales_name": self.scales_name,
        }


@dataclass(frozen=True)
class ClassDecision:
    tensor_class: str
    quantize: bool
    reason: str
    tensor_count: int
    original_bytes: int
    planned_bytes: int
    saved_bytes: int


@dataclass(frozen=True)
class KLinearQuantizationPlan:
    profile: QuantizationProfile
    tensors: tuple[TensorDecision, ...]

    @property
    def original_bytes(self) -> int:
        return sum(tensor.original_bytes for tensor in self.tensors)

    @property
    def planned_bytes(self) -> int:
        return sum(tensor.planned_bytes for tensor in self.tensors)

    @property
    def saved_bytes(self) -> int:
        return self.original_bytes - self.planned_bytes

    @property
    def quantized_tensor_count(self) -> int:
        return sum(tensor.quantize for tensor in self.tensors)

    def by_class(self) -> tuple[ClassDecision, ...]:
        classes: dict[tuple[str, bool, str], list[TensorDecision]] = {}
        for tensor in self.tensors:
            key = (tensor.tensor_class, tensor.quantize, tensor.reason)
            classes.setdefault(key, []).append(tensor)
        reports = []
        for (tensor_class, quantize, reason), tensors in sorted(classes.items()):
            original = sum(tensor.original_bytes for tensor in tensors)
            planned = sum(tensor.planned_bytes for tensor in tensors)
            reports.append(
                ClassDecision(
                    tensor_class=tensor_class,
                    quantize=quantize,
                    reason=reason,
                    tensor_count=len(tensors),
                    original_bytes=original,
                    planned_bytes=planned,
                    saved_bytes=original - planned,
                )
            )
        return tuple(reports)

    def as_dict(self) -> dict:
        return {
            "profile": {
                "name": self.profile.name,
                "description": self.profile.description,
                "quantized_tensor_classes": sorted(
                    self.profile.quantized_tensor_classes
                ),
            },
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
                "quantized_tensor_count": self.quantized_tensor_count,
                "tensor_count": len(self.tensors),
                "results_projection_bytes": RESULTS_PROJECTION_BYTES,
                "planned_minus_results_projection_bytes": (
                    self.planned_bytes - RESULTS_PROJECTION_BYTES
                ),
            },
            "classes": [asdict(report) for report in self.by_class()],
            "tensors": [tensor.as_dict() for tensor in self.tensors],
        }


def w4a16_storage_bytes(shape: tuple[int, ...]) -> int:
    """Return packed plus BF16-scale bytes for a real tensor shape."""
    if len(shape) < 2:
        raise ValueError("W4A16 storage applies only to weight matrices")
    reduction = shape[-1]
    if reduction % GROUP_SIZE:
        raise ValueError(
            f"reduction dimension {reduction} is not divisible by group size {GROUP_SIZE}"
        )
    rows = prod(shape[:-1])
    elements = rows * reduction
    return elements // 2 + rows * (reduction // GROUP_SIZE) * 2


def _layer_index(name: str) -> int | None:
    match = _LAYER_PATTERN.search(name)
    return int(match.group(1)) if match else None


def _is_matrix_weight(metadata: TensorMetadata) -> bool:
    return len(metadata.shape) >= 2 and metadata.name.endswith(".weight")


def _is_embedding(name: str) -> bool:
    return any(
        marker in name
        for marker in (
            ".embed_tokens.weight",
            ".tok_embeddings.weight",
            ".word_embeddings.weight",
        )
    )


def _is_router(name: str) -> bool:
    return any(
        marker in name
        for marker in (
            ".block_sparse_moe.gate.weight",
            ".mlp.gate.weight",
            ".router.weight",
            ".router.gate.weight",
            ".shared_expert_gate.weight",
        )
    )


def _is_norm(name: str) -> bool:
    lowered = name.lower()
    return "layernorm" in lowered or ".norm." in lowered or "_norm." in lowered


def _classify(
    metadata: TensorMetadata,
    *,
    kda_layers: frozenset[int],
    profile: QuantizationProfile,
) -> tuple[str, bool, str]:
    name = metadata.name
    layer = _layer_index(name)

    if name.endswith(".bias") or "e_score_correction_bias" in name:
        return (
            "biases",
            False,
            "Bias tensors are small and remain in source precision.",
        )
    if _is_norm(name):
        return (
            "normalization",
            False,
            "Normalization tensors are small and numerically sensitive.",
        )
    if _is_embedding(name):
        return (
            "token embedding",
            False,
            "Keep the input embedding in BF16 to avoid vocabulary representation loss.",
        )
    if name.endswith("lm_head.weight"):
        return (
            "lm head",
            False,
            (
                "Keep the untied vocabulary head in BF16 because output logits are "
                "quantization sensitive."
            ),
        )
    if _is_router(name):
        return (
            "router gate",
            False,
            "Router error changes discrete expert selection, so router weights remain in BF16.",
        )
    if layer in kda_layers and any(marker in name for marker in _KDA_SENSITIVE_MARKERS):
        return (
            "KDA gates, state, and convolutions",
            False,
            "Keep KDA recurrent controls and short convolutions in source precision.",
        )
    if ".experts." in name or ".routed_expert" in name:
        tensor_class = _ROUTED_EXPERT_CLASS
        return (
            tensor_class,
            profile.quantizes(tensor_class),
            "Routed experts dominate resident bytes and are the primary fit target.",
        )
    if ".shared_experts." in name or ".shared_expert." in name:
        tensor_class = _SHARED_EXPERT_CLASS
        quantize = profile.quantizes(tensor_class)
        return (
            tensor_class,
            quantize,
            (
                "Shared expert matrices are large token-path projections and must be "
                "compressed for fit."
                if quantize
                else (
                    "Retain shared expert projections in BF16 for the "
                    "routing-stability hypothesis test."
                )
            ),
        )
    if layer == 0 and any(name.endswith(suffix) for suffix in _DENSE_MLP_PROJECTIONS):
        tensor_class = _DENSE_LAYER_0_MLP_CLASS
        return (
            tensor_class,
            profile.quantizes(tensor_class),
            "The first dense MLP is a large matrix bank and is part of the fit target.",
        )
    if name.endswith("kv_a_proj_with_mqa.weight"):
        return (
            "MLA latent down-projection",
            False,
            (
                "This projection produces the compressed KV latent that is written to "
                "the cache, so error here persists for the whole sequence instead of "
                "perturbing one token's activation. It is also only 576 by 2304 across "
                "seven layers, roughly 18.6 MB in BF16, so quantizing it does nothing "
                "for fit. Quantize for fit, not for its own sake."
            ),
        )
    if (
        (".self_attn." in name or ".attention." in name)
        and name.endswith("_proj.weight")
    ):
        tensor_class = _ATTENTION_CLASS
        return (
            tensor_class,
            profile.quantizes(tensor_class),
            "Quantize large attention projection matrices while preserving KDA controls.",
        )
    if not _is_matrix_weight(metadata):
        return (
            "vectors and non-matrix state",
            False,
            "Only matrix weights in an approved class use W4A16.",
        )
    return (
        "other checkpoint matrices",
        False,
        "The matrix is outside the approved fit-driven classes and remains unchanged.",
    )


def build_klinear_quantization_plan(
    tensors: Iterable[TensorMetadata],
    *,
    profile: str = DEFAULT_PROFILE_NAME,
) -> KLinearQuantizationPlan:
    """Classify every real checkpoint tensor and compute projected storage."""
    selected_profile = get_klinear_quantization_profile(profile)
    metadata = tuple(sorted(tensors, key=lambda tensor: tensor.name))
    if not metadata:
        raise ValueError("the checkpoint tensor manifest is empty")
    names = [tensor.name for tensor in metadata]
    if len(names) != len(set(names)):
        raise ValueError("the checkpoint tensor manifest contains duplicate names")

    kda_layers = frozenset(
        layer
        for tensor in metadata
        if any(marker in tensor.name for marker in _KDA_MARKERS)
        if (layer := _layer_index(tensor.name)) is not None
    )
    decisions = []
    for tensor in metadata:
        tensor_class, quantize, reason = _classify(
            tensor,
            kda_layers=kda_layers,
            profile=selected_profile,
        )
        if quantize:
            if tensor.dtype != "BF16":
                raise TypeError(
                    f"approved tensor {tensor.name} must be BF16, got {tensor.dtype}"
                )
            if not _is_matrix_weight(tensor):
                raise ValueError(
                    f"approved tensor {tensor.name} is not a matrix weight: {tensor.shape}"
                )
            planned = w4a16_storage_bytes(tensor.shape)
        else:
            planned = tensor.original_bytes
        decisions.append(
            TensorDecision(
                name=tensor.name,
                shape=tensor.shape,
                dtype=tensor.dtype,
                source_file=tensor.source_file,
                tensor_class=tensor_class,
                quantize=quantize,
                reason=reason,
                original_bytes=tensor.original_bytes,
                planned_bytes=planned,
            )
        )
    return KLinearQuantizationPlan(
        profile=selected_profile,
        tensors=tuple(decisions),
    )
