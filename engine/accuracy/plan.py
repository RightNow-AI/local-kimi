"""Kimi Linear tensor selection with fail-closed canonical reconciliation."""

from __future__ import annotations

import hashlib
import importlib
import inspect
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Mapping, Sequence

from engine.quant.w4a16 import GROUP_SIZE

CANONICAL_PLAN_MODULE = "engine.quant.klinear_plan"
LOCAL_POLICY_VERSION = "kimi-linear-local-fallback-v1"


@dataclass(frozen=True)
class TensorSpec:
    name: str
    shard: str
    shape: tuple[int, ...]
    dtype: str


@dataclass(frozen=True)
class TensorDecision:
    name: str
    tensor_class: str
    shape: tuple[int, ...]
    dtype: str
    quantize: bool
    reason: str


@dataclass(frozen=True)
class PlanResolution:
    source: str
    source_version: str | None
    source_file_sha256: str | None
    reconciliation_status: str
    publication_allowed: bool
    canonical_matches_local_fallback: bool | None
    decisions: tuple[TensorDecision, ...]

    @property
    def selected_names(self) -> tuple[str, ...]:
        return tuple(item.name for item in self.decisions if item.quantize)

    @property
    def digest(self) -> str:
        payload = json.dumps(
            [asdict(item) for item in self.decisions],
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def as_dict(self, *, include_decisions: bool = True) -> dict[str, Any]:
        output = {
            "source": self.source,
            "source_version": self.source_version,
            "source_file_sha256": self.source_file_sha256,
            "reconciliation_status": self.reconciliation_status,
            "publication_allowed": self.publication_allowed,
            "canonical_matches_local_fallback": self.canonical_matches_local_fallback,
            "digest": self.digest,
            "selected_tensor_count": len(self.selected_names),
            "total_tensor_count": len(self.decisions),
        }
        if include_decisions:
            output["decisions"] = [asdict(item) for item in self.decisions]
        return output


def is_router_tensor(name: str) -> bool:
    """Identify gate weights and correction bias used for expert selection."""
    return bool(
        re.search(r"(?:^|\.)block_sparse_moe\.gate(?:\.|$)", name)
        or re.search(r"(?:^|\.)mlp\.gate(?:\.|$)", name)
        or name.endswith(".e_score_correction_bias")
    )


def _tensor_class(name: str, shape: tuple[int, ...]) -> str:
    if is_router_tensor(name):
        return "router"
    if "embed_tokens" in name or "word_embeddings" in name:
        return "embedding"
    if name.endswith("lm_head.weight"):
        return "lm_head"
    if ".experts." in name:
        return "routed_expert"
    if ".shared_experts." in name or ".shared_expert." in name:
        return "shared_expert"
    if ".self_attn." in name:
        return "attention_projection" if len(shape) == 2 else "attention_state"
    if ".mlp." in name or name.endswith((".w1.weight", ".w2.weight", ".w3.weight")):
        return "dense_mlp"
    if "res_proj" in name:
        return "residual_projection"
    if len(shape) == 1:
        return "vector_or_norm"
    if len(shape) == 2:
        return "other_matrix"
    return "other_tensor"


def _local_decision(spec: TensorSpec) -> TensorDecision:
    tensor_class = _tensor_class(spec.name, spec.shape)
    if is_router_tensor(spec.name):
        return TensorDecision(
            spec.name,
            tensor_class,
            spec.shape,
            spec.dtype,
            False,
            "Router tensors stay at source precision so expert selection is not directly quantized.",
        )
    if spec.dtype != "BF16":
        return TensorDecision(
            spec.name,
            tensor_class,
            spec.shape,
            spec.dtype,
            False,
            "The existing W4A16 codec accepts BF16 source weights only.",
        )
    if len(spec.shape) != 2:
        return TensorDecision(
            spec.name,
            tensor_class,
            spec.shape,
            spec.dtype,
            False,
            "The existing W4A16 codec applies only to two-dimensional dense matrices.",
        )
    if spec.shape[-1] % GROUP_SIZE:
        return TensorDecision(
            spec.name,
            tensor_class,
            spec.shape,
            spec.dtype,
            False,
            f"The reduction axis is not divisible by the codec group size {GROUP_SIZE}.",
        )
    if tensor_class == "embedding":
        return TensorDecision(
            spec.name,
            tensor_class,
            spec.shape,
            spec.dtype,
            False,
            "The local fallback keeps token embeddings at BF16 until the canonical plan decides otherwise.",
        )
    return TensorDecision(
        spec.name,
        tensor_class,
        spec.shape,
        spec.dtype,
        True,
        "Eligible BF16 dense matrix selected by the local fallback policy.",
    )


def _file_sha256(path: str | None) -> str | None:
    if not path:
        return None
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _invoke_with_context(function: Callable[..., Any], context: Mapping[str, Any]) -> Any:
    signature = inspect.signature(function)
    kwargs: dict[str, Any] = {}
    for parameter in signature.parameters.values():
        if parameter.kind in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD):
            continue
        if parameter.name in context:
            kwargs[parameter.name] = context[parameter.name]
            continue
        if parameter.default is not inspect.Parameter.empty:
            continue
        raise TypeError(
            f"unsupported required parameter {parameter.name!r} on "
            f"{function.__module__}.{function.__name__}"
        )
    return function(**kwargs)


def _decision_from_value(spec: TensorSpec, value: Any, *, default_reason: str) -> TensorDecision:
    if isinstance(value, bool):
        quantize = value
        reason = default_reason
        tensor_class = _tensor_class(spec.name, spec.shape)
    elif isinstance(value, Mapping):
        quantize = bool(value.get("quantize", value.get("selected", False)))
        reason = str(value.get("reason", default_reason))
        tensor_class = str(
            value.get("tensor_class", value.get("category", _tensor_class(spec.name, spec.shape)))
        )
    else:
        quantize = bool(getattr(value, "quantize", getattr(value, "selected", False)))
        reason = str(getattr(value, "reason", default_reason))
        tensor_class = str(
            getattr(value, "tensor_class", getattr(value, "category", _tensor_class(spec.name, spec.shape)))
        )
    return TensorDecision(
        name=spec.name,
        tensor_class=tensor_class,
        shape=spec.shape,
        dtype=spec.dtype,
        quantize=quantize,
        reason=reason,
    )


def _value_quantizes(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, Mapping):
        return bool(value.get("quantize", value.get("selected", False)))
    return bool(getattr(value, "quantize", getattr(value, "selected", False)))


def _mapping_from_plan(plan: Any) -> Mapping[str, Any] | None:
    if hasattr(plan, "tensors"):
        plan = plan.tensors
    if isinstance(plan, Mapping):
        return plan
    if isinstance(plan, Sequence) and not isinstance(plan, (str, bytes)):
        if all(isinstance(item, str) for item in plan):
            return {str(item): True for item in plan}
        mapping: dict[str, Any] = {}
        for item in plan:
            name = getattr(item, "name", getattr(item, "tensor_name", None))
            if name is None:
                return None
            mapping[str(name)] = item
        return mapping
    return None


def _canonical_context(
    module: ModuleType,
    specs: tuple[TensorSpec, ...],
    *,
    source_dir: Path,
    config: Mapping[str, Any],
    index: Mapping[str, Any],
    profile: str = "default",
) -> dict[str, Any]:
    """Build the argument context the canonical plan factory is called with.

    Separated from `_canonical_decisions` so a test can assert what this
    produces without needing a checkpoint on disk. That matters: this contract
    has broken twice, both times only discovered on a rented GPU after a
    multi-minute model load, with

        TypeError: unsupported required parameter 'tensors' on
        engine.quant.klinear_plan.build_klinear_quantization_plan

    The canonical factory takes an iterable of ITS OWN TensorMetadata, which
    this module has no reason to know about statically. Both types carry the
    same four facts, so translate rather than duplicate the type, and do it
    defensively so a canonical module without TensorMetadata still resolves
    through the other keys.
    """
    specs_by_name = {spec.name: spec for spec in specs}
    context: dict[str, Any] = {
        "checkpoint_dir": source_dir,
        "model_dir": source_dir,
        "source_dir": source_dir,
        "config": config,
        "index": index,
        "weight_map": index.get("weight_map", {}),
        "tensor_specs": specs_by_name,
        "tensor_names": tuple(specs_by_name),
        "weight_names": tuple(specs_by_name),
        # Named quantization profile. The canonical factory defaults this, so
        # passing it here is what lets an experiment measure a profile other
        # than the shipped one without editing either module.
        "profile": profile,
    }
    metadata_type = getattr(module, "TensorMetadata", None)
    if callable(metadata_type):
        translated = tuple(
            metadata_type(
                name=spec.name,
                shape=tuple(spec.shape),
                dtype=spec.dtype,
                source_file=spec.shard,
            )
            for spec in specs
        )
        context["tensors"] = translated
        context["metadata"] = translated
        context["tensor_metadata"] = translated
    return context


def _canonical_decisions(
    module: ModuleType,
    specs: tuple[TensorSpec, ...],
    *,
    source_dir: Path,
    config: Mapping[str, Any],
    index: Mapping[str, Any],
    profile: str = "default",
) -> tuple[TensorDecision, ...]:
    specs_by_name = {spec.name: spec for spec in specs}
    context = _canonical_context(
        module,
        specs,
        source_dir=source_dir,
        config=config,
        index=index,
        profile=profile,
    )

    for factory_name in (
        "build_klinear_quantization_plan",
        "build_quantization_plan",
        "build_plan",
    ):
        factory = getattr(module, factory_name, None)
        if not callable(factory):
            continue
        plan = _invoke_with_context(factory, context)
        mapping = _mapping_from_plan(plan)
        if mapping is None:
            raise TypeError(f"{CANONICAL_PLAN_MODULE}.{factory_name} returned an unsupported plan")
        unknown = sorted(set(mapping) - set(specs_by_name))
        if unknown:
            raise ValueError(
                f"canonical plan names tensors absent from the checkpoint: {unknown[:10]}"
            )
        return tuple(
            _decision_from_value(
                spec,
                mapping.get(spec.name, False),
                default_reason=(
                    "Selected by the canonical Kimi Linear quantization plan."
                    if spec.name in mapping and _value_quantizes(mapping[spec.name])
                    else "Not selected by the canonical Kimi Linear quantization plan."
                ),
            )
            for spec in specs
        )

    for selector_name in (
        "should_quantize_tensor",
        "should_quantize",
        "select_tensor",
    ):
        selector = getattr(module, selector_name, None)
        if not callable(selector):
            continue
        decisions = []
        for spec in specs:
            selector_context = dict(context)
            selector_context.update(
                {
                    "name": spec.name,
                    "tensor_name": spec.name,
                    "spec": spec,
                    "shape": spec.shape,
                    "dtype": spec.dtype,
                }
            )
            value = _invoke_with_context(selector, selector_context)
            decisions.append(
                _decision_from_value(
                    spec,
                    value,
                    default_reason=f"Decision returned by canonical selector {selector_name}.",
                )
            )
        return tuple(decisions)

    for constant_name in (
        "QUANTIZED_TENSOR_NAMES",
        "SELECTED_TENSOR_NAMES",
        "KLINEAR_QUANTIZED_TENSORS",
    ):
        selected = getattr(module, constant_name, None)
        if selected is None:
            continue
        mapping = {str(name): True for name in selected}
        unknown = sorted(set(mapping) - set(specs_by_name))
        if unknown:
            raise ValueError(
                f"canonical tensor-name constant contains absent tensors: {unknown[:10]}"
            )
        return tuple(
            _decision_from_value(
                spec,
                mapping.get(spec.name, False),
                default_reason=(
                    f"Selected by canonical constant {constant_name}."
                    if spec.name in mapping
                    else f"Not selected by canonical constant {constant_name}."
                ),
            )
            for spec in specs
        )

    raise AttributeError(
        f"{CANONICAL_PLAN_MODULE} exists but exposes no supported plan contract"
    )


def _validate_codec_eligibility(decisions: tuple[TensorDecision, ...]) -> None:
    for decision in decisions:
        if not decision.quantize:
            continue
        if is_router_tensor(decision.name):
            raise ValueError(f"canonical plan attempts to quantize router tensor {decision.name}")
        if decision.dtype != "BF16":
            raise ValueError(f"selected tensor {decision.name} is {decision.dtype}, not BF16")
        if len(decision.shape) != 2:
            raise ValueError(f"selected tensor {decision.name} is not a two-dimensional matrix")
        if decision.shape[-1] % GROUP_SIZE:
            raise ValueError(
                f"selected tensor {decision.name} reduction axis is not divisible by {GROUP_SIZE}"
            )


def resolve_plan(
    specs: tuple[TensorSpec, ...],
    *,
    source_dir: Path,
    config: Mapping[str, Any],
    index: Mapping[str, Any],
    profile: str = "default",
) -> PlanResolution:
    """Use the canonical plan when present, otherwise emit a blocked fallback."""
    local = tuple(_local_decision(spec) for spec in specs)
    try:
        module = importlib.import_module(CANONICAL_PLAN_MODULE)
    except ModuleNotFoundError as exc:
        if exc.name != CANONICAL_PLAN_MODULE:
            raise
        _validate_codec_eligibility(local)
        return PlanResolution(
            source=LOCAL_POLICY_VERSION,
            source_version=LOCAL_POLICY_VERSION,
            source_file_sha256=None,
            reconciliation_status="BLOCKED_CANONICAL_PLAN_MISSING",
            publication_allowed=False,
            canonical_matches_local_fallback=None,
            decisions=local,
        )

    canonical = _canonical_decisions(
        module,
        specs,
        source_dir=source_dir,
        config=config,
        index=index,
        profile=profile,
    )
    _validate_codec_eligibility(canonical)
    local_selected = {item.name for item in local if item.quantize}
    canonical_selected = {item.name for item in canonical if item.quantize}
    matches = local_selected == canonical_selected
    module_path = getattr(module, "__file__", None)
    return PlanResolution(
        source=CANONICAL_PLAN_MODULE,
        source_version=str(
            getattr(module, "PLAN_VERSION", getattr(module, "VERSION", "unversioned"))
        ),
        source_file_sha256=_file_sha256(module_path),
        reconciliation_status=(
            "RECONCILED_CANONICAL_MATCHES_LOCAL_FALLBACK"
            if matches
            else "RECONCILED_CANONICAL_OVERRIDES_LOCAL_FALLBACK"
        ),
        publication_allowed=True,
        canonical_matches_local_fallback=matches,
        decisions=canonical,
    )
