"""Fail-closed loading and validation for benchmark candidate adapters."""

from __future__ import annotations

import importlib
from collections.abc import Mapping, Sequence
from typing import Any, Callable


def load_candidate_factory(spec: str) -> Callable[..., Any]:
    """Load exactly the requested factory, with no reference fallback."""
    if ":" not in spec:
        raise ValueError("candidate factory must use module.path:function_name syntax")
    module_name, function_name = spec.split(":", 1)
    if not module_name or not function_name:
        raise ValueError("candidate factory must name both a module and a function")
    module = importlib.import_module(module_name)
    factory = getattr(module, function_name)
    if not callable(factory):
        raise TypeError(f"candidate factory {spec!r} is not callable")
    return factory


def require_candidate_coverage(runner: Any) -> dict[str, Any]:
    """Require an adapter to disclose the exact path it replaced."""
    coverage = getattr(runner, "coverage", None)
    if not isinstance(coverage, Mapping):
        raise TypeError("candidate runner must expose a coverage mapping")
    router_keys = coverage.get("measured_router_keys")
    if (
        not isinstance(router_keys, Sequence)
        or isinstance(router_keys, (str, bytes))
        or not router_keys
        or not all(isinstance(key, str) and key for key in router_keys)
        or len(set(router_keys)) != len(router_keys)
    ):
        raise ValueError("candidate coverage must list measured_router_keys")
    if not isinstance(coverage.get("full_model_candidate"), bool):
        raise ValueError("candidate coverage must state full_model_candidate")
    label = coverage.get("candidate_label")
    if not isinstance(label, str) or not label.strip():
        raise ValueError("candidate coverage must include candidate_label")
    return dict(coverage)
