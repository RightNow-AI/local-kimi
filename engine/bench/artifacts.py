"""Offline-safe schema checks for saved benchmark artifacts."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

IMMUTABLE_REVISION = re.compile(r"[0-9a-f]{40}")


def validate_reference_artifact(
    artifact: Any,
    *,
    schema_version: int,
    model_id: str,
    prompt_fingerprint: str,
) -> dict[str, Any]:
    """Validate identity fields before any candidate measurement is allowed."""
    if not isinstance(artifact, Mapping):
        raise TypeError("reference artifact must be a mapping")
    if artifact.get("schema_version") != schema_version:
        raise ValueError("reference artifact schema does not match this benchmark code")
    if artifact.get("model_id") != model_id:
        raise ValueError("reference artifact model does not match the benchmark model")
    resolved_revision = artifact.get("resolved_revision")
    if not isinstance(resolved_revision, str) or IMMUTABLE_REVISION.fullmatch(resolved_revision) is None:
        raise ValueError("reference artifact is missing an immutable resolved revision")
    if artifact.get("prompt_fingerprint") != prompt_fingerprint:
        raise ValueError("reference artifact prompt fingerprint does not match this benchmark code")
    requested_revision = artifact.get("requested_revision")
    if not isinstance(requested_revision, str) or not requested_revision:
        raise ValueError("reference artifact is missing the requested revision")
    snapshot_path = artifact.get("snapshot_path")
    if not isinstance(snapshot_path, str) or not snapshot_path:
        raise ValueError("reference artifact is missing the saved snapshot path")
    for field in ("download_seconds", "reference_seconds"):
        value = artifact.get(field)
        if not isinstance(value, (int, float)) or value < 0:
            raise ValueError(f"reference artifact {field} must be nonnegative")

    router_config = artifact.get("router_config")
    if not isinstance(router_config, Mapping):
        raise ValueError("reference artifact does not record router configuration")
    for field in ("router_count", "experts_per_token", "expert_count"):
        value = router_config.get(field)
        if not isinstance(value, int) or value < 1:
            raise ValueError(f"reference artifact router_config.{field} must be positive")

    for field in ("prompt_records", "heldout_records"):
        records = artifact.get(field)
        if (
            not isinstance(records, Sequence)
            or isinstance(records, (str, bytes))
            or not records
            or not all(isinstance(record, Mapping) for record in records)
        ):
            raise ValueError(f"reference artifact {field} must be a nonempty record sequence")
    return dict(artifact)
