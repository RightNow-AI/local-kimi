import pytest

from engine.bench.adapters import load_candidate_factory
from engine.bench.artifacts import validate_reference_artifact


def _artifact():
    return {
        "schema_version": 2,
        "model_id": "moonshotai/Kimi-Linear-48B-A3B-Instruct",
        "requested_revision": "main",
        "resolved_revision": "a" * 40,
        "snapshot_path": "/volume/snapshots/" + "a" * 40,
        "prompt_fingerprint": "fixed-prompts",
        "download_seconds": 10.0,
        "reference_seconds": 20.0,
        "router_config": {
            "router_count": 2,
            "experts_per_token": 4,
            "expert_count": 64,
        },
        "prompt_records": [{"id": "prompt"}],
        "heldout_records": [{"id": "heldout"}],
    }


def _validate(artifact):
    return validate_reference_artifact(
        artifact,
        schema_version=2,
        model_id="moonshotai/Kimi-Linear-48B-A3B-Instruct",
        prompt_fingerprint="fixed-prompts",
    )


def test_valid_artifact_identity_fields_pass():
    assert _validate(_artifact())["resolved_revision"] == "a" * 40


def test_artifact_schema_rejects_missing_resolved_revision():
    artifact = _artifact()
    del artifact["resolved_revision"]

    with pytest.raises(ValueError, match="resolved revision"):
        _validate(artifact)


def test_artifact_schema_rejects_different_prompt_fingerprint():
    artifact = _artifact()
    artifact["prompt_fingerprint"] = "different-prompts"

    with pytest.raises(ValueError, match="prompt fingerprint"):
        _validate(artifact)


def test_candidate_factory_import_failure_raises_without_reference_fallback():
    with pytest.raises(ModuleNotFoundError):
        load_candidate_factory("engine.bench.module_that_does_not_exist:build_runner")
