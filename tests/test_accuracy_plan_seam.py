"""Pin the seam between the accuracy plan resolver and the canonical plan.

These two modules were written by separate lanes that never saw each other's
code. They agreed on intent and disagreed on calling convention, and nothing
caught it until an H200 job had already loaded and died with

    TypeError: unsupported required parameter 'tensors' on
    engine.quant.klinear_plan.build_klinear_quantization_plan

The accuracy resolver passes arguments by matching parameter NAMES against a
context dictionary, so the canonical factory can rename or add a required
parameter and break the integration without either file changing in a way a
reviewer would notice. That is what these tests exist to catch, on CPU, in
milliseconds, instead of on a rented GPU after a checkpoint load.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from engine.accuracy.plan import TensorSpec, _canonical_context, _invoke_with_context
from engine.quant import klinear_plan
from engine.quant.klinear_plan import TensorMetadata, build_klinear_quantization_plan


def _specs() -> tuple[TensorSpec, ...]:
    """A minimal but structurally real slice of the checkpoint contract."""
    return (
        TensorSpec(
            name="model.layers.1.block_sparse_moe.experts.0.w1.weight",
            shard="model-00002-of-00020.safetensors",
            shape=(1024, 2304),
            dtype="BF16",
        ),
        TensorSpec(
            name="model.layers.1.block_sparse_moe.gate.weight",
            shard="model-00002-of-00020.safetensors",
            shape=(256, 2304),
            dtype="BF16",
        ),
        TensorSpec(
            name="model.layers.3.self_attn.kv_a_proj_with_mqa.weight",
            shard="model-00004-of-00020.safetensors",
            shape=(576, 2304),
            dtype="BF16",
        ),
        TensorSpec(
            name="model.layers.1.input_layernorm.weight",
            shard="model-00002-of-00020.safetensors",
            shape=(2304,),
            dtype="BF16",
        ),
        # The shared expert runs on EVERY token, unlike a routed expert which
        # sees roughly 8 of 256, so it is the class the shared-experts-bf16
        # profile exists to retain.
        TensorSpec(
            name="model.layers.1.block_sparse_moe.shared_experts.down_proj.weight",
            shard="model-00002-of-00020.safetensors",
            shape=(2304, 1024),
            dtype="BF16",
        ),
    )


def _context_for(specs: tuple[TensorSpec, ...]) -> dict[str, object]:
    """The REAL context builder, not a reimplementation of it.

    An earlier version of this file built the context itself. That was a false
    positive: it proved the translation logic was correct while the production
    path did not perform it at all, so the same TypeError reappeared on a rented
    GPU with the suite green. A seam test that does not call the seam is not a
    seam test.
    """
    return _canonical_context(
        klinear_plan,
        specs,
        source_dir=Path("/weights/Kimi-Linear-48B-A3B-Instruct"),
        config={"num_hidden_layers": 27},
        index={"weight_map": {spec.name: spec.shard for spec in specs}},
    )


def test_context_satisfies_every_required_parameter_of_the_canonical_factory():
    """Fails the moment the canonical factory requires something unsupplied.

    This is the direct pin for the observed failure. It asserts the property
    that actually matters rather than a specific parameter name, so renaming
    ``tensors`` to something else still fails here first.
    """
    signature = inspect.signature(build_klinear_quantization_plan)
    context = _context_for(_specs())
    missing = [
        parameter.name
        for parameter in signature.parameters.values()
        if parameter.kind not in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD)
        and parameter.default is inspect.Parameter.empty
        and parameter.name not in context
    ]
    assert not missing, (
        "the accuracy resolver context cannot satisfy "
        f"build_klinear_quantization_plan; unsupplied required parameters: {missing}"
    )


def test_resolver_actually_invokes_the_canonical_plan():
    """End to end through the real invocation path, not just the signature."""
    plan = _invoke_with_context(build_klinear_quantization_plan, _context_for(_specs()))
    decisions = {decision.name: decision for decision in plan.tensors}
    assert set(decisions) == {spec.name for spec in _specs()}


def test_canonical_decisions_survive_translation():
    """The translated metadata must reproduce the plan's real policy.

    A translation that silently lost the shape or dtype would still return a
    plan, and that plan would be wrong in exactly the way nobody notices: the
    expert would be skipped, or the router would be quantized.
    """
    plan = _invoke_with_context(build_klinear_quantization_plan, _context_for(_specs()))
    decisions = {decision.name: decision for decision in plan.tensors}

    expert = decisions["model.layers.1.block_sparse_moe.experts.0.w1.weight"]
    assert expert.quantize is True, "routed experts are the primary fit target"

    router = decisions["model.layers.1.block_sparse_moe.gate.weight"]
    assert router.quantize is False, (
        "routing is discrete, so router error changes which computation runs"
    )

    latent = decisions["model.layers.3.self_attn.kv_a_proj_with_mqa.weight"]
    assert latent.quantize is False, (
        "the MLA latent down-projection writes the cached KV, so its error "
        "persists for a whole sequence"
    )

    norm = decisions["model.layers.1.input_layernorm.weight"]
    assert norm.quantize is False


def test_translation_rejects_a_dtype_the_canonical_plan_cannot_represent():
    """Translation must not launder an unsupported dtype into the plan."""
    with pytest.raises(ValueError):
        TensorMetadata(
            name="model.layers.1.block_sparse_moe.experts.0.w1.weight",
            shape=(1024, 2304),
            dtype="NOT_A_DTYPE",
            source_file="model-00002-of-00020.safetensors",
        )


def test_canonical_module_exposes_the_translation_target():
    """The resolver translates defensively; this pins what it depends on."""
    assert hasattr(klinear_plan, "TensorMetadata")
    assert callable(klinear_plan.TensorMetadata)


def test_the_named_profile_reaches_the_canonical_plan():
    """A profile that does not arrive would silently measure the wrong artifact.

    The canonical factory defaults ``profile``, so a context that omits it
    produces a valid plan for the WRONG policy rather than an error. The whole
    shared-expert experiment depends on this argument arriving, and its absence
    would look like a null result instead of a plumbing bug.
    """
    specs = _specs()
    context = _canonical_context(
        klinear_plan,
        specs,
        source_dir=Path("/weights/Kimi-Linear-48B-A3B-Instruct"),
        config={"num_hidden_layers": 27},
        index={"weight_map": {spec.name: spec.shard for spec in specs}},
        profile="shared-experts-bf16",
    )
    assert context["profile"] == "shared-experts-bf16"

    plan = _invoke_with_context(build_klinear_quantization_plan, context)
    decisions = {decision.name: decision for decision in plan.tensors}

    # The one decision the profile is supposed to change, and one it is not.
    shared = decisions["model.layers.1.block_sparse_moe.shared_experts.down_proj.weight"]
    assert shared.quantize is False, "shared-experts-bf16 must retain shared experts"
    routed = decisions["model.layers.1.block_sparse_moe.experts.0.w1.weight"]
    assert routed.quantize is True, "routed experts stay quantized under this profile"


def test_an_unknown_profile_is_refused_rather_than_defaulted():
    """Falling back on a typo would measure the shipped artifact silently."""
    specs = _specs()
    context = _canonical_context(
        klinear_plan,
        specs,
        source_dir=Path("/weights/Kimi-Linear-48B-A3B-Instruct"),
        config={"num_hidden_layers": 27},
        index={"weight_map": {spec.name: spec.shard for spec in specs}},
        profile="shared-experts-bf-16",
    )
    with pytest.raises(ValueError, match="unknown"):
        _invoke_with_context(build_klinear_quantization_plan, context)
