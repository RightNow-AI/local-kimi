"""Modal quality benchmark for Kimi-Linear reference and engine outputs.

The reference path is HuggingFace Transformers. The candidate path is a strict
adapter contract so this lane does not pretend the existing K3 layer probe is an
end-to-end Kimi-Linear engine. Both paths consume identical saved token IDs.

    modal run engine/modal_bench.py --action reference --gpu H100
    modal run engine/modal_bench.py --action compare --gpu H100 \
        --engine-factory engine.runtime:build_benchmark_runner
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import modal

APP = modal.App("k3-engine-bench")

BASE_IMAGE = modal.Image.debian_slim(python_version="3.12")
BENCH_IMAGE = (
    BASE_IMAGE.pip_install(
        "torch>=2.5",
        "transformers>=4.56.0",
        "accelerate>=1.2",
        "safetensors>=0.4",
        "einops>=0.8",
        "packaging>=24.0",
        "fla-core",
    )
    .pip_install("flash-attn>=2.7", extra_options="--no-build-isolation")
    .add_local_dir("engine", remote_path="/root/engine")
)

WEIGHTS = modal.Volume.from_name("k3-weights", create_if_missing=True)
VOL = "/weights"
ARTIFACT_DIR = f"{VOL}/bench/kimi-linear-48b-a3b"
REFERENCE_PATH = f"{ARTIFACT_DIR}/reference.pt"
REPORT_PATH = f"{ARTIFACT_DIR}/comparison.json"
HF_CACHE = f"{VOL}/huggingface"

MODEL_ID = "moonshotai/Kimi-Linear-48B-A3B-Instruct"
SCHEMA_VERSION = 1
DEFAULT_GPU = "H100"
ALLOWED_GPU_TYPES = {"H100", "H200", "B200"}

# These are acceptance policy limits, not measured loss values.
MAX_MEAN_KL_NATS = 1e-4
MIN_TOP1_AGREEMENT = 0.999
MIN_ROUTING_AGREEMENT = 1.0
MAX_ABS_PERPLEXITY_RELATIVE_DELTA = 1e-3

PROMPTS: tuple[dict[str, str], ...] = (
    {"id": "arithmetic", "text": "Compute 37 * 19 and explain one check on the result."},
    {"id": "code", "text": "Write a Python function that merges two sorted integer lists."},
    {"id": "json", "text": "Return JSON with keys answer and confidence for: Is 97 prime?"},
    {"id": "instruction", "text": "Give exactly three concise reasons to test numerical kernels."},
    {"id": "factual", "text": "Explain why the sky appears blue without using an equation."},
    {"id": "arabic", "text": "اشرح باختصار الفرق بين الذاكرة والتخزين في الحاسوب."},
    {"id": "reasoning", "text": "A box has 4 red and 6 blue balls. What is the chance of red in one draw?"},
    {"id": "long_context", "text": "Alpha precedes beta. Beta precedes gamma. Gamma precedes delta. Which item is second?"},
)

HELDOUT_TEXTS: tuple[dict[str, str], ...] = (
    {
        "id": "heldout_science",
        "text": "A controlled experiment changes one factor while holding other relevant conditions fixed. Repeated observations help separate a stable effect from random variation.",
    },
    {
        "id": "heldout_code",
        "text": "A reliable service validates inputs, records failures with enough context to diagnose them, and keeps retries safe by making repeated requests idempotent.",
    },
    {
        "id": "heldout_reasoning",
        "text": "If every cedar is a tree and no tree is a mineral, then no cedar is a mineral. The conclusion follows from the two stated relations.",
    },
    {
        "id": "heldout_arabic",
        "text": "تتحسن دقة القياس عندما نكرر التجربة ونقارن النتائج بمرجع ثابت ونوثق طريقة الحساب بوضوح.",
    },
)


def dequantize_mxfp4(packed: Any, scale: Any) -> Any:
    """Apply the already-proven K3 E2M1 plus E8M0 group-32 decode.

    This keeps the exact nibble order and 2^(byte-127) arithmetic from
    research/verify_lossless.py. It is exposed for candidate engine adapters so
    the benchmark does not introduce a second interpretation of packed weights.
    """
    import torch

    if packed.dtype != torch.uint8 or scale.dtype != torch.uint8:
        raise TypeError("MXFP4 packed values and scales must both be uint8 tensors")
    if packed.ndim != 2 or scale.ndim != 2 or packed.shape[0] != scale.shape[0]:
        raise ValueError("MXFP4 packed values and scales must be aligned rank-2 tensors")
    columns = packed.shape[1] * 2
    if columns % scale.shape[1] != 0 or columns // scale.shape[1] != 32:
        raise ValueError("MXFP4 requires one E8M0 exponent for each group of 32 values")
    table = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
         -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
        dtype=torch.float32,
        device=packed.device,
    )
    codes = torch.empty((packed.shape[0], columns), dtype=torch.long, device=packed.device)
    codes[:, 0::2] = (packed & 0x0F).long()
    codes[:, 1::2] = (packed >> 4).long()
    exponent = torch.exp2(scale.to(torch.int16).float() - 127.0)
    return table[codes] * exponent.repeat_interleave(32, dim=1)


def _validated_gpu(gpu: str) -> str:
    match = re.fullmatch(r"(H100|H200|B200)(?::([12]))?", gpu)
    if match is None or match.group(1) not in ALLOWED_GPU_TYPES:
        raise ValueError("gpu must be H100, H100:2, H200, H200:2, B200, or B200:2")
    return gpu


def _prompt_fingerprint() -> str:
    payload = json.dumps(
        {"prompts": PROMPTS, "heldout": HELDOUT_TEXTS},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _load_reference_model(revision: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        revision=revision,
        trust_remote_code=True,
        cache_dir=HF_CACHE,
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        revision=revision,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
        attn_implementation="flash_attention_2",
        cache_dir=HF_CACHE,
    )
    model.eval()
    return tokenizer, model


def _router_modules(model: Any) -> dict[str, Any]:
    routers = {
        name: module
        for name, module in model.named_modules()
        if name.endswith(".gate")
        and hasattr(module, "top_k")
        and hasattr(module, "num_experts")
        and hasattr(module, "e_score_correction_bias")
    }
    if not routers:
        raise RuntimeError("no Kimi MoE gate modules were found for routing capture")
    return routers


def _validate_reference_routers(model: Any) -> dict[str, int]:
    routers = _router_modules(model)
    top_k_values = {int(module.top_k) for module in routers.values()}
    expert_counts = {int(module.num_experts) for module in routers.values()}
    if len(top_k_values) != 1 or next(iter(top_k_values)) < 1:
        raise RuntimeError(f"reference router top-k differs across layers: {sorted(top_k_values)}")
    if len(expert_counts) != 1:
        raise RuntimeError(f"reference router expert counts differ across layers: {sorted(expert_counts)}")
    return {
        "router_count": len(routers),
        "experts_per_token": next(iter(top_k_values)),
        "expert_count": next(iter(expert_counts)),
    }


def _capture_forward(model: Any, input_ids: Any, attention_mask: Any) -> tuple[Any, dict[str, Any]]:
    import torch

    captured: dict[str, list[Any]] = {}
    handles = []

    def hook_for(name: str):
        def hook(_module, _inputs, output):
            if not isinstance(output, tuple) or len(output) < 1:
                raise RuntimeError(f"router {name} did not return top-k indices")
            topk_idx = output[0]
            captured.setdefault(name, []).append(topk_idx.detach().to("cpu", dtype=torch.int16))

        return hook

    for name, module in _router_modules(model).items():
        handles.append(module.register_forward_hook(hook_for(name)))
    try:
        device = model.get_input_embeddings().weight.device
        with torch.inference_mode():
            output = model(
                input_ids=input_ids.to(device),
                attention_mask=attention_mask.to(device),
                use_cache=False,
                return_dict=True,
            )
    finally:
        for handle in handles:
            handle.remove()
    routes = {
        name: torch.cat(chunks, dim=0)
        for name, chunks in captured.items()
    }
    if len(routes) != len(_router_modules(model)):
        raise RuntimeError("one or more router modules produced no routing capture")
    return output.logits.detach(), routes


def _chat_inputs(tokenizer: Any, text: str, max_input_tokens: int) -> dict[str, Any]:
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": text}],
        tokenize=False,
        add_generation_prompt=True,
    )
    return tokenizer(
        rendered,
        return_tensors="pt",
        add_special_tokens=False,
        truncation=True,
        max_length=max_input_tokens,
    )


def _plain_inputs(tokenizer: Any, text: str, max_input_tokens: int) -> dict[str, Any]:
    return tokenizer(
        text,
        return_tensors="pt",
        add_special_tokens=True,
        truncation=True,
        max_length=max_input_tokens,
    )


def _resolved_revision(model: Any, requested_revision: str) -> str:
    commit_hash = str(getattr(model.config, "_commit_hash", "") or "")
    if re.fullmatch(r"[0-9a-fA-F]{40}", commit_hash):
        return commit_hash.lower()
    if re.fullmatch(r"[0-9a-fA-F]{40}", requested_revision):
        return requested_revision.lower()
    raise RuntimeError("HuggingFace did not expose an immutable 40-character checkpoint revision")


def _gpu_inventory() -> dict[str, Any]:
    import torch

    names = [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
    return {"count": len(names), "names": names}


@APP.function(
    image=BENCH_IMAGE,
    gpu=DEFAULT_GPU,
    volumes={VOL: WEIGHTS},
    timeout=60 * 60 * 6,
    cpu=8.0,
    memory=65536,
)
def capture_reference(
    revision: str = "main",
    max_input_tokens: int = 128,
    scored_prompt_positions: int = 8,
    max_heldout_tokens: int = 64,
) -> dict[str, Any]:
    """Capture HuggingFace logits and router choices on the fixed corpus."""
    import torch

    if max_input_tokens < scored_prompt_positions or scored_prompt_positions < 1:
        raise ValueError("scored_prompt_positions must be between 1 and max_input_tokens")
    if max_heldout_tokens < 2:
        raise ValueError("max_heldout_tokens must allow at least one next-token prediction")

    started = time.time()
    tokenizer, model = _load_reference_model(revision)
    router_config = _validate_reference_routers(model)
    prompt_records = []
    for prompt in PROMPTS:
        inputs = _chat_inputs(tokenizer, prompt["text"], max_input_tokens)
        logits, routes = _capture_forward(model, inputs["input_ids"], inputs["attention_mask"])
        positions = min(scored_prompt_positions, logits.shape[1])
        prompt_records.append(
            {
                "id": prompt["id"],
                "text": prompt["text"],
                "input_ids": inputs["input_ids"].to(torch.int64),
                "attention_mask": inputs["attention_mask"].to(torch.int64),
                "logits": logits[:, -positions:, :].float().cpu(),
                "routes": routes,
                "scored_positions": positions,
            }
        )

    heldout_records = []
    for sample in HELDOUT_TEXTS:
        inputs = _plain_inputs(tokenizer, sample["text"], max_heldout_tokens)
        logits, routes = _capture_forward(model, inputs["input_ids"], inputs["attention_mask"])
        if logits.shape[1] < 2:
            raise RuntimeError(f"heldout sample {sample['id']} has no next-token prediction")
        heldout_records.append(
            {
                "id": sample["id"],
                "text": sample["text"],
                "input_ids": inputs["input_ids"].to(torch.int64),
                "attention_mask": inputs["attention_mask"].to(torch.int64),
                "logits": logits[:, :-1, :].float().cpu(),
                "targets": inputs["input_ids"][:, 1:].to(torch.int64),
                "routes": routes,
            }
        )

    resolved_revision = _resolved_revision(model, revision)
    artifact = {
        "schema_version": SCHEMA_VERSION,
        "model_id": MODEL_ID,
        "requested_revision": revision,
        "resolved_revision": resolved_revision,
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "hardware": _gpu_inventory(),
        "prompt_fingerprint": _prompt_fingerprint(),
        "router_config": router_config,
        "prompt_records": prompt_records,
        "heldout_records": heldout_records,
        "capture_arithmetic": {
            "prompt_count": f"{len(PROMPTS)} fixed prompts",
            "scored_prompt_positions": (
                f"sum(record.scored_positions) = {sum(record['scored_positions'] for record in prompt_records)}"
            ),
            "heldout_prediction_count": (
                "sum(token_count - 1) = "
                f"{sum(record['targets'].numel() for record in heldout_records)}"
            ),
        },
    }
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    temporary = f"{REFERENCE_PATH}.tmp"
    torch.save(artifact, temporary)
    os.replace(temporary, REFERENCE_PATH)
    WEIGHTS.commit()
    result = {
        "reference_path": REFERENCE_PATH,
        "model_id": MODEL_ID,
        "resolved_revision": resolved_revision,
        "prompt_fingerprint": artifact["prompt_fingerprint"],
        "prompt_count": len(prompt_records),
        "heldout_count": len(heldout_records),
        "router_config": router_config,
        "seconds": round(time.time() - started, 1),
    }
    print(json.dumps(result, indent=2))
    return result


def _load_factory(spec: str):
    if ":" not in spec:
        raise ValueError("engine_factory must use module.path:function_name syntax")
    module_name, function_name = spec.split(":", 1)
    if "/root" not in sys.path:
        sys.path.insert(0, "/root")
    factory = getattr(importlib.import_module(module_name), function_name)
    if not callable(factory):
        raise TypeError(f"engine factory {spec!r} is not callable")
    return factory


def _candidate_output(runner: Any, record: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
    output = runner.run(
        input_ids=record["input_ids"],
        attention_mask=record["attention_mask"],
        capture_routing=True,
    )
    if not isinstance(output, dict) or "logits" not in output or "routes" not in output:
        raise TypeError("engine runner must return a dict containing logits and routes")
    if not isinstance(output["routes"], dict):
        raise TypeError("engine runner routes must be a mapping from router name to top-k indices")
    return output["logits"], output["routes"]


def _aligned_candidate_logits(candidate: Any, reference: Any) -> Any:
    if candidate.ndim != 3 or reference.ndim != 3:
        raise ValueError("reference and candidate logits must both have shape [batch, tokens, vocab]")
    if candidate.shape[0] != reference.shape[0] or candidate.shape[2] != reference.shape[2]:
        raise ValueError("reference and candidate batch or vocabulary dimensions differ")
    if candidate.shape[1] < reference.shape[1]:
        raise ValueError("candidate returned fewer logit positions than the saved reference")
    return candidate[:, -reference.shape[1]:, :].float().cpu()


def _validate_route_keys(reference: dict[str, Any], candidate: dict[str, Any]) -> None:
    reference_keys = set(reference)
    candidate_keys = set(candidate)
    if reference_keys != candidate_keys:
        missing = sorted(reference_keys - candidate_keys)
        extra = sorted(candidate_keys - reference_keys)
        raise ValueError(f"router key mismatch; missing={missing}, extra={extra}")


def _route_decision_count(routes: dict[str, Any]) -> int:
    return sum(int(route.reshape(-1, route.shape[-1]).shape[0]) for route in routes.values())


@APP.function(
    image=BENCH_IMAGE,
    gpu=DEFAULT_GPU,
    volumes={VOL: WEIGHTS},
    timeout=60 * 60 * 6,
    cpu=8.0,
    memory=65536,
)
def compare_engine(
    engine_factory: str,
    max_mean_kl_nats: float = MAX_MEAN_KL_NATS,
    min_top1_agreement: float = MIN_TOP1_AGREEMENT,
    min_routing_agreement: float = MIN_ROUTING_AGREEMENT,
    max_abs_perplexity_relative_delta: float = MAX_ABS_PERPLEXITY_RELATIVE_DELTA,
) -> dict[str, Any]:
    """Run the candidate engine on saved inputs and apply fail-closed gates.

    The factory must return an object with run(input_ids, attention_mask,
    capture_routing=True). Its result must contain full logits and a routes
    mapping keyed exactly like the HuggingFace router modules.
    """
    import torch

    if "/root" not in sys.path:
        sys.path.insert(0, "/root")
    from engine.bench.metrics import (
        perplexity,
        routing_agreement,
        token_kl_divergence,
        top1_agreement,
    )

    if not os.path.exists(REFERENCE_PATH):
        raise FileNotFoundError(f"{REFERENCE_PATH} is missing; run capture_reference first")
    reference = torch.load(REFERENCE_PATH, map_location="cpu", weights_only=False)
    if reference.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("reference artifact schema does not match this benchmark code")
    if reference.get("model_id") != MODEL_ID:
        raise ValueError("reference artifact model does not match the benchmark model")
    if reference.get("prompt_fingerprint") != _prompt_fingerprint():
        raise ValueError("fixed prompt corpus changed after the reference capture")
    router_config = reference.get("router_config")
    if not isinstance(router_config, dict) or not isinstance(router_config.get("experts_per_token"), int):
        raise ValueError("reference artifact does not record a valid router configuration")
    experts_per_token = router_config["experts_per_token"]

    factory = _load_factory(engine_factory)
    runner = factory(
        model_id=reference["model_id"],
        revision=reference["resolved_revision"],
        cache_dir=HF_CACHE,
        dequantize_mxfp4=dequantize_mxfp4,
    )

    prompt_results = []
    all_reference_logits = []
    all_candidate_logits = []
    all_reference_routes: dict[str, list[Any]] = {}
    all_candidate_routes: dict[str, list[Any]] = {}
    for record in reference["prompt_records"]:
        candidate_logits, candidate_routes = _candidate_output(runner, record)
        candidate_logits = _aligned_candidate_logits(candidate_logits, record["logits"])
        _validate_route_keys(record["routes"], candidate_routes)
        candidate_routes = {name: route.detach().cpu() for name, route in candidate_routes.items()}
        route_total = _route_decision_count(record["routes"])
        route_value = routing_agreement(
            record["routes"],
            candidate_routes,
            expected_experts_per_token=experts_per_token,
        )
        token_count = int(record["logits"].shape[0] * record["logits"].shape[1])
        top1_value = top1_agreement(record["logits"], candidate_logits)
        kl_value = token_kl_divergence(record["logits"], candidate_logits)
        prompt_results.append(
            {
                "id": record["id"],
                "mean_kl_nats": kl_value,
                "mean_kl_arithmetic": f"{kl_value * token_count:.12g} total token KL / {token_count} positions = {kl_value:.12g}",
                "top1_agreement": top1_value,
                "top1_arithmetic": f"{round(top1_value * token_count)} matches / {token_count} positions = {top1_value:.12g}",
                "routing_agreement": route_value,
                "routing_arithmetic": f"{round(route_value * route_total)} matches / {route_total} token-layer decisions = {route_value:.12g}",
            }
        )
        all_reference_logits.append(record["logits"])
        all_candidate_logits.append(candidate_logits)
        for name in record["routes"]:
            all_reference_routes.setdefault(name, []).append(record["routes"][name])
            all_candidate_routes.setdefault(name, []).append(candidate_routes[name])

    heldout_reference_logits = []
    heldout_candidate_logits = []
    heldout_targets = []
    for record in reference["heldout_records"]:
        candidate_logits, candidate_routes = _candidate_output(runner, record)
        candidate_logits = candidate_logits[:, :-1, :] if candidate_logits.shape[1] == record["input_ids"].shape[1] else candidate_logits
        candidate_logits = _aligned_candidate_logits(candidate_logits, record["logits"])
        _validate_route_keys(record["routes"], candidate_routes)
        routing_agreement(
            record["routes"],
            candidate_routes,
            expected_experts_per_token=experts_per_token,
        )
        heldout_reference_logits.append(record["logits"])
        heldout_candidate_logits.append(candidate_logits)
        heldout_targets.append(record["targets"])

    reference_logits = torch.cat(all_reference_logits, dim=1)
    candidate_logits = torch.cat(all_candidate_logits, dim=1)
    reference_routes = {
        name: torch.cat(chunks, dim=0) for name, chunks in all_reference_routes.items()
    }
    candidate_routes = {
        name: torch.cat(chunks, dim=0) for name, chunks in all_candidate_routes.items()
    }
    prompt_token_count = int(reference_logits.shape[0] * reference_logits.shape[1])
    route_total = _route_decision_count(reference_routes)
    mean_kl = token_kl_divergence(reference_logits, candidate_logits)
    top1 = top1_agreement(reference_logits, candidate_logits)
    routing = routing_agreement(
        reference_routes,
        candidate_routes,
        expected_experts_per_token=experts_per_token,
    )

    reference_ppl_logits = torch.cat(heldout_reference_logits, dim=1)
    candidate_ppl_logits = torch.cat(heldout_candidate_logits, dim=1)
    targets = torch.cat(heldout_targets, dim=1)
    prediction_count = int(targets.numel())
    reference_ppl = perplexity(reference_ppl_logits, targets)
    candidate_ppl = perplexity(candidate_ppl_logits, targets)
    ppl_relative_delta = (candidate_ppl - reference_ppl) / reference_ppl

    gates = {
        "mean_kl": mean_kl <= max_mean_kl_nats,
        "top1": top1 >= min_top1_agreement,
        "routing": routing >= min_routing_agreement,
        "perplexity": abs(ppl_relative_delta) <= max_abs_perplexity_relative_delta,
    }
    report = {
        "schema_version": SCHEMA_VERSION,
        "model_id": reference["model_id"],
        "resolved_revision": reference["resolved_revision"],
        "engine_factory": engine_factory,
        "compared_at_utc": datetime.now(timezone.utc).isoformat(),
        "hardware": _gpu_inventory(),
        "prompt_fingerprint": reference["prompt_fingerprint"],
        "router_config": router_config,
        "metrics": {
            "mean_token_kl_nats": mean_kl,
            "mean_token_kl_arithmetic": f"{mean_kl * prompt_token_count:.12g} total token KL / {prompt_token_count} positions = {mean_kl:.12g}",
            "top1_agreement": top1,
            "top1_arithmetic": f"{round(top1 * prompt_token_count)} matches / {prompt_token_count} positions = {top1:.12g}",
            "routing_agreement": routing,
            "routing_arithmetic": f"{round(routing * route_total)} matches / {route_total} token-layer decisions = {routing:.12g}",
            "reference_perplexity": reference_ppl,
            "candidate_perplexity": candidate_ppl,
            "perplexity_arithmetic": (
                f"exp(mean negative log likelihood over {prediction_count} predictions); "
                f"relative delta = ({candidate_ppl:.12g} - {reference_ppl:.12g}) / {reference_ppl:.12g} = {ppl_relative_delta:.12g}"
            ),
            "perplexity_relative_delta": ppl_relative_delta,
        },
        "policy_limits": {
            "max_mean_kl_nats": max_mean_kl_nats,
            "min_top1_agreement": min_top1_agreement,
            "min_routing_agreement": min_routing_agreement,
            "max_abs_perplexity_relative_delta": max_abs_perplexity_relative_delta,
        },
        "gates": gates,
        "passed": all(gates.values()),
        "prompt_results": prompt_results,
        "coverage": {
            "proves": "candidate engine agreement with HuggingFace on fixed Kimi-Linear inputs",
            "does_not_prove": "full Kimi K3 end-to-end quality",
        },
    }
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    temporary = f"{REPORT_PATH}.tmp"
    Path(temporary).write_text(json.dumps(report, indent=2), encoding="utf-8")
    os.replace(temporary, REPORT_PATH)
    WEIGHTS.commit()
    print(json.dumps(report, indent=2))
    return report


@APP.local_entrypoint()
def main(
    action: str = "reference",
    gpu: str = DEFAULT_GPU,
    revision: str = "main",
    engine_factory: str = "",
):
    selected_gpu = _validated_gpu(gpu)
    if action == "reference":
        capture_reference.with_options(gpu=selected_gpu).remote(revision=revision)
    elif action == "compare":
        if not engine_factory:
            raise ValueError("--engine-factory is required for compare; no reference fallback is allowed")
        compare_engine.with_options(gpu=selected_gpu).remote(engine_factory=engine_factory)
    else:
        raise ValueError("action must be reference or compare")
