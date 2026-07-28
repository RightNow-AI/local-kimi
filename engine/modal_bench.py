"""Modal benchmark for HuggingFace and the disclosed engine.k3ref candidate.

Run the actions in order:

    modal run engine/modal_bench.py --action download --revision main
    modal run engine/modal_bench.py --action reference --gpu H200 --revision main
    modal run engine/modal_bench.py --action compare --gpu H200 \
        --engine-factory engine.bench.candidate:build_kimi_linear_runner

The compare action rewrites the local RESULTS.md and LOSS-LEDGER.md only after
Modal returns a real measured report.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import modal

APP = modal.App("kimi-linear-engine-bench")

BASE_IMAGE = modal.Image.debian_slim(python_version="3.12")
DOWNLOAD_IMAGE = BASE_IMAGE.pip_install("huggingface_hub>=0.34")
BENCH_IMAGE = (
    BASE_IMAGE.pip_install(
        "torch>=2.5",
        "transformers>=4.56.0",
        "accelerate>=1.2",
        "huggingface_hub>=0.34",
        "safetensors>=0.4",
        "einops>=0.8",
        "packaging>=24.0",
        "fla-core",
    )
    .pip_install("flash-attn>=2.7", extra_options="--no-build-isolation")
    .add_local_dir("engine", remote_path="/root/engine")
)

KIMI_LINEAR_WEIGHTS = modal.Volume.from_name(
    "kimi-linear-weights", create_if_missing=True
)
VOL = "/kimi-linear"
HF_CACHE = f"{VOL}/huggingface"
ARTIFACT_DIR = f"{VOL}/bench/kimi-linear-48b-a3b"
DOWNLOAD_MANIFEST_PATH = f"{ARTIFACT_DIR}/download.json"
REFERENCE_PATH = f"{ARTIFACT_DIR}/reference.pt"
REPORT_PATH = f"{ARTIFACT_DIR}/comparison.json"
VOLUME_RESULTS_PATH = f"{ARTIFACT_DIR}/RESULTS.md"
VOLUME_LEDGER_PATH = f"{ARTIFACT_DIR}/LOSS-LEDGER.md"

LOCAL_RESULTS_PATH = Path("engine/bench/RESULTS.md")
LOCAL_LEDGER_PATH = Path("engine/bench/LOSS-LEDGER.md")

MODEL_ID = "moonshotai/Kimi-Linear-48B-A3B-Instruct"
SCHEMA_VERSION = 2
DEFAULT_GPU = "H200"
DEFAULT_ENGINE_FACTORY = "engine.bench.candidate:build_kimi_linear_runner"

# One H100 cannot hold the roughly 96 GB BF16 checkpoint.
ALLOWED_GPU_SHAPES = {"H100:2", "H200", "H200:2", "B200", "B200:2"}

# These are policy limits, not measured values.
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


def _validated_gpu(gpu: str) -> str:
    if gpu not in ALLOWED_GPU_SHAPES:
        allowed = ", ".join(sorted(ALLOWED_GPU_SHAPES))
        raise ValueError(f"gpu must be one of: {allowed}")
    return gpu


def _prompt_fingerprint() -> str:
    payload = json.dumps(
        {"prompts": PROMPTS, "heldout": HELDOUT_TEXTS},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _atomic_text(path: str, text: str) -> None:
    os.makedirs(str(Path(path).parent), exist_ok=True)
    temporary = f"{path}.tmp"
    Path(temporary).write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _atomic_json(path: str, value: dict[str, Any]) -> None:
    _atomic_text(path, json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def _resolved_snapshot_revision(snapshot_path: str) -> str:
    revision = Path(snapshot_path).name.lower()
    if re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise RuntimeError("HuggingFace snapshot path does not end in an immutable revision")
    return revision


@APP.function(
    image=DOWNLOAD_IMAGE,
    volumes={VOL: KIMI_LINEAR_WEIGHTS},
    timeout=60 * 60 * 6,
    cpu=4.0,
    memory=8192,
)
def download_model(revision: str = "main") -> dict[str, Any]:
    """Download one immutable Kimi-Linear snapshot into its dedicated Volume."""
    from huggingface_hub import snapshot_download

    started = time.time()
    KIMI_LINEAR_WEIGHTS.reload()
    snapshot_path = snapshot_download(
        repo_id=MODEL_ID,
        revision=revision,
        cache_dir=HF_CACHE,
    )
    resolved_revision = _resolved_snapshot_revision(snapshot_path)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "model_id": MODEL_ID,
        "requested_revision": revision,
        "resolved_revision": resolved_revision,
        "snapshot_path": snapshot_path,
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "seconds": round(time.time() - started, 1),
        "volume": "kimi-linear-weights",
    }
    _atomic_json(DOWNLOAD_MANIFEST_PATH, manifest)
    KIMI_LINEAR_WEIGHTS.commit()
    print(json.dumps(manifest, indent=2))
    return manifest


def _download_manifest(revision: str) -> dict[str, Any]:
    if not os.path.exists(DOWNLOAD_MANIFEST_PATH):
        raise FileNotFoundError(
            f"{DOWNLOAD_MANIFEST_PATH} is missing; run the download action first"
        )
    manifest = json.loads(Path(DOWNLOAD_MANIFEST_PATH).read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("download manifest schema does not match this benchmark code")
    if manifest.get("model_id") != MODEL_ID:
        raise ValueError("download manifest model does not match the benchmark model")
    resolved = manifest.get("resolved_revision")
    if not isinstance(resolved, str) or re.fullmatch(r"[0-9a-f]{40}", resolved) is None:
        raise ValueError("download manifest is missing an immutable revision")
    if revision not in {manifest.get("requested_revision"), resolved}:
        raise ValueError(
            f"downloaded revision is {resolved}; rerun download for requested revision {revision}"
        )
    snapshot_path = manifest.get("snapshot_path")
    if not isinstance(snapshot_path, str) or not Path(snapshot_path).is_dir():
        raise FileNotFoundError("download manifest snapshot_path is missing from the Volume")
    if _resolved_snapshot_revision(snapshot_path) != resolved:
        raise ValueError("download manifest revision and snapshot path disagree")
    return manifest


def _load_reference_model(snapshot_path: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        snapshot_path,
        trust_remote_code=True,
        local_files_only=True,
        cache_dir=HF_CACHE,
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        snapshot_path,
        trust_remote_code=True,
        local_files_only=True,
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
        def hook(_module: Any, _inputs: Any, output: Any) -> None:
            if not isinstance(output, tuple) or not output:
                raise RuntimeError(f"router {name} did not return top-k indices")
            captured.setdefault(name, []).append(
                output[0].detach().to("cpu", dtype=torch.int16)
            )

        return hook

    routers = _router_modules(model)
    for name, module in routers.items():
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
    routes = {name: torch.cat(chunks, dim=0) for name, chunks in captured.items()}
    if set(routes) != set(routers):
        missing = sorted(set(routers) - set(routes))
        raise RuntimeError(f"reference produced no routing capture for {missing}")
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


def _gpu_inventory(requested_gpu: str) -> dict[str, Any]:
    import torch

    names = [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
    return {"requested_gpu": requested_gpu, "count": len(names), "names": names}


@APP.function(
    image=BENCH_IMAGE,
    gpu=DEFAULT_GPU,
    volumes={VOL: KIMI_LINEAR_WEIGHTS},
    timeout=60 * 60 * 6,
    cpu=8.0,
    memory=65536,
)
def capture_reference(
    revision: str = "main",
    requested_gpu: str = DEFAULT_GPU,
    max_input_tokens: int = 128,
    scored_prompt_positions: int = 8,
    max_heldout_tokens: int = 64,
) -> dict[str, Any]:
    """Capture HuggingFace logits and router choices for the fixed corpus."""
    import torch

    if max_input_tokens < scored_prompt_positions or scored_prompt_positions < 1:
        raise ValueError("scored_prompt_positions must be between 1 and max_input_tokens")
    if max_heldout_tokens < 2:
        raise ValueError("max_heldout_tokens must allow at least one next-token prediction")

    started = time.time()
    KIMI_LINEAR_WEIGHTS.reload()
    download = _download_manifest(revision)
    tokenizer, model = _load_reference_model(download["snapshot_path"])
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

    reference_seconds = round(time.time() - started, 1)
    artifact = {
        "schema_version": SCHEMA_VERSION,
        "model_id": MODEL_ID,
        "requested_revision": revision,
        "resolved_revision": download["resolved_revision"],
        "snapshot_path": download["snapshot_path"],
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "hardware": _gpu_inventory(requested_gpu),
        "prompt_fingerprint": _prompt_fingerprint(),
        "router_config": router_config,
        "prompt_records": prompt_records,
        "heldout_records": heldout_records,
        "download_seconds": download["seconds"],
        "reference_seconds": reference_seconds,
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
    KIMI_LINEAR_WEIGHTS.commit()
    result = {
        "reference_path": REFERENCE_PATH,
        "model_id": MODEL_ID,
        "resolved_revision": artifact["resolved_revision"],
        "prompt_fingerprint": artifact["prompt_fingerprint"],
        "prompt_count": len(prompt_records),
        "heldout_count": len(heldout_records),
        "router_config": router_config,
        "seconds": reference_seconds,
        "hardware": artifact["hardware"],
    }
    print(json.dumps(result, indent=2))
    return result


def _candidate_output(runner: Any, record: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
    output = runner.run(
        input_ids=record["input_ids"],
        attention_mask=record["attention_mask"],
        capture_routing=True,
    )
    if not isinstance(output, dict) or "logits" not in output or "routes" not in output:
        raise TypeError("candidate runner must return a dict containing logits and routes")
    if not isinstance(output["routes"], dict):
        raise TypeError("candidate routes must map router names to top-k indices")
    return output["logits"], output["routes"]


def _aligned_candidate_logits(candidate: Any, reference: Any) -> Any:
    if candidate.ndim != 3 or reference.ndim != 3:
        raise ValueError("reference and candidate logits must have shape [batch, tokens, vocab]")
    if candidate.shape[0] != reference.shape[0] or candidate.shape[2] != reference.shape[2]:
        raise ValueError("reference and candidate batch or vocabulary dimensions differ")
    if candidate.shape[1] < reference.shape[1]:
        raise ValueError("candidate returned fewer logit positions than the reference")
    return candidate[:, -reference.shape[1] :, :].float().cpu()


def _cpu_routes(routes: dict[str, Any]) -> dict[str, Any]:
    converted = {}
    for name, route in routes.items():
        if not hasattr(route, "detach"):
            raise TypeError(f"candidate route {name} is not a tensor")
        converted[name] = route.detach().cpu()
    return converted


def _validate_route_keys(reference: dict[str, Any], candidate: dict[str, Any]) -> None:
    if set(reference) != set(candidate):
        missing = sorted(set(reference) - set(candidate))
        extra = sorted(set(candidate) - set(reference))
        raise ValueError(f"router key mismatch; missing={missing}, extra={extra}")


def _covered_routes(routes: dict[str, Any], keys: list[str]) -> dict[str, Any]:
    missing = sorted(set(keys) - set(routes))
    if missing:
        raise ValueError(f"coverage names router keys absent from artifact: {missing}")
    return {key: routes[key] for key in keys}


def _route_decision_count(routes: dict[str, Any]) -> int:
    return sum(int(route.reshape(-1, route.shape[-1]).shape[0]) for route in routes.values())


@APP.function(
    image=BENCH_IMAGE,
    gpu=DEFAULT_GPU,
    volumes={VOL: KIMI_LINEAR_WEIGHTS},
    timeout=60 * 60 * 12,
    cpu=8.0,
    memory=65536,
)
def compare_engine(
    engine_factory: str,
    requested_gpu: str = DEFAULT_GPU,
    max_mean_kl_nats: float = MAX_MEAN_KL_NATS,
    min_top1_agreement: float = MIN_TOP1_AGREEMENT,
    min_routing_agreement: float = MIN_ROUTING_AGREEMENT,
    max_abs_perplexity_relative_delta: float = MAX_ABS_PERPLEXITY_RELATIVE_DELTA,
) -> dict[str, Any]:
    """Run the candidate on saved inputs, measure it, and render artifacts."""
    import torch

    if "/root" not in sys.path:
        sys.path.insert(0, "/root")
    from engine.bench.adapters import load_candidate_factory, require_candidate_coverage
    from engine.bench.artifacts import validate_reference_artifact
    from engine.bench.metrics import (
        perplexity,
        routing_agreement,
        token_kl_divergence,
        top1_agreement,
    )
    from engine.bench.results import render_measured_results
    from engine.k3ref.dequant import dequantize_mxfp4

    started = time.time()
    KIMI_LINEAR_WEIGHTS.reload()
    if not os.path.exists(REFERENCE_PATH):
        raise FileNotFoundError(f"{REFERENCE_PATH} is missing; run the reference action first")
    reference = torch.load(REFERENCE_PATH, map_location="cpu", weights_only=False)
    reference = validate_reference_artifact(
        reference,
        schema_version=SCHEMA_VERSION,
        model_id=MODEL_ID,
        prompt_fingerprint=_prompt_fingerprint(),
    )
    if _resolved_snapshot_revision(reference["snapshot_path"]) != reference["resolved_revision"]:
        raise ValueError("reference artifact snapshot path and resolved revision disagree")
    router_config = reference["router_config"]
    experts_per_token = router_config["experts_per_token"]

    factory = load_candidate_factory(engine_factory)
    runner = factory(
        model_id=reference["model_id"],
        revision=reference["resolved_revision"],
        cache_dir=HF_CACHE,
        snapshot_path=reference["snapshot_path"],
        dequantize_mxfp4=dequantize_mxfp4,
    )
    coverage = require_candidate_coverage(runner)
    if coverage.get("model_id") != reference["model_id"]:
        raise ValueError("candidate coverage model does not match the reference artifact")
    if coverage.get("resolved_revision") != reference["resolved_revision"]:
        raise ValueError("candidate coverage revision does not match the reference artifact")
    measured_router_keys = list(coverage["measured_router_keys"])

    prompt_results = []
    all_reference_logits = []
    all_candidate_logits = []
    all_reference_routes: dict[str, list[Any]] = {}
    all_candidate_routes: dict[str, list[Any]] = {}

    def add_routes(reference_routes: dict[str, Any], candidate_routes: dict[str, Any]) -> None:
        selected_reference = _covered_routes(reference_routes, measured_router_keys)
        selected_candidate = _covered_routes(candidate_routes, measured_router_keys)
        for name in measured_router_keys:
            all_reference_routes.setdefault(name, []).append(selected_reference[name])
            all_candidate_routes.setdefault(name, []).append(selected_candidate[name])

    for record in reference["prompt_records"]:
        candidate_logits, candidate_routes = _candidate_output(runner, record)
        candidate_logits = _aligned_candidate_logits(candidate_logits, record["logits"])
        candidate_routes = _cpu_routes(candidate_routes)
        _validate_route_keys(record["routes"], candidate_routes)
        selected_reference = _covered_routes(record["routes"], measured_router_keys)
        selected_candidate = _covered_routes(candidate_routes, measured_router_keys)
        route_total = _route_decision_count(selected_reference)
        route_value = routing_agreement(
            selected_reference,
            selected_candidate,
            expected_experts_per_token=experts_per_token,
        )
        token_count = int(record["logits"].shape[0] * record["logits"].shape[1])
        top1_value = top1_agreement(record["logits"], candidate_logits)
        kl_value = token_kl_divergence(record["logits"], candidate_logits)
        prompt_results.append(
            {
                "id": record["id"],
                "mean_kl_nats": kl_value,
                "mean_kl_arithmetic": (
                    f"{kl_value * token_count:.12g} total token KL / "
                    f"{token_count} positions = {kl_value:.12g}"
                ),
                "top1_agreement": top1_value,
                "top1_arithmetic": (
                    f"{round(top1_value * token_count)} matches / "
                    f"{token_count} positions = {top1_value:.12g}"
                ),
                "routing_agreement": route_value,
                "routing_arithmetic": (
                    f"{round(route_value * route_total)} matches / "
                    f"{route_total} covered token-layer decisions = {route_value:.12g}"
                ),
            }
        )
        all_reference_logits.append(record["logits"])
        all_candidate_logits.append(candidate_logits)
        add_routes(record["routes"], candidate_routes)

    heldout_reference_logits = []
    heldout_candidate_logits = []
    heldout_targets = []
    for record in reference["heldout_records"]:
        candidate_logits, candidate_routes = _candidate_output(runner, record)
        if candidate_logits.shape[1] == record["input_ids"].shape[1]:
            candidate_logits = candidate_logits[:, :-1, :]
        candidate_logits = _aligned_candidate_logits(candidate_logits, record["logits"])
        candidate_routes = _cpu_routes(candidate_routes)
        _validate_route_keys(record["routes"], candidate_routes)
        add_routes(record["routes"], candidate_routes)
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
    comparison_seconds = round(time.time() - started, 1)
    report = {
        "schema_version": SCHEMA_VERSION,
        "model_id": reference["model_id"],
        "resolved_revision": reference["resolved_revision"],
        "engine_factory": engine_factory,
        "compared_at_utc": datetime.now(timezone.utc).isoformat(),
        "requested_gpu": requested_gpu,
        "hardware": _gpu_inventory(requested_gpu),
        "prompt_fingerprint": reference["prompt_fingerprint"],
        "router_config": router_config,
        "metrics": {
            "mean_token_kl_nats": mean_kl,
            "mean_token_kl_arithmetic": (
                f"{mean_kl * prompt_token_count:.12g} total token KL / "
                f"{prompt_token_count} positions = {mean_kl:.12g}"
            ),
            "top1_agreement": top1,
            "top1_arithmetic": (
                f"{round(top1 * prompt_token_count)} matches / "
                f"{prompt_token_count} positions = {top1:.12g}"
            ),
            "routing_agreement": routing,
            "routing_arithmetic": (
                f"{round(routing * route_total)} matches / "
                f"{route_total} covered token-layer decisions = {routing:.12g}"
            ),
            "reference_perplexity": reference_ppl,
            "candidate_perplexity": candidate_ppl,
            "perplexity_arithmetic": (
                f"exp(mean negative log likelihood over {prediction_count} predictions); "
                f"relative delta = ({candidate_ppl:.12g} - {reference_ppl:.12g}) / "
                f"{reference_ppl:.12g} = {ppl_relative_delta:.12g}"
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
        "coverage": coverage,
        "download_seconds": reference["download_seconds"],
        "reference_seconds": reference["reference_seconds"],
        "comparison_seconds": comparison_seconds,
    }
    results_markdown, ledger_markdown = render_measured_results(report)
    _atomic_json(REPORT_PATH, report)
    _atomic_text(VOLUME_RESULTS_PATH, results_markdown)
    _atomic_text(VOLUME_LEDGER_PATH, ledger_markdown)
    KIMI_LINEAR_WEIGHTS.commit()
    print(json.dumps(report, indent=2))
    return {
        "report": report,
        "results_markdown": results_markdown,
        "ledger_markdown": ledger_markdown,
    }


@APP.local_entrypoint()
def main(
    action: str = "reference",
    gpu: str = DEFAULT_GPU,
    revision: str = "main",
    engine_factory: str = DEFAULT_ENGINE_FACTORY,
) -> None:
    if action == "download":
        download_model.remote(revision=revision)
        return

    selected_gpu = _validated_gpu(gpu)
    if action == "reference":
        capture_reference.with_options(gpu=selected_gpu).remote(
            revision=revision,
            requested_gpu=selected_gpu,
        )
    elif action == "compare":
        if not engine_factory:
            raise ValueError("--engine-factory is required; no reference fallback is allowed")
        payload = compare_engine.with_options(gpu=selected_gpu).remote(
            engine_factory=engine_factory,
            requested_gpu=selected_gpu,
        )
        LOCAL_RESULTS_PATH.write_text(payload["results_markdown"], encoding="utf-8")
        LOCAL_LEDGER_PATH.write_text(payload["ledger_markdown"], encoding="utf-8")
        print(f"wrote {LOCAL_RESULTS_PATH} and {LOCAL_LEDGER_PATH}")
    else:
        raise ValueError("action must be download, reference, or compare")
