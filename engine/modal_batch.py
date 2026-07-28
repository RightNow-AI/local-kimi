"""Modal harness for measured Kimi K3 router unions under batched generation.

The analytic model lives in ``engine/batching/union_model.py``. This file is the
measurement gate. It loads a real model from the existing ``k3-weights`` Modal
Volume, hooks the router modules, and records the actual selected expert union
for each layer and decode step.

Throughput and routing are collected in separate matched passes. Router hooks
force device synchronization and would otherwise contaminate the timing result.
The timed pass uses only a one-shot decode-start hook. The trace pass records
the selected indices but its elapsed time is never reported as throughput.

Example, once a complete local model exists at /weights/Kimi-K3:

    modal run engine/modal_batch.py --concurrencies 1,2,4,8,16,32,64,128

For a custom engine, set ``K3_BATCH_LOADER=package.module:function``. The
function must accept a model path and return ``(model, tokenizer)``. The model
must expose ``generate`` and router modules whose forward output includes the
actual integer top-k expert indices.
"""

from __future__ import annotations

import importlib
import json
import math
import os
import re
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import modal


APP = modal.App("k3-engine-batch")
IMAGE = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "accelerate>=1.0",
        "numpy>=2.0",
        "safetensors>=0.5",
        "sentencepiece>=0.2",
        "torch>=2.5",
        "transformers>=4.53",
    )
)
WEIGHTS = modal.Volume.from_name("k3-weights", create_if_missing=True)
VOL = "/weights"

TOTAL_EXPERTS = 896
EXPERTS_PER_TOKEN = 16
MOE_LAYERS = 92
EXPERT_BYTES = 17_547_264
LAYER_PATTERN = re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)")

DEFAULT_PROMPTS = (
    "Implement a bounded async worker pool in Python with cancellation and tests.",
    "Review this API design for idempotency, retry safety, and race conditions.",
    "Write a TypeScript parser for a streaming JSON-lines protocol with backpressure.",
    "Diagnose a deadlock in a multithreaded cache and propose the smallest safe fix.",
    "Create a SQL migration that adds an online unique constraint without downtime.",
    "Refactor a CUDA benchmark harness so warmup and timed iterations cannot mix.",
    "Explain why this distributed lease can admit two owners and repair the protocol.",
    "Add property tests for a serializer that must preserve unknown fields exactly.",
)


def _parse_concurrencies(spec: str) -> tuple[int, ...]:
    values = tuple(int(item.strip()) for item in spec.split(",") if item.strip())
    if not values or any(value <= 0 for value in values):
        raise ValueError("concurrencies must be a comma-separated list of positive integers")
    return values


def _load_custom_loader(spec: str) -> Callable[[str], tuple[Any, Any]]:
    if ":" not in spec:
        raise ValueError("K3_BATCH_LOADER must have package.module:function form")
    module_name, function_name = spec.split(":", 1)
    module = importlib.import_module(module_name)
    loader = getattr(module, function_name)
    if not callable(loader):
        raise TypeError(f"custom loader {spec!r} is not callable")
    return loader


def _load_transformers(model_path: str) -> tuple[Any, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    offload = f"{VOL}/.batch-offload"
    os.makedirs(offload, exist_ok=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        local_files_only=True,
        low_cpu_mem_usage=True,
        offload_folder=offload,
        offload_state_dict=True,
        torch_dtype="auto",
        trust_remote_code=True,
    )
    model.eval()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return model, tokenizer


def _load_model(model_path: str) -> tuple[Any, Any]:
    custom = os.environ.get("K3_BATCH_LOADER")
    if custom:
        return _load_custom_loader(custom)(model_path)
    return _load_transformers(model_path)


def _module_layer(name: str) -> int | None:
    match = LAYER_PATTERN.search(name)
    return int(match.group(1)) if match else None


def _is_router_module(name: str, module: Any) -> bool:
    """Prefer full router modules, never their final linear projection."""
    class_name = type(module).__name__.lower()
    if class_name == "linear":
        return False
    router_name = "router" in class_name or "moegate" in class_name
    router_attributes = any(
        hasattr(module, attribute)
        for attribute in ("n_routed_experts", "num_experts", "top_k", "topk")
    )
    path_hint = name.endswith(".gate") or name.endswith(".router")
    return _module_layer(name) is not None and (router_name or (path_hint and router_attributes))


def _find_routers(model: Any) -> dict[int, tuple[str, Any]]:
    routers: dict[int, tuple[str, Any]] = {}
    duplicates: dict[int, list[str]] = defaultdict(list)
    for name, module in model.named_modules():
        if not _is_router_module(name, module):
            continue
        layer = _module_layer(name)
        if layer is None or layer == 0:
            continue
        if layer in routers:
            duplicates[layer].extend((routers[layer][0], name))
        else:
            routers[layer] = (name, module)
    if duplicates:
        detail = {layer: sorted(set(names)) for layer, names in duplicates.items()}
        raise RuntimeError(f"multiple router modules found for layers: {detail}")
    expected = set(range(1, MOE_LAYERS + 1))
    actual = set(routers)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise RuntimeError(
            f"expected routers for layers 1..{MOE_LAYERS}; missing={missing}, extra={extra}"
        )
    return routers


def _walk_values(value: Any):
    if isinstance(value, dict):
        preferred = (
            "topk_idx",
            "topk_indices",
            "expert_indices",
            "selected_experts",
            "routing_indices",
        )
        for key in preferred:
            if key in value:
                yield value[key]
        for key, item in value.items():
            if key not in preferred:
                yield from _walk_values(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _walk_values(item)
    else:
        yield value


def _extract_selected_indices(output: Any):
    import torch

    candidates = []
    integer_dtypes = {
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    }
    for value in _walk_values(output):
        if not torch.is_tensor(value) or value.dtype not in integer_dtypes:
            continue
        if value.numel() == 0 or value.numel() % EXPERTS_PER_TOKEN != 0:
            continue
        reshaped = value.reshape(-1, EXPERTS_PER_TOKEN)
        if int(reshaped.min().item()) < 0 or int(reshaped.max().item()) >= TOTAL_EXPERTS:
            continue
        candidates.append(reshaped)
    if len(candidates) != 1:
        shapes = [list(candidate.shape) for candidate in candidates]
        raise RuntimeError(
            "router hook must expose exactly one integer top-16 expert-index tensor; "
            f"found {len(candidates)} candidates with shapes {shapes}"
        )
    return candidates[0]


class _DecodeStartTimer:
    """Timestamp the second call of one router, which starts cached decoding."""

    def __init__(self) -> None:
        self.calls = 0
        self.decode_start: float | None = None

    def hook(self, _module: Any, _inputs: Any) -> None:
        self.calls += 1
        if self.calls == 2:
            self.decode_start = time.perf_counter()


class RouterTraceCollector:
    def __init__(self, routers: dict[int, tuple[str, Any]]) -> None:
        self.routers = routers
        self.calls: dict[int, int] = defaultdict(int)
        self.unions: dict[int, list[int]] = defaultdict(list)
        self.active_tokens: dict[int, list[int]] = defaultdict(list)
        self.counts: dict[int, list[int]] = {
            layer: [0] * TOTAL_EXPERTS for layer in routers
        }
        self.handles = []

    def _hook_for(self, layer: int):
        import torch

        def hook(_module: Any, _inputs: Any, output: Any) -> None:
            self.calls[layer] += 1
            if self.calls[layer] == 1:
                return  # The first call is prompt prefill, not decode.
            indices = _extract_selected_indices(output)
            self.unions[layer].append(int(torch.unique(indices).numel()))
            self.active_tokens[layer].append(int(indices.shape[0]))
            counts = torch.bincount(indices.flatten(), minlength=TOTAL_EXPERTS)
            host_counts = counts.detach().to("cpu").tolist()
            destination = self.counts[layer]
            for expert, count in enumerate(host_counts):
                destination[expert] += int(count)

        return hook

    def __enter__(self) -> "RouterTraceCollector":
        for layer, (_name, module) in self.routers.items():
            self.handles.append(module.register_forward_hook(self._hook_for(layer)))
        return self

    def __exit__(self, _exc_type: Any, _exc: Any, _traceback: Any) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    @staticmethod
    def _quantiles(values: list[int]) -> dict[str, float]:
        import numpy as np

        array = np.asarray(values, dtype=np.float64)
        return {
            "mean": float(array.mean()),
            "min": float(array.min()),
            "p05": float(np.quantile(array, 0.05)),
            "p50": float(np.quantile(array, 0.50)),
            "p95": float(np.quantile(array, 0.95)),
            "max": float(array.max()),
        }

    def summary(self, generated_tokens: int) -> dict:
        import numpy as np

        missing = [layer for layer in self.routers if not self.unions[layer]]
        if missing:
            raise RuntimeError(f"no decode router records for layers {missing}")
        all_unions = [value for layer in sorted(self.unions) for value in self.unions[layer]]
        total_union_bytes = sum(all_unions) * EXPERT_BYTES
        pooled_counts = np.sum(
            np.asarray([self.counts[layer] for layer in sorted(self.counts)], dtype=np.int64),
            axis=0,
        )
        probabilities = pooled_counts / pooled_counts.sum()
        positive = probabilities[probabilities > 0]
        entropy = float(-np.sum(positive * np.log(positive)))
        top = np.argsort(pooled_counts)[::-1][:32]
        return {
            "router_calls_including_prefill": {
                str(layer): self.calls[layer] for layer in sorted(self.calls)
            },
            "decode_records": len(all_unions),
            "union_across_layer_steps": self._quantiles(all_unions),
            "routed_traffic_gb_per_generated_token": (
                total_union_bytes / generated_tokens / 1e9
            ),
            "per_layer": {
                str(layer): {
                    "union": self._quantiles(self.unions[layer]),
                    "active_tokens": self._quantiles(self.active_tokens[layer]),
                }
                for layer in sorted(self.unions)
            },
            "pooled_expert_frequency": {
                "normalized_entropy": entropy / math.log(TOTAL_EXPERTS),
                "effective_experts": math.exp(entropy),
                "top_32": [
                    {
                        "expert": int(expert),
                        "selections": int(pooled_counts[expert]),
                        "fraction": float(probabilities[expert]),
                    }
                    for expert in top
                ],
            },
        }


def _prompts(concurrency: int) -> list[str]:
    return [DEFAULT_PROMPTS[index % len(DEFAULT_PROMPTS)] for index in range(concurrency)]


def _model_input_device(model: Any):
    try:
        return model.get_input_embeddings().weight.device
    except Exception:
        return next(model.parameters()).device


def _prepare_inputs(model: Any, tokenizer: Any, concurrency: int, max_prompt_tokens: int):
    encoded = tokenizer(
        _prompts(concurrency),
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_prompt_tokens,
    )
    device = _model_input_device(model)
    return {key: value.to(device) for key, value in encoded.items()}


def _synchronize() -> None:
    import torch

    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _generate(model: Any, tokenizer: Any, inputs: dict, new_tokens: int):
    import torch

    with torch.inference_mode():
        return model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=new_tokens,
            min_new_tokens=new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            use_cache=True,
        )


def _timed_pass(
    *,
    model: Any,
    tokenizer: Any,
    routers: dict[int, tuple[str, Any]],
    inputs: dict,
    new_tokens: int,
) -> dict:
    first_layer = min(routers)
    timer = _DecodeStartTimer()
    handle = routers[first_layer][1].register_forward_pre_hook(timer.hook)
    try:
        _synchronize()
        wall_start = time.perf_counter()
        output = _generate(model, tokenizer, inputs, new_tokens)
        _synchronize()
        end = time.perf_counter()
    finally:
        handle.remove()
    if timer.decode_start is None:
        raise RuntimeError("generation never reached a cached decode step")
    total_generated = int(
        output.shape[0] * (output.shape[1] - inputs["input_ids"].shape[1])
    )
    # Prefill produces the first new token. The second router call starts the
    # timed cached-decode region, so count only the remaining generated tokens.
    decoded = total_generated - int(output.shape[0])
    decode_seconds = end - timer.decode_start
    return {
        "total_generated_tokens": total_generated,
        "timed_decode_tokens": decoded,
        "full_generation_seconds": end - wall_start,
        "decode_seconds": decode_seconds,
        "aggregate_tokens_per_second": decoded / decode_seconds,
        "per_agent_tokens_per_second": decoded / decode_seconds / output.shape[0],
        "timing_note": (
            "Decode timer starts at the second call of layer-1 router. It excludes "
            "prompt prefill and includes the remainder of generation."
        ),
    }


def _trace_pass(
    *,
    model: Any,
    tokenizer: Any,
    routers: dict[int, tuple[str, Any]],
    inputs: dict,
    new_tokens: int,
) -> dict:
    with RouterTraceCollector(routers) as collector:
        output = _generate(model, tokenizer, inputs, new_tokens)
        _synchronize()
    total_generated = int(
        output.shape[0] * (output.shape[1] - inputs["input_ids"].shape[1])
    )
    decoded = total_generated - int(output.shape[0])
    summary = collector.summary(decoded)
    summary["total_generated_tokens"] = total_generated
    summary["traced_decode_tokens"] = decoded
    summary["trace_note"] = (
        "The first generated token is produced during prefill and is excluded "
        "from both traced router traffic and the decode-token denominator."
    )
    return summary


def _runtime_info() -> dict:
    import torch

    return {
        "cuda_available": torch.cuda.is_available(),
        "cuda_devices": [
            torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())
        ],
        "torch_version": torch.__version__,
    }


@APP.function(
    image=IMAGE,
    gpu="H200:8",
    volumes={VOL: WEIGHTS},
    timeout=60 * 60 * 12,
)
def measure_batch_curve(
    model_dir: str = "Kimi-K3",
    concurrencies: str = "1,2,4,8,16,32,64,128",
    new_tokens: int = 64,
    max_prompt_tokens: int = 512,
    warmup_tokens: int = 4,
) -> dict:
    """Measure matched uninstrumented throughput and instrumented router unions."""
    import torch

    model_path = f"{VOL}/{model_dir.strip('/')}"
    if not os.path.isdir(model_path):
        raise FileNotFoundError(
            f"{model_path} is missing; place a complete local model on k3-weights first"
        )
    if new_tokens < 2:
        raise ValueError("new_tokens must be at least 2 so decode follows prefill")
    values = _parse_concurrencies(concurrencies)
    model, tokenizer = _load_model(model_path)
    routers = _find_routers(model)

    results = {
        "status": "measured on the reported Modal runtime",
        "model_path": model_path,
        "measurement_mode": "static synchronized batch, matched timing and trace passes",
        "runtime": _runtime_info(),
        "router_modules": {
            str(layer): name for layer, (name, _module) in sorted(routers.items())
        },
        "constants": {
            "total_experts": TOTAL_EXPERTS,
            "experts_per_token": EXPERTS_PER_TOKEN,
            "moe_layers": MOE_LAYERS,
            "expert_bytes": EXPERT_BYTES,
        },
        "points": [],
        "limitations": [
            "Static synchronized batches do not model continuous slot refill.",
            "Modal throughput is not local EPYC plus RTX 5090 throughput.",
            "The local capacity model should consume measured unions, not Modal tok/s.",
            "Timed and trace passes are separate so tracing overhead cannot inflate latency.",
        ],
    }

    for concurrency in values:
        inputs = _prepare_inputs(model, tokenizer, concurrency, max_prompt_tokens)
        try:
            if warmup_tokens > 0:
                _generate(model, tokenizer, inputs, warmup_tokens)
                _synchronize()
            timing = _timed_pass(
                model=model,
                tokenizer=tokenizer,
                routers=routers,
                inputs=inputs,
                new_tokens=new_tokens,
            )
            point = {"concurrency": concurrency, "timing": timing}
            try:
                point["routing"] = _trace_pass(
                    model=model,
                    tokenizer=tokenizer,
                    routers=routers,
                    inputs=inputs,
                    new_tokens=new_tokens,
                )
            except RuntimeError as exc:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                point["routing_error"] = f"{type(exc).__name__}: {exc}"
        except RuntimeError as exc:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            point = {
                "concurrency": concurrency,
                "error": f"{type(exc).__name__}: {exc}",
            }
        results["points"].append(point)
        print(json.dumps(point, indent=2), flush=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = Path(VOL) / "batch-measurements"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"k3-batch-{timestamp}.json"
    results["output_path"] = str(output_path)
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    WEIGHTS.commit()
    print(f"wrote {output_path}")
    return results


@APP.local_entrypoint()
def main(
    model_dir: str = "Kimi-K3",
    concurrencies: str = "1,2,4,8,16,32,64,128",
    new_tokens: int = 64,
    max_prompt_tokens: int = 512,
    warmup_tokens: int = 4,
):
    measure_batch_curve.remote(
        model_dir=model_dir,
        concurrencies=concurrencies,
        new_tokens=new_tokens,
        max_prompt_tokens=max_prompt_tokens,
        warmup_tokens=warmup_tokens,
    )
