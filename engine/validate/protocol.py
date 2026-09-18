"""Build the checkpoint-locked input protocol shared by both implementations."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from engine.klinear.weights import CheckpointKind, detect_checkpoint_kind
from engine.serve.contracts import ChatPrompt
from engine.serve.klinear_engine import KimiChatTokenizer
from engine.validate.prompts import (
    build_validation_prompt_set,
    validation_prompt_set_sha256,
)
from engine.validate.thresholds import ENGINE_PARITY_V1, INTERPRETATION_RULES

PROTOCOL_SCHEMA = "runinfra.kimi_linear.engine_parity_protocol.v1"
SEED = 20260729
MAX_MODEL_LEN = 8192


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return value


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def checkpoint_record(directory: Path) -> dict[str, Any]:
    resolved = directory.resolve(strict=True)
    if not resolved.is_dir():
        raise NotADirectoryError(resolved)
    config_path = resolved / "config.json"
    index_path = resolved / "model.safetensors.index.json"
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    if not index_path.is_file():
        raise FileNotFoundError(index_path)
    index = read_json(index_path)
    kind = detect_checkpoint_kind(index)
    if kind is not CheckpointKind.BF16:
        raise ValueError(
            "engine parity must use the source BF16 checkpoint, never a quantized artifact"
        )
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError("checkpoint index has no non-empty weight_map")
    return {
        "directory": str(resolved),
        "kind": kind.value,
        "config_sha256": sha256_file(config_path),
        "index_sha256": sha256_file(index_path),
        "tensor_count": len(weight_map),
        "declared_total_size_bytes": (index.get("metadata") or {}).get("total_size"),
    }


def build_protocol(checkpoint_dir: Path) -> dict[str, Any]:
    checkpoint = checkpoint_record(checkpoint_dir)
    tokenizer = KimiChatTokenizer.from_directory(checkpoint["directory"])
    prompts = build_validation_prompt_set()
    prompt_records = []
    for prompt in prompts:
        token_ids = tokenizer.encode_prompt(
            ChatPrompt(messages=({"role": "user", "content": prompt.text},))
        )
        if len(token_ids) + ENGINE_PARITY_V1.greedy_max_tokens > MAX_MODEL_LEN:
            raise ValueError(
                f"prompt {prompt.prompt_id} plus generation exceeds max model length"
            )
        prompt_records.append(
            {
                "prompt_id": prompt.prompt_id,
                "category": prompt.category,
                "text_sha256": hashlib.sha256(prompt.text.encode("utf-8")).hexdigest(),
                "token_ids": token_ids,
            }
        )

    measurement = {
        "checkpoint": checkpoint,
        "prompt_set_sha256": validation_prompt_set_sha256(prompts),
        "prompt_token_ids_sha256": sha256_json(
            [record["token_ids"] for record in prompt_records]
        ),
        "prompt_count": len(prompt_records),
        "greedy": {
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": 0,
            "max_tokens": ENGINE_PARITY_V1.greedy_max_tokens,
            "ignore_eos": True,
        },
        "first_token_distribution": {
            "positions_per_prompt": 1,
            "scope": "exact full vocabulary",
            "kl_direction": "KL(vLLM reference || engine.klinear candidate)",
        },
        "max_model_len": MAX_MODEL_LEN,
        "seed": SEED,
        "threshold": ENGINE_PARITY_V1.as_dict(),
        "interpretation_rules": list(INTERPRETATION_RULES),
    }
    protocol = {
        "schema_version": PROTOCOL_SCHEMA,
        "created_at": utc_now(),
        "measurement": measurement,
        "prompts": prompt_records,
    }
    protocol["protocol_fingerprint"] = sha256_json(protocol)
    return protocol


__all__ = [
    "MAX_MODEL_LEN",
    "PROTOCOL_SCHEMA",
    "SEED",
    "build_protocol",
    "checkpoint_record",
    "read_json",
    "sha256_file",
    "sha256_json",
    "utc_now",
    "write_json",
]
