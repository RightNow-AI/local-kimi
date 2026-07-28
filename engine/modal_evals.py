"""Run capability evaluations against an already-served OpenAI endpoint.

Example:

    modal run engine/modal_evals.py \
        --base-url https://example.modal.run \
        --model-id moonshotai/Kimi-Linear-48B-A3B-Instruct \
        --endpoint-identity bf16-baseline \
        --output bf16-capability.json

Run the same command against the INT4 endpoint with a different endpoint identity,
then compare the two records with ``engine.evals.compare.compare_capability_records``.
"""

from __future__ import annotations

import json
from pathlib import Path

import modal

app = modal.App("kimi-linear-capability-evals")

IMAGE = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("httpx>=0.27")
    .add_local_dir(Path(__file__).parent / "evals", remote_path="/root/engine/evals")
    .add_local_dir(Path(__file__).parent.parent / "k3", remote_path="/root/k3")
)


@app.function(image=IMAGE, cpu=2.0, memory=4096, timeout=60 * 60 * 4)
def run_endpoint_evaluation(
    *,
    base_url: str,
    model_id: str,
    endpoint_identity: str | None = None,
    api_key: str | None = None,
    timeout_s: float = 180.0,
    instruction_max_tokens: int = 512,
    tool_max_tokens: int = 512,
) -> dict:
    from engine.evals.runner import run_capability_evaluation

    return run_capability_evaluation(
        base_url=base_url,
        model_id=model_id,
        endpoint_identity=endpoint_identity,
        api_key=api_key,
        timeout_s=timeout_s,
        instruction_max_tokens=instruction_max_tokens,
        tool_max_tokens=tool_max_tokens,
    )


@app.local_entrypoint()
def main(
    base_url: str,
    model_id: str,
    endpoint_identity: str = "",
    api_key: str = "",
    output: str = "",
    timeout_s: float = 180.0,
    instruction_max_tokens: int = 512,
    tool_max_tokens: int = 512,
) -> None:
    record = run_endpoint_evaluation.remote(
        base_url=base_url,
        model_id=model_id,
        endpoint_identity=endpoint_identity or None,
        api_key=api_key or None,
        timeout_s=timeout_s,
        instruction_max_tokens=instruction_max_tokens,
        tool_max_tokens=tool_max_tokens,
    )
    serialized = json.dumps(record, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    if output:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(serialized, encoding="utf-8")
    print(serialized, end="")


__all__ = ["app", "run_endpoint_evaluation"]
