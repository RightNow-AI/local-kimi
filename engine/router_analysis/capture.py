"""Non-invasive capture through the engine's existing auxiliary model output."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch

from .records import EncodedPrompt, LayerRoutingTrace, PromptRoutingTrace, RoutingRun

EXPECTED_ROUTER_CONFIG = {
    "num_experts": 256,
    "num_experts_per_token": 8,
    "activation": "sigmoid",
    "renormalize": True,
    "routed_scaling_factor": 2.446,
    "use_grouped_topk": True,
    "num_expert_group": 1,
    "topk_group": 1,
}


def router_config_from_model(model: object) -> dict[str, object]:
    config = model.config
    captured = {
        "num_experts": int(config.num_experts),
        "num_experts_per_token": int(config.num_experts_per_token),
        "activation": str(config.moe_router_activation_func),
        "renormalize": bool(config.moe_renormalize),
        "routed_scaling_factor": float(config.routed_scaling_factor),
        "use_grouped_topk": bool(config.use_grouped_topk),
        "num_expert_group": int(config.num_expert_group),
        "topk_group": int(config.topk_group),
    }
    if captured != EXPECTED_ROUTER_CONFIG:
        raise ValueError(
            "loaded checkpoint router configuration differs from the pinned "
            f"Kimi-Linear router: {captured}"
        )
    return captured


def capture_routing_run(
    model: object,
    *,
    checkpoint: str,
    prompt_set_sha256: str,
    prompts: Sequence[EncodedPrompt],
    device: torch.device,
) -> RoutingRun:
    """Run every prompt and copy only router IDs and MoE weights to CPU arrays."""

    router_config = router_config_from_model(model)
    prompt_traces = []
    with torch.inference_mode():
        for prompt in prompts:
            input_ids = torch.tensor(
                [prompt.token_ids],
                dtype=torch.long,
                device=device,
            )
            output = model(input_ids=input_ids)
            layers = []
            for layer_index, (expert_ids, expert_weights) in enumerate(
                zip(output.router_indices, output.router_weights, strict=True)
            ):
                if expert_ids.ndim != 2 or expert_weights.ndim != 2:
                    raise ValueError("engine router output must have shape [tokens, top_k]")
                if expert_ids.shape != expert_weights.shape:
                    raise ValueError("engine router ID and weight shapes differ")
                if expert_ids.shape[1] == 0:
                    if expert_weights.numel() != 0:
                        raise ValueError("dense layer returned router weights without IDs")
                    continue
                if expert_ids.shape != (
                    len(prompt.token_ids),
                    router_config["num_experts_per_token"],
                ):
                    raise ValueError(
                        f"prompt {prompt.prompt_id} layer {layer_index} returned "
                        f"unexpected router shape {tuple(expert_ids.shape)}"
                    )
                ids_cpu = (
                    expert_ids.detach()
                    .to(device="cpu", dtype=torch.int16)
                    .numpy()
                    .copy()
                )
                weights_cpu = (
                    expert_weights.detach()
                    .to(device="cpu", dtype=torch.float32)
                    .numpy()
                    .copy()
                )
                layers.append(
                    LayerRoutingTrace(
                        layer_index=layer_index,
                        expert_ids=np.asarray(ids_cpu),
                        expert_weights=np.asarray(weights_cpu),
                    )
                )
            prompt_traces.append(
                PromptRoutingTrace(
                    prompt_id=prompt.prompt_id,
                    token_ids=prompt.token_ids,
                    layers=tuple(layers),
                )
            )
            del output, input_ids
    return RoutingRun(
        checkpoint=checkpoint,
        prompt_set_sha256=prompt_set_sha256,
        prompts=tuple(prompt_traces),
        router_config=router_config,
    )
