"""Built-in mixed-path candidate for Kimi-Linear quality measurement.

The adapter keeps HuggingFace as the model shell, then replaces every compatible
KDA and latent-MoE module with the validated plain-PyTorch implementations in
``engine.k3ref``. Coverage is explicit because embeddings, dense layers, MLA,
residual plumbing, final normalization, and the LM head remain HuggingFace.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import torch
from torch import nn

from engine.k3ref.attention import KDAAttention
from engine.k3ref.config import K3LayerConfig
from engine.k3ref.moe import LatentMoE
from engine.k3ref.router import K3Router


class _KDAForwardAdapter(nn.Module):
    """Present the Moonshot attention call shape around the engine KDA path."""

    def __init__(self, core: KDAAttention) -> None:
        super().__init__()
        self.core = core

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        cache_params: Any | None = None,
        **_kwargs: Any,
    ) -> torch.Tensor:
        if cache_params is not None:
            raise NotImplementedError("the benchmark candidate measures non-cached forwards only")
        return self.core(hidden_states, attention_mask=attention_mask)


def _first_parameter_dtype(module: nn.Module) -> torch.dtype:
    try:
        return next(module.parameters()).dtype
    except StopIteration as exc:
        raise ValueError(f"{module.__class__.__name__} has no parameters") from exc


def _assign_state(target: nn.Module, source: nn.Module, component: str) -> None:
    source_state = source.state_dict()
    target_state = target.state_dict()
    missing = sorted(set(target_state) - set(source_state))
    extra = sorted(set(source_state) - set(target_state))
    mismatched = sorted(
        name
        for name in set(target_state) & set(source_state)
        if tuple(target_state[name].shape) != tuple(source_state[name].shape)
    )
    if missing or extra or mismatched:
        raise ValueError(
            f"{component} state mismatch: missing={missing}, extra={extra}, "
            f"shape_mismatch={mismatched}"
        )
    try:
        target.load_state_dict(source_state, strict=True, assign=True)
    except RuntimeError as exc:
        raise ValueError(f"{component} state assignment failed: {exc}") from exc
    target.eval()


def _layer_config(config: Any) -> K3LayerConfig:
    linear = config.linear_attn_config
    return K3LayerConfig(
        hidden_size=config.hidden_size,
        num_attention_heads=config.num_attention_heads,
        num_key_value_heads=config.num_key_value_heads,
        q_lora_rank=config.q_lora_rank,
        kv_lora_rank=config.kv_lora_rank,
        qk_nope_head_dim=config.qk_nope_head_dim,
        qk_rope_head_dim=config.qk_rope_head_dim,
        v_head_dim=config.v_head_dim,
        mla_use_output_gate=config.mla_use_output_gate,
        kda_head_dim=linear["head_dim"],
        kda_num_heads=linear["num_heads"],
        short_conv_kernel_size=linear["short_conv_kernel_size"],
        kda_gate_lower_bound=linear.get("gate_lower_bound"),
        routed_expert_hidden_size=config.routed_expert_hidden_size,
        moe_intermediate_size=config.moe_intermediate_size,
        num_experts=config.num_experts,
        num_experts_per_token=config.num_experts_per_token,
        num_shared_experts=config.num_shared_experts or 0,
        num_expert_group=getattr(config, "num_expert_group", 1),
        topk_group=getattr(config, "topk_group", 1),
        moe_renormalize=config.moe_renormalize,
        routed_scaling_factor=config.routed_scaling_factor,
        rms_norm_eps=config.rms_norm_eps,
        activation_situ_beta=config.activation_situ_beta,
        activation_situ_linear_beta=getattr(config, "activation_situ_linear_beta", None),
        attn_res_block_size=getattr(config, "attn_res_block_size", None),
        full_attention_layers=tuple(linear["full_attn_layers"]),
    )


def _replacement_kda(source: nn.Module, config: K3LayerConfig) -> _KDAForwardAdapter:
    core = KDAAttention(
        config.hidden_size,
        config.kda_num_heads,
        config.kda_head_dim,
        conv_size=config.short_conv_kernel_size,
        gate_lower_bound=config.kda_gate_lower_bound,
        rms_norm_eps=config.rms_norm_eps,
        device="meta",
        dtype=_first_parameter_dtype(source),
    )
    _assign_state(core, source, "KDA")
    return _KDAForwardAdapter(core)


def _replacement_moe(source: nn.Module, config: Any) -> LatentMoE:
    if not getattr(source, "use_latent_moe", False):
        raise ValueError("engine LatentMoE does not cover a non-latent expert block")
    if not getattr(source, "latent_moe_use_norm", False):
        raise ValueError("engine LatentMoE currently requires the routed latent norm")
    replacement = LatentMoE(
        hidden_size=config.hidden_size,
        latent_size=config.routed_expert_hidden_size,
        expert_intermediate_size=config.moe_intermediate_size,
        num_experts=config.num_experts,
        top_k=config.num_experts_per_token,
        num_shared_experts=config.num_shared_experts or 0,
        num_expert_group=getattr(config, "num_expert_group", 1),
        topk_group=getattr(config, "topk_group", 1),
        renormalize=config.moe_renormalize,
        routed_scaling_factor=config.routed_scaling_factor,
        rms_norm_eps=config.rms_norm_eps,
        situ_beta=config.activation_situ_beta,
        situ_linear_beta=getattr(config, "activation_situ_linear_beta", None),
        device="meta",
        dtype=_first_parameter_dtype(source),
    )
    replacement.gate.activation = config.moe_router_activation_func
    _assign_state(replacement, source, "latent MoE")
    return replacement


def _replacement_router(source: nn.Module, config: Any) -> K3Router:
    replacement = K3Router(
        hidden_size=config.hidden_size,
        num_experts=config.num_experts,
        top_k=config.num_experts_per_token,
        num_expert_group=getattr(config, "num_expert_group", 1),
        topk_group=getattr(config, "topk_group", 1),
        renormalize=config.moe_renormalize,
        routed_scaling_factor=config.routed_scaling_factor,
        activation=config.moe_router_activation_func,
        device="meta",
        dtype=_first_parameter_dtype(source),
    )
    _assign_state(replacement, source, "router")
    return replacement


def _router_modules(model: nn.Module) -> dict[str, nn.Module]:
    routers = {
        name: module
        for name, module in model.named_modules()
        if name.endswith(".gate")
        and hasattr(module, "top_k")
        and hasattr(module, "num_experts")
        and hasattr(module, "e_score_correction_bias")
    }
    if not routers:
        raise RuntimeError("candidate model exposes no Kimi router modules")
    return routers


class KimiLinearK3RefRunner:
    """Run a disclosed mixed HuggingFace and engine.k3ref candidate graph."""

    def __init__(self, model: nn.Module, coverage: dict[str, Any]) -> None:
        self.model = model
        self.coverage = coverage

    def run(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        capture_routing: bool,
    ) -> dict[str, Any]:
        if not capture_routing:
            raise ValueError("the benchmark requires routing capture")
        captured: dict[str, list[torch.Tensor]] = {}
        handles = []

        def hook_for(name: str):
            def hook(_module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
                if not isinstance(output, tuple) or not output:
                    raise RuntimeError(f"candidate router {name} did not return expert indices")
                captured.setdefault(name, []).append(
                    output[0].detach().to("cpu", dtype=torch.int16)
                )

            return hook

        routers = _router_modules(self.model)
        for name, module in routers.items():
            handles.append(module.register_forward_hook(hook_for(name)))
        try:
            device = self.model.get_input_embeddings().weight.device
            with torch.inference_mode():
                output = self.model(
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
            raise RuntimeError(f"candidate produced no routing capture for {missing}")
        return {"logits": output.logits.detach(), "routes": routes}


def build_kimi_linear_runner(
    *,
    model_id: str,
    revision: str,
    cache_dir: str,
    snapshot_path: str,
    dequantize_mxfp4: Any,
) -> KimiLinearK3RefRunner:
    """Build the in-repo partial candidate, or raise without a fallback."""
    from transformers import AutoModelForCausalLM

    if re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise ValueError("candidate requires an immutable 40-character revision")
    snapshot = Path(snapshot_path)
    if not snapshot.is_dir() or snapshot.name != revision:
        raise FileNotFoundError("candidate snapshot path does not match the resolved revision")
    if dequantize_mxfp4 is None:
        raise ValueError("candidate factory did not receive the canonical MXFP4 decoder")

    model = AutoModelForCausalLM.from_pretrained(
        str(snapshot),
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
        attn_implementation="flash_attention_2",
        cache_dir=cache_dir,
    )
    model.eval()
    if model.config.model_type != "kimi_linear":
        raise ValueError(f"candidate expected kimi_linear, got {model.config.model_type!r}")

    config = model.config
    engine_config = _layer_config(config)
    layers = getattr(getattr(model, "model", None), "layers", None)
    if layers is None:
        raise TypeError("candidate model does not expose model.layers")

    kda_layers: list[int] = []
    moe_layers: list[int] = []
    router_only_layers: list[int] = []
    uncovered: list[dict[str, Any]] = []
    replaced_gate_ids: set[int] = set()

    for layer_index, layer in enumerate(layers):
        if getattr(layer, "is_linear_attn", False):
            try:
                layer.self_attn = _replacement_kda(layer.self_attn, engine_config)
                kda_layers.append(layer_index)
            except (KeyError, TypeError, ValueError) as exc:
                uncovered.append(
                    {"layer": layer_index, "component": "kda", "reason": str(exc)}
                )

        source_moe = getattr(layer, "block_sparse_moe", None)
        if source_moe is None:
            continue
        try:
            replacement_moe = _replacement_moe(source_moe, config)
            layer.block_sparse_moe = replacement_moe
            replaced_gate_ids.add(id(replacement_moe.gate))
            moe_layers.append(layer_index)
        except (KeyError, TypeError, ValueError) as moe_exc:
            try:
                replacement_gate = _replacement_router(source_moe.gate, config)
                source_moe.gate = replacement_gate
                replaced_gate_ids.add(id(replacement_gate))
                router_only_layers.append(layer_index)
                uncovered.append(
                    {
                        "layer": layer_index,
                        "component": "latent_moe",
                        "reason": str(moe_exc),
                        "measured_instead": "engine.k3ref router only",
                    }
                )
            except (KeyError, TypeError, ValueError) as router_exc:
                uncovered.append(
                    {
                        "layer": layer_index,
                        "component": "latent_moe_and_router",
                        "reason": f"MoE: {moe_exc}; router: {router_exc}",
                    }
                )

    routers = _router_modules(model)
    measured_router_keys = sorted(
        name for name, module in routers.items() if id(module) in replaced_gate_ids
    )
    if not kda_layers and not moe_layers and not measured_router_keys:
        raise RuntimeError("candidate adapter could not install any engine.k3ref component")

    coverage = {
        "candidate_label": "engine.k3ref mixed-path Kimi-Linear candidate",
        "full_model_candidate": False,
        "scope": "partial",
        "model_id": model_id,
        "resolved_revision": revision,
        "kda_layers_replaced": kda_layers,
        "latent_moe_layers_replaced": moe_layers,
        "router_only_layers_replaced": router_only_layers,
        "measured_router_keys": measured_router_keys,
        "uncovered": uncovered,
        "huggingface_components_retained": [
            "token embeddings",
            "dense decoder layers",
            "MLA attention",
            "decoder residual plumbing and layer norms",
            "final normalization",
            "LM head",
        ],
        "logit_interpretation": (
            "end-to-end logits from a mixed graph containing the listed engine.k3ref "
            "replacements and the listed retained HuggingFace components"
        ),
        "mxfp4_decoder_used": False,
    }
    return KimiLinearK3RefRunner(model, coverage)
