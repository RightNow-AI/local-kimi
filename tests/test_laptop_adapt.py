from engine.k3ref.config import K3LayerConfig
from engine.k3ref.manifest import expected_layer_tensor_shapes
from engine.laptop.adapt import adapt_from_header


def _kimi_linear_config():
    return {
        "model_type": "kimi_linear",
        "vocab_size": 163_840,
        "hidden_size": 2_304,
        "intermediate_size": 9_216,
        "num_hidden_layers": 27,
        "num_attention_heads": 32,
        "num_key_value_heads": 32,
        "q_lora_rank": None,
        "kv_lora_rank": 512,
        "qk_nope_head_dim": 128,
        "qk_rope_head_dim": 64,
        "v_head_dim": 128,
        "mla_use_output_gate": False,
        "linear_attn_config": {
            "head_dim": 128,
            "num_heads": 32,
            "short_conv_kernel_size": 4,
            "gate_lower_bound": None,
            "use_full_rank_gate": False,
            "full_attn_layers": [4, 8, 12, 16, 20, 24, 27],
        },
        "moe_intermediate_size": 1_024,
        "num_experts": 256,
        "num_experts_per_token": 8,
        "num_shared_experts": 1,
        "num_expert_group": 1,
        "topk_group": 1,
        "first_k_dense_replace": 1,
        "moe_layer_freq": 1,
        "moe_renormalize": True,
        "routed_scaling_factor": 1.0,
        "rms_norm_eps": 1e-6,
    }


def test_kimi_header_builds_model_specific_manifest_and_names_loader_gaps():
    payload = _kimi_linear_config()
    config = K3LayerConfig.from_mapping(payload)
    header = {
        f"model.layers.1.{name}": {
            "shape": list(options[0]),
            "dtype": "BF16",
            "data_offsets": [0, 0],
        }
        for name, options in expected_layer_tensor_shapes(config, 1).items()
    }
    header["model.layers.1.self_attn.A_log"]["shape"] = [32]
    for projection, shape in {
        "w1": (1_024, 2_304),
        "w2": (2_304, 1_024),
        "w3": (1_024, 2_304),
    }.items():
        header[f"model.layers.1.block_sparse_moe.experts.0.{projection}.weight"] = {
            "shape": list(shape),
            "dtype": "BF16",
            "data_offsets": [0, 0],
        }

    adaptation = adapt_from_header(payload, header, layer_idx=1)
    requirements = "\n".join(adaptation.required_engine_changes)

    assert adaptation.existing_loader_is_compatible is False
    assert "model-specific manifest" in requirements
    assert "low-rank KDA" in requirements
    assert "per-head KDA A_log" in requirements
    assert "direct MLA q_proj" in requirements
    assert "non-latent routed experts" in requirements
    assert "ordinary expert weights" in requirements
