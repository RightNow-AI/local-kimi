import json
import struct

import pytest
import torch

from engine.klinear.attention import KDAAttention
from engine.klinear.config import KLinearConfig
from engine.klinear.manifest import (
    BF16,
    F32,
    REAL_CHECKPOINT_MANIFEST,
    REAL_EXPERT_TEMPLATE_MANIFEST,
    REAL_KDA_ATTENTION_MANIFEST,
    REAL_MLA_ATTENTION_MANIFEST,
    REAL_MODEL_TENSOR_MANIFEST,
    REAL_UNRESOLVED_TENSORS,
)
from engine.klinear.weights import SafetensorIndexStore


def _tiny_config() -> KLinearConfig:
    return KLinearConfig(
        vocab_size=13,
        hidden_size=4,
        head_dim=2,
        intermediate_size=8,
        num_hidden_layers=3,
        num_attention_heads=2,
        num_key_value_heads=2,
        kv_lora_rank=2,
        qk_nope_head_dim=1,
        qk_rope_head_dim=1,
        v_head_dim=2,
        kda_num_heads=2,
        kda_head_dim=2,
        short_conv_kernel_size=2,
        num_experts=4,
        num_experts_per_token=2,
        num_shared_experts=1,
        moe_intermediate_size=3,
        first_k_dense_replace=1,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
        full_attention_layers=(3,),
        kda_layers=(1, 2),
    )


def test_layer_kind_resolver_is_total_and_uses_one_based_attention_membership():
    config = KLinearConfig()

    kinds = tuple(config.layer_kind(layer_idx) for layer_idx in range(27))

    assert kinds[0] == "dense"
    assert {index for index, kind in enumerate(kinds) if kind == "mla"} == {
        3,
        7,
        11,
        15,
        19,
        23,
        26,
    }
    assert kinds.count("dense") == 1
    assert kinds.count("kda") == 19
    assert kinds.count("mla") == 7
    assert config.attention_kind(0) == "kda"
    assert config.attention_kind(26) == "mla"
    with pytest.raises(IndexError):
        config.layer_kind(-1)
    with pytest.raises(IndexError):
        config.layer_kind(27)


def test_layer_kind_configuration_rejects_unclassified_or_overlapping_layers():
    base = _tiny_config()
    values = dict(base.__dict__)
    values["kda_layers"] = (1,)
    with pytest.raises(ValueError, match="classify every layer exactly once"):
        KLinearConfig(**values)

    values["kda_layers"] = (1, 3)
    with pytest.raises(ValueError, match="overlap"):
        KLinearConfig(**values)


def test_real_manifest_pins_every_indexed_tensor_name_and_observed_shape():
    assert len(REAL_CHECKPOINT_MANIFEST) == 20_493
    assert REAL_UNRESOLVED_TENSORS == ()
    assert REAL_MODEL_TENSOR_MANIFEST["model.embed_tokens.weight"].shape == (
        163_840,
        2_304,
    )
    assert REAL_MODEL_TENSOR_MANIFEST["model.norm.weight"].shape == (2_304,)
    assert REAL_MODEL_TENSOR_MANIFEST["lm_head.weight"].shape == (163_840, 2_304)
    assert REAL_KDA_ATTENTION_MANIFEST["self_attn.A_log"].shape == (1, 1, 32, 1)
    assert REAL_KDA_ATTENTION_MANIFEST["self_attn.A_log"].dtype == F32
    assert REAL_KDA_ATTENTION_MANIFEST["self_attn.dt_bias"].dtype == F32
    assert REAL_KDA_ATTENTION_MANIFEST["self_attn.q_conv1d.weight"].dtype == BF16
    assert REAL_MLA_ATTENTION_MANIFEST["self_attn.q_proj.weight"].shape == (
        6_144,
        2_304,
    )
    assert set(REAL_EXPERT_TEMPLATE_MANIFEST) == {
        "block_sparse_moe.experts.{expert}.w1.weight",
        "block_sparse_moe.experts.{expert}.w2.weight",
        "block_sparse_moe.experts.{expert}.w3.weight",
    }
    assert REAL_EXPERT_TEMPLATE_MANIFEST[
        "block_sparse_moe.experts.{expert}.w1.weight"
    ].shape == (1_024, 2_304)
    assert REAL_EXPERT_TEMPLATE_MANIFEST[
        "block_sparse_moe.experts.{expert}.w2.weight"
    ].shape == (2_304, 1_024)


def test_kda_recurrent_state_size_is_constant_across_sequence_lengths():
    attention = KDAAttention(
        hidden_size=4,
        num_heads=2,
        head_dim=2,
        conv_size=2,
        dtype=torch.float32,
    ).eval()

    _, short_state = attention(
        torch.randn(1, 1, 4),
        return_state=True,
    )
    _, long_state = attention(
        torch.randn(1, 9, 4),
        state=short_state,
        return_state=True,
    )

    assert short_state.recurrent.shape == (1, 2, 2, 2)
    assert long_state.recurrent.shape == short_state.recurrent.shape
    assert long_state.recurrent.numel() == short_state.recurrent.numel()
    assert long_state.q_conv.shape == short_state.q_conv.shape == (1, 4, 2)
    assert long_state.k_conv.shape == short_state.k_conv.shape == (1, 4, 2)
    assert long_state.v_conv.shape == short_state.v_conv.shape == (1, 4, 2)


def test_safetensor_index_store_reads_real_format_header_offsets(tmp_path):
    shard_name = "model-00001-of-00001.safetensors"
    header = {
        "sample.weight": {
            "dtype": "F32",
            "shape": [2],
            "data_offsets": [0, 8],
        }
    }
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    shard = len(header_bytes).to_bytes(8, "little") + header_bytes
    shard += struct.pack("<2f", 1.25, -2.5)
    (tmp_path / shard_name).write_bytes(shard)
    index = {"metadata": {"total_size": 8}, "weight_map": {"sample.weight": shard_name}}
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(index), encoding="utf-8"
    )

    store = SafetensorIndexStore(tmp_path, validate_real_layout=False)
    tensor = store.load("sample.weight")

    assert store.spec("sample.weight").shape == (2,)
    assert store.spec("sample.weight").dtype == F32
    assert torch.equal(tensor, torch.tensor([1.25, -2.5]))
