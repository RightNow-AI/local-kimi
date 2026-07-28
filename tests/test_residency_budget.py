from __future__ import annotations

import pytest

from engine.residency.budget import (
    BF16,
    INT4_WEIGHTS,
    KIMI_LINEAR_SHAPE,
    MLACachePolicy,
    ResidencyBudgetExceeded,
    RuntimeHeadroom,
    build_residency_budget,
    mla_kv_bytes_per_token_per_sequence,
    require_envelope_fits,
    solve_residency_frontier,
)

NO_HEADROOM = RuntimeHeadroom(activation_bytes=0, workspace_bytes=0)


def test_kda_state_pool_scales_with_max_num_seqs_not_max_model_len() -> None:
    short_context = build_residency_budget(
        INT4_WEIGHTS,
        max_num_seqs=3,
        max_model_len=128,
        headroom=NO_HEADROOM,
    )
    long_context = build_residency_budget(
        INT4_WEIGHTS,
        max_num_seqs=3,
        max_model_len=65_536,
        headroom=NO_HEADROOM,
    )
    twice_the_pool = build_residency_budget(
        INT4_WEIGHTS,
        max_num_seqs=6,
        max_model_len=128,
        headroom=NO_HEADROOM,
    )

    assert short_context.kda_recurrent_state_bytes == long_context.kda_recurrent_state_bytes
    assert short_context.short_conv_state_bytes == long_context.short_conv_state_bytes
    assert (
        twice_the_pool.kda_recurrent_state_bytes
        == 2 * short_context.kda_recurrent_state_bytes
    )
    assert twice_the_pool.short_conv_state_bytes == 2 * short_context.short_conv_state_bytes


def test_mla_kv_cache_scales_with_both_sequences_and_context() -> None:
    base = build_residency_budget(
        INT4_WEIGHTS,
        max_num_seqs=2,
        max_model_len=1_024,
        headroom=NO_HEADROOM,
    )
    twice_sequences = build_residency_budget(
        INT4_WEIGHTS,
        max_num_seqs=4,
        max_model_len=1_024,
        headroom=NO_HEADROOM,
    )
    twice_context = build_residency_budget(
        INT4_WEIGHTS,
        max_num_seqs=2,
        max_model_len=2_048,
        headroom=NO_HEADROOM,
    )

    assert twice_sequences.mla_kv_cache_bytes == 2 * base.mla_kv_cache_bytes
    assert twice_context.mla_kv_cache_bytes == 2 * base.mla_kv_cache_bytes


def test_mla_cache_policies_have_distinct_source_derived_token_costs() -> None:
    expanded = mla_kv_bytes_per_token_per_sequence(
        cache_policy=MLACachePolicy.EXPANDED,
        dtype=BF16,
    )
    compressed = mla_kv_bytes_per_token_per_sequence(
        cache_policy=MLACachePolicy.COMPRESSED_LATENT,
        dtype=BF16,
    )
    compressed_elements_per_layer = (
        KIMI_LINEAR_SHAPE.mla_kv_lora_rank
        + KIMI_LINEAR_SHAPE.mla_qk_rope_head_dim
    )

    assert expanded == 143_360
    assert compressed == 8_064
    assert compressed == (
        KIMI_LINEAR_SHAPE.mla_layers
        * compressed_elements_per_layer
        * BF16.bytes_per_element
    )
    assert expanded != compressed

    compressed_budget = build_residency_budget(
        INT4_WEIGHTS,
        max_num_seqs=1,
        max_model_len=1,
        mla_cache_policy=MLACachePolicy.COMPRESSED_LATENT,
        headroom=NO_HEADROOM,
    )
    assert compressed_budget.mla_cache_policy == "compressed_latent"
    assert compressed_budget.mla_kv_cache_bytes == compressed


def test_solver_and_guard_never_return_an_over_budget_envelope() -> None:
    one_token = build_residency_budget(
        INT4_WEIGHTS,
        max_num_seqs=1,
        max_model_len=1,
        headroom=NO_HEADROOM,
    )
    too_small = one_token.total_bytes - 1

    assert solve_residency_frontier(
        too_small,
        INT4_WEIGHTS,
        max_num_seqs_values=(1,),
        headroom=NO_HEADROOM,
    ) == ()
    with pytest.raises(ResidencyBudgetExceeded):
        require_envelope_fits(
            too_small,
            INT4_WEIGHTS,
            max_num_seqs=1,
            max_model_len=1,
            headroom=NO_HEADROOM,
        )


def test_known_breakdown_matches_hand_computed_bytes() -> None:
    headroom = RuntimeHeadroom(activation_bytes=1_234, workspace_bytes=5_678)
    budget = build_residency_budget(
        INT4_WEIGHTS,
        max_num_seqs=2,
        max_model_len=1_024,
        headroom=headroom,
    )

    assert budget.weights_bytes == 24_561_340_864
    assert budget.kda_recurrent_state_bytes == 83_886_080
    assert budget.short_conv_state_bytes == 3_932_160
    assert budget.mla_kv_cache_bytes == 293_601_280
    assert budget.activation_headroom_bytes == 1_234
    assert budget.workspace_headroom_bytes == 5_678
    assert budget.total_bytes == 24_942_767_296
