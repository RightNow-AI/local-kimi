import math

from engine.k3ref.config import K3LayerConfig
from engine.k3ref.manifest import K3_LAYER_TENSOR_MANIFEST, layer_tensor_manifest
from engine.laptop.profile import (
    KIMI_LINEAR_PARAMETER_COUNTS,
    build_profiles,
)


def _profiles_by_format():
    return {profile.format_key: profile for profile in build_profiles()}


def test_kimi_linear_bf16_reproduces_98_2_gb_from_parameter_count():
    counts = KIMI_LINEAR_PARAMETER_COUNTS
    bf16 = _profiles_by_format()["bf16"]

    assert counts.total_parameters == 49_122_681_728
    assert bf16.resident_bytes == counts.total_parameters * 2
    assert round(bf16.resident_bytes / 1_000_000_000, 1) == 98.2


def test_weight_only_int4_is_half_fp8_and_quarter_bf16():
    profiles = _profiles_by_format()
    bf16 = profiles["bf16"].resident_bytes
    fp8 = profiles["fp8"].resident_bytes
    int4 = profiles["int4-weight-only"].resident_bytes

    assert int4 * 2 == fp8
    assert int4 * 4 == bf16


def test_non_fitting_profile_is_reported_without_clamping():
    bf16 = _profiles_by_format()["bf16"]
    fit_64 = next(row for row in bf16.memory_fits if row.capacity_gib == 64)

    assert fit_64.fits is False
    assert fit_64.headroom_bytes < 0
    assert fit_64.resident_bytes == bf16.resident_bytes


def test_generalized_header_manifest_preserves_exact_k3_layer_12_contract():
    header = {
        f"model.layers.12.{name}": {
            "shape": list(spec.shape),
            "dtype": spec.dtype,
            "data_offsets": [0, 0],
        }
        for name, spec in K3_LAYER_TENSOR_MANIFEST.items()
    }

    actual = layer_tensor_manifest(K3LayerConfig(), header, layer_idx=12)

    assert actual == K3_LAYER_TENSOR_MANIFEST


def test_active_bytes_use_active_a3b_path_not_total_parameters():
    counts = KIMI_LINEAR_PARAMETER_COUNTS
    int4 = _profiles_by_format()["int4-weight-only"]

    assert counts.active_parameters_per_token == 3_106_974_848
    assert int4.active_bytes_per_token == 1_553_487_424
    assert int4.active_bytes_per_token < int4.resident_bytes

    ddr5 = next(
        row for row in int4.throughput if row.hardware_key == "laptop-ddr5-100gb-s"
    )
    assert math.isclose(
        ddr5.physical_ceiling_tokens_per_second,
        100_000_000_000 / int4.active_bytes_per_token,
    )
    assert math.isclose(
        ddr5.projected_tokens_per_second_at_attainment,
        ddr5.physical_ceiling_tokens_per_second * 0.60,
    )
