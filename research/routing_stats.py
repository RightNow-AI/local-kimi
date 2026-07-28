"""Inspect Kimi K3 router tensors as a prior for batching experiments.

This script downloads only each MoE layer's gate weight and noaux_tc correction
bias by safetensors HTTP range read. It reuses the retrying range reader from
``research/expert_spectrum_v2.py`` so the repository has one transport path.

The output is evidence about the router parameters, not a routing measurement.
Actual expert frequencies depend on hidden-state inputs, sigmoid scores,
correction biases, top-k selection, and generation workload. Use
``engine/modal_batch.py`` to measure those decisions during generation.
"""

from __future__ import annotations

import argparse
import json
import math
import struct
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

try:
    from research.expert_spectrum_v2 import BASE, http
except ModuleNotFoundError:  # Direct execution from the research directory.
    from expert_spectrum_v2 import BASE, http


DEFAULT_LAYERS = tuple(range(1, 93))


@dataclass(frozen=True)
class TensorRef:
    layer: int
    role: str
    name: str
    shard: str


def _parse_layers(spec: str) -> tuple[int, ...]:
    layers: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_text, end_text = part.split("-", 1)
            start, end = int(start_text), int(end_text)
            if end < start:
                raise ValueError(f"invalid descending layer range {part!r}")
            layers.update(range(start, end + 1))
        else:
            layers.add(int(part))
    if not layers:
        raise ValueError("at least one layer is required")
    if min(layers) < 1 or max(layers) > 92:
        raise ValueError("K3 MoE layers are 1 through 92")
    return tuple(sorted(layers))


def _load_index(cache: Path) -> dict[str, str]:
    path = cache / "model.safetensors.index.json"
    if not path.exists():
        path.write_bytes(http(f"{BASE}/model.safetensors.index.json"))
    return json.loads(path.read_text(encoding="utf-8"))["weight_map"]


def _load_header(shard: str, cache: Path) -> tuple[dict, int]:
    """Read the standard safetensors header using the shared range reader."""
    path = cache / f"{shard}.header.json"
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload["header"], payload["data_start"]

    header_bytes = struct.unpack("<Q", http(f"{BASE}/{shard}", (0, 7)))[0]
    header = json.loads(http(f"{BASE}/{shard}", (8, 8 + header_bytes - 1)))
    header.pop("__metadata__", None)
    data_start = 8 + header_bytes
    path.write_text(
        json.dumps({"header": header, "data_start": data_start}),
        encoding="utf-8",
    )
    return header, data_start


def _find_tensor(weight_map: dict[str, str], layer: int, suffix: str) -> TensorRef:
    prefix = f"language_model.model.layers.{layer}."
    candidates = [
        name
        for name in weight_map
        if name.startswith(prefix)
        and name.endswith(suffix)
        and ".gate." in name
    ]
    if len(candidates) != 1:
        raise KeyError(
            f"expected one layer {layer} tensor ending in {suffix!r}, found {candidates}"
        )
    role = "correction_bias" if suffix.endswith("e_score_correction_bias") else "weight"
    return TensorRef(layer, role, candidates[0], weight_map[candidates[0]])


def _decode_tensor(raw: bytes, metadata: dict) -> np.ndarray:
    shape = tuple(metadata["shape"])
    dtype = metadata["dtype"]
    if dtype == "BF16":
        words = np.frombuffer(raw, dtype="<u2")
        values = (words.astype(np.uint32) << 16).view(np.float32)
    elif dtype == "F32":
        values = np.frombuffer(raw, dtype="<f4").astype(np.float32, copy=False)
    elif dtype == "F16":
        values = np.frombuffer(raw, dtype="<f2").astype(np.float32)
    else:
        raise ValueError(f"unsupported router tensor dtype {dtype!r}")
    expected = math.prod(shape)
    if values.size != expected:
        raise ValueError(
            f"decoded {values.size} values for shape {shape}, expected {expected}"
        )
    return values.reshape(shape)


def _cache_name(ref: TensorRef) -> str:
    return f"layer{ref.layer}_{ref.role}.npy"


def _download_tensor(
    ref: TensorRef,
    headers: dict[str, tuple[dict, int]],
    cache: Path,
) -> Path:
    output = cache / _cache_name(ref)
    if output.exists():
        return output
    header, data_start = headers[ref.shard]
    metadata = header[ref.name]
    start, end = metadata["data_offsets"]
    raw = http(f"{BASE}/{ref.shard}", (data_start + start, data_start + end - 1))
    values = _decode_tensor(raw, metadata)
    np.save(output, values)
    return output


def download_router_tensors(
    *, cache: Path, layers: Iterable[int], threads: int
) -> dict[int, dict[str, Path]]:
    """Range-download router weights and biases, returning local cache paths."""
    cache.mkdir(parents=True, exist_ok=True)
    weight_map = _load_index(cache)
    refs: list[TensorRef] = []
    for layer in layers:
        refs.append(_find_tensor(weight_map, layer, ".gate.weight"))
        refs.append(_find_tensor(weight_map, layer, ".gate.e_score_correction_bias"))

    headers = {
        shard: _load_header(shard, cache)
        for shard in sorted({ref.shard for ref in refs})
    }
    results: dict[int, dict[str, Path]] = {}

    def fetch(ref: TensorRef) -> tuple[TensorRef, Path]:
        return ref, _download_tensor(ref, headers, cache)

    with ThreadPoolExecutor(max_workers=threads) as pool:
        for index, (ref, path) in enumerate(pool.map(fetch, refs), start=1):
            results.setdefault(ref.layer, {})[ref.role] = path
            if index % 16 == 0 or index == len(refs):
                print(f"downloaded or cached {index}/{len(refs)} router tensors", flush=True)
    return results


def _distribution(values: np.ndarray) -> dict[str, float | int]:
    flat = np.asarray(values, dtype=np.float64).ravel()
    quantiles = np.quantile(flat, [0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0])
    return {
        "count": int(flat.size),
        "mean": float(flat.mean()),
        "std": float(flat.std()),
        "rms": float(np.sqrt(np.mean(flat * flat))),
        "min": float(quantiles[0]),
        "p01": float(quantiles[1]),
        "p05": float(quantiles[2]),
        "p25": float(quantiles[3]),
        "p50": float(quantiles[4]),
        "p75": float(quantiles[5]),
        "p95": float(quantiles[6]),
        "p99": float(quantiles[7]),
        "max": float(quantiles[8]),
        "range": float(quantiles[8] - quantiles[0]),
    }


def _randomized_spectrum(
    matrix: np.ndarray,
    *,
    rank: int,
    oversample: int,
    power_iterations: int,
    seed: int,
) -> dict:
    rows, columns = matrix.shape
    target = min(rank + oversample, rows, columns)
    if rank <= 0 or target <= 0:
        raise ValueError("spectrum rank must be positive")
    rng = np.random.default_rng(seed)
    omega = rng.standard_normal((columns, target), dtype=np.float32)
    sample = matrix @ omega
    for _ in range(power_iterations):
        sample = matrix @ (matrix.T @ sample)
    basis, _ = np.linalg.qr(sample, mode="reduced")
    compressed = basis.T @ matrix
    singular = np.linalg.svd(compressed, compute_uv=False)[:rank]
    frobenius_sq = float(np.sum(matrix.astype(np.float64) ** 2))
    captured = np.cumsum(singular.astype(np.float64) ** 2) / frobenius_sq
    return {
        "method": "randomized_svd",
        "requested_rank": rank,
        "oversample": oversample,
        "power_iterations": power_iterations,
        "singular_values": singular.astype(float).tolist(),
        "normalized_to_top": (singular / singular[0]).astype(float).tolist(),
        "captured_frobenius_energy": captured.astype(float).tolist(),
        "top_singular_value": float(singular[0]),
        "stable_rank_estimate": float(frobenius_sq / singular[0] ** 2),
        "note": (
            "Top spectrum is randomized. Stable rank is approximate because the "
            "top singular value is approximate."
        ),
    }


def _sample_pairwise_cosines(
    matrix: np.ndarray, *, pairs: int, seed: int
) -> dict[str, float | int]:
    rng = np.random.default_rng(seed)
    rows = matrix.shape[0]
    left = rng.integers(0, rows, size=pairs)
    right = rng.integers(0, rows - 1, size=pairs)
    right = right + (right >= left)
    norms = np.linalg.norm(matrix, axis=1)
    values = np.empty(pairs, dtype=np.float32)
    chunk = 128
    for start in range(0, pairs, chunk):
        end = min(start + chunk, pairs)
        dots = np.einsum(
            "ij,ij->i",
            matrix[left[start:end]],
            matrix[right[start:end]],
            optimize=True,
        )
        values[start:end] = dots / (norms[left[start:end]] * norms[right[start:end]])
    out = _distribution(values)
    out["absolute_mean"] = float(np.mean(np.abs(values)))
    out["absolute_p95"] = float(np.quantile(np.abs(values), 0.95))
    return out


def analyze_layer(
    *,
    layer: int,
    weight_path: Path,
    bias_path: Path,
    spectrum_rank: int,
    oversample: int,
    power_iterations: int,
    cosine_pairs: int,
    seed: int,
) -> dict:
    weight = np.load(weight_path, mmap_mode="r")
    bias = np.load(bias_path, mmap_mode="r")
    if weight.shape != (896, 7168):
        raise ValueError(f"layer {layer} gate weight has unexpected shape {weight.shape}")
    if bias.shape != (896,):
        raise ValueError(f"layer {layer} correction bias has unexpected shape {bias.shape}")

    weight_array = np.asarray(weight, dtype=np.float32)
    bias_array = np.asarray(bias, dtype=np.float32)
    row_norms = np.linalg.norm(weight_array, axis=1)
    norm_std = float(row_norms.std())
    bias_norm_correlation = (
        float(np.corrcoef(bias_array, row_norms)[0, 1])
        if bias_array.std() > 0.0 and norm_std > 0.0
        else 0.0
    )
    mean_direction = weight_array.mean(axis=0)
    rms_row_norm = float(np.sqrt(np.mean(row_norms.astype(np.float64) ** 2)))

    return {
        "layer": layer,
        "weight_shape": list(weight.shape),
        "bias_shape": list(bias.shape),
        "correction_bias": _distribution(bias_array),
        "gate_row_norm": _distribution(row_norms),
        "bias_row_norm_correlation": bias_norm_correlation,
        "mean_direction_to_rms_row_norm": float(
            np.linalg.norm(mean_direction) / rms_row_norm
        ),
        "sampled_pairwise_row_cosine": _sample_pairwise_cosines(
            weight_array, pairs=cosine_pairs, seed=seed + layer
        ),
        "singular_spectrum": _randomized_spectrum(
            weight_array,
            rank=spectrum_rank,
            oversample=oversample,
            power_iterations=power_iterations,
            seed=seed + layer,
        ),
    }


def _pooled_report(layer_reports: list[dict], cache_paths: dict[int, dict[str, Path]]) -> dict:
    biases = np.concatenate(
        [np.load(cache_paths[layer]["correction_bias"]) for layer in sorted(cache_paths)]
    )
    top_normalized = np.asarray(
        [report["singular_spectrum"]["normalized_to_top"] for report in layer_reports]
    )
    bias_ranges = np.asarray(
        [report["correction_bias"]["range"] for report in layer_reports]
    )
    stable_ranks = np.asarray(
        [report["singular_spectrum"]["stable_rank_estimate"] for report in layer_reports]
    )
    correlations = np.asarray(
        [report["bias_row_norm_correlation"] for report in layer_reports]
    )
    return {
        "layers": len(layer_reports),
        "correction_bias": _distribution(biases),
        "layer_bias_range": _distribution(bias_ranges),
        "stable_rank_estimate": _distribution(stable_ranks),
        "bias_row_norm_correlation": _distribution(correlations),
        "mean_normalized_top_singular_spectrum": top_normalized.mean(axis=0).tolist(),
        "p05_normalized_top_singular_spectrum": np.quantile(
            top_normalized, 0.05, axis=0
        ).tolist(),
        "p95_normalized_top_singular_spectrum": np.quantile(
            top_normalized, 0.95, axis=0
        ).tolist(),
        "layers_by_bias_range_desc": [
            report["layer"]
            for report in sorted(
                layer_reports,
                key=lambda item: item["correction_bias"]["range"],
                reverse=True,
            )
        ],
    }


def _render_markdown(results: dict) -> str:
    pooled = results["pooled"]
    bias = pooled["correction_bias"]
    stable = pooled["stable_rank_estimate"]
    lines = [
        "# Kimi K3 router tensor statistics",
        "",
        "> Parameter evidence only. This is not a measurement of expert selection frequencies.",
        "",
        "## Pooled correction bias",
        "",
        f"Across {bias['count']:,} expert biases: mean {bias['mean']:.6g}, standard deviation {bias['std']:.6g}, minimum {bias['min']:.6g}, median {bias['p50']:.6g}, maximum {bias['max']:.6g}, range {bias['range']:.6g}.",
        "",
        "The spread is direct evidence that noaux_tc learned or maintained expert-specific corrective pressure. It is not a direct estimate of natural routing imbalance. A large bias can counteract a naturally popular or unpopular expert, and the sign cannot be read as traffic without the pre-bias scores.",
        "",
        "## Gate-weight structure",
        "",
        f"Approximate stable rank across layers: median {stable['p50']:.1f}, p05 {stable['p05']:.1f}, p95 {stable['p95']:.1f}.",
        "",
        "The JSON contains each layer's randomized top singular spectrum, row-norm distribution, sampled row-cosine distribution, mean-direction strength, and bias-to-row-norm correlation. Low stable rank or strongly aligned rows would support a low-dimensional router-score predictor. Flat spectra and near-orthogonal rows would argue against that shortcut.",
        "",
        "## What these tensors cannot answer",
        "",
        "- They cannot give expert frequencies for coding prompts without hidden-state samples.",
        "- They cannot determine the union across concurrent tokens without joint routing traces.",
        "- Correction-bias magnitude cannot be converted into a Zipf exponent by itself.",
        "- Gate-weight singular vectors describe the score map, not the workload distribution in its input space.",
        "- The real decision number must come from instrumented generation using the same prompts, sampling settings, and concurrency policy as the product.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layers", default="1-92")
    parser.add_argument("--cache", type=Path, default=Path("research/.routing-cache"))
    parser.add_argument("--output", type=Path, default=Path("research/routing_stats.json"))
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--spectrum-rank", type=int, default=64)
    parser.add_argument("--oversample", type=int, default=16)
    parser.add_argument("--power-iterations", type=int, default=1)
    parser.add_argument("--cosine-pairs", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--download-only", action="store_true")
    args = parser.parse_args()

    layers = _parse_layers(args.layers)
    cache_paths = download_router_tensors(
        cache=args.cache,
        layers=layers,
        threads=args.threads,
    )
    if args.download_only:
        return 0

    reports = []
    for index, layer in enumerate(layers, start=1):
        paths = cache_paths[layer]
        reports.append(
            analyze_layer(
                layer=layer,
                weight_path=paths["weight"],
                bias_path=paths["correction_bias"],
                spectrum_rank=args.spectrum_rank,
                oversample=args.oversample,
                power_iterations=args.power_iterations,
                cosine_pairs=args.cosine_pairs,
                seed=args.seed,
            )
        )
        print(f"analyzed {index}/{len(layers)} router layers", flush=True)

    results = {
        "status": "parameter prior, not measured routing",
        "model": "moonshotai/Kimi-K3",
        "layers": reports,
        "pooled": _pooled_report(reports, cache_paths),
        "interpretation": {
            "can_tell": [
                "correction-bias spread and layer variation",
                "gate-map singular structure",
                "expert row-norm and row-alignment structure",
                "whether a low-dimensional score predictor is plausible",
            ],
            "cannot_tell": [
                "real expert selection frequencies",
                "concurrent-token expert unions",
                "a defensible routing-skew exponent without hidden states",
                "generation throughput",
            ],
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    markdown_path = args.output.with_suffix(".md")
    markdown_path.write_text(_render_markdown(results), encoding="utf-8")
    print(f"wrote {args.output} and {markdown_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

