"""GPU benchmark for cold prefill against restored prefix state.

The warm request appends one token to an exact cached prefix. This matches the
coding-session case where a stable system and repository prefix gains a new
turn. Warm time to first token includes pinned-host restore, one-token suffix
prefill, and greedy selection. Target-state allocation is outside the timed
path because production keeps the CUDA-graph-bound buffers preallocated.

This feature does not make decode faster. Tokens per second after the first
token is unchanged because warm and cold requests use the same decode path.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

import torch

from engine.klinear.generate import prefill
from engine.klinear.model import KLinearModel
from engine.klinear.state import KLinearDecodeState, MLALayerState
from engine.statecache.key import fingerprint_model, prefix_key
from engine.statecache.store import StateCache


_T = TypeVar("_T")
PREFIX_LENGTHS = (512, 2048, 8192, 32768)


def _cuda_time(call: Callable[[], _T]) -> tuple[_T, float]:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    result = call()
    end.record()
    end.synchronize()
    return result, start.elapsed_time(end)


def _mla_bytes_per_token(state: KLinearDecodeState) -> tuple[int, int]:
    stored = 0
    live = 0
    for layer in state.layer_states:
        if not isinstance(layer, MLALayerState):
            continue
        if layer.key_pass is None or layer.value is None:
            raise ValueError("benchmark requires complete live MLA state")
        compressed = (
            layer.compressed_kv[:, :1].numel()
            * layer.compressed_kv.element_size()
        )
        rotary = layer.rotary_key[:, :1].numel() * layer.rotary_key.element_size()
        projected_key = layer.key_pass[:, :, :1].numel() * layer.key_pass.element_size()
        projected_value = layer.value[:, :, :1].numel() * layer.value.element_size()
        stored += compressed + rotary
        live += compressed + rotary + projected_key + projected_value
    return stored, live


def benchmark(
    checkpoint: Path,
    *,
    byte_budget: int,
    spill_directory: Path | None,
    seed: int,
) -> dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("BENCH-STATECACHE requires CUDA")

    device = torch.device("cuda")
    torch.manual_seed(seed)
    model = KLinearModel.from_directory(
        checkpoint,
        device=device,
        dtype=torch.bfloat16,
    )
    fingerprint = fingerprint_model(model)
    cache = StateCache(
        byte_budget=byte_budget,
        disk_spill=spill_directory is not None,
        spill_directory=spill_directory,
    )

    maximum = max(PREFIX_LENGTHS) + 1
    tokens = torch.randint(
        low=3,
        high=model.config.vocab_size - 1,
        size=(1, maximum),
        dtype=torch.long,
        device=device,
    )
    tokens[:, 0] = model.config.bos_token_id
    rows: list[dict[str, object]] = []

    for length in PREFIX_LENGTHS:
        prefix = tokens[:, :length]
        suffix = tokens[:, length : length + 1]
        full_request = tokens[:, : length + 1]

        def cold_first_token() -> tuple[object, torch.Tensor]:
            output = prefill(model, full_request)
            return output, output.logits[:, -1].argmax(dim=-1)

        (cold_output, cold_token), cold_ttft_ms = _cuda_time(cold_first_token)
        del cold_output
        torch.cuda.empty_cache()

        prefix_output, prefill_ms = _cuda_time(lambda: prefill(model, prefix))
        mla_stored_bytes_per_token, mla_live_bytes_per_token = _mla_bytes_per_token(
            prefix_output.state
        )
        key = prefix_key(prefix, fingerprint)

        def snapshot() -> bool:
            return cache.save(
                key,
                prefix_output.state,
                length,
                token_ids=prefix,
                model_fingerprint=fingerprint,
            )

        saved, snapshot_ms = _cuda_time(snapshot)
        if not saved:
            raise RuntimeError(
                f"snapshot at {length} tokens exceeds the configured byte budget"
            )
        cache.synchronize(key)
        snapshot_bytes = cache.entry_bytes(key)
        if snapshot_bytes is None:
            raise RuntimeError("snapshot disappeared before restore")
        del prefix_output
        torch.cuda.empty_cache()

        target = cache.allocate_state(
            key,
            model=model,
            device=device,
            additional_tokens=1,
        )
        loaded, restore_total_ms = _cuda_time(
            lambda: cache.load(key, target, model=model)
        )
        if not loaded:
            raise RuntimeError("snapshot disappeared during restore benchmark")
        _, rebuild_ms = _cuda_time(lambda: cache.rebuild_mla(model, target))
        derived_transfer_ms = max(0.0, restore_total_ms - rebuild_ms)

        def warm_first_token() -> tuple[object, torch.Tensor]:
            if not cache.load(key, target, model=model):
                raise RuntimeError("snapshot disappeared during warm request")
            output = prefill(model, suffix, state=target)
            return output, output.logits[:, -1].argmax(dim=-1)

        (warm_output, warm_token), warm_ttft_ms = _cuda_time(warm_first_token)
        identical = torch.equal(cold_token, warm_token)
        if not identical:
            raise AssertionError(
                f"restored state changed the first generated token at length {length}"
            )

        rows.append(
            {
                "prefix_tokens": length,
                "prefill_ms": prefill_ms,
                "snapshot_ms": snapshot_ms,
                "restore_total_ms": restore_total_ms,
                "standalone_rebuild_ms": rebuild_ms,
                "derived_transfer_ms": derived_transfer_ms,
                "cold_time_to_first_token_ms": cold_ttft_ms,
                "warm_time_to_first_token_ms": warm_ttft_ms,
                "time_to_first_token_speedup": cold_ttft_ms / warm_ttft_ms,
                "snapshot_host_bytes": snapshot_bytes,
                "total_host_bytes_held": cache.host_bytes,
                "mla_stored_bytes_per_token": mla_stored_bytes_per_token,
                "mla_live_bytes_per_token": mla_live_bytes_per_token,
                "first_token_identical": identical,
                "decode_tokens_per_second": "unchanged, same decode path",
            }
        )
        del warm_output, target
        torch.cuda.empty_cache()

    return {
        "checkpoint": str(checkpoint.resolve()),
        "model_fingerprint": fingerprint,
        "byte_budget": byte_budget,
        "disk_spill": spill_directory is not None,
        "disk_spill_directory": (
            None if spill_directory is None else str(spill_directory.resolve())
        ),
        "timing_contract": (
            "restore_total_ms includes pinned-host transfer and MLA projection "
            "rebuild. standalone_rebuild_ms repeats the rebuild to expose its "
            "cost, and derived_transfer_ms is total minus that standalone timing. "
            "Warm time to first token includes total restore, one-token suffix "
            "prefill, and greedy selection. Target allocation is excluded."
        ),
        "cache_match_contract": (
            "A hit requires an exact token prefix. A change near the start "
            "invalidates every longer snapshot."
        ),
        "decode_contract": (
            "State caching removes repeated prefix prefill. It does not change "
            "decode tokens per second."
        ),
        "rows": rows,
    }


def _print_table(result: dict[str, object]) -> None:
    rows = result["rows"]
    first = rows[0]
    print(
        "MLA bytes per token: "
        f"stored={first['mla_stored_bytes_per_token']:,}, "
        f"live={first['mla_live_bytes_per_token']:,}. "
        "Every restore total below includes the projected-cache rebuild."
    )
    print(
        f"{'prefix':>8} {'cold TTFT':>12} {'warm TTFT':>12} "
        f"{'speedup':>9} {'prefill':>10} {'snapshot':>10} "
        f"{'rebuild':>10} {'restore':>10} {'stored MiB':>12}"
    )
    for row in rows:
        print(
            f"{row['prefix_tokens']:>8} "
            f"{row['cold_time_to_first_token_ms']:>10.2f}ms "
            f"{row['warm_time_to_first_token_ms']:>10.2f}ms "
            f"{row['time_to_first_token_speedup']:>8.2f}x "
            f"{row['prefill_ms']:>8.2f}ms "
            f"{row['snapshot_ms']:>8.2f}ms "
            f"{row['standalone_rebuild_ms']:>8.2f}ms "
            f"{row['restore_total_ms']:>8.2f}ms "
            f"{row['snapshot_host_bytes'] / (1024**2):>12.2f}"
        )
    print()
    print(result["decode_contract"])
    print(result["cache_match_contract"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--byte-budget-gib", type=float, default=16.0)
    parser.add_argument("--disk-spill-directory", type=Path)
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()
    if args.byte_budget_gib < 0:
        parser.error("--byte-budget-gib cannot be negative")

    result = benchmark(
        args.checkpoint,
        byte_budget=int(args.byte_budget_gib * 1024**3),
        spill_directory=args.disk_spill_directory,
        seed=args.seed,
    )
    _print_table(result)
    if args.json_output is not None:
        args.json_output.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
