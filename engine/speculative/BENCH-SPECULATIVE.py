"""Run the KLinear greedy speculative decoding benchmark on an L40S.

The benchmark reports ordinary CUDA graph decode, speculative wall time per
emitted token, mean accepted draft tokens per round, acceptance rate, and
measured snapshot and restore costs. It always runs one repetitive prompt and
one non-repetitive prompt because prompt-lookup acceptance is entirely workload
dependent. A result from one prompt says little about another workload.

Run with:

    modal run engine/speculative/BENCH-SPECULATIVE.py
    modal run engine/speculative/BENCH-SPECULATIVE.py --k-values 2,4,8 --ngrams 2,3,4

Current KLinear wiring accepts only one token with fixed-capacity state. This
benchmark therefore presents the fixed cache as a read-only growing view for
the one-pass draft verification, then commits an all-accepted result or restores
and replays the corrected prefix. Production wiring should add a fixed-shape
``verify_k`` entry point inside KLinear so the verification and commit remain
inside captured device work. No KLinear source change is hidden in this file.
"""

from __future__ import annotations

import json
from pathlib import Path

import modal

app = modal.App("kimi-linear-speculative-bench")

VOLUME = modal.Volume.from_name("kimi-linear-quantized", create_if_missing=False)
MOUNT = "/weights"
MODEL_DIR = f"{MOUNT}/Kimi-Linear-48B-A3B-Instruct-W4A16"
BUDGET_BYTES = 34_359_738_368

IMAGE = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-devel-ubuntu22.04",
        add_python="3.12",
    )
    .entrypoint([])
    .apt_install("git")
    .pip_install(
        "torch>=2.5",
        "triton>=3.1",
        "safetensors>=0.4.5",
        "transformers>=4.48",
        "numpy>=2.0",
        "tiktoken>=0.9",
        "blobfile>=3.0",
    )
    .env({"CUDA_HOME": "/usr/local/cuda"})
    .add_local_dir(Path(__file__).parents[1], remote_path="/root/engine")
)

PROMPTS = {
    "repetitive": """Complete this repeated Python pattern with the same structure.

def add_one(x):
    return x + 1

def add_two(x):
    return x + 2

def add_one(x):
    return x + 1

def add_two(x):
    return x + 2

def add_three(x):
""",
    "non_repetitive": (
        "Explain how a lunar eclipse differs from a solar eclipse, then give one "
        "safe observation tip and one historical reason eclipses were useful to "
        "astronomers. Use concise prose without repeating phrases."
    ),
}


@app.function(
    image=IMAGE,
    gpu="L40S",
    cpu=16.0,
    memory=65536,
    timeout=60 * 60,
    volumes={MOUNT: VOLUME},
)
def run_benchmark(
    k_values: list[int],
    ngrams: list[int],
    new_tokens: int = 64,
    repeats: int = 3,
    cap_memory: bool = True,
) -> dict:
    import gc
    import statistics
    import time

    import torch
    from transformers import AutoTokenizer

    from engine.klinear.generate import CUDAGraphDecodeRunner, decode, prefill
    from engine.klinear.model import KLinearModel
    from engine.klinear.state import KDALayerState, KLinearDecodeState, MLALayerState
    from engine.speculative.draft import propose
    from engine.speculative.state_checkpoint import DecodeCheckpoint
    from engine.speculative.verify import align_verification_logits, verify_greedy

    if not torch.cuda.is_available():
        raise RuntimeError("the speculative benchmark requires CUDA")
    if not k_values or any(k <= 0 for k in k_values):
        raise ValueError("k_values must contain positive integers")
    if not ngrams or any(ngram <= 0 for ngram in ngrams):
        raise ValueError("ngrams must contain positive integers")
    if new_tokens <= 0 or repeats <= 0:
        raise ValueError("new_tokens and repeats must be positive")

    device = torch.device("cuda:0")
    total_bytes = torch.cuda.get_device_properties(device).total_memory
    if cap_memory:
        torch.cuda.set_per_process_memory_fraction(BUDGET_BYTES / total_bytes, device)

    load_started = time.perf_counter()
    model = KLinearModel.from_directory(MODEL_DIR, device=device, dtype=torch.bfloat16)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_DIR,
        trust_remote_code=True,
        local_files_only=True,
    )
    torch.cuda.synchronize(device)
    load_seconds = time.perf_counter() - load_started

    def verification_view(state: KLinearDecodeState) -> KLinearDecodeState:
        """Expose the live prefix without mutating fixed-capacity cache storage."""

        prefix = state.tokens_seen
        layers = []
        for layer in state.layer_states:
            if layer is None:
                layers.append(None)
            elif isinstance(layer, KDALayerState):
                layers.append(
                    KDALayerState(
                        layer.q_conv,
                        layer.k_conv,
                        layer.v_conv,
                        layer.recurrent,
                        is_static=False,
                    )
                )
            elif isinstance(layer, MLALayerState):
                layers.append(
                    MLALayerState(
                        layer.compressed_kv[:, :prefix],
                        layer.rotary_key[:, :prefix],
                        layer.key_pass[:, :, :prefix],
                        layer.value[:, :, :prefix],
                    )
                )
            else:
                raise TypeError("unsupported KLinear layer state")
        prefix_mask = (
            None
            if state.attention_mask is None
            else state.attention_mask[:, :prefix]
        )
        return KLinearDecodeState(tuple(layers), prefix, prefix_mask)

    def commit_verified_state(
        target: KLinearDecodeState,
        verified: KLinearDecodeState,
        token_count: int,
    ) -> KLinearDecodeState:
        """Commit an all-accepted dynamic verification into fixed buffers."""

        start = target.tokens_seen
        end = start + token_count
        for target_layer, source_layer in zip(
            target.layer_states,
            verified.layer_states,
            strict=True,
        ):
            if target_layer is None or source_layer is None:
                if target_layer is not source_layer:
                    raise ValueError("verification layer kinds disagree")
            elif isinstance(target_layer, KDALayerState) and isinstance(
                source_layer, KDALayerState
            ):
                target_layer.q_conv.copy_(source_layer.q_conv)
                target_layer.k_conv.copy_(source_layer.k_conv)
                target_layer.v_conv.copy_(source_layer.v_conv)
                target_layer.recurrent.copy_(source_layer.recurrent)
            elif isinstance(target_layer, MLALayerState) and isinstance(
                source_layer, MLALayerState
            ):
                target_layer.compressed_kv[:, start:end].copy_(
                    source_layer.compressed_kv[:, start:end]
                )
                target_layer.rotary_key[:, start:end].copy_(
                    source_layer.rotary_key[:, start:end]
                )
                target_layer.key_pass[:, :, start:end].copy_(
                    source_layer.key_pass[:, :, start:end]
                )
                target_layer.value[:, :, start:end].copy_(
                    source_layer.value[:, :, start:end]
                )
                target_layer.position.add_(token_count)
            else:
                raise ValueError("verification layer kinds disagree")
        if target.attention_mask is not None:
            target.attention_mask[:, start:end].fill_(1)
        target.position.add_(token_count)
        return target.with_tokens_seen(end)

    def ordinary_baseline(input_ids: torch.Tensor) -> tuple[list[int], float]:
        output = prefill(model, input_ids)
        runner = CUDAGraphDecodeRunner(model, input_ids, output, new_tokens)
        runner.reset()
        for _ in range(4):
            runner.graph.replay()
        torch.cuda.synchronize(device)

        timings = []
        for _ in range(repeats):
            runner.reset()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(new_tokens):
                runner.graph.replay()
            end.record()
            torch.cuda.synchronize(device)
            timings.append(start.elapsed_time(end))
        timings.sort()
        generated = runner.result().generated_ids[0].detach().cpu().tolist()
        median_ms = timings[len(timings) // 2]
        del runner, output
        gc.collect()
        torch.cuda.empty_cache()
        return generated, median_ms

    @torch.inference_mode()
    def one_speculative_run(
        input_ids: torch.Tensor,
        k: int,
        ngram: int,
    ) -> dict:
        output = prefill(model, input_ids)
        state = output.state.reserve_decode_capacity(new_tokens + k)
        next_logits = output.logits[:, -1]
        checkpoint = DecodeCheckpoint(state, max_speculative_tokens=k)

        checkpoint.snapshot(state)
        state = checkpoint.restore(state)
        torch.cuda.synchronize(device)
        micro_iterations = 20
        micro_snapshot_start = torch.cuda.Event(enable_timing=True)
        micro_snapshot_end = torch.cuda.Event(enable_timing=True)
        micro_snapshot_start.record()
        for _ in range(micro_iterations):
            checkpoint.snapshot(state)
        micro_snapshot_end.record()
        micro_restore_start = torch.cuda.Event(enable_timing=True)
        micro_restore_end = torch.cuda.Event(enable_timing=True)
        micro_restore_start.record()
        for _ in range(micro_iterations):
            state = checkpoint.restore(state)
        micro_restore_end.record()
        torch.cuda.synchronize(device)
        isolated_snapshot_ms = (
            micro_snapshot_start.elapsed_time(micro_snapshot_end) / micro_iterations
        )
        isolated_restore_ms = (
            micro_restore_start.elapsed_time(micro_restore_end) / micro_iterations
        )

        context = input_ids[0].detach().cpu().tolist()
        generated: list[int] = []
        rounds = 0
        drafted_rounds = 0
        proposed_tokens = 0
        accepted_tokens = 0
        restore_count = 0
        snapshot_events = [
            (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            for _ in range(new_tokens)
        ]
        restore_events = [
            (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            for _ in range(new_tokens)
        ]
        token_buffer = input_ids.new_empty(1, 1)
        draft_buffer = input_ids.new_empty(1, k)

        torch.cuda.synchronize(device)
        started = time.perf_counter()
        while len(generated) < new_tokens:
            rounds += 1
            remaining = new_tokens - len(generated)
            draft = propose(context, k, ngram) if remaining >= k else []
            if len(draft) != k:
                token = int(next_logits.argmax(dim=-1).detach().cpu().tolist()[0])
                emitted = [token]
                token_buffer.fill_(token)
                output = decode(model, token_buffer, state)
                state = output.state
                next_logits = output.logits[:, -1]
            else:
                event_start, event_end = snapshot_events[drafted_rounds]
                event_start.record()
                checkpoint.snapshot(state)
                event_end.record()
                drafted_rounds += 1
                proposed_tokens += k

                draft_buffer.copy_(
                    torch.tensor([draft], dtype=input_ids.dtype, device=device)
                )
                verified_output = model(
                    draft_buffer,
                    state=verification_view(state),
                )
                aligned = align_verification_logits(
                    next_logits,
                    verified_output.logits,
                )
                verification = verify_greedy(draft_buffer, aligned)
                accepted = int(
                    verification.accepted_prefix_length.detach().cpu().tolist()[0]
                )
                emitted_count = int(
                    verification.emitted_count.detach().cpu().tolist()[0]
                )
                emitted = (
                    verification.emitted_ids[0, :emitted_count]
                    .detach()
                    .cpu()
                    .tolist()
                )
                accepted_tokens += accepted

                if accepted == k:
                    state = commit_verified_state(state, verified_output.state, k)
                    next_logits = verified_output.logits[:, -1]
                else:
                    restore_start, restore_end = restore_events[restore_count]
                    restore_start.record()
                    state = checkpoint.restore(state)
                    restore_end.record()
                    restore_count += 1
                    for token in emitted:
                        token_buffer.fill_(token)
                        output = decode(model, token_buffer, state)
                        state = output.state
                        next_logits = output.logits[:, -1]

            generated.extend(emitted)
            context.extend(emitted)

        torch.cuda.synchronize(device)
        wall_ms = (time.perf_counter() - started) * 1000.0
        snapshot_ms = sum(
            start.elapsed_time(end)
            for start, end in snapshot_events[:drafted_rounds]
        )
        restore_ms = sum(
            start.elapsed_time(end)
            for start, end in restore_events[:restore_count]
        )
        return {
            "generated_token_ids": generated,
            "rounds": rounds,
            "drafted_rounds": drafted_rounds,
            "proposed_tokens": proposed_tokens,
            "accepted_tokens": accepted_tokens,
            "restore_count": restore_count,
            "wall_ms": wall_ms,
            "snapshot_ms": snapshot_ms,
            "restore_ms": restore_ms,
            "checkpoint_bytes": checkpoint.snapshot_bytes,
            "isolated_snapshot_ms": isolated_snapshot_ms,
            "isolated_restore_ms": isolated_restore_ms,
        }

    prompt_results = []
    for prompt_kind, prompt in PROMPTS.items():
        input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
        reference_ids, baseline_ms = ordinary_baseline(input_ids)
        baseline_ms_per_token = baseline_ms / new_tokens
        rows = []
        for k in k_values:
            for ngram in ngrams:
                runs = [one_speculative_run(input_ids, k, ngram) for _ in range(repeats)]
                for run in runs:
                    if run["generated_token_ids"] != reference_ids:
                        raise RuntimeError(
                            f"speculative output changed for {prompt_kind}, k={k}, "
                            f"ngram={ngram}"
                        )
                wall_ms = statistics.median(run["wall_ms"] for run in runs)
                representative = min(runs, key=lambda run: abs(run["wall_ms"] - wall_ms))
                rounds = representative["rounds"]
                proposed = representative["proposed_tokens"]
                mean_round_ms = wall_ms / rounds
                row = {
                    "k": k,
                    "ngram": ngram,
                    "mean_accepted_tokens_per_round": (
                        representative["accepted_tokens"] / rounds
                    ),
                    "mean_emitted_tokens_per_round": new_tokens / rounds,
                    "acceptance_rate": (
                        representative["accepted_tokens"] / proposed
                        if proposed
                        else 0.0
                    ),
                    "drafted_round_fraction": (
                        representative["drafted_rounds"] / rounds
                    ),
                    "wall_ms": wall_ms,
                    "wall_ms_per_emitted_token": wall_ms / new_tokens,
                    "speedup_over_ordinary": baseline_ms / wall_ms,
                    "ordinary_ms_per_emitted_token": baseline_ms_per_token,
                    "checkpoint_bytes": representative["checkpoint_bytes"],
                    "snapshot_ms_per_drafted_round": (
                        representative["snapshot_ms"]
                        / representative["drafted_rounds"]
                        if representative["drafted_rounds"]
                        else 0.0
                    ),
                    "restore_ms_per_restore": (
                        representative["restore_ms"]
                        / representative["restore_count"]
                        if representative["restore_count"]
                        else 0.0
                    ),
                    "isolated_snapshot_ms": representative[
                        "isolated_snapshot_ms"
                    ],
                    "isolated_restore_ms": representative["isolated_restore_ms"],
                    "isolated_snapshot_share_of_mean_round_percent": (
                        100.0
                        * representative["isolated_snapshot_ms"]
                        / mean_round_ms
                    ),
                    "isolated_restore_share_of_mean_round_percent": (
                        100.0
                        * representative["isolated_restore_ms"]
                        / mean_round_ms
                    ),
                    "snapshot_share_of_wall_percent": (
                        100.0 * representative["snapshot_ms"] / wall_ms
                    ),
                    "restore_share_of_wall_percent": (
                        100.0 * representative["restore_ms"] / wall_ms
                    ),
                    "mean_round_ms": mean_round_ms,
                    "rounds": rounds,
                    "drafted_rounds": representative["drafted_rounds"],
                    "restore_count": representative["restore_count"],
                    "token_ids_identical": True,
                }
                rows.append(row)
                print(
                    f"{prompt_kind:<15} k={k:<2} n={ngram:<2} "
                    f"accepted/round={row['mean_accepted_tokens_per_round']:.2f} "
                    f"acceptance={100.0 * row['acceptance_rate']:.1f}% "
                    f"ms/token={row['wall_ms_per_emitted_token']:.3f} "
                    f"speedup={row['speedup_over_ordinary']:.3f}x"
                )
        prompt_results.append(
            {
                "prompt_kind": prompt_kind,
                "prompt": prompt,
                "prompt_tokens": int(input_ids.shape[1]),
                "ordinary_cuda_graph_ms": baseline_ms,
                "ordinary_ms_per_emitted_token": baseline_ms_per_token,
                "rows": rows,
            }
        )

    kda_recurrent_bytes = 20 * 32 * 128 * 128 * 4
    payload = {
        "gpu": torch.cuda.get_device_name(device),
        "gpu_total_bytes": total_bytes,
        "memory_capped_to_bytes": BUDGET_BYTES if cap_memory else None,
        "load_seconds": load_seconds,
        "resident_weight_bytes": model.resident_weight_bytes,
        "new_tokens": new_tokens,
        "repeats": repeats,
        "k_values": k_values,
        "ngrams": ngrams,
        "kda_recurrent_bytes": kda_recurrent_bytes,
        "kda_copy_brief_rounded_ms": 0.046,
        "kda_copy_floor_ms_at_864_gbps": (
            1000.0 * kda_recurrent_bytes / 864_000_000_000
        ),
        "summary": (
            "Prompt lookup has no draft weights, but its acceptance depends on "
            "repetition. Compare both prompt classes and do not generalize from "
            "one prompt."
        ),
        "prompts": prompt_results,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return payload


@app.local_entrypoint()
def main(
    k_values: str = "2,4,8",
    ngrams: str = "2,3,4",
    new_tokens: int = 64,
    repeats: int = 3,
) -> None:
    payload = run_benchmark.remote(
        [int(value) for value in k_values.split(",") if value.strip()],
        [int(value) for value in ngrams.split(",") if value.strip()],
        new_tokens=new_tokens,
        repeats=repeats,
    )
    print("\nSpeculative decoding summary")
    print(
        f"GPU {payload['gpu']} | KDA snapshot floor "
        f"{payload['kda_copy_floor_ms_at_864_gbps']:.4f} ms"
    )
    for prompt in payload["prompts"]:
        print(
            f"\n{prompt['prompt_kind']} | ordinary "
            f"{prompt['ordinary_ms_per_emitted_token']:.3f} ms/token"
        )
        print(
            f"{'k':>3} {'n':>3} {'acc/round':>10} {'accept':>9} "
            f"{'ms/token':>10} {'speedup':>9} {'snap%':>8} {'restore%':>9}"
        )
        for row in prompt["rows"]:
            print(
                f"{row['k']:>3} {row['ngram']:>3} "
                f"{row['mean_accepted_tokens_per_round']:>10.2f} "
                f"{100.0 * row['acceptance_rate']:>8.1f}% "
                f"{row['wall_ms_per_emitted_token']:>10.3f} "
                f"{row['speedup_over_ordinary']:>8.3f}x "
                f"{row['isolated_snapshot_share_of_mean_round_percent']:>7.2f}% "
                f"{row['isolated_restore_share_of_mean_round_percent']:>8.2f}%"
            )
