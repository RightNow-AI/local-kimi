# Routing-aware batch composition results

> Status: implementation complete, simulation not executed in this lane.

No composition win is claimed in this file yet. The lane brief assigns simulation execution to
the orchestrator, and this lane ran no test runner, simulation, GPU, network, weight, Modal, or git
command.

## Arithmetic that is already fixed

- Batch-1 routed traffic is `16 * 17,547,264 * 92 = 25,829,572,608` bytes, or
  `25.829572608` GB per token.
- For an observed run, saved bytes per token are
  `(random expert-layer pairs - greedy expert-layer pairs) * 17,547,264 / tokens served`.
- Per-layer unions are measured separately. Summing them counts each expert-layer pair exactly
  once, so the byte formula does not multiply by 92 again.
- Modeled throughput is produced by `HardwareConfig.predict` in
  `engine/batching/union_model.py`, with only the observed union-derived routed-byte and dequant
  terms replaced.

## Required orchestrator run

Run this from the repository root to replace this template with a 64-round simulated sweep:

```powershell
python -m engine.scheduling.simulate --rounds 64 --format markdown --output engine/scheduling/RESULTS.md
```

For a more stable report, use 256 rounds:

```powershell
python -m engine.scheduling.simulate --rounds 256 --format markdown --output engine/scheduling/RESULTS.md
```

The generated report includes:

- uniform, Zipf, and Dirichlet simulated routing;
- saturated, bursty, and sparse round-based arrivals;
- `B = 8, 16, 32` and `P = B, 2B, 4B`;
- same-pool and end-to-end union reduction;
- analytic, random, and greedy union per layer;
- bytes saved per token and modeled throughput feedback;
- worst-case deferral rounds and wall-time waits for random and greedy policies;
- the first pool size where incremental modeled throughput is under 1% while worst-case deferral
  increases.

## Honest read before execution

Whether this belongs in a real scheduler is unresolved. A positive simulation is only enough to
justify collecting real per-layer route traces and prototyping a bounded, age-capped scheduler.
Router lookahead latency, continuous-time arrivals, request cancellation, and real routing
correlations remain unmeasured. The simulator also does not reconstruct grouped-top-k correlation
or charge composer CPU time. A small or negative end-to-end reduction should stop the work.
