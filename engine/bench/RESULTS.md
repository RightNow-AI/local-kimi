# Kimi-Linear engine benchmark results

Status: **UNMEASURED**

No Modal run has been executed in this worktree. The benchmark code is prepared for the orchestrator, but no number below may be treated as measured until the compare command completes and rewrites this file.

## Run order

```powershell
modal run engine/modal_bench.py --action download --revision main
modal run engine/modal_bench.py --action reference --gpu H200 --revision main
modal run engine/modal_bench.py --action compare --gpu H200 --engine-factory engine.bench.candidate:build_kimi_linear_runner
```

`H200` is the default because the roughly 96 GB BF16 checkpoint does not fit one 80 GB H100. The GPU remains a parameter and no 8-GPU shape is hardcoded.

## Modelled runtime, not measured

- Download: MODELLED 30 to 120 minutes, depending on HuggingFace and Modal Volume throughput.
- HuggingFace reference capture: MODELLED 30 to 120 minutes on one H200.
- Mixed engine candidate comparison: MODELLED 2 to 8 hours on one H200 because the plain-PyTorch KDA recurrence and expert dispatch prioritize correctness over speed.

The Modal timeout is deliberately longer than these modelled ranges. Actual durations will replace this section after the run.

## Claim boundary

A passing result will measure `moonshotai/Kimi-Linear-48B-A3B-Instruct`, not full Kimi K3. The built-in candidate is intentionally partial: it replaces compatible KDA and latent-MoE components with `engine.k3ref`, records router-only fallbacks, and leaves embeddings, dense layers, MLA, residual plumbing, final normalization, and the LM head on HuggingFace. The generated report will list exact covered layers and any incompatibility.
