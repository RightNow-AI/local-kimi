# Routing-aware batch composition results

> Status: SIMULATED, not measured on real K3 router traces or a real scheduler.

## Fixed arithmetic

- Batch-1 routed traffic is `16 * 17,547,264 * 92 = 25,829,572,608` bytes, or `25.829572608` GB per token.
- Simulated bytes saved per token are `(random expert-layer pairs - greedy expert-layer pairs) * 17,547,264 / tokens served`.
- The throughput feedback uses `HardwareConfig.predict` from `engine/batching/union_model.py` with `epyc-12ch-5090` and replaces only the observed union-derived routed-byte and dequant terms.

## Simulation method

- Each row runs `256` scheduling rounds with seed-controlled routes.
- Uniform routing is exact uniform top-k sampling.
- Skewed routing uses the union model's inclusion probabilities with randomized systematic fixed-size sampling. It is not a measured K3 trace.
- Real grouped-top-k routing correlations are not modeled.
- The optimization objective is the sum of per-layer unions, which equals average union times layer count and directly tracks routed bytes.
- Fairness includes served and still-outstanding tokens. Worst deferral is the maximum number of completed scheduling rounds a token waited.

## Simulated sweep

| prior | arrivals | B | P | analytic union/layer | random union/layer | greedy union/layer | local/end-to-end reduction | saved MB/token | throughput gain | worst rounds random/greedy | worst wait ms random/greedy | guard fallbacks |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| uniform | saturated | 8 | 8 | 120.279 | 120.303 | 120.303 | 0.00%/0.00% | 0.000 | 0.00% | 0/0 | 0.000/0.000 | 0 |
| uniform | saturated | 8 | 16 | 120.279 | 120.263 | 119.602 | 0.57%/0.55% | 133.361 | 0.35% | 9/5 | 6173.220/3418.566 | 1 |
| uniform | saturated | 8 | 32 | 120.279 | 120.268 | 119.231 | 0.87%/0.86% | 209.316 | 0.55% | 34/10 | 23317.134/6822.354 | 0 |
| uniform | saturated | 16 | 16 | 224.412 | 224.467 | 224.467 | 0.00%/0.00% | 0.000 | 0.00% | 0/0 | 0.000/0.000 | 0 |
| uniform | saturated | 16 | 32 | 224.412 | 224.448 | 222.531 | 0.85%/0.85% | 193.388 | 0.65% | 13/6 | 13776.145/6313.905 | 0 |
| uniform | saturated | 16 | 64 | 224.412 | 224.329 | 221.530 | 1.27%/1.25% | 282.500 | 0.96% | 37/13 | 39186.182/13632.842 | 0 |
| uniform | saturated | 32 | 32 | 392.619 | 392.529 | 392.529 | 0.00%/0.00% | 0.000 | 0.00% | 0/0 | 0.000/0.000 | 0 |
| uniform | saturated | 32 | 64 | 392.619 | 392.546 | 388.156 | 1.12%/1.12% | 221.455 | 0.96% | 14/8 | 23279.432/13174.070 | 0 |
| uniform | saturated | 32 | 128 | 392.619 | 392.630 | 385.853 | 1.72%/1.73% | 341.915 | 1.48% | 39/15 | 64849.293/24586.159 | 0 |
| uniform | bursty | 8 | 8 | 120.279 | 120.295 | 120.295 | 0.00%/0.00% | 0.000 | 0.00% | 1/1 | 687.620/687.620 | 0 |
| uniform | bursty | 8 | 16 | 120.279 | 120.319 | 119.599 | 0.56%/0.60% | 145.408 | 0.38% | 10/6 | 6859.905/4102.131 | 0 |
| uniform | bursty | 8 | 32 | 120.279 | 120.267 | 119.215 | 0.85%/0.87% | 212.238 | 0.55% | 25/11 | 17145.200/7507.167 | 0 |
| uniform | bursty | 16 | 16 | 224.412 | 224.432 | 224.432 | 0.00%/0.00% | 0.000 | 0.00% | 1/1 | 1065.939/1065.939 | 0 |
| uniform | bursty | 16 | 32 | 224.412 | 224.394 | 222.535 | 0.84%/0.83% | 187.532 | 0.63% | 13/6 | 13770.919/6324.239 | 0 |
| uniform | bursty | 16 | 64 | 224.412 | 224.401 | 221.566 | 1.26%/1.26% | 286.068 | 0.97% | 39/14 | 41305.427/14691.762 | 0 |
| uniform | bursty | 32 | 32 | 392.619 | 392.716 | 392.716 | 0.00%/0.00% | 0.000 | 0.00% | 1/1 | 1669.877/1669.877 | 0 |
| uniform | bursty | 32 | 64 | 392.619 | 392.601 | 388.225 | 1.11%/1.11% | 220.770 | 0.95% | 12/8 | 19970.522/13178.437 | 0 |
| uniform | bursty | 32 | 128 | 392.619 | 392.681 | 385.927 | 1.70%/1.72% | 340.745 | 1.48% | 29/15 | 48240.908/24590.059 | 0 |
| uniform | sparse | 8 | 8 | 62.306 | 62.304 | 62.304 | 0.00%/0.00% | 0.000 | 0.00% | 0/0 | 0.000/0.000 | 0 |
| uniform | sparse | 8 | 16 | 62.306 | 62.304 | 62.304 | 0.00%/0.00% | 0.000 | 0.00% | 0/0 | 0.000/0.000 | 0 |
| uniform | sparse | 8 | 32 | 62.306 | 62.304 | 62.304 | 0.00%/0.00% | 0.000 | 0.00% | 0/0 | 0.000/0.000 | 0 |
| uniform | sparse | 16 | 16 | 120.279 | 120.278 | 120.278 | 0.00%/0.00% | 0.000 | 0.00% | 0/0 | 0.000/0.000 | 0 |
| uniform | sparse | 16 | 32 | 120.279 | 120.278 | 120.278 | 0.00%/0.00% | 0.000 | 0.00% | 0/0 | 0.000/0.000 | 0 |
| uniform | sparse | 16 | 64 | 120.279 | 120.278 | 120.278 | 0.00%/0.00% | 0.000 | 0.00% | 0/0 | 0.000/0.000 | 0 |
| uniform | sparse | 32 | 32 | 224.412 | 224.358 | 224.358 | 0.00%/0.00% | 0.000 | 0.00% | 0/0 | 0.000/0.000 | 0 |
| uniform | sparse | 32 | 64 | 224.412 | 224.358 | 224.358 | 0.00%/0.00% | 0.000 | 0.00% | 0/0 | 0.000/0.000 | 0 |
| uniform | sparse | 32 | 128 | 224.412 | 224.358 | 224.358 | 0.00%/0.00% | 0.000 | 0.00% | 0/0 | 0.000/0.000 | 0 |
| zipf-1 | saturated | 8 | 8 | 87.828 | 87.813 | 87.813 | 0.00%/0.00% | 0.000 | 0.00% | 0/0 | 0.000/0.000 | 0 |
| zipf-1 | saturated | 8 | 16 | 87.828 | 87.820 | 87.313 | 1.12%/0.58% | 102.430 | 0.32% | 10/7 | 5697.730/3979.614 | 2 |
| zipf-1 | saturated | 8 | 32 | 87.828 | 87.836 | 87.039 | 1.73%/0.91% | 160.796 | 0.50% | 26/15 | 14809.895/8506.353 | 0 |
| zipf-1 | saturated | 16 | 16 | 147.375 | 147.381 | 147.381 | 0.00%/0.00% | 0.000 | 0.00% | 0/0 | 0.000/0.000 | 0 |
| zipf-1 | saturated | 16 | 32 | 147.375 | 147.375 | 146.358 | 1.54%/0.69% | 102.563 | 0.47% | 12/10 | 9405.782/7794.433 | 0 |
| zipf-1 | saturated | 16 | 64 | 147.375 | 147.398 | 145.800 | 2.34%/1.08% | 161.168 | 0.74% | 27/25 | 21129.758/19470.524 | 0 |
| zipf-1 | saturated | 32 | 32 | 238.647 | 238.678 | 238.678 | 0.00%/0.00% | 0.000 | 0.00% | 0/0 | 0.000/0.000 | 0 |
| zipf-1 | saturated | 32 | 64 | 238.647 | 238.650 | 236.612 | 1.89%/0.85% | 102.775 | 0.66% | 12/13 | 13333.562/14351.929 | 0 |
| zipf-1 | saturated | 32 | 128 | 238.647 | 238.618 | 235.489 | 2.89%/1.31% | 157.838 | 1.02% | 27/36 | 29999.354/39593.901 | 0 |
| zipf-1 | bursty | 8 | 8 | 87.828 | 87.810 | 87.810 | 0.00%/0.00% | 0.000 | 0.00% | 1/1 | 576.215/576.215 | 0 |
| zipf-1 | bursty | 8 | 16 | 87.828 | 87.837 | 87.325 | 1.10%/0.58% | 103.210 | 0.32% | 14/8 | 7976.775/4539.568 | 1 |
| zipf-1 | bursty | 8 | 32 | 87.828 | 87.852 | 87.047 | 1.65%/0.92% | 162.321 | 0.51% | 26/16 | 14803.266/9066.230 | 0 |
| zipf-1 | bursty | 16 | 16 | 147.375 | 147.375 | 147.375 | 0.00%/0.00% | 0.000 | 0.00% | 1/1 | 790.252/790.252 | 0 |
| zipf-1 | bursty | 16 | 32 | 147.375 | 147.382 | 146.392 | 1.51%/0.67% | 99.941 | 0.46% | 15/12 | 11738.521/9362.889 | 0 |
| zipf-1 | bursty | 16 | 64 | 147.375 | 147.352 | 145.813 | 2.31%/1.04% | 155.291 | 0.71% | 34/24 | 26620.815/18666.663 | 0 |
| zipf-1 | bursty | 32 | 32 | 238.647 | 238.756 | 238.756 | 0.00%/0.00% | 0.000 | 0.00% | 1/1 | 1118.659/1118.659 | 0 |
| zipf-1 | bursty | 32 | 64 | 238.647 | 238.771 | 236.651 | 1.79%/0.89% | 106.976 | 0.69% | 14/13 | 15558.597/14356.530 | 0 |
| zipf-1 | bursty | 32 | 128 | 238.647 | 238.809 | 235.511 | 2.73%/1.38% | 166.378 | 1.08% | 32/33 | 35560.753/36282.766 | 0 |
| zipf-1 | sparse | 8 | 8 | 50.921 | 50.918 | 50.918 | 0.00%/0.00% | 0.000 | 0.00% | 0/0 | 0.000/0.000 | 0 |
| zipf-1 | sparse | 8 | 16 | 50.921 | 50.918 | 50.918 | 0.00%/0.00% | 0.000 | 0.00% | 0/0 | 0.000/0.000 | 0 |
| zipf-1 | sparse | 8 | 32 | 50.921 | 50.918 | 50.918 | 0.00%/0.00% | 0.000 | 0.00% | 0/0 | 0.000/0.000 | 0 |
| zipf-1 | sparse | 16 | 16 | 87.828 | 87.868 | 87.868 | 0.00%/0.00% | 0.000 | 0.00% | 0/0 | 0.000/0.000 | 0 |
| zipf-1 | sparse | 16 | 32 | 87.828 | 87.868 | 87.868 | 0.00%/0.00% | 0.000 | 0.00% | 0/0 | 0.000/0.000 | 0 |
| zipf-1 | sparse | 16 | 64 | 87.828 | 87.868 | 87.868 | 0.00%/0.00% | 0.000 | 0.00% | 0/0 | 0.000/0.000 | 0 |
| zipf-1 | sparse | 32 | 32 | 147.375 | 147.359 | 147.359 | 0.00%/0.00% | 0.000 | 0.00% | 0/0 | 0.000/0.000 | 0 |
| zipf-1 | sparse | 32 | 64 | 147.375 | 147.359 | 147.359 | 0.00%/0.00% | 0.000 | 0.00% | 0/0 | 0.000/0.000 | 0 |
| zipf-1 | sparse | 32 | 128 | 147.375 | 147.359 | 147.359 | 0.00%/0.00% | 0.000 | 0.00% | 0/0 | 0.000/0.000 | 0 |
| dirichlet-0.3 | saturated | 8 | 8 | 102.215 | 102.170 | 102.170 | 0.00%/0.00% | 0.000 | 0.00% | 0/0 | 0.000/0.000 | 0 |
| dirichlet-0.3 | saturated | 8 | 16 | 102.215 | 102.187 | 101.297 | 1.00%/0.87% | 179.620 | 0.52% | 11/5 | 6834.950/3094.020 | 1 |
| dirichlet-0.3 | saturated | 8 | 32 | 102.215 | 102.164 | 100.844 | 1.51%/1.29% | 266.268 | 0.77% | 29/12 | 18007.635/7390.265 | 0 |
| dirichlet-0.3 | saturated | 16 | 16 | 167.696 | 167.646 | 167.646 | 0.00%/0.00% | 0.000 | 0.00% | 0/0 | 0.000/0.000 | 0 |
| dirichlet-0.3 | saturated | 16 | 32 | 167.696 | 167.690 | 166.013 | 1.33%/1.00% | 169.175 | 0.71% | 10/7 | 8564.056/5962.689 | 0 |
| dirichlet-0.3 | saturated | 16 | 64 | 167.696 | 167.651 | 165.148 | 2.00%/1.49% | 252.538 | 1.06% | 31/16 | 26516.268/13560.279 | 0 |
| dirichlet-0.3 | saturated | 32 | 32 | 250.323 | 250.343 | 250.343 | 0.00%/0.00% | 0.000 | 0.00% | 0/0 | 0.000/0.000 | 0 |
| dirichlet-0.3 | saturated | 32 | 64 | 250.323 | 250.384 | 247.809 | 1.58%/1.03% | 129.910 | 0.81% | 11/10 | 12692.071/11451.244 | 0 |
| dirichlet-0.3 | saturated | 32 | 128 | 250.323 | 250.349 | 246.488 | 2.41%/1.54% | 194.802 | 1.22% | 33/25 | 38034.217/28479.602 | 0 |
| dirichlet-0.3 | bursty | 8 | 8 | 102.215 | 102.225 | 102.225 | 0.00%/0.00% | 0.000 | 0.00% | 1/1 | 625.386/625.386 | 0 |
| dirichlet-0.3 | bursty | 8 | 16 | 102.215 | 102.229 | 101.354 | 0.96%/0.86% | 176.535 | 0.51% | 13/6 | 8070.515/3711.295 | 2 |
| dirichlet-0.3 | bursty | 8 | 32 | 102.215 | 102.190 | 100.863 | 1.54%/1.30% | 267.664 | 0.77% | 28/13 | 17374.879/8014.130 | 1 |
| dirichlet-0.3 | bursty | 16 | 16 | 167.696 | 167.773 | 167.773 | 0.00%/0.00% | 0.000 | 0.00% | 1/1 | 861.572/861.572 | 0 |
| dirichlet-0.3 | bursty | 16 | 32 | 167.696 | 167.759 | 166.099 | 1.33%/0.99% | 167.436 | 0.70% | 14/8 | 11994.124/6804.374 | 0 |
| dirichlet-0.3 | bursty | 16 | 64 | 167.696 | 167.704 | 165.170 | 1.98%/1.51% | 255.669 | 1.07% | 39/18 | 33375.779/15267.787 | 0 |
| dirichlet-0.3 | bursty | 32 | 32 | 250.323 | 250.337 | 250.337 | 0.00%/0.00% | 0.000 | 0.00% | 1/1 | 1159.759/1159.759 | 0 |
| dirichlet-0.3 | bursty | 32 | 64 | 250.323 | 250.342 | 247.761 | 1.60%/1.03% | 130.176 | 0.81% | 13/11 | 14982.070/12575.908 | 0 |
| dirichlet-0.3 | bursty | 32 | 128 | 250.323 | 250.365 | 246.470 | 2.41%/1.56% | 196.481 | 1.23% | 42/26 | 48392.538/29600.952 | 0 |
| dirichlet-0.3 | sparse | 8 | 8 | 57.662 | 57.662 | 57.662 | 0.00%/0.00% | 0.000 | 0.00% | 0/0 | 0.000/0.000 | 0 |
| dirichlet-0.3 | sparse | 8 | 16 | 57.662 | 57.662 | 57.662 | 0.00%/0.00% | 0.000 | 0.00% | 0/0 | 0.000/0.000 | 0 |
| dirichlet-0.3 | sparse | 8 | 32 | 57.662 | 57.662 | 57.662 | 0.00%/0.00% | 0.000 | 0.00% | 0/0 | 0.000/0.000 | 0 |
| dirichlet-0.3 | sparse | 16 | 16 | 102.215 | 102.276 | 102.276 | 0.00%/0.00% | 0.000 | 0.00% | 0/0 | 0.000/0.000 | 0 |
| dirichlet-0.3 | sparse | 16 | 32 | 102.215 | 102.276 | 102.276 | 0.00%/0.00% | 0.000 | 0.00% | 0/0 | 0.000/0.000 | 0 |
| dirichlet-0.3 | sparse | 16 | 64 | 102.215 | 102.276 | 102.276 | 0.00%/0.00% | 0.000 | 0.00% | 0/0 | 0.000/0.000 | 0 |
| dirichlet-0.3 | sparse | 32 | 32 | 167.696 | 167.733 | 167.733 | 0.00%/0.00% | 0.000 | 0.00% | 0/0 | 0.000/0.000 | 0 |
| dirichlet-0.3 | sparse | 32 | 64 | 167.696 | 167.733 | 167.733 | 0.00%/0.00% | 0.000 | 0.00% | 0/0 | 0.000/0.000 | 0 |
| dirichlet-0.3 | sparse | 32 | 128 | 167.696 | 167.733 | 167.733 | 0.00%/0.00% | 0.000 | 0.00% | 0/0 | 0.000/0.000 | 0 |

## Where a larger composition pool stops paying

- At `P <= B`, every pending token is selected, so composition has no choice and must produce exactly the random union.
- `uniform`, `saturated`, B=8: stop at P=8 before P=16 under the declared 1% rule. Incremental throughput was 0.37% and worst deferral increased by 5 rounds.
- `uniform`, `saturated`, B=16: stop at P=16 before P=32 under the declared 1% rule. Incremental throughput was 0.66% and worst deferral increased by 6 rounds.
- `uniform`, `saturated`, B=32: stop at P=32 before P=64 under the declared 1% rule. Incremental throughput was 0.95% and worst deferral increased by 8 rounds.
- `uniform`, `bursty`, B=8: stop at P=8 before P=16 under the declared 1% rule. Incremental throughput was 0.37% and worst deferral increased by 5 rounds.
- `uniform`, `bursty`, B=16: stop at P=16 before P=32 under the declared 1% rule. Incremental throughput was 0.65% and worst deferral increased by 5 rounds.
- `uniform`, `bursty`, B=32: stop at P=32 before P=64 under the declared 1% rule. Incremental throughput was 0.98% and worst deferral increased by 7 rounds.
- `zipf-1`, `saturated`, B=8: stop at P=8 before P=16 under the declared 1% rule. Incremental throughput was 0.32% and worst deferral increased by 7 rounds.
- `zipf-1`, `saturated`, B=16: stop at P=16 before P=32 under the declared 1% rule. Incremental throughput was 0.47% and worst deferral increased by 10 rounds.
- `zipf-1`, `saturated`, B=32: stop at P=32 before P=64 under the declared 1% rule. Incremental throughput was 0.67% and worst deferral increased by 13 rounds.
- `zipf-1`, `bursty`, B=8: stop at P=8 before P=16 under the declared 1% rule. Incremental throughput was 0.31% and worst deferral increased by 7 rounds.
- `zipf-1`, `bursty`, B=16: stop at P=16 before P=32 under the declared 1% rule. Incremental throughput was 0.45% and worst deferral increased by 11 rounds.
- `zipf-1`, `bursty`, B=32: stop at P=32 before P=64 under the declared 1% rule. Incremental throughput was 0.68% and worst deferral increased by 12 rounds.
- `dirichlet-0.3`, `saturated`, B=8: stop at P=8 before P=16 under the declared 1% rule. Incremental throughput was 0.51% and worst deferral increased by 5 rounds.
- `dirichlet-0.3`, `saturated`, B=16: stop at P=16 before P=32 under the declared 1% rule. Incremental throughput was 0.69% and worst deferral increased by 7 rounds.
- `dirichlet-0.3`, `saturated`, B=32: stop at P=32 before P=64 under the declared 1% rule. Incremental throughput was 0.79% and worst deferral increased by 10 rounds.
- `dirichlet-0.3`, `bursty`, B=8: stop at P=8 before P=16 under the declared 1% rule. Incremental throughput was 0.51% and worst deferral increased by 5 rounds.
- `dirichlet-0.3`, `bursty`, B=16: stop at P=16 before P=32 under the declared 1% rule. Incremental throughput was 0.71% and worst deferral increased by 7 rounds.
- `dirichlet-0.3`, `bursty`, B=32: stop at P=32 before P=64 under the declared 1% rule. Incremental throughput was 0.81% and worst deferral increased by 10 rounds.

## Honest read

- The best simulated modeled throughput gain is 1.48% for `uniform`, `saturated`, B=32, P=128. This is enough to justify a bounded trace-driven prototype, not production integration.
- The largest simulated fairness cost is +9 worst-case deferral rounds relative to random. A real scheduler needs an explicit age or deadline cap.
- Real-model routing traces, router lookahead latency, request cancellation, and continuous-time arrivals remain unmeasured.
- Composer CPU time and real grouped-top-k routing correlations are also unmeasured, so a small modeled throughput gain may disappear in implementation overhead.
