# local-kimi

`local-kimi` contains two related pieces of work:

- `k3/` is a local protocol adapter for sending Anthropic Messages, OpenAI Chat
  Completions, and OpenAI Responses requests to a Kimi K3 endpoint.
- `engine/` and `research/` contain reference implementations, measurement
  tools, and models used to investigate local Kimi serving.

The proxy is alpha software. The engine work is research code, not a complete
or production-qualified Kimi K3 serving engine.

## Current evidence boundary

- There is no measured full-model Kimi K3 throughput result in this repository.
- There is no measured comparison with vLLM or any other serving engine.
- The Kimi-Linear benchmark report is currently marked `UNMEASURED`.
- Laptop throughput values are projections from a bandwidth model, not laptop
  measurements.
- Routing-aware batch composition has an implementation, but the checked-in
  result file says its simulation was not executed and claims no win.

The source of each statement is linked below. A projection or model output is
not presented as a measurement.

## Repository layout

| Path | Purpose |
| --- | --- |
| `k3/` | Protocol detection, request translation, streaming translation, tool parsing, reasoning preservation, recording, and replay |
| `tests/` | Offline unit, regression, conformance, and protocol tests |
| `engine/` | Kimi K3 and Kimi-Linear reference work, measurement code, and analytic models |
| `research/` | Exploratory scripts whose conclusions require the evidence stated in each file |
| `reference/` | Third-party Moonshot reference material used as a test oracle, under separate upstream terms |

## Install and run the proxy

Install the locked development environment with [uv](https://docs.astral.sh/uv/):

```bash
uv sync --frozen
uv run k3 --help
```

Run the protocol path against the built-in mock upstream without model weights
or a GPU:

```bash
uv run k3 serve --mock
```

To use a real engine, point the proxy at an OpenAI-compatible Kimi K3 endpoint:

```bash
uv run k3 serve --upstream http://127.0.0.1:8000/v1 --model k3
```

The proxy binds to `127.0.0.1` by default. If it is bound to a non-loopback
address, configure `--api-key` and review the exposure settings before use.

## Measured findings

The measurements below come from `engine/modal_kernelbench.py`. Warmup was
discarded and CUDA was synchronized. They are component measurements, not a
full serving benchmark.

| Finding | Hardware and conditions | Result | Source |
| --- | --- | --- | --- |
| PCIe expert streaming is not a viable Kimi K3 decode design | H100 80 GB HBM3; measured host-to-device bandwidth of 53.7 GB/s for 17.5 MB expert-sized transfers; 25.83 GB of routed expert bytes per token | 2.08 tok/s bound | [`engine/MEASUREMENTS.md`](engine/MEASUREMENTS.md#pcie-expert-streaming-is-dead-measured) |
| Naive dequantization dominates one routed-expert tensor path | H100 80 GB HBM3; naive PyTorch MXFP4 dequant of one `w1` tensor compared with a 3072 x 3584 batch-1 expert GEMM | 0.342 ms dequant versus 0.023 ms GEMM, about 15 times the GEMM time | [`engine/MEASUREMENTS.md`](engine/MEASUREMENTS.md#dequantization-dominates-the-expert-path-by-an-order-of-magnitude) |
| Expert GEMM amortizes across a larger batch | H100 80 GB HBM3; the same expert GEMM at batch 1 and batch 32 | 23.3 microseconds at batch 1 and 22.6 microseconds at batch 32 | [`engine/MEASUREMENTS.md`](engine/MEASUREMENTS.md#batching-is-nearly-free-on-the-compute-side) |
| The partial real-weight path executes, but is not a throughput result | H100 80 GB HBM3; actual Moonshot checkpoint; layers 11 through 13 only; four generated tokens; network-volume reads; no fusion | 11.010 seconds generation time and 25.245 GB peak allocation | [`engine/MEASUREMENTS.md`](engine/MEASUREMENTS.md#real-tokens-from-real-kimi-k3-weights) |

The three-layer run is a correctness result. Its timing must not be extrapolated
to all 93 layers, and its generated tokens are not evidence of model quality.

## Projected and unmeasured work

The Kimi-Linear laptop model uses exact parameter and byte arithmetic, followed
by a projected bandwidth roofline. It has not been measured on a laptop.

| Projection | Hardware and conditions | Projected result | Source |
| --- | --- | --- | --- |
| Kimi-Linear INT4 weight-only batch-1 decode | Complete 24.561 GB packed weight bank resident in a 100 GB/s DDR5 path; 60 percent attainment transferred from a different K3 calibration | 38.62 tok/s projected | [`engine/laptop/RESULTS.md`](engine/laptop/RESULTS.md#bandwidth-roofline) |
| Kimi-Linear INT4 weight-only batch-1 decode | Complete 24.561 GB packed weight bank and runtime state resident in a 900 GB/s dGPU memory path; 60 percent transferred attainment | 347.61 tok/s projected | [`engine/laptop/RESULTS.md`](engine/laptop/RESULTS.md#bandwidth-roofline) |

The laptop report also states that INT4 accuracy was not measured and that the
existing K3 loader is not a complete Kimi-Linear loader. The end-to-end
Kimi-Linear benchmark remains `UNMEASURED`; its checked-in runtime ranges are
modelled planning estimates only. See
[`engine/bench/RESULTS.md`](engine/bench/RESULTS.md).

`engine/batching/` contains analytic throughput models calibrated with measured
component inputs. Its generated throughput tables are modelled, not measured.
`engine/scheduling/RESULTS.md` currently records that the routing-aware
composition simulation was not run, so no scheduling throughput gain is
claimed here.

## Tests and supported matrix

The package declares Python 3.10 through 3.13 on Linux, macOS, and Windows.
`.github/workflows/ci.yml` enforces that contract with this matrix:

| Operating system runner | Python versions |
| --- | --- |
| Ubuntu 24.04 | 3.10, 3.11, 3.12, 3.13 |
| macOS 14 | 3.10, 3.11, 3.12, 3.13 |
| Windows Server 2022 | 3.10, 3.11, 3.12, 3.13 |

The workflow sets Python UTF-8 mode, installs from the lockfile with uv, runs
Ruff, builds the package, and runs the offline suite with tests marked `gpu`,
`network`, or `weights` deselected. This describes the CI contract, not a claim
that a public CI run has already passed.

Run the same blocking checks locally with:

```bash
uv sync --frozen
uv run --no-sync ruff check .
uv run --no-sync python -m pytest -m "not gpu and not network and not weights"
```

## Third-party reference material

The files under `reference/` include Moonshot code and a chat template retained
so tests can compare this project's rendering with Moonshot's implementation.
They are not imported by the `k3` runtime and they are not relicensed under this
project's Apache-2.0 licence.

See [`reference/PROVENANCE.md`](reference/PROVENANCE.md) for exact upstream
revisions, file relationships, fetched licence text, and redistribution terms.

## Contributing and security

See [CONTRIBUTING.md](CONTRIBUTING.md) for the development and evidence rules.
Report security issues through the process in [SECURITY.md](SECURITY.md).

## Licence

Original project code and documentation are licensed under Apache-2.0. See
[LICENSE](LICENSE). Third-party files in `reference/` remain under the upstream
terms documented in [reference/PROVENANCE.md](reference/PROVENANCE.md).
