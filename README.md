# local-kimi

![local-kimi](assets/banner.jpeg)

`local-kimi` is primarily `k3/`, a protocol adapter between a local Kimi endpoint
and clients that speak Anthropic Messages, OpenAI Chat Completions, or OpenAI
Responses. It detects the client dialect per request, translates tools and
streams, and carries reasoning through client round trips when the backend
provides a separate or recognised inline reasoning channel.

Use it when llama.cpp already serves Kimi but Claude Code, Codex, or an OpenAI
SDK client needs its own wire protocol. With llama.cpp listening on port 8000,
the adapter command is:

```bash
uv run k3 serve --upstream http://127.0.0.1:8000/v1 --model kimi-linear --reasoning-field inline
```

Start with the [five-minute quickstart](docs/QUICKSTART.md). Worked client setup
is in [docs/CLAUDE-CODE.md](docs/CLAUDE-CODE.md),
[docs/CODEX.md](docs/CODEX.md), and
[docs/OPENAI-SDK.md](docs/OPENAI-SDK.md).

## Backend position

For local Kimi-Linear-48B inference, use llama.cpp as the backend. A GGUF exists
at
[AaryanK/Kimi-Linear-48B-A3B-Instruct-GGUF](https://huggingface.co/AaryanK/Kimi-Linear-48B-A3B-Instruct-GGUF)
with Q4_K_S at 27.9 GB, and
[llama.cpp PR 17592](https://github.com/ggml-org/llama.cpp/pull/17592) reports
roughly 32 tokens per second on an RTX 3090. The engine in this repository is
research code and measures 0.67 to 3.39 tokens per second. It is not the
recommended local backend.

The distinct component here is `k3/`. llama.cpp provides a raw OpenAI-compatible
endpoint. `k3` adds Anthropic Messages and OpenAI Responses surfaces alongside
Chat Completions, per-request dialect detection, client-shaped streaming and
errors, tool-call translation, and a reasoning ledger. See
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Current evidence boundary

What is measured:

- The Kimi-Linear INT4 artifact exists and was built on an H100 from the real
  BF16 checkpoint. 98,245,528,576 bytes of source tensors become
  28,803,304,448, a 3.41x reduction in weight bytes. Planned and actual byte
  totals agree exactly. See `engine/quant/QUANTIZATION-RESULTS.md`.
- This engine loads and runs that artifact. On one H100 80GB, all 27 layers and
  all 256 experts per MoE layer, resident weight bytes equal the checkpoint's
  tensor storage exactly at 28,803,304,448, peak reserved device memory during
  generation is 30,511,464,448 bytes (28.42 GiB), and the continuation it
  produces is coherent English. One short greedy generation from one prompt, so
  it establishes that the stack is wired correctly and nothing about quality or
  throughput. See `engine/klinear/INT4-SERVING-RESULTS.md`.
- Stock vLLM 0.26.0 will not serve this model below BF16. On one H200, the
  as-shipped checkpoint loads and generates, while the bitsandbytes 4-bit path
  is refused at load with `Model KimiLinearForCausalLM does not support
  BitsAndBytes quantization yet. No 'packed_modules_mapping' found.` This tests
  bitsandbytes, the only 4-bit path needing no pre-built artifact; AWQ, GPTQ and
  compressed-tensors all require a 4-bit checkpoint that does not exist publicly
  for this model.
- Routing-aware batch composition has now been simulated at 256 rounds. Its best
  case is a 1.72 percent union reduction and 1.48 percent modelled throughput,
  costing up to 39 rounds of deferral and 64.8 seconds of worst-case wait. It is
  recorded as a negative result. It remains a simulation, not a measurement on
  real router traces.

What is not measured:

- There is no measured full-model Kimi K3 throughput result in this repository.
- There is no measured speed comparison with vLLM or any other serving engine.
  The footprint statement above is about bytes, not about speed.
- There is no measured quality result for the INT4 artifact yet. Weight-space
  error is not model quality. `engine/accuracy/` exists to settle it.
- Laptop throughput values are projections from a bandwidth model, not laptop
  measurements.
- The residency frontier is derived from source, not measured on a device.

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

To use a real backend, point the proxy at an OpenAI-compatible Kimi endpoint:

```bash
uv run k3 serve --upstream http://127.0.0.1:8000/v1 --model kimi-linear --reasoning-field inline
```

The complete llama.cpp setup is in [docs/QUICKSTART.md](docs/QUICKSTART.md).

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
