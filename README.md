# local-kimi

![local-kimi](assets/banner.jpeg)

<sub>The sketch is illustrative. The model behind this repository's measured 32 GiB result is Kimi-Linear-48B on an NVIDIA L40S under a hard 32 GiB process cap. Kimi K3 is 2.78T parameters and roughly 1.56 TB of weights. It does not run on a laptop, and this project does not claim that it does.</sub>

Point any coding agent at a local Kimi model and it works: `k3` translates between the agent's protocol and the model's.

## The problem

Local model servers usually speak OpenAI Chat Completions. Claude Code speaks Anthropic Messages. Codex speaks OpenAI Responses. Point the wrong client at the wrong endpoint and the request fails. `k3` sits between them, detects the caller on each request, and translates the request and response.

## Protocol support

The protocol landscape has changed. Current releases of [vLLM](https://docs.vllm.ai/en/stable/serving/online_serving/), [llama.cpp](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md), and [Ollama](https://docs.ollama.com/api/openai-compatibility) now document all three protocol families. `k3` is different because it is a standalone adapter for an existing OpenAI Chat Completions backend.

| Capability | vLLM | llama.cpp | Ollama | `k3` |
| --- | --- | --- | --- | --- |
| OpenAI Chat Completions | Documented | Documented | Documented | Verified |
| OpenAI Responses | Documented | Documented | Documented, non-stateful | Verified |
| Anthropic Messages | Documented | Documented | Documented | Verified |
| Serves model inference itself | Yes | Yes | Yes | No, proxy only |
| Detects a client preset from route, headers, user agent, and body hints | Not documented | Not documented | Not documented | Verified |
| Translates tool calls to and from a separate OpenAI Chat upstream | Not applicable | Not applicable | Not applicable | Verified |
| Ledger-backed reasoning restoration across client turns | Not documented | Not documented | Not documented | Verified when the backend supplies reasoning |

The vendor columns describe their public server documentation checked on July 29, 2026. "Not documented" is not a claim that the behavior is impossible. Ollama's [Anthropic compatibility](https://docs.ollama.com/api/anthropic-compatibility) is documented separately. The `k3` column is verified against [`k3/dialects/`](k3/dialects/), [`k3/detect.py`](k3/detect.py), [`k3/presets.py`](k3/presets.py), and [`k3/reasoning.py`](k3/reasoning.py).

## Quickstart with Claude Code

This path assumes an OpenAI-compatible Kimi server is already listening at `http://127.0.0.1:8000/v1`.

```bash
git clone https://github.com/RightNow-AI/local-kimi
cd local-kimi
uv sync --frozen
uv run k3 serve --upstream http://127.0.0.1:8000/v1 --model kimi-linear --reasoning-field inline
```

In another terminal, from the project Claude Code should work on:

```bash
export ANTHROPIC_BASE_URL=http://localhost:8080
export ANTHROPIC_AUTH_TOKEN=local
claude
```

The serve flags above are defined in [`k3/cli.py`](k3/cli.py). See the [full quickstart](docs/QUICKSTART.md) for llama.cpp setup, the model download, and troubleshooting.

## Measured here

- `k3` added **0.231 ms per request** in a local CPU-only ASGI benchmark that sent the same request directly to a stub backend and through the proxy. The benchmark used no socket or network. The host hardware was not recorded, so this number describes that run rather than a portable latency guarantee.
- The selective INT4 artifact is **28,803,304,448 bytes**, which is **3.41x smaller** than the **98,245,528,576-byte BF16 source tensor storage**. It was built and verified from the real checkpoint on one NVIDIA H100. See [`engine/quant/QUANTIZATION-RESULTS.md`](engine/quant/QUANTIZATION-RESULTS.md).
- The engine decoded at **35.71 tokens per second** on one NVIDIA L40S. The run used one stream, an 8-token prompt, 64 generated tokens, greedy decoding, three repeats, and a hard 32 GiB process cap. Generated token ids were byte-identical to its pre-optimisation reference. See [`engine/klinear/DECODE-BENCHMARK.md`](engine/klinear/DECODE-BENCHMARK.md).

The **35.71 tokens per second** L40S engine result and llama.cpp's reported **roughly 32 tokens per second** on an RTX 3090 are not a comparison. The hardware and harnesses differ.

The INT4 artifact is not equivalent in quality to the BF16 source. [`engine/accuracy/RESULTS.md`](engine/accuracy/RESULTS.md) records the measured difference.

## Built by RunInfra

This project is built and maintained by [RunInfra](https://runinfra.ai/).

## Repository contents

| Path | Contents |
| --- | --- |
| `k3/` | Protocol detection, translation, streaming, tool calls, and reasoning restoration |
| `docs/` | Setup guides for Claude Code, Codex, the OpenAI SDK, and the adapter architecture |
| `engine/` | Kimi and Kimi-Linear experiments, measurements, and reference serving work |
| `tests/` | Protocol, regression, conformance, and overhead checks |
| `research/` | Exploratory scripts and recorded conclusions |
| `scripts/` | Repository helper scripts |
| `reference/` | Third-party Moonshot material used as a test oracle under its upstream terms |
| `assets/` | Images used by the documentation |

## Licence

See [LICENSE](LICENSE) and [LICENCE-DECISION.md](LICENCE-DECISION.md). Third-party files in `reference/` keep their upstream terms.
