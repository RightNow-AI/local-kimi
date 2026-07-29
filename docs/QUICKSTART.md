# Quickstart

This guide starts a Q4_K_S Kimi-Linear-48B GGUF with llama.cpp, places `k3` in
front of it, and points Claude Code at the Anthropic Messages endpoint exposed by
`k3`.

The commands below target Linux or WSL with an NVIDIA GPU. llama.cpp supports
other build targets, but those commands are outside this repository and were not
executed in this documentation lane.

## 1. Install the command-line tools

Install `uv` and Claude Code using their native installers:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
curl -fsSL https://claude.ai/install.sh | bash
```

These installer commands come from the
[uv installation guide](https://docs.astral.sh/uv/getting-started/installation/)
and [Claude Code setup guide](https://code.claude.com/docs/en/setup).

Install the Hugging Face CLI:

```bash
python3 -m pip install --upgrade huggingface_hub
```

You also need Git, CMake, a C++ compiler, the CUDA toolkit, Python 3.10 through
3.13, enough disk space for a 27.9 GB model, and enough CPU or GPU memory for the
model plus context.

## 2. Build llama.cpp

```bash
git clone --depth 1 https://github.com/ggml-org/llama.cpp
cmake -S llama.cpp -B llama.cpp/build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build llama.cpp/build --config Release -j
```

The build options and `llama-server` executable come from llama.cpp's current
[build guide](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md)
and [server guide](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md).
They were checked on 2026-07-29 but were not run in this lane.

## 3. Download the GGUF

```bash
mkdir -p models/kimi-linear
hf download AaryanK/Kimi-Linear-48B-A3B-Instruct-GGUF Kimi-Linear-48B-A3B-Instruct.q4_k_s.gguf --local-dir models/kimi-linear
```

The repository and filename above were checked against the
[Hugging Face model file](https://huggingface.co/AaryanK/Kimi-Linear-48B-A3B-Instruct-GGUF/blob/main/Kimi-Linear-48B-A3B-Instruct.q4_k_s.gguf).
The download was not performed in this lane.

## 4. Start llama.cpp

In terminal 1:

```bash
./llama.cpp/build/bin/llama-server -m models/kimi-linear/Kimi-Linear-48B-A3B-Instruct.q4_k_s.gguf --alias kimi-linear --host 127.0.0.1 --port 8000 -c 32768 -ngl 99 --jinja
```

`--jinja` enables llama.cpp's template path used for tool definitions. `-ngl 99`
requests GPU offload; llama.cpp may retain layers on the CPU if the model and
runtime state do not fit in VRAM.

Confirm that the OpenAI-compatible server is reachable:

```bash
curl http://127.0.0.1:8000/v1/models
```

## 5. Install and start k3

In terminal 2:

```bash
git clone https://github.com/jaberjaber23/local-kimi
cd local-kimi
uv sync --frozen
uv run k3 serve --upstream http://127.0.0.1:8000/v1 --model kimi-linear --reasoning-field inline
```

The `k3` flags above are defined in `k3/cli.py`. `k3/server.py` registers
`/v1/messages`, `/v1/chat/completions`, and `/v1/responses` on port 8080 by
default. `k3/upstream.py` appends `/chat/completions` to the configured upstream
base URL.

Check both processes through `k3`:

```bash
curl http://localhost:8080/health
```

## 6. Point Claude Code at k3

In terminal 3, from the project Claude Code should work on:

```bash
export ANTHROPIC_BASE_URL=http://localhost:8080
export ANTHROPIC_AUTH_TOKEN=local
claude
```

Then enter this request:

```text
Reply with exactly: local Kimi is connected
```

Those are the two connection variables. `k3` accepts any token when it is
started without `--api-key`, and it maps whichever Claude model name the client
requests to the upstream model configured by `--model kimi-linear`.

## Reasoning behavior with llama.cpp

The proxy does not require any k3-specific response field. For a non-streaming
response, `k3/pipeline.py` reads ordinary `choices[0].message.content`; for a
stream it reads ordinary `choices[].delta.content`. A missing
`reasoning_content` field is not an error. Response ids, model names, usage, and
native `tool_calls` are all handled when present, but the visible text path does
not depend on them.

The quickstart uses `--reasoning-field inline` because a backend may place
reasoning inside `<think>...</think>` in `content`. In that mode `k3` separates
the inner text, sends it to Claude Code as a thinking block, signs it, and puts it
back inside `<think>` tags when the assistant turn returns upstream.

If llama.cpp emits a separate `reasoning_content` field, `k3` reads that too. If
the backend emits neither a separate reasoning field nor `<think>` tags, the
request still works, but there are no distinct reasoning bytes to preserve. The
ledger can preserve only the visible assistant content and tool calls in that
case. Do not describe that path as byte-exact reasoning round-trip.

## What was and was not verified

Verified by repository inspection:

- every `k3 serve` flag used above exists in `k3/cli.py`;
- the default `k3` bind is `127.0.0.1:8080` in `k3/cli.py` and `k3/server.py`;
- the upstream URL becomes `/v1/chat/completions` in `k3/upstream.py`;
- no separate reasoning field is required by `k3/pipeline.py`;
- inline `<think>` extraction and restoration are implemented in
  `k3/pipeline.py` and `k3/reasoning.py`;
- `tests/test_full_chain.py` drives an Anthropic-shaped request through `k3`, an
  HTTP OpenAI-compatible upstream, and back without a socket.

Not verified in this lane:

- the GGUF was not downloaded;
- llama.cpp was not built or started;
- Claude Code was not launched against the proxy;
- no repository test was run.

The llama.cpp command is therefore checked against current upstream command
documentation and the local proxy contract, but it is not a claimed live test.

Continue with the [Claude Code worked example](CLAUDE-CODE.md), or configure
[Codex](CODEX.md) or the [OpenAI Python SDK](OPENAI-SDK.md) against the same `k3`
process.
