# Quickstart

This guide starts Kimi-Linear-48B with llama.cpp, puts `k3` in front of it, and connects Claude Code through Anthropic Messages.

The commands target Linux or WSL with an NVIDIA GPU. The model file is 27.9 GB. You also need memory for context and runtime state. This guide does not claim that Kimi K3 runs on a laptop.

## If your model server is already running

If an OpenAI-compatible Kimi server is listening at `http://127.0.0.1:8000/v1`, skip to [Start k3](#start-k3).

## Install the command-line tools

Install `uv` and Claude Code:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
curl -fsSL https://claude.ai/install.sh | bash
```

Install the Hugging Face CLI:

```bash
python3 -m pip install --upgrade huggingface_hub
```

You also need Git, CMake, a C++ compiler, the CUDA toolkit, and Python 3.10 through 3.13.

Sources: [uv installation](https://docs.astral.sh/uv/getting-started/installation/), [Claude Code setup](https://code.claude.com/docs/en/setup), and [Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/guides/cli).

## Build llama.cpp

```bash
git clone --depth 1 https://github.com/ggml-org/llama.cpp
cmake -S llama.cpp -B llama.cpp/build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build llama.cpp/build --config Release -j
```

See the current llama.cpp [build guide](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md) and [server guide](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md) for other platforms.

## Download Kimi-Linear-48B

```bash
mkdir -p models/kimi-linear
hf download AaryanK/Kimi-Linear-48B-A3B-Instruct-GGUF Kimi-Linear-48B-A3B-Instruct.q4_k_s.gguf --local-dir models/kimi-linear
```

The file is published at [AaryanK/Kimi-Linear-48B-A3B-Instruct-GGUF](https://huggingface.co/AaryanK/Kimi-Linear-48B-A3B-Instruct-GGUF/blob/main/Kimi-Linear-48B-A3B-Instruct.q4_k_s.gguf).

## Start llama.cpp

In terminal 1:

```bash
./llama.cpp/build/bin/llama-server -m models/kimi-linear/Kimi-Linear-48B-A3B-Instruct.q4_k_s.gguf --alias kimi-linear --host 127.0.0.1 --port 8000 -c 32768 -ngl 99 --jinja
```

`--jinja` enables the template path used for tool definitions. `-ngl 99` requests GPU offload. llama.cpp can keep layers on the CPU if the model and runtime state do not fit in VRAM.

Check the backend:

```bash
curl http://127.0.0.1:8000/v1/models
```

## Start k3

In terminal 2:

```bash
git clone https://github.com/RightNow-AI/local-kimi
cd local-kimi
uv sync --frozen
uv run k3 serve --upstream http://127.0.0.1:8000/v1 --model kimi-linear --reasoning-field inline
```

These `k3 serve` flags are defined in `k3/cli.py`. The proxy listens on `127.0.0.1:8080` by default and sends upstream requests to `/v1/chat/completions`.

Check the proxy and backend together:

```bash
curl http://localhost:8080/health
```

## Connect Claude Code

In terminal 3, change to the project Claude Code should work on:

```bash
export ANTHROPIC_BASE_URL=http://localhost:8080
export ANTHROPIC_AUTH_TOKEN=local
claude
```

Enter this request:

```text
Reply with exactly: local Kimi is connected
```

Do not add `/v1` to `ANTHROPIC_BASE_URL`. Claude Code adds the Messages route. The token value can be any non-empty string while `k3` is started without `--api-key`.

## Reasoning behavior

The `--reasoning-field inline` setting handles backends that place reasoning inside `<think>...</think>` tags. `k3` separates that text, sends it to Claude Code as a thinking block, and restores it when the assistant turn returns upstream.

`k3` also reads a separate `reasoning_content` field when the backend provides one. If the backend provides no reasoning field and no recognised inline reasoning, normal text and tool calls still work, but there are no separate reasoning bytes to preserve.

## Next steps

- [Claude Code details](CLAUDE-CODE.md)
- [Codex setup](CODEX.md)
- [OpenAI Python SDK setup](OPENAI-SDK.md)
- [Architecture](ARCHITECTURE.md)

Keep the default loopback bind for local use. If you bind `k3` to another interface, set `--api-key` and review who can reach the port.
