# Codex

Preset: `codex`

This preset uses the OpenAI Responses dialect at `/v1/responses`. The config
below is generated from the preset body in `k3/presets.py`.

Start `k3` against the local llama.cpp server:

```bash
uv run k3 serve --upstream http://127.0.0.1:8000/v1 --model kimi-linear --reasoning-field inline
```

Add this to `~/.codex/config.toml`:

```toml
model = "kimi-linear"
model_provider = "k3"

[model_providers.k3]
name = "k3"
base_url = "http://localhost:8080/v1"
wire_api = "responses"
env_key = "K3_API_KEY"
```

Set the token and launch Codex:

```bash
export K3_API_KEY=local
codex
```

Enter one request:

```text
Inspect this repository and explain the request path from the HTTP route to the upstream model.
```

The token value is arbitrary while `k3` is started without `--api-key`.

Codex replays reasoning as Responses `reasoning` items. `k3` places its signature
in `encrypted_content` and uses the reasoning ledger as the primary recovery
path. The implementation is in `k3/dialects/openai_responses.py` and
`k3/reasoning.py`.
