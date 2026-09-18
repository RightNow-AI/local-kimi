# Claude Code

The `claude-code` preset uses the Anthropic Messages dialect and the routes
`/v1/messages`, `/v1/messages/count_tokens`, and `/v1/models`. The preset name,
environment variables, routes, tool parser, and reasoning policy are defined in
`k3/presets.py`.

Complete the [quickstart](QUICKSTART.md) first if the model server is not already
running.

Start `k3` against the local llama.cpp server:

```bash
uv run k3 serve --upstream http://127.0.0.1:8000/v1 --model kimi-linear --reasoning-field inline
```

Set the two connection variables and launch Claude Code:

```bash
export ANTHROPIC_BASE_URL=http://localhost:8080
export ANTHROPIC_AUTH_TOKEN=local
claude
```

Enter one request:

```text
List the files in this directory, then tell me which file looks like the project entry point.
```

`ANTHROPIC_BASE_URL` does not include `/v1`. Claude Code adds the Messages route.
The token value is arbitrary while `k3` is started without `--api-key`.

The preset also knows `ANTHROPIC_MODEL` and `ANTHROPIC_SMALL_FAST_MODEL`, but they
are not required for this setup. `k3` resolves the model name requested by Claude
Code to the model passed to `k3 serve --model`.

Reasoning is emitted to Claude Code as `thinking` blocks. The block signature
contains a self-contained copy when small enough and a reasoning-ledger id. On
the next turn, `k3/reasoning.py` restores the original reasoning and, on a ledger
hit, the complete upstream assistant message.

See [QUICKSTART.md](QUICKSTART.md#reasoning-behavior) for the
llama.cpp limitation when the backend provides no distinct reasoning channel.
