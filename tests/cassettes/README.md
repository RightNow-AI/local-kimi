# Cassettes

Recorded client traffic, replayed by `tests/test_conformance.py` and `k3 replay`.

Each cassette holds one request: what the client sent, what we sent the engine,
what the engine streamed back, and the exact bytes we returned. Replaying stubs
the engine out with its own recording, so a change in any preset, dialect,
parser, or the reasoning translation shows up as a diff.

Files are gzipped JSON (`.json.gz`); real client traffic is large, mostly
system prompt. `Cassette.load` reads plain `.json` too.

## Provenance

Every cassette carries a `source` field.

**`recorded`** — the real client produced these bytes.

| cassette | how it was captured |
|---|---|
| `claude-code-turn1-tools-stream` | `claude` CLI against `k3 serve --record`, over a socket |
| `claude-code-turn2-thinking-restore-stream` | second turn of that same session — carries a `thinking` block and a `tool_result` back, and the recorded engine payload shows the reasoning restored |
| `claude-code-background-json` | a background request from that session |
| `codex-turn1-tools-stream` | `codex exec` against `k3 serve --record`, over a socket |
| `codex-turn2-tool-output-stream` | second turn of that session, carrying a `function_call_output` |
| `openai-sdk-tools-json` | the official `openai` Python SDK, via `scripts/capture_openai.py` |
| `openai-sdk-turn2-json` | second turn, echoing the assistant message back with reasoning stripped |
| `openai-sdk-stream` | SDK streaming with `stream_options.include_usage` |

The OpenAI SDK captures run over `httpx.ASGITransport` rather than a socket —
the bytes are the SDK's, the round trip just stays in-process.

**`synthetic`** — request bodies written to match the client's documented wire
format, replayed against the mock engine. These pin behaviour but do not prove
it. They exist for clients not installed on the machine that built this set:
`kimi-code`, `cline`, `opencode`, `aider`.

`tests/test_conformance.py::test_stable_presets_have_recorded_traffic` enforces
that a preset marked `stable` in `k3/presets.py` has a `recorded` cassette
behind it. Promoting a preset means capturing real traffic, not editing a label.

## Regenerating

```bash
# curate a raw recording session into the fixture set
k3 serve --record ./session      # then use your client normally
uv run python scripts/build_fixtures.py --raw ./session

# re-capture the OpenAI SDK cassettes
uv run python scripts/capture_openai.py
```

`build_fixtures.py` deletes and rebuilds `synthetic` cassettes; `recorded` ones
are left alone unless you pass a `--raw` directory that produces them.

## When one fails

Read the diff before touching the cassette. A cassette is a statement about what
a real client accepted — if the diff looks like an improvement, it still needs a
fresh recording from the client to back it up.
