# Architecture

`k3` is an HTTP protocol adapter. The inference backend remains an
OpenAI-compatible Chat Completions server such as llama.cpp.

```text
client request
  -> server route
  -> per-request preset detection
  -> dialect ingress
  -> canonical IR
  -> OpenAI Chat Completions upstream
  -> canonical stream events
  -> dialect egress
  -> client response
```

## Entry points and detection

`k3/cli.py` builds `ServerConfig` and `UpstreamConfig`, prints client setup, and
starts the FastAPI application. `k3/server.py` registers every supported route.
Unless `--client` pins a preset, each request goes through `k3/detect.py`.

Detection scores strong signals such as a user agent or client-specific header.
Weak body keys only break ties. Each route has a fallback in `k3/presets.py`, so
an unrecognised `/v1/messages`, `/v1/responses`, or `/v1/chat/completions`
request still reaches the matching protocol family. The selected preset carries
the dialect, tool parser, reasoning policy, defaults, model aliases, routes, and
copy-paste setup.

## Translation path

The selected module under `k3/dialects/` lowers the client body into the data
types in `k3/ir.py`. The IR keeps raw tool argument strings and reasoning text so
translation does not normalise bytes that must survive a turn.

`k3/upstream.py` turns the canonical request into an OpenAI Chat Completions
payload and sends it to `{upstream}/chat/completions`. `k3/template.py` handles
system text, native tool definitions, optional prompt-rendered tools, and Kimi
XTML helpers.

`k3/pipeline.py` consumes either a normal completion or SSE chunks. It produces a
single canonical event stream for reasoning, visible text, tool calls, usage,
and termination. `k3/toolcalls.py` can accept native OpenAI tool calls or parse
model text, including Kimi K3 XTML. The original dialect then raises the events
into Anthropic Messages SSE, OpenAI Chat Completions SSE, or OpenAI Responses
named events.

`tests/test_full_chain.py` covers the whole transport shape. It connects `k3` to
an ASGI OpenAI-compatible server through HTTP transport, then sends Anthropic
and OpenAI requests through the proxy.

## Reasoning ledger

`k3/reasoning.py` implements three recovery paths:

1. A self-contained `k3r1` signature stores the reasoning text and ledger id.
   Claude Code carries it in a thinking-block `signature`; Codex carries it in
   `encrypted_content`.
2. The in-memory ledger stores the complete upstream assistant message. A hit
   restores reasoning plus raw tool-call argument strings without rebuilding
   them from client objects.
3. Clients using plain Chat Completions may strip reasoning. For those turns,
   the ledger indexes a fingerprint of visible text and tool calls and can find
   the original assistant message when it returns.

The default ledger is bounded to 4,096 entries with a six-hour TTL in
`k3/server.py`. A process restart loses ledger-only state, but a self-contained
signature can still restore its reasoning text. It cannot restore an entire raw
assistant message after a restart.

The backend sets the upper bound on this mechanism. If it emits
`reasoning_content`, `reasoning`, recognised Kimi XTML reasoning, or inline
`<think>` content while `--reasoning-field inline` is active, `k3` can create a
reasoning part and carry it through the client. If the backend exposes no
reasoning representation, the ledger records an empty reasoning string. Visible
content and tool calls still work, but byte-exact reasoning preservation does
not apply.

## Files to start with

- `k3/server.py`: HTTP routes and request orchestration
- `k3/detect.py` and `k3/presets.py`: per-request client selection
- `k3/dialects/`: ingress and egress for each client protocol
- `k3/ir.py`: canonical request, response, parts, and stream events
- `k3/upstream.py`: OpenAI-compatible backend payload and transport
- `k3/pipeline.py`: upstream response parsing and canonical event production
- `k3/reasoning.py`: signatures, ledger, fingerprints, and restoration
- `k3/toolcalls.py`: native and text tool-call parsing
- `k3/template.py`: system, tool, and Kimi XTML rendering
