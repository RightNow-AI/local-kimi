# local-kimi

Running Kimi K3 locally: a serving proxy that speaks every coding agent's wire
format, and the engine research behind actually executing a 2.78T-parameter
model on hardware you can own.

Two halves:

| | |
|---|---|
| **`k3/`** | The proxy. Auto-detects Claude Code / Codex / OpenAI clients and speaks each one's dialect. Working, tested, and [deployed](#live-endpoint). |
| **`engine/`, `research/`** | The engine. Reference implementation, weight access, and the measurements that decide what is buildable. Early. |

## Live endpoint

```
https://rightnow-ai--k3-serve-api.modal.run
```

Deployed on Modal, token-gated. Verified end to end against the real `claude`
CLI and the real `codex` CLI over the internet, plus the official `openai`
Python SDK: all three dialects, auth enforced, streaming, and the reasoning
signature round-tripping. See `engine/modal_serve.py`.

## What is actually established about Kimi K3

Every number here was read from the checkpoint this session, not assumed:

| | |
|---|---|
| Size | 2.78T params, 1,560.9 GB, 96 shards, 497,220 tensors |
| Layers | 93 — layer 0 dense, 1–92 MoE |
| Experts | 896 per layer, **16 routed + 2 shared** active per token |
| MoE shape | **latent**: 7168 → 3584 → per-expert → 7168, each expert 33,030,144 params |
| Expert format | already 4-bit — `weight_packed` U8 + `weight_scale` U8, group 32, **exactly 4.250 bits/param** |
| The rest | BF16 — attention, shared experts, latent projections, embeddings: **114.4 GB** |
| Attention | hybrid — 24 MLA layers, **69 KDA linear-attention** layers, 1M context |
| Activation | `situ`, **not** SwiGLU (`activation_situ_linear_beta` 25.0) |
| Prompt format | XTML (`<\|open\|>` / `<\|close\|>` / `<\|sep\|>`), **not** the K2 control tokens |

Two consequences worth stating plainly:

**Moonshot ships K3 already quantized, so their release is the reference.** There
is no higher-precision K3 to lose ground against. Reading it losslessly costs
nothing; the loss ledger starts only when we change something.

**"K3 is 4-bit" is over-stated.** Only the *routed* experts are. 2 of the 18
experts active per token are the BF16 shared experts.

## Status, honestly

This project has an adversarial verification pass (`wf_9f96ac7d-99b`) that
attacked its own load-bearing claims. Three of four failed. What that changed:

- **Performance numbers are under revision.** An earlier figure of 8.4–9.6 tok/s
  for batch-1 decode was refuted: it implies the 114.4 GB of BF16 dense weights
  move at 3.6× theoretical peak DRAM. The corrected figure is closer to
  ~3.3 tok/s. **Do not price hardware off the old number.**
- **`research/verify_lossless.py` proves nothing.** It multiplies and divides by
  the same power of two, so its check is a tautology that returns PROVEN for a
  swapped nibble order. The *conclusion* survives on independent evidence (two
  ports of `compressed-tensors` agreeing bit-identically on real bytes); the
  script does not.
- **`research/expert_spectrum_v2.py`'s verdict must not be cited.** Its statistic
  saturates at ≈ min(sketch², n) and its thresholds sit below the checkpoint's
  own ~16% quantization noise floor. The decision it reached — that a global
  shared-basis low-rank codec is not worth pursuing — still stands, but on
  participation-ratio evidence, not on that script.

Nothing above was found by a customer. It was found by pointing agents at our
own work and asking them to break it.

---

# k3 — the proxy

Client presets for the K3 inference engine.

vLLM makes you assemble `--tool-call-parser`, `--chat-template`, and a reasoning
parser yourself, then hope the coding agent you're pointing at it agrees. `k3`
bundles that per client instead, detects which client is calling from the
request itself, and prints the line you paste to start using it.

```
$ k3 serve
  k3 0.1.0   serving on http://localhost:8080
  engine  http://127.0.0.1:8000/v1   model k3
  client  auto-detect (7 presets)

  Claude Code

    export ANTHROPIC_BASE_URL=http://localhost:8080
    export ANTHROPIC_AUTH_TOKEN=local
    export ANTHROPIC_MODEL=k3
    claude
```

No engine yet? `k3 serve --mock` runs the whole path — dialects, tool parsing,
reasoning translation — against a scripted stand-in.

## Install

```bash
uv sync
uv run k3 --help
```

## Use

```bash
k3 serve                          # auto-detect the client per request
k3 serve --client claude-code     # pin a preset
k3 serve --mock                   # no GPU required
k3 serve --record ./session       # record traffic for the conformance suite

k3 presets -v                     # what each preset bundles
k3 detect -H 'user-agent: claude-cli/1.0.60'   # why a request resolves where it does
k3 doctor                         # presets validate, engine reachable?
k3 replay ./session               # did anything change?
```

Point the engine somewhere else with `--upstream http://host:8000/v1 --model k3`.

### Exposure

`k3` binds `127.0.0.1` and runs open by default, which is right for a local
engine. Two things are deliberately not permissive:

- **CORS is off** unless you pass `--cors-origin https://…` (repeatable). A
  wildcard would let any page you visit drive your engine and read the replies.
- **`--api-key TOKEN`** gates every route except `/health`, which stays
  reachable for container health checks but reports only `{"status": …}`
  without credentials.

If you bind `--host 0.0.0.0`, set `--api-key`.

## What a preset bundles

Six things, per client:

| | |
|---|---|
| **route + dialect** | which endpoints to expose; Messages vs Chat Completions vs Responses |
| **tool parser** | Kimi control tokens, hermes tags, bare JSON, pythonic — or `passthrough` when the engine already parsed them |
| **chat template** | how messages and tool definitions render into K3's prompt |
| **reasoning translation** | both directions — see below |
| **defaults** | reasoning effort, max tokens, streaming shape |
| **model aliasing** | whatever model string the client asks for resolves to K3 |

```
$ k3 presets

preset       status        dialect              tools  reasoning          routes
claude-code  stable        anthropic_messages   kimi   thinking_blocks    /v1/messages …
openai       stable        openai_chat          kimi   strip              /v1/chat/completions …
codex        stable        openai_responses     kimi   responses_item     /v1/responses …
kimi-code    provisional   openai_chat          kimi   reasoning_content  /v1/chat/completions …
cline        provisional   openai_chat          kimi   reasoning_content  /v1/chat/completions …
opencode     provisional   openai_chat          kimi   reasoning_content  /v1/chat/completions …
aider        provisional   openai_chat          kimi   inline_tags        /v1/chat/completions …
```

`stable` means there is recorded client traffic pinning the behaviour.
`provisional` means the preset is built from the client's documented wire format
but hasn't been pinned by a recording yet. Promote one by recording real traffic
and dropping the cassette in — see [Conformance](#conformance).

Today that means:

- **claude-code** — three cassettes from a real `claude` CLI session against
  this server, including the second turn of an agent loop where a `thinking`
  block came back and its reasoning was restored into the engine payload.
- **codex** — two cassettes from a real `codex exec` session, covering a
  `function_call` and the `function_call_output` turn that follows it.
- **openai** — three cassettes recorded from the official `openai` Python SDK,
  the reference implementation of the dialect.

`kimi-code`, `cline`, `opencode`, and `aider` are pinned by synthetic bodies
until someone records them. See [`tests/cassettes/README.md`](tests/cassettes/README.md).

**Kimi Code is the correctness reference.** Moonshot says K3 works best with it,
so if another preset diverges from Kimi Code's behaviour, the preset is wrong,
not the model.

## Auto-detection

You can tell who's calling from the request itself, so `--client` is an override
rather than a requirement.

- `POST /v1/messages` with an `anthropic-version` header → Claude Code
- `POST /v1/responses` → Codex
- `POST /v1/chat/completions` → generic OpenAI, unless a user-agent or vendor
  header (`http-referer: cline.bot`, `x-msh-client`, …) says otherwise

A *strong* signal — user-agent or vendor header — makes a preset eligible; weak
signals only break ties; every route has a fallback so an unrecognised client
still gets served. Every decision carries a reason you can read:

```
$ k3 detect --path /v1/messages -H 'user-agent: claude-cli/1.0.60' -H 'anthropic-version: 2023-06-01'
  preset   claude-code  (Claude Code)
  why      user-agent ~ /claude-cli//, header anthropic-version
  score    95
  dialect  anthropic_messages
```

## Reasoning translation

This is the part that's hard, and the part that makes the project real.

K3 emits `reasoning_content` and requires the complete assistant message passed
back **verbatim** on the next turn. Every client wants that in a different
shape — Claude Code wants `thinking` blocks, Codex wants reasoning items,
OpenAI chat wants it stripped. Lose it and K3 degrades across agent loops, and
every user blames your quantization.

So `k3` converts K3's reasoning out to the client's format, then converts the
client's echo back into exactly what K3 expects, byte for byte. Three vehicles,
tried in order:

1. **Ledger.** A signature carries an id that resolves to the *complete*
   upstream assistant message — including raw tool-call argument strings, which
   no client can round-trip losslessly because it parses them into objects. When
   this hits, the bytes are the original bytes.
2. **Self-contained signature.** `k3r1.<base64url(zlib(json))>` carries the
   reasoning text itself. Survives a restart, survives the client dropping the
   visible text, needs no shared state. This is what rides in Anthropic's
   `signature` field and the Responses API's `encrypted_content`.
3. **Fingerprint.** For clients that strip reasoning entirely, hash the parts
   every client *does* round-trip — text, tool names, normalised arguments — and
   look the reasoning up by that. This is what keeps K3 from degrading on a
   plain OpenAI client.

If all three miss, we fall back to whatever visible text there is, which is
exactly the pre-`k3` behaviour — never worse.

One related trap, handled: Anthropic `tool_use` ids and OpenAI `tool_calls` ids
live in different namespaces. `k3` uses a deterministic reversible prefix
(`toolu_k3_…`) rather than minting fresh ids and keeping a map, because a map
doesn't survive a restart and a mismatch silently breaks every agent loop.

## Conformance

Claude Code, Codex, and the rest all ship updates, and a preset that worked last
month can break silently. Without recorded traffic you find out from GitHub
issues.

```bash
k3 serve --record ./session    # then use your client normally
k3 replay ./session            # after any change
```

A cassette records what the client sent, what we sent the engine, what the
engine streamed back, and the exact bytes we returned. Replaying re-runs it with
the engine stubbed out by its own recording, so a change in any preset, dialect,
or parser shows up as a diff — in the client-facing bytes *and* in the payload
we send K3. Auth headers are redacted on write; cassettes are meant to be
committed. Add `--record-compress` for real sessions, which are large.

Cassettes carry their provenance. `source: recorded` is real traffic captured
off the wire from the actual client; `source: synthetic` is a request body
written to match the client's documented format. Both pin behaviour, only one
proves it, and `tests/test_conformance.py` enforces that a `stable` preset has
recorded traffic behind it.

Non-deterministic fields (ids, timestamps, the random ledger id inside a
signature) are normalised rather than ignored: a signature is decoded and
replaced with a hash of *the reasoning it carries*, so the comparison still
fails if the reasoning payload changes. Tool-call argument whitespace is allowed
to differ, since a cold replay rebuilds arguments from the client's parsed copy.

One honest limitation: presets whose reasoning policy is `strip` have no
client-side vehicle at all, so restoration runs entirely off the server-side
fingerprint index — which a cold replay cannot reproduce by construction. Those
are exempted from the upstream reasoning comparison; every other preset carries
a signature through the client and *must* restore reasoning even on a cold
server, which the suite does assert.

```bash
uv run pytest                                              # everything
uv run python scripts/build_fixtures.py --raw ./session    # curate a session
```

## Architecture

```
client body ──ingress──▶ CanonicalRequest ──build_payload──▶ engine
                                                                │
client SSE ◀──egress── StreamEvent* ◀──pipeline.run── engine chunks
```

| file | |
|---|---|
| `k3/ir.py` | canonical IR; raw tool arguments and raw reasoning never get reserialized |
| `k3/reasoning.py` | signature codec, ledger, restoration — the hard part |
| `k3/toolcalls.py` | incremental parsers; a control token split across chunks never leaks |
| `k3/template.py` | system prompt and tool-definition rendering |
| `k3/presets.py` | the seven presets |
| `k3/detect.py` | scoring, with a reason for every decision |
| `k3/dialects/` | `anthropic_messages`, `openai_chat`, `openai_responses` |
| `k3/pipeline.py` | engine chunks → canonical events (streaming and not, one path) |
| `k3/upstream.py` | engine adapter + the mock engine |
| `k3/record.py`, `k3/replay.py` | cassettes |
| `k3/server.py`, `k3/cli.py` | HTTP surface, command line |

Streaming and non-streaming share `pipeline.run` — a non-streaming engine
response becomes a short synthetic chunk list — so the two paths can't drift.

## Engine expectations

`k3` talks to any OpenAI-compatible `/v1/chat/completions` endpoint that serves
K3. Configure how it carries reasoning:

```bash
--reasoning-field reasoning_content   # vLLM / SGLang / Moonshot (default)
--reasoning-field inline              # <think>…</think> inside content
--reasoning-field none                # engine has no reasoning channel
```

If the engine was started with its own tool-call parser, set the preset's parser
to `passthrough` and native `tool_calls` deltas are used instead. Otherwise `k3`
parses the model's raw text — the default, since it keeps the format contract in
one place.

## Status

Three presets are pinned by recorded traffic — **claude-code**, **codex**, and
**openai** — which covers most agent usage. The rest are built from documented
wire formats and are expected to work; they get promoted as recordings land. The
conformance harness was built with the second preset, not the sixth, which is
the only reason the third and fourth were cheap.

Licensed Apache-2.0 — see [`LICENSE`](LICENSE).
