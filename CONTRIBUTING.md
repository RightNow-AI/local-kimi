# Contributing to k3

## Set up the locked environment

Install uv, then create the development environment from the repository root:

```bash
uv sync --frozen
```

Run the blocking lint check and the complete offline suite with:

```bash
uv run --no-sync ruff check .
uv run --no-sync python -m pytest -m "not gpu and not network and not weights"
```

The explicit marker expression matches CI. Tests marked `gpu`, `network`, or
`weights` require hardware, an external service, or local model weights and are
not part of the portable package gate.

To run one of those categories deliberately, clear the repository default and
select the marker:

```bash
uv run --no-sync python -m pytest -o addopts= -m gpu
```

The conformance suite is part of the offline gate. Run it directly when changing
presets, dialects, replay normalization, tool parsing, or reasoning translation:

```bash
uv run --no-sync python -m pytest tests/test_conformance.py
```

CI runs the offline suite on Ubuntu 24.04, macOS 14, and Windows Server 2022
with Python 3.10, 3.11, 3.12, and 3.13. Code and tests must use UTF-8 explicitly
for text fixtures and must not assume Windows path separators.

## Evidence rules

Benchmark and research changes must label each result as one of:

- measured;
- projected;
- simulated;
- modelled; or
- unmeasured.

Every reported performance number must name its hardware, workload, and
conditions. A component measurement must not be presented as full-model
throughput. A comparison with another engine requires a side-by-side run using
the same model revision, prompts, decoding settings, hardware, and measurement
method.

Do not replace a negative result with an estimate that looks better. Negative
results are part of the repository's evidence.

## Record and curate cassettes

Read `tests/cassettes/README.md` before recording or changing fixture coverage.
It documents cassette provenance, compression, replay behavior, and the
difference between recorded and synthetic traffic.

Capture a real client session against the recording server:

```bash
k3 serve --record ./session
```

Use the client normally, then replay the raw session before curating it:

```bash
k3 replay ./session
uv run --no-sync python scripts/build_fixtures.py --raw ./session
```

For official OpenAI Python SDK captures, use:

```bash
uv run --no-sync python scripts/capture_openai.py
```

Review every cassette before committing it. Confirm that credentials are
redacted, provenance is accurate, and the request came from the named client.
A stable preset requires real recorded traffic, not a synthetic request that
resembles the client's documented format.

Never edit a cassette to make a test pass. If intentional behavior changes
invalidate a recording, reproduce the interaction with the real client, record
fresh traffic, inspect the replay diff, and replace the cassette with evidence
from that new session.

## Third-party reference material

Read `reference/PROVENANCE.md` before changing anything in `reference/`.
Third-party files retain their upstream terms and are not covered by the
project's Apache-2.0 licence. An upstream sync must record the exact revision,
the relationship between local and upstream text, and the licence text that was
actually checked.

## Keep the proxy lightweight

The `k3` proxy must remain installable without GPU or model dependencies. Keep
Modal, NumPy, Torch, and similar engine-only packages in the `engine` optional
dependency extra. Add a package to the development group when the test suite or
repository tooling imports it.
