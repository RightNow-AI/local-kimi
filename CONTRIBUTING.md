# Contributing to k3

## Set up a development environment

Install uv, then create the project environment from the repository root:

```bash
uv sync
```

Run the complete offline suite with:

```bash
uv run pytest
```

The default pytest configuration excludes tests marked `gpu`, `network`, or
`weights`. Those categories are reserved for tests that require hardware,
external services, or the local model weights. To run one of those categories
explicitly, clear the default options:

```bash
uv run pytest -o addopts= -m gpu
```

The client conformance suite is part of the normal test run. Run it directly
when changing presets, dialects, replay normalization, tool parsing, or
reasoning translation:

```bash
uv run pytest tests/test_conformance.py
```

Run the blocking lint check and the current formatting audit with:

```bash
uv run ruff check .
uv run ruff format --check .
```

CI runs the suite on Ubuntu, macOS, and Windows with Python 3.10, 3.11, 3.12,
and 3.13.

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
uv run python scripts/build_fixtures.py --raw ./session
```

For the official OpenAI Python SDK captures, use:

```bash
uv run python scripts/capture_openai.py
```

Review every cassette before committing it. Confirm that credentials are
redacted, provenance is accurate, and the request came from the named client.
A stable preset requires real recorded traffic, not a synthetic request that
resembles the client's documented format.

A cassette is a statement about what a real client accepted. Never edit a
cassette to make a test pass. If intentional behavior changes invalidate a
recording, reproduce the interaction with the real client, record fresh
traffic, inspect the replay diff, and replace the cassette with evidence from
that new session.

## Keep the proxy lightweight

The `k3` proxy must remain installable without GPU or model dependencies. Keep
Modal, NumPy, Torch, and similar engine-only packages in the `engine` optional
dependency extra. Add a package to the development group when the test suite or
repository tooling imports it.
