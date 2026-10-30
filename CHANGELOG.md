# Changelog

All notable changes to this project will be documented in this file.

The format follows Keep a Changelog, and the project uses Semantic Versioning.

## [Unreleased]

### Fixed

- An `image` block inside a Claude Code `tool_result` was serialised into the
  tool message verbatim, base64 payload included, so one screenshot from the
  Read tool or an MCP tool put hundreds of kilobytes of unreadable text into
  the prompt and could overflow the engine's context on its own. Blocks that
  carry a base64 payload now become a one-line description with the block
  type, media type and payload size.

## [0.2.0] - 2026-07-29

Decode throughput for Kimi-Linear-48B-A3B on one NVIDIA L40S under a hard 32 GiB
process cap went from 35.76 to 113.83 tokens per second, a 3.18 times gain, with
peak reserved memory falling from 29.56 GiB to 27.63 GiB.

### Added

- Kernel registry with an equivalence harness. Each operation has one reference
  implementation and any number of variants, selection runs through a
  programmatic override, then `KIMI_KERNELS`, then the shipped default, then the
  reference, and every variant can be compared against the reference before use.
- Fused batch-1 grouped W4A16 GEMV for the mixture-of-experts path. Measured at
  51.7 percent of the L40S peak bandwidth against 8.3 percent for the kernel it
  replaces.
- Fused batch-1 dense W4A16 GEMV for the attention and projection path, with a
  fallback to the existing tensor-core path when more than one token is being
  processed, so prefill is unaffected.
- Fused Kimi Delta Attention decode step, collapsing five whole-state passes over
  the 2 MiB recurrent state into one kernel.
- Fused Kimi Delta Attention preparation, replacing 47 small launches per layer
  with 2 across the twenty KDA layers.
- Per-kernel decode profiler, kernel benchmark runner, variant-sweeping decode
  benchmark, and a teacher-forced equivalence runner.
- Six figures generated from the measured numbers, in Computer Modern, written
  as both PNG and vector PDF.

### Changed

- Grouped kernel launch configuration chosen by sweeping every candidate inside
  one decode process rather than by an untested estimate of grid occupancy.
- Cross-platform continuous integration for Python 3.10 through 3.13.
- Ruff lint and formatting reports.
- Typed-package and release metadata.

### Fixed

- The fast kernels were registered but nothing selected them, so an ordinary run
  used the reference path while the benchmarks, which select variants
  explicitly, reported the fast one. The registry now has a shipped-default tier
  and `tests/test_kernel_defaults.py` asserts it.
- The decode benchmark reported byte-identical output using a filler prompt that
  made the model emit one token 64 times. It now refuses to run when the
  reference produces fewer than three distinct tokens, and equivalence is
  measured by teacher forcing rather than free-running greedy decode.
- Decode benchmark reserved capacity for one loop while running three, which
  failed as a device-side assert attributed to the profiler.

### Notes

- The fused kernels are **not** bit-identical to the reference path. Teacher
  forced they agree on 96.9 percent of next-token choices at 0.0036 nats mean
  KL, roughly one tenth of the divergence INT4 quantisation itself introduces.
  `engine/kernels/RESULTS.md` records the measurement.
- Two fusions were built, measured and left switched off because they lost end
  to end despite winning an isolated benchmark. Both remain registered so the
  result is reproducible.
- All measurements are one card, one stream, one prompt, greedy decoding. None
  of this is a serving throughput claim, and the engine has not been measured on
  a consumer card.

## [0.1.0] - 2026-07-28

### Added

- Initial K3 client preset proxy and conformance suite.
- Kimi K3 XTML tool-call format support.
- Modal engine harness and expert-spectrum research tools.
- MXFP4 volume filename correction and real-weight dequantization validation.
- Quality baseline and lossless weight-path verification.

[Unreleased]: https://github.com/RightNow-AI/local-kimi/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/RightNow-AI/local-kimi/releases/tag/v0.2.0
[0.1.0]: https://github.com/RightNow-AI/local-kimi/releases/tag/v0.1.0
