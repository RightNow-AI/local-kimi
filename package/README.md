# Catalog packaging layer

This directory defines the catalog-facing package contract for the Kimi-Linear serving engine.
It does not perform optimization, pricing, signing, publishing, or release-gate work.

The package class is explicit. A `recipe` package changes no weight bytes and points to an
immutable source revision. An `optimized_weights` package changes weight bytes and therefore
requires redistribution permission plus paired quality evidence before it can be listable.

## Evidence contract

`claims.py` accepts the factory artifacts that already exist:

- `scorecard.py` output with `schemaVersion: 1`;
- a signed `catalog-benchmark-receipt.v1` produced by `build_kit.py` and signed before use;
- `engine/laptop/RESULTS.md` for computed structural footprint facts only.

A claim names an exact selector in one of those files. Construction reads the selected value and
rejects the claim when the file is absent, the artifact shape is wrong, or the claimed value does
not equal the value on disk. Performance claims also require both runtime identities, versions,
hardware, concurrency, a request profile, and measured evidence. Structural arithmetic cannot be
used as performance evidence.

The renderer accepts a validated `Claim`, not a number. This keeps claim wording and measurement
conditions attached to the same object. Every rendered figure therefore carries hardware,
concurrency, request profile, runtime identity, runtime version, and its evidence path inline.

## Kimi-Linear position

`build_kimi_linear_definition` reads the resident and active byte counts plus the single-card
memory requirement from `engine/laptop/RESULTS.md`. It does not repeat those values in Python
source. The resulting card presents footprint as computed structural evidence.

The card states that the serving engine was written from scratch for this architecture rather
than being a configuration of an existing engine. It also states that vLLM supports the
architecture and that this package is not a speed comparison against vLLM. No laptop projection
is converted into a measured performance claim.

## Refusals

The layer refuses:

- missing evidence files, because prose is not evidence;
- a claim value that differs from the selected on-disk value, because copied numbers drift;
- structural arithmetic used as performance evidence, because a roofline is not a benchmark;
- comparative claims missing either runtime side, because an unnamed counterfactual is not a
  comparison;
- a recipe that modifies weights, or an optimized-weights package that says it did not;
- listability for optimized weights without verified redistribution rights and measured paired
  quality evidence;
- authored positioning that asserts first support, fastest serving, or day-0 support.

`PackageDefinition.assert_listable()` is a local preflight. It does not replace
`Model-Factory/tools/check_gate.py`, the signed kit receipt, founder pricing from
`price_package.py`, or the catalog publish gate.
