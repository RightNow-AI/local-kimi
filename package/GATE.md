# Kimi-Linear pre-listing gate map

Status: **BLOCKED. This package is not ready to list.**

This checklist maps the current lane to `Model-Factory/gate/release-criteria.json`
and `Model-Factory/tools/check_gate.py`. A source document can be ready while the
factory gate remains unsatisfied because the check runs against a rendered kit,
its package metadata, its accuracy document, and its signatures.

Status meanings:

- **SATISFIED**: the required evidence exists in the form the gate consumes.
- **EVIDENCE READY, GATE UNSATISFIED**: source evidence exists, but it has not been
  rendered into the gate input or another required condition is missing.
- **UNSATISFIED**: the required evidence or artifact does not exist.
- **FAILED**: evidence exists and records a failing verdict.

## Scorecard stage

| Condition | Status | Evidence or blocker |
|---|---|---|
| Five factory dimensions are present | **SATISFIED by assembler shape** | `package/scorecard.py` emits latency, throughput, tokensPerSec, accuracyRetained, and stability. |
| Missing dimensions render as `NOT_MEASURED` | **SATISFIED by assembler shape** | Latency, throughput, tokens per second, and stability remain explicit absences. |
| Every emitted figure comes from evidence on disk | **SATISFIED by assembler shape** | `package/scorecard.py` reads measured, source-derived, and projected values only from the five committed JSON evidence files. Every claim carries a JSON path and selector. Companion Markdown is a consistency check, never the published value source. |
| JSON and human Markdown agree | **SATISFIED where the companion document is present** | The assembler normalizes layout-only Markdown differences and refuses a value disagreement. `engine/accuracy/SHARED-EXPERT-EXPERIMENT.md` is not present in this worktree, so its committed JSON records the result but cannot yet report a verified human-document cross-check. |
| Accuracy verdict remains visible | **FAILED** | [`engine/accuracy/results.json`](../engine/accuracy/results.json) and [`engine/accuracy/shared-expert-experiment.json`](../engine/accuracy/shared-expert-experiment.json) both record `FAIL`; the assembler carries both verdicts verbatim and marks the overall scorecard blocked. |
| Stock vLLM capability below BF16 | **MEASURED ENABLEMENT** | [`engine/klinear/int4-serving-results.json`](../engine/klinear/int4-serving-results.json) records that stock vLLM 0.26.0 refuses this model below BF16. Its 98,245,528,576-byte weight floor cannot fit on a 32 GB-class card, while the 28,803,304,448-byte INT4 artifact loads and runs in this engine. This is a binary capability result. |
| Measured speed comparison against vLLM | **UNSATISFIED** | No matched benchmark receipt exists. No throughput comparison against vLLM has been measured, none is claimed, and the capability result must not be described as a speed result. |
| Engine correctness against a reference implementation | **UNSATISFIED** | [`engine/accuracy/RESULTS.md`](../engine/accuracy/RESULTS.md) states that this remains unmeasured. |

## Measurement contract

| Factory condition | Status | Evidence or blocker |
|---|---|---|
| Measured basis only | **EVIDENCE READY, GATE UNSATISFIED** | Footprint, serving memory, and both paired INT4 quality profiles are machine-readable. Every JSON value has `MEASURED`, `SOURCE_DERIVED`, or `PROJECTED` status. The 32 GiB envelope remains projected from measured weights, source-derived state, and a policy reserve. No projected value is promoted to measured. |
| Environment discloses GPU name, driver version, and serving runtime | **UNSATISFIED** | Accuracy records one H200 and vLLM 0.26.0, but the complete factory environment block, including driver version, has not been assembled into a signed receipt. |
| Equal sample count on both accuracy sides | **EVIDENCE READY, GATE UNSATISFIED** | [`engine/accuracy/RESULTS.md`](../engine/accuracy/RESULTS.md) describes identical prompts and protocols, but it is not the factory accuracy schema consumed by `check_gate.py`. |
| Confidence intervals or statistical test | **UNSATISFIED** | The committed accuracy report has point estimates and a predeclared screen, not factory confidence intervals for a catalog recovery claim. |
| No mixed sample sizes in one comparison table | **EVIDENCE READY, GATE UNSATISFIED** | The local report discloses its different metric coverage, but no gate-ready accuracy table exists. |

## Accuracy gate for optimized weights

| Factory condition | Status | Evidence or blocker |
|---|---|---|
| Accuracy document contains a `THIS PACKAGE` row | **UNSATISFIED** | No factory accuracy JSON exists. |
| Core-task recovery is at least the factory floor | **UNSATISFIED** | No gsm8k recovery row exists. The separate Kimi-Linear accuracy screen records **FAIL**. |
| Required suite includes gsm8k | **UNSATISFIED** | No result on disk. |
| Required suite includes ifeval | **UNSATISFIED** | No result on disk. |
| Required suite includes tool-calling JSON validity | **UNSATISFIED** | No result on disk. |
| Required suite includes safety refusal delta | **UNSATISFIED** | No result on disk. Safety is blocking under the factory criteria. |
| Required eval entries are complete and comparison-valid where allowed | **UNSATISFIED** | No factory eval-suite document exists. |
| Buyer protocol is the gated protocol | **UNSATISFIED** | No signed buyer-protocol accuracy receipt exists. |
| Default INT4 screen passes its predeclared thresholds | **FAILED** | [`engine/accuracy/results.json`](../engine/accuracy/results.json) records the verbatim `FAIL` verdict. |
| Shared-experts-bf16 screen passes its predeclared thresholds | **FAILED** | [`engine/accuracy/shared-expert-experiment.json`](../engine/accuracy/shared-expert-experiment.json) records the verbatim `FAIL` verdict. Holding shared experts in BF16 does not make the artifact behaviourally equivalent to the original. |

No suite exemption applies to this package.

## Package class and weights integrity

This is an `optimized_weights` package because the weight bytes changed. It must
follow the bundled-weights gate and cannot use recipe output identity as a way to
avoid paired accuracy evidence.

| Factory condition | Status | Evidence or blocker |
|---|---|---|
| Rendered manifest consistently identifies bundled optimized weights | **UNSATISFIED** | No rendered `manifest.json` exists. |
| Weights-tree SHA256 is computed with the factory walk | **UNSATISFIED** | No `weights_digest.py` result exists for the persisted INT4 artifact. |
| Manifest and signed benchmark receipt carry the same weights digest | **UNSATISFIED** | Neither artifact exists. |
| Full file manifest is covered by the integrity statement | **UNSATISFIED** | No kit has been rendered or signed. |
| Detached signature is Ed25519 and binds slug, version, kit hash, and size | **UNSATISFIED** | No detached kit signature exists. |
| In-kit benchmark receipt is Ed25519 signed with no `UNSIGNED` placeholders | **UNSATISFIED** | No benchmark receipt exists. |
| Public key is distributed and verifies the signed bytes | **UNSATISFIED** | No rendered kit exists. |
| Container image is pinned by digest | **UNSATISFIED** | No gate-ready accuracy image digest or rendered compose file exists. |

## Serving gate

| Factory condition | Status | Evidence or blocker |
|---|---|---|
| Endpoint binds to `127.0.0.1` by default | **UNSATISFIED** | No rendered `docker-compose.yml` exists. |
| API key is required | **UNSATISFIED** | No rendered compose contract exists. |
| `no-new-privileges` is set | **UNSATISFIED** | No rendered compose contract exists. |
| Unevaluated modalities are disabled | **UNSATISFIED** | No gate-ready `recipe.json`, engine arguments, or compose file exists. |
| Ready-to-host kit starts without manual configuration beyond the documented key | **UNSATISFIED** | No kit or deployment guides have been assembled. |
| Stable OpenAI-compatible API is proven | **UNSATISFIED** | A short coherent generation is not an API conformance or stability result. |

## Disclosure gate

| Factory condition | Status | Evidence or blocker |
|---|---|---|
| Card shows both recovery and relative-loss framing | **EVIDENCE READY, GATE UNSATISFIED** | The two local accuracy JSON files record the quality losses and both `FAIL` verdicts, but no rendered model card exists and no factory recovery row exists. |
| Card contains a rejected-variant ledger | **EVIDENCE READY, GATE UNSATISFIED** | [`engine/bench/LOSS-LEDGER.md`](../engine/bench/LOSS-LEDGER.md) records the failed INT4 quality candidate, rejected wrong decoders, and rejected simulated scheduler direction. It has not been rendered into a kit card. |
| Card contains cost per token | **UNSATISFIED** | No measured throughput or sourced pricing derivation exists. |
| Tier names are neutral | **UNSATISFIED** | No tier or final definition has been approved. |
| Known limits state unmeasured speed and correctness plainly | **EVIDENCE READY, GATE UNSATISFIED** | `package/definition_draft.py` states that no speed or throughput comparison against vLLM has been measured, does not imply this engine is faster, refuses behavioural equivalence, and keeps reference correctness unmeasured. No kit card exists. |

## Build-kit inputs

| Factory input | Status | Evidence or blocker |
|---|---|---|
| `recipe.json` with model, scheme, target, and immutable revision | **UNSATISFIED** | No factory package directory has been assembled. |
| Baseline and optimized benchmark JSON with matching request profile and concurrency | **UNSATISFIED** | The benchmark lanes are still in flight. |
| Accuracy JSON with task, protocol, rows, disclosures, rejected techniques, known limits, calibration, licence, production date, and image digest | **UNSATISFIED** | The local machine-readable accuracy evidence is not the complete factory accuracy schema consumed by `check_gate.py`. |
| Engine version recorded in the measured environment | **UNSATISFIED** | The purpose-built engine has no gate-ready versioned receipt. |
| Weights SHA256 supplied before render | **UNSATISFIED** | No weights digest result exists. |
| Benchmark receipt signed before archive assembly | **UNSATISFIED** | No receipt or archive exists. |

## Licence and redistribution position

| Condition | Status | Evidence or blocker |
|---|---|---|
| Licence instrument identified with location, immutable commit, and SPDX id | **SATISFIED** | [`LICENCE-DECISION.md`](../LICENCE-DECISION.md) records the companion repository, pinned commit, and MIT SPDX id. |
| Founder accepts the companion-repository instrument | **SATISFIED** | [`LICENCE-DECISION.md`](../LICENCE-DECISION.md) records the founder decision dated 2026-07-29. |
| Required MIT notice travels with redistributed material | **UNSATISFIED** | No kit has been assembled and checked for the notice. |
| Final catalog licence approval | **UNSATISFIED** | The recorded legal position is evidence, not a completed catalog approval artifact. |

## Pricing and publication

| Factory condition | Status | Evidence or blocker |
|---|---|---|
| Measured speed value or qualifying clean enablement evidence | **UNSATISFIED** | There is no measured speed comparison and no clean soak receipt. |
| Accuracy recovery has sample count and confidence evidence | **UNSATISFIED** | The factory pricing inputs do not exist. |
| Price derivation returns `PRICE`, not `REFUSE` | **UNSATISFIED** | Current inputs would be refused because speed or qualified enablement, recovery N, and confidence evidence are missing. |
| Founder sets price | **UNSATISFIED** | Pending. |
| Founder approves listing | **UNSATISFIED** | Pending. |
| Final gate passes against the signed kit | **UNSATISFIED** | No signed kit exists, and the committed accuracy verdict is FAIL. |

## Current blocking summary

The immediate hard blockers are the committed default INT4 **FAIL** and the committed
shared-experts-bf16 **FAIL**. The stock-vLLM result is capability, not performance:
vLLM 0.26.0 refuses the model below its 98,245,528,576-byte BF16 weight floor, while
this engine loads and runs the 28,803,304,448-byte INT4 artifact. No speed comparison
is claimed. Independent blockers remain the absent engine reference-correctness result,
absent release soak, absent required evaluation suite, absent weights digest, absent
signed receipt and kit, absent cost-per-token evidence, and pending founder price and
listing approvals.
