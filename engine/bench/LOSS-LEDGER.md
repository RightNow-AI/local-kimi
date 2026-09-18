# Kimi-Linear optimization loss ledger

Status: **MEASURED where stated, SIMULATED where stated**

This ledger records losses, gains, and rejected directions. A missing measurement is
never written as zero. Machine-readable JSON is the value source. Companion Markdown
is the human view and is cross-checked where present.

| Transformation or variant | Evidence status | Quality or fairness cost | What it bought | Disposition | Evidence |
|---|---|---|---|---|---|
| Selective W4A16 checkpoint | MEASURED | Perplexity moved from 12.712 to 12.815, a +0.81% increase. Top-1 agreement was 85.16%, mean KL was 0.0555 nats, greedy identity was 37.69%, and router set agreement was 34.30%. The predeclared accuracy verdict is **FAIL**. | Source tensor storage fell from 98,245,528,576 bytes to 28,803,304,448 bytes, a 3.41x reduction. The planned and actual tensor bytes matched exactly. | **REJECTED for listing in its current form.** The evidence supports an `optimized_weights` disclosure shape, not a recipe or behavioural-equivalence claim. | [`engine/quant/quantization-results.json`](../quant/quantization-results.json), [`engine/accuracy/results.json`](../accuracy/results.json) |
| Shared-experts-bf16 W4A16 checkpoint | MEASURED | Perplexity moved from 12.712 to 12.960, a +1.95% increase. Top-1 agreement was 91.41%, mean KL was 0.0398 nats, greedy identity was 40.77%, and router set agreement was 36.74%. The predeclared accuracy verdict is **FAIL**. | Tensor storage was 29,067,840,512 bytes, exactly 264,536,064 bytes above the default artifact. Planned and actual bytes matched. 20,072 tensors were quantized and 421 retained. | **REJECTED for listing in its current form.** Retaining shared experts in BF16 improved some agreement metrics but did not pass the declared accuracy gate and does not support behavioural equivalence. | [`engine/accuracy/shared-expert-experiment.json`](../accuracy/shared-expert-experiment.json) |
| Packed INT4 serving path | MEASURED | This serving run did not measure additional quality loss or reference correctness. The paired accuracy results above remain the quality authority. | Resident weight bytes matched checkpoint tensor storage exactly. Peak reserved memory during the short H100 generation was 30,511,464,448 bytes, and all 27 layers and 256 experts per MoE layer ran with coherent output. | **KEPT as capability and footprint evidence only.** Stock vLLM 0.26.0 refuses this model below BF16, whose weights require 98,245,528,576 bytes. This engine loads and runs the 28,803,304,448-byte INT4 artifact. No speed comparison is measured or claimed. | [`engine/klinear/int4-serving-results.json`](../klinear/int4-serving-results.json) |
| Wrong INT4 decoders: group axis, scale, and nibble order | MEASURED negative controls | Each deliberately wrong decoder diverged on a real checkpoint tensor. | They proved the round-trip verifier can reject incorrect decoding rather than merely agreeing with itself. | **REJECTED as required.** All three wrong decoders failed. | [`engine/quant/quantization-results.json`](../quant/quantization-results.json) |
| Routing-aware batch composition, uniform saturated B=32 and P=128 | SIMULATED | The best-case union result came with up to 64.8 seconds of worst-case deferral in the simulated sweep. Real routing traces, lookahead latency, cancellations, continuous arrivals, composer cost, and real grouped-top-k correlations remain unmeasured. | Best-case union reduction was 1.72%, with 1.48% modeled throughput gain. | **REJECTED for production integration and for any catalog performance claim.** The result only justifies a bounded trace-driven prototype. | [`engine/scheduling/RESULTS.md`](../scheduling/RESULTS.md) |

## Current decision

The default selective INT4 artifact and the shared-experts-bf16 artifact are real
footprint results and failed quality candidates at the same time. Both verdicts are
**FAIL**. Neither may be described as behaviourally equivalent to the BF16 checkpoint.

The vLLM distinction is capability, not speed. Stock vLLM 0.26.0 refuses to serve this
model below the 98,245,528,576-byte BF16 weight floor, so a 32 GB-class card cannot run
that option at any speed. This engine loads and runs the 28,803,304,448-byte INT4 artifact.
No throughput comparison against vLLM has been measured, none is claimed, and nothing
here implies that this engine is faster. This engine's correctness against a reference
implementation also remains unmeasured.
