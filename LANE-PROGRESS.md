# k3fmt lane progress

## 2026-07-28 initial oracle pass

- Read the full authoritative `reference/encoding_k3.py`, `reference/run_oracle.py`, the golden prompt, `k3/toolcalls.py`, `k3/reasoning.py`, and `k3/ir.py` with line numbers.
- Verified K3 uses `<|open|>`, `<|close|>`, `<|sep|>`, and `<|end_of_msg|>`; K2 tokens are absent from the golden prompt.
- Verified assistant structure is `think`, `response`, then optional `tools`; `think` is emitted even when empty in thinking mode.
- Verified normal argument JSON is parsed into typed `argument` elements, while `_xtml_json_block` alone selects raw `json` emission.
- Verified tool results are matched by opaque id, reordered to the assistant call order, and rendered with function name plus 1-based index only.
- Chosen implementation: preserve `KimiToolParser` unchanged for K2, add a separate incremental `KimiK3ToolParser`, add K3 assistant normalization in `reasoning.py`, and put the byte-exact full prompt emitter in `template.py`.
- No git command or test runner has been used.

## 2026-07-28 implementation checkpoint

- Added incremental `KimiK3ToolParser` and registered `kimi_k3` without changing `kimi` or `kimi_k2`.
- Added K3 assistant/tool-result normalization plus a full XTML prompt renderer.
- Added `kimi_k3` prompted-tool documentation and kept every preset default on `kimi`.
- Added configurable K3 output to `MockUpstream`; its default remains K2.
- Added `tests/test_k3_oracle.py` and `tests/test_toolcalls_k3.py`, plus the minimal existing registry expectation update.
- Invoked `reference/encoding_k3.py` directly with the uv-managed stdlib Python. The primary fixture was byte-equal at 1383 bytes, and three extra cases also matched: all typed arguments, reversed opaque-id tool results, and a raw JSON block.
- Compiled all changed Python sources with `compile()` and verified a representative parser stream at all 274 split points. No test runner was invoked.
- Bare oracle Python cannot import `k3.upstream` because project dependency `httpx` is not installed in that interpreter; the MockUpstream source compiles, but its runtime check remains for the orchestrator suite.
- CodeRabbit CLI is not installed, so the skill's automated second-opinion review is unavailable. Continuing with a manual read-only audit.

## 2026-07-28 final static audit

- Corrected the prompted K3 tool declaration to the oracle's markdown plus fenced compact JSON body.
- Removed duplicate K3 call serialization from MockUpstream; it now uses the same canonical call renderer as the full prompt emitter.
- Rechecked compilation after the refactor and re-ran direct oracle comparisons.
- Final direct evidence: the primary checked fixture remains byte-equal; typed/empty-think, opaque-id result reorder, raw JSON, image arguments, non-thinking generation, dynamic tools, tool choice, response formats, escaped attributes, and thinking effort all matched `reference/encoding_k3.py` exactly.
- Directly verified K3 mock output parses into one `ParsedToolCall` and the default mock path still emits K2 tokens, using a harmless `httpx` import stub because the bare oracle interpreter has no project dependencies.
- No git command, pytest invocation, push, commit, or change outside the allowed source/tests/progress files was performed.
