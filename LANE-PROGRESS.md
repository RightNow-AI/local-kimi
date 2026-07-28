# Harness lane progress

## 2026-07-28

- Read `k3/replay.py`, `k3/record.py`, `tests/test_record_replay.py`, `tests/test_conformance.py`, and `tests/cassettes/README.md` in full.
- Confirmed `normalize()` sends structured ID fields and free-string regex matches through `_norm_id()`, which currently erases every recognized value to a prefix token.
- Confirmed signatures preserve carried reasoning through a stable digest, timestamps normalize to zero, and `diff_json()` reports path-anchored differences.
- Confirmed `Recorder.finish()` appends every saved path to `self.written`, with no bound, and `written` is otherwise only asserted by tests in the inspected scope.
- Planned fix: one ID map per `normalize()` call, assigned in traversal order and shared by structured fields and free strings; retain only the 100 most recent recorder paths.
- Added RED-first coverage for relationship divergence, random-value equivalence, shared field/string mapping, deterministic call-local numbering, and bounded recorder history.
- Replaced prefix-token ID erasure with per-call `<id:N>` numbering in `k3/replay.py`; timestamps and signature handling were left unchanged.
- Updated the free-string regex to derive from the same recognized prefix table and use the same per-call mapping as structured fields.
- Bounded `Recorder.written` in place to the 100 most recent saved paths without changing constructor, `finish()`, save/load, gzip, or plain JSON behavior.
- No Git commands or test runners used.
