# Background worker CPU and memory

2026-10-18. Performance-only; TH-01 through TH-12 apply.

## Problem and objective

V2 rebuilds an in-memory set by reading its roughly 400 MB decision trace on
every append. The RL collector re-encodes roughly 515 MB of candidate records
when updating a subset of their labels. Remove repeated work while preserving
collection, learning, independent evaluation and shadow observation.

## Contract

- Trace uniqueness remains the decoded `sym|tf|bar_ts` key over the entire file,
  including old bars and restarts. A rebuildable SQLite index under `.runtime`
  streams history once and then indexes only appended bytes. Bound SQLite's page
  cache; never retain all trace rows/keys or read the whole trace into a string.
  Serialize participating writers with an index transaction. Recover from a
  missing index, append-before-commit failure, truncation, atomic replacement
  and same-size edits. External prefix edits combined with growth are outside
  the append/atomic-replace writer contract. An unterminated final row remains
  readable and is separated from a newly appended event with a newline.
- Candidate rewrites may take an explicit scope matching the mutator's decoded
  pair guard. Decode and validate all rows, preserving malformed-row cleanup,
  ID-cache population, ordering, locks, preflight, strict failures and atomic
  publication. Copy out-of-scope valid rows without re-encoding. Always serialize
  in-scope rows even if the mutator returns false (provenance side effects).
  Batch label updates and single-pair updates scope to the decoded pair and at
  least one missing `ret_3/5/10`. Fully mature rows have no forward-label mutation
  or provenance side effects and are copied. Preserve zero-valued mature labels.
- No polling cadence, universe, feature, label definition, training gate, BUY,
  SELL, model, metric definition, alert rule or promotion permission changes.
  Downtime remains unknown, not negative training evidence.

## Verification and risks

Focused tests precede implementation: full-history/restart deduplication,
incremental decode counts, malformed and unterminated history, concurrent writes,
replacement/truncation/same-size edits, crash recovery, scoped/unscoped label
parity, false-return side effects and failed publication cleanup. Run existing
worker, observer, provenance and dataset tests and `git diff --check`.
Benchmark equal synthetic inputs with row/byte counts, CPU time and peak Python
allocation; these are component diagnostics, not whole-process or profit claims.
The first trace index rebuild costs one full streaming scan and additional disk
space; subsequent appends are bounded by new bytes plus a small disk lookup.

No policy relaxation is proposed; a maximum-period strategy replay cannot measure
a serialization/index optimization and is not a release gate here. Full Harness
currently FAILs TH-11 (portfolio artifact source hash drift); retain that finding.
After tests, restore RL with its collector and V2, restore RL Windows autostart,
and verify identity, fresh heartbeats and a short CPU/RAM canary. A short sample
does not certify steady-state savings during training or scheduled daily reports.

## Rollback

Revert this source change and restart the two workers with their existing launchers.
JSONL remains authoritative and compatible. Delete the derived index only while
V2 is stopped if a rebuild is required. No runtime files or models are committed.

## Verification evidence, 2026-10-18

- 127 focused/related regression tests PASS. The separately checked existing
  `test_collector_recovers_only_known_permission_failure` fails identically on
  HEAD: it expects retry/cancellation where the current collector stops collection.
  This pre-existing failure is retained, not hidden as a clean full-suite pass.
- Instrumented synthetic trace: 12,000 rows / 3,780,890 bytes; eight appends.
  Legacy CPU 11.28125 s, indexed warm CPU 0.125 s; peak Python allocations
  11,361,930 vs 17,631 bytes. Cold index build CPU 2.765625 s, peak Python
  allocations 257,531 bytes. Native SQLite pages are outside tracemalloc.
- Instrumented synthetic candidate rewrite: 4,000 rows / 24,882,950 bytes,
  20 pending rows. Legacy CPU 73.65625 s vs scoped 38.03125 s; peak Python
  allocations 492,738 vs 461,431 bytes. Decoded outputs match in both benchmarks.
  CPU timings include identical tracemalloc instrumentation and are diagnostic.
- Change Harness PASS; full Harness FAIL TH-11 as recorded above. Runtime
  benchmark inputs/results and rebuilt indexes are excluded from the commit.
