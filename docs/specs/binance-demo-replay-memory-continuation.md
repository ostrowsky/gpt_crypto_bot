# FMR-1: memory-bounded continuation of the frozen MRP-2 experiment

2026-11-18. SPEC -> RED -> CODE. Independent research only; TH-01..12.
MRP-2-20261118-v1 failed with MemoryError while reading an invariant report,
after87 COMPLETE portfolios. Preserve registration, reports, source bundle,
status and incident. This is a new declared continuation, not a renamed PASS.

FM01: Economic semantics remain byte-identical: trade_minute_core.py,
rl_direction_core/runner, minute_replay_core candidate semantics, fixed model
features/thresholds/costs/splits/cohort/risk and all168 tasks. No new training or
hyperparameter selection. Verify prior complete report hashes and financial
checks. Only COMPLETE reports may be explicitly reused with source path/SHA,
original registration SHA and REUSED_PRIOR evidence; no partial account restore.
All84 STOP_FIRST reports must exist. Recompute only missing affected fixed-target
arms. Nonfixed TARGET_FIRST reuses the matching STOP_FIRST result; absence of
that evidence fails closed. No neural policy/model/memo load is needed or allowed
for this continuation. Cache/buffer memory goal<=256MiB, reports one at a time.

FM02: Reverify old data with the original data-audit implementation before any
continuation. Pin its closed public journal/audit proof and every used/imported
response, accepted frame and raw-prefetch extraction. No refetching old packets.
Candidate features remain isolated from execution presence. Old raw-frame values
must equal the verified packet's exact timestamp slice. Fine-data acceptance
and empty-minute execution stay TA03..07, including unresolved same-minute cases.

FM03: Old collection expiry remains expired and unchanged. A NEW read-only
continuation window may run7200s from registration, solely to complete missing
fixed-target cases after computational failure. Preserve the original total
2000 new-native-attempt quota across the lineage: derive old actual attempts
from paired durable REST_BEGIN/END, new scope attempt_limit=2000-old_attempts.
Never reset attempts or omit errors; no remaining budget means no new request.
Record old expiry, inherited count, new window and reason explicitly. This is
not a change to trading/pilot expiry, capital, order or loss limits. Same public
Demo GET host/path/pacing/timeouts/PRC commitments; no credentials or orders.

FM04: Frozen input/model/source hashes are checked before and after computation.
Use the prior immutable source bundle for historical documentation. Only an
EOF-whitespace normalization of a .md file may remap to the pinned snapshot;
any executable/data change invalidates reuse. Register both historical and
current documentation hashes. Parent files and old results are not modified.
New metadata never pretends the reused portfolios were newly executed. The
top-level data_scope_sha256 identifies the NEW own scope, not its parent.

FM05: Produce all168 conditional reports, source/ledger/calendar checks and
paired graphics, exact same seed grouping. Report recomputed/prior/invariant
counts separately. No winner on incomplete source/objective, no daily-income or
actual Demo-profit claim. Root full Truth Harness FAIL stays FAIL; old MRP-1
and MRP-2 incomplete states stay visible. Runtime/market/model files uncommitted.

FM06: Tests before code: cumulative budget/exhaustion, immutable code versus
EOF-only docs, incomplete report rejection, missing nonfixed source rejection
and remaining tasks requiring zero neural policies. Full app tests, diff/staged
harness/review, intended-only commit and push. Rollback: stop only own finite
research continuation; no production behavior was changed.

FM07: Per-request guards hash an immutable imported-file manifest instead of
rehashing thousands of imported cache packets for every GET. Individual cache
hashes remain in the before/final full input verification and in the sealed
manifest; source/spec/root/journal files stay in the live guard. No weakening
of wire/frame validation. A local pre-GET timeout with zero query plans/REST
records is not a native attempt; retain it as an incomplete controller run.
The network/child timeout and cumulative2000 quota are unchanged.
