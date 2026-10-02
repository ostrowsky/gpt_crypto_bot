# Frozen full-policy paired replay

Date: 2026-10-02. Status: diagnostic infrastructure, not live strategy approval.
TH-01..TH-12 apply. Full Harness at registration: FAIL TH-11 (current source hash).

The independent evaluator must compare full admission, capacity/replacement,
cooldown and exit paths, not ranker T+5 returns. A runner freezes both raw model
artifacts, source digests, archive manifest and all closed market receipts before
execution. The historical phase uses the entire maximum recovered window including
warmup. Separate arm candidate populations are rebuilt because model score changes
can change admission; never reuse champion-selected candidates for the challenger.
Both arms use the same closed market grid, ten positions, positive fees/slippage,
BTC benchmark, full strategy exits and forced boundary liquidation. No mutable
temporal scout logs or learned model files may silently enter the run.

Supported challenger is exactly the existing bounded ranker overlay (delta +/-1
point, champion EV/replacement components unchanged); this is not a general SELL
learner. All other live flags remain unchanged. If the ranker is disabled, its
weight is zero, or champion provenance fails the live loader, preflight returns
BLOCKED: training alone cannot change that strategy. Never bypass these failures
by setting runtime_eligible on a model or enabling a production config.

Pure account recomputation is shared with the signed independent gate. Unsigned
numerical PASS is diagnostic-only and cannot issue tickets. Incomplete historical
population, missing independent raw/PIT provenance and model exposure remain
UNKNOWN. Historical partitions are not a sealed cohort. A future shadow period
cannot be backfilled or called prospective by renaming this historical bundle.

Successful release still requires independently certified historical/sealed/shadow
and then canary phases, adequate periods/counts, positive paired lower CI, risk
budgets, current Full Harness PASS and existing signed CAS rollout. No winning
candidate is promised. Failure retains champion; crash-safe pointer locking,
ticket expiry and rollback remain mandatory. Operator must supply actual missing
evidence, not fabricate authority flags. Automated collection of matched full
live shadow admissions/fills is not yet delivered by this historical runner.

Tests: effective-policy preflight, source/model/input drift, isolated overlay
budget and restored flags, separate arm rebuilds, immutable outputs, unsigned
comparison never authorizes; full existing signed activation/rollback regressions.
Rollback: stop replay runner; production behavior is untouched.

Actual preflight on 2026-10-02: bound maximum recovered archive Apr 1--Sep 28
(181 evaluation days; 93/105 complete symbols), frozen current champion and
registered candidate. All archive receipts were verified. Result BLOCKED with
three causes: runtime disabled, weight zero, champion lacks runtime provenance.
No paired portfolio backtest or candidate victory is claimed for this run. Full
future shadow/canary capture and live strategy activation remain incomplete.
