# Closed-grid policy replay and valuation integrity

Date: 2026-10-01
Status: offline current-rule baseline diagnostic; no production adoption
Parent: continuous-signal-improvement.md; TH-01 through TH-12 apply.

## Measurement repair

Canonical portfolio contract becomes v2. Window span does not prove candle
coverage: require every aligned 15-minute BTC close within the requested window,
including boundaries. Count observed/expected points and missing points explicitly.
Open-position marks cannot carry prices older than 15 minutes; a stale mark is
missing evidence, not a valid valuation. Missing benchmark closes or stale holding
marks make the result incomplete and not decision-grade. Publish drawdown only
when the closed-grid holding curve is complete; otherwise null. Diagnostic cash
returns remain explicitly simulated. Complete cadence alone is not a release gate.
Existing sparse-daily fixtures must not claim decision-grade at 15-minute cadence.

## Maximum-archive runner

Read the entire recovered archive summary; derive its full start/end and universe
from it, never a chosen day/symbol subset. Verify each series content SHA, identity,
closed aligned OHLCV, ordering and every expected timestamp against its manifest.
Copy accepted input bytes into a new immutable run; record incomplete series and
counts. Do not overwrite recovery sources or existing output. Malformed identities
or hashes fail closed. Current rules are frozen by source SHA before inference;
changes during inference fail publication. Artifact result has a receipt.

Replay the entire archived period on symbols with complete 15m AND 1h histories,
derive complete 4h context, build the shared strategy candidate stream with warmup,
then simulate existing entry guards, mode exits, ATR trail, SELL and cooldown,
cluster capacity and replacement rules in one ten-slot account. No gate ablation.
Exclude warmup admissions and include last close in the event clock. Use existing
rule-only baseline context to disable unbound mutable learned artifacts ONLY
within this standalone research process. No active config/model/position writes.
Measure a 15m liquidation curve and same-window BTC buy-and-hold after costs.

Incomplete requested symbols are NOT considered unsuccessful trades or zero
capture. Partial-population results are diagnostic only, never evidence permitting
BUY/SELL relaxation. Report partial-population status, current-rules rather than
historical champion parity, learned scoring disabled, missing historical universe
and agent-policy parity, and local interior candle provenance not independent
raw-exchange certification. Never claim sealed model validation or improvement.
Always runtime_eligible=false and achievement_claimed=false. Controller unchanged.

Rollback: stop offline runner; production unchanged. Next gates: certify exchange
provenance/point-in-time universe, snapshot complete champion and challenger policy
inputs, causal learned scoring parity, sealed paired portfolio evaluation and
forward/canary validation. Do not silently shorten the maximum archive period.

## Verification

Tests: endpoint-only BTC gap, stale holding marks, complete 15m curve with costs,
malformed/nonfinite bars, archive SHA mismatch, missing series, immutable output,
warmup/end-boundary event clock and source drift. Run maximum archive baseline,
focused regression tests, full/staged Truth Harness and diff check; commit/push
source/spec/tests only. Generated snapshots, trades and reports remain runtime.

Verification note (2026-10-01): the full Truth Harness is FAIL at TH-11 after
the valuation correction, because the previous canonical portfolio artifact is
bound to the old evaluator source hash. Preserve this invalidation; do not patch
artifact hashes or substitute a partial-population diagnostic as decision-grade
proof. A fresh independently verified, complete v2 evaluation is required.
