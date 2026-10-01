# Independent ten-slot replay foundation

Date: 2026-10-01
Status: retrospective diagnostic only; production adoption blocked
Parent: continuous-signal-improvement.md. Applies TH-01 through TH-12.

## Contract

Consume the entire immutable causal-entry reconstruction, verify its receipts,
exchange response bindings and source hashes, and independently recompute the
post-exposure cohort using the frozen model. Never train or replace a candidate.
Do not reset the prospective experiment. Report the complete available range,
exposure exclusions, incomplete groups and the eligible replay range. Newer live
observations outside this frozen snapshot are explicitly not covered.

Compare candidate-score ordering with frozen-model ordering on identical
observations. This baseline is NOT the deployed champion strategy. Rank only
observations available before the same simulated next-minute-open execution;
never pool different decision times using hindsight. Strip labels, teacher,
label provenance and execution reconstruction before inference.

Simulate a single cash account, initial equity 1, ten slots, fixed initial-capital
allocation 0.1 per entry, no leverage, one open position per symbol across all
timeframes. Process scheduled T+5 exits before entries at the same timestamp.
Apply explicit nonnegative per-side fees and adverse slippage to both policies.
Preserve fills, skipped events, fees, cash/equity reconciliation and batch counts.
T+5 close execution is an assumed fill, NOT an actual exchange execution. Exits
are fixed-horizon, NOT production SELL. All observation actions are research
candidates; recorded admission/cooldown/replacement rules are NOT replayed.

All invalid rows or incomplete feature-bar groups remain excluded, with counts.
Only validated raw OHLCV produces prices; never invert rounded return labels.
Fail on unknown IDs, malformed/nonfinite transactions, negative costs, duplicate
IDs, inconsistent timing, changed source or damaged artifacts. Output is an
immutable run directory with input/source hashes and receipt; existing output
cannot be overwritten. No active model/config/position writes or network calls.

## Evidence boundary and next gates

After-cost fixed-horizon portfolio returns are simulated diagnostics. Settled
equity drawdown is NOT mark-to-market drawdown. Continuous price coverage, actual
admission/SELL parity, point-in-time universe, benchmark prices, funding, sealed
historical protocol, exact execution and forward/canary adapter remain missing.
The result must always be BLOCKED, runtime_eligible=false, achievement_claimed=false.
It must never satisfy the existing safe-release portfolio gate. No policy is
relaxed; rollback is to stop invoking this offline diagnostic, live unchanged.

Next: reconstruct verified closed-bar paths and immutable admission/exit policy
snapshots; replay champion SELL/cooldown/rotation on both arms, add BTC same-window
benchmark and continuous equity/drawdown, preregister benefit/risk thresholds,
then independent forward/canary validation. Missing history stays UNKNOWN.

## Tests and validation

Tests cover cash conservation, ten-slot capacity, symbol overlap, adverse costs,
same-time exit/entry ordering, no future candidate pooling, deterministic ID ties,
invalid numeric/timing/duplicate inputs, complete liquidation and immutable output.
Run the entire maximum available verified frozen snapshot, no symbol/date filter.
Run focused tests, full and staged Truth Harness, diff check, commit and push.
Runtime reports and frozen model artifacts are not committed.
