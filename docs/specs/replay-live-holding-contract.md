# Replay/live holding-period alignment

2026-10-04. Research-engine correction; production BUY/SELL/config unchanged.
TH-01 through TH-12 apply. This does not approve or deploy a candidate.

## Contract

The replay entry picker must use the existing live base holding limit:
`MAX_HOLD_BARS_15M` (default 48) for 15m and `MAX_HOLD_BARS` (default 16)
for other supported timeframes, for trend/strong-trend, surge, impulse and
alignment entries. Breakout/retest keep their own limits (6/10 defaults).
Preserve detector priority, trail multipliers and early-continuation flag.
The separate confirmed-leader continuation modifier is outside this base probe.

## Verification and evidence limits

Focused tests exercise every branch with changed config values and compare the
actual replay picker with the source-bound live assignment. Conditional detector
fixtures are NOT full `monitor._poll_coin` BUY parity. External portfolio
occupancy, group/candidate gates, feature fetch horizon and ordering remain
unverified. A base-hold PASS cannot certify the complete strategy.

Repeat paired historical portfolios on the maximum available immutable archive,
freezing current sources and models into a NEW output directory. Keep old FAIL,
PASS, registrations and receipts unchanged. Record UNKNOWN for unsigned/PIT
coverage or overlapping model exposure, even if numerical comparison succeeds.
No production relaxation until full live-path and independent portfolio/forward
gates pass. Do not reuse old forward certificates after source drift.

One heavy task at a time, one logical CPU, BelowNormal, library threads=1.
The paired runner emits flushed phase/arm/symbol progress to its redirected
stdout, including before each expensive series build and completed counts.
RUNNING logs and completed series are not a final result or approval.
Pause the owned learning scheduler gracefully during maintenance/replay; do not
stop Telegram or unrelated Python workers. A process exit without result and
receipt is UNKNOWN, not completion. Resume the scheduler only after the replay
finishes; existing frozen cohorts must reject source drift rather than be rebound.

Rollback: revert the replay-only correction and start a fresh research run;
never rewrite prior evidence. Production champion remains unchanged.
