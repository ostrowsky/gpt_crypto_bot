# Negative-day rebound score audit

Status: offline research only; no production relaxation.
Registered: 2026-10-01, before execution.

## Hypothesis and arms

A locally rising 15m breakout/retest can be blocked because its local-day return
is negative. Compare baseline with `extra_penalty_off` (remove only the -8
penalty below -0.25%) and `negative_terms_off` (also neutralize the negative
clamped day-return contribution). Keep threshold 34, exits, capacity, chase,
cooldown and all other admission rules unchanged. Nonnegative days and other
modes/timeframes are unchanged. Never substitute general candidate score for
top-gainer score. In today's logged TRX case even restoring approximately
14.04 points leaves 8.79 below 34: this hypothesis alone cannot admit that case.

## Evidence contract

Use the maximum recovered complete closed-candle archive, all eligible symbols,
10-day warmup, immutable input/source hashes. Exclude missing series explicitly.
Build causal current-rule candidates once, then copy them per arm without
mutating the baseline. Use full BUY/SELL replay with ten-slot unified cash
account, configured fees and 5bps slippage, BTC benchmark. Report full-period
and chronological 60/20/20 windows with identical population/bounds. These are
historical diagnostic partitions, NOT sealed independent holdouts.
Truncate candle/feature arrays at each boundary, retaining warmup and original
indices; mark open positions at boundary rather than the archive's future end.
Exclude mutable temporal scout logs from this reproducible current-rule audit.

Report trade counts, added entry identities, net-losing-trade counts (not
top-mover false-positive rate), TRX entries, portfolio return/drawdown and BTC
comparison. Preserve all arms, including rejected results. No zero-denominator
ratios. T+5 candidate labels cannot approve trading changes.

Current-rule reconstructed population is not proven live population parity;
learned scores are disabled offline, raw exchange/PIT-universe certification
is unavailable. Therefore results are UNKNOWN for production eligibility even
if an arm improves. Full Truth Harness FAIL remains visible. Require independent
population certification, untouched OOS and forward canary before relaxation.

## Safety and verification

Standalone research command only; no config edits, model loads, deployment or
bot restart. Hash changes abort; exclusive output directory prevents overwrites.
Focused tests cover exact adjustment boundaries, unchanged other modes, copy
isolation, chronological window coverage, missing/zero denominators. Run relevant
tests, diff check, staged truth harness, commit and push source/spec/tests only.
Rollback: stop offline command; production was never modified.
