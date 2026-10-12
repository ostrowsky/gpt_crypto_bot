# Requested capacity/chase removal: historical validation

2026-10-12. User requested removing the overall ten-position entry block and
the top-gainer chase guard. This stage is research-only, not live enablement.
TH-01..12 apply. Do not touch the independent Binance Demo application.

## Preregistered experiment

Evaluate control, no_capacity, no_chase and both on the maximum available
immutable closed-candle archive. Freeze current sources, config and model bytes
before calculation; verify input and output hashes. Regenerate two full entry
populations: existing chase settings and chase disabled. Do not reuse the
already filtered control population to estimate the removed chase guard.

Capacity-disabled replay bypasses only the overall capacity admission/replacement
branch. Preserve score floors, cooldown, mode/cluster limits, exits, entry price,
sorting and model settings. Keep ten as the reference capacity for other rules;
do not simulate 'unlimited' by setting an arbitrary enormous integer.
This is a per-run simulator option, default enabled, not a production switch.
With capacity bypass, a full cluster remains blocked rather than using the
replace-only exception and then adding an extra position without a replacement.

Evaluate all generated trades with the same initial 10,000 quote capital,
ten-slot allocation, positive fees/slippage and BTC benchmark. If removing the
signal limit causes cash or capacity violations, retain the violations and mark
the financial evidence invalid; do not silently invent leverage or replace the
capital-allocation contract. Candidate/trade counts alone do not prove profit.
Any new capital-allocation scheme requires a separate specification and audit.

## Acceptance and limitations

Publish each arm's counts, canonical account diagnostics and violations, and
compare against the unchanged control. History is reconstructed, models may
have seen the period, and live external-agent/group/candidate parity is not
certified. Overall release eligibility is false, even if numerical diagnostics
improve. Enabling in live requires completed maximum-history validation,
independent causal/forward confirmation, downside guardrails and rollback.
Do not waive these because ICP rose after one blocked entry.

Run one research worker, one CPU, BelowNormal, numerical-library threads=1.
Emit flushed phase/symbol/frame progress. Preserve old proofs and use a fresh
directory; no Telegram restart or model training. Result and receipt are both
required; vanished process without them is UNKNOWN.

Tests: default capacity remains blocking, research bypass, isolated four-arm
mapping, chase settings restored, accounting violation rejection, hash/bounds
checks and no release authority. Rollback: stop/revert research tooling only;
live protections remain enabled until approved validation is complete.
