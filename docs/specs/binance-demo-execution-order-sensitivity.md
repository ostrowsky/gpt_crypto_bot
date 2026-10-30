# EOS-1: frozen-policy execution-order sensitivity

Date 2026-10-30. SPEC -> RED -> CODE. Independent Demo app research, TH-01..12.
This is a counterfactual convention study, NOT RDE-2B native intrabar replay.
The incomplete historical1s/trade prerequisites remain incomplete.

## Registered question and population

EOS01: On RDE-1-MAX519-20261029, compare STOP_FIRST and TARGET_FIRST when a
fixed-position protection stop AND fixed take-profit target are both touched
inside one15m execution candle. Keep all519 current-survivor source symbols,
maximum original histories, global TRAIN/TUNE/TEST boundaries, cutoff,
candidate screen, feature cache, predictors, policies, five seeds, thresholds,
fees/slippage, capital, risk and decision cadence frozen. No retraining/HPO or
new signal selection. Both conventions are uniformly applied across ALL
ambiguous candles reached by each continuous portfolio, not only old losers.

EOS02: Recompute21 original portfolios for each TUNE/TEST segment and base/
stress cost case under each convention:168 full portfolio reports. Keep every
seed11/23/37/53/71, cash and BTC benchmark. Original TRAIN is not re-evaluated
or modified. Same chronological evaluation periods as RDE-1, no all-symbol
common-start filter. Do not pick a seed or tune a threshold from new outcomes.

EOS03: Both are SIMULATION / RETROSPECTIVE_REUSED_HOLDOUT. TARGET_FIRST is a
deliberate15m idealization; it may conflict with already observed minute-level
stop-first evidence. Neither convention is certified historical execution,
and they are NOT mathematical upper/lower portfolio PnL bounds: changing an
exit can change subsequent admissions and losses. No real/future price path
is manufactured. No profitability or daily-income proof from this sensitivity.

## Accounting contract

EOS04: Preserve RDE-1 next-open entry and entry fees/slip, resting OPEN gap
protection before discretionary EXIT, target-capped favorable gaps, adverse
stop/market fills, continuous cash, one position/max30 debit/min5, hard stop.5%,
fixed target.5%,16bar holding limit, terminal MTM and latched interval1USDT
loss/drawdown entry kill. No midnight/capital/seed reset or risk relaxation.
Trailing uses prior15m peak and has no fixed TP; learned exits likewise have
no fixed TP. These arms must be numerically invariant to the convention.

EOS05: Change ONLY the unresolved intrabar both-touch branch in fixed mode:
STOP_FIRST resolves at the original stop rule; TARGET_FIRST resolves at its
fixed target. Record candle/symbol/levels/chosen reason. Open-triggered
protection or a discretionary exit completed at open is already resolved and
cannot be overridden by later high/low. Preserve original event timestamps
(15m proxy); do not label them observed touch times. Missing selected/held
bars invalidate the objective without substitute symbols or omitted days.
Freeze each decision before requesting any execution bar.

EOS06: Reevaluate subsequent decisions from the changed portfolio state. No
patching old trades and summing hypothetical wins. Reuse frozen RDE-1 threshold
and policy observation logic. R5 common-entry signal stream must have exactly
the same full-period count/SHA for every exit/seed/convention. Precomputation
of these causal cached signals is allowed; no execution/label availability
filter may influence them.

EOS07: Once entry_killed AND flat, economic state is absorbing. The runner may
skip later policy inference, record SYSTEM_RISK_LOCK instead of pretending a
model said WAIT, and append every original15m constant-cash accounting mark.
Prove economic equivalence against the unoptimized runner. Intent/action/veto
counters after this lock are explicitly not comparable with original inferred
intents; fills, financial ledger, equity and day coverage must remain equal.
Never skip inference while holding or before the latch. Cash marks after the
lock are a known accounting transition, not imputation of missing market data.

## Binding and acceptance

EOS08: Register before evaluation: parent registration SHA, original source/
spec hashes, prepared manifest/data SHA, all symbol array SHAs, screen/cache
manifest and file SHAs, predictor/policy manifests and files, frozen threshold/
encoder/policy bindings, all84 reference portfolio SHAs, new spec/source hashes
and full Truth Harness report at registration. Validate source identity and
registration links, archive page-vector hashes, cohort/splits and actual seed/
gamma identities. No source/model/report mutation in RDE-1. Changed resume or
corrupt source/artifact rejects; successful child reports are sealed separately.

EOS09: STOP_FIRST must reproduce ALL84 original financial portfolios within
absolute1e-8USDT tolerance: all equity marks/days, trades/events, fees, cash,
drawdown, exposure, fill counts and common signal hashes. Exclude only the
explicit skipped policy-intent diagnostics. Any mismatch keeps the study
INCOMPLETE, not successful. Independently check cash+exposure-initial=net and
closed trades plus terminal open-position MTM reconcile to net. TARGET_FIRST
no-fixed-target arms must also match their original financial references.

EOS10: Report both conventions by segment/cost/arm, five-seed means with all
individual results, net interval USDT first, complete versus partial calendar
days, active days, fills, risk-lock timing, ambiguous executions, open exposure
and costs. Rank against cash descriptively; no winner/learning claim from a
positive partial/sparse mean. Show changed admissions/trades and explicitly
state whether the overall non-profitability conclusion changes. Full-period
equity and paired net-result plots use identical calendar axes/units. Full
Truth Harness FAIL remains FAIL even if baseline parity passes.

## Resources, delivery and rollback

EOS11: Offline only; no network, credentials, orders, training or worker changes.
Use existing isolated research Python and exact installed packages, <=2CPU
threads. Stream one portfolio at a time, mmap inputs, compressed detailed
reports and immutable per-report progress. Resume only matching completed
reports after integrity checks; no restoring partial/ambiguous financial state.
Interrupted work is INCOMPLETE with its failure retained. No automatic promotion.
Rollback is leaving all original production/RDE-1 files and policies untouched.

EOS12: Tests precede implementation. Cover both-touch outcomes, exact original
STOP_FIRST parity, gap and discretionary priorities, nonambiguous/nonfixed
invariance, missing data, causal lookup ordering, subsequent admission changes,
latched-risk shortcut equivalence and signal coverage, financial reconciliation,
immutable registration/report tampering and full168-arm schedule. Real CPU
saved-policy loading occurs only in the isolated research runtime. Run relevant
regressions, diff checks, selective staged Truth Harness, commit/push intended
spec/source/tests only. Runtime, reports and model weights remain uncommitted.
