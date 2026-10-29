# Feature-source financial ablation (FSA-1)

Status: preregistered research only, 2026-10-29. No live policy change.
Objective: daily net portfolio equity PnL in USDT, not accuracy.
Reopen basis: new Spot Demo source, financial target and causal matched-feature
comparison; old futures direction/execution hypotheses remain rejected.

## Frozen historical protocol

Use the maximum available verified BTCUSDT and ETHUSDT 15m archive in the
independent application's all-usdt/history.sqlite through 2026-10-28 14:30 UTC.
This liquid diagnostic cohort does not represent all 519 current-survivor pairs.
Read only, verify compressed and canonical page hashes, record page-vector hashes,
source/receipt times, regular/shortened/missing counts. Never borrow old aggregates.
Receipts collected now cannot certify historical publication latency or membership.
Retrospective test periods have been exposed to earlier studies: research only.

Global calendar-duration split 70/15/15 from earliest common open to cutoff;
train 2017-08-17 04:00 to 2024-01-24 18:30 UTC; tune ends 2025-06-11
16:30 UTC; test ends cutoff. No parameter or threshold search, refit or test
selection. Exclude training labels not mature before train cutoff and windows
crossing a missing/shortened bar; report exclusions. Do not interpolate gaps.

C0 = frozen trend-breakout-v1 rule diagnostic control.
A0 = CatBoost with causal OHLCV features on the same trend-rule candidates.
A1 = A0 plus closed BTC/ETH returns/volatility and relative strength.
A2 = A1 plus candle-aggregated taker buy volume ratios. These are a flow proxy,
not raw trades, second-level order flow or L2. No L2 claim from candle data.

All learned arms use identical candidate rows, labels, model parameters, risk,
execution and zero predicted-return gate, ranking by predicted net return then
symbol. Candidate availability uses only prior closed valid 32-bar windows.
A0 features: log returns at 1/2/4/8/24 bars, range, realized volatility,
MA gaps, relative volume, log trade count and close location.
A1 adds both instruments' corresponding closed returns/volatility, relative
return and breadth, with common valid context requirement for all learned arms.
A2 adds signed taker quote imbalance over 1/4/8/24 bars and zero-volume flag.
Zero actual volume is valid; invalid/missing volumes block the common row.

CatBoostRegressor 1.2.10: iterations=300, depth=5, learning_rate=.05,
l2_leaf_reg=10, loss_function=RMSE, random_seed=42, thread_count=2,
bootstrap_type=No, has_time=True, allow_writing_files=False. No early stopping.
Target: isolated counterfactual candidate trade net return with same fixed exit
policy, next-open entry, maximum 16 bars. Reserve 16 bars for label maturity,
including when a trade closes sooner. This label is a training proxy; only the
shared-cash portfolio evaluation measures the financial objective.

## Common causal simulator

New research modules only; do not alter frozen trend_research/trend_runner or
running pilot. 100 USDT initial capital per arm per segment, shared cash,
one position, entry debit <=30 including fee, 4-bar cooldown, daily loss and
peak drawdown kill=1 USDT, max holding=16 bars, stop=1%, TP=4%, trailing=1.5%
after prior peak gains 1%. Weakening fast8<slow24 is known before next open.
Base fee=10bp per fill, adverse market slip=5bp; stress=20bp/10bp.
TP is a limit fill at target (no adverse market slip below its limit).
If TP and stop touch in the same bar, stop wins; gap stop fills at worse open.
Trailing uses prior bar peak, never same-bar high before low. Mark open exposure
at final close; include unrealized PnL and separately disclose open exposure.
Never select using execution bar presence or values: decide on past histories,
then resolve the selected next open. Missing selected/held execution or invalid
price makes the segment INCOMPLETE with objective null; no substitute symbol.
Require full per-pair regular grid in tune/test; train gaps reset warmup.

Budapest calendar MTM daily net PnL including fees/slippage, DST-aware boundaries,
complete days only; include zero-trade days. Report requested/observed/excluded
days, total net, daily mean, profitable n/N, trades, fees, turnover, max drawdown,
open exposure and stress, against C0 and USDT no-trade=0.
Primary contrast A2-A1; secondary A1-A0. Paired circular 7-day block bootstrap,
5000 draws, seed42; 98.75% two-sided intervals for four predeclared bounds (two deltas and two absolute earnings intervals), Bonferroni family coverage >=95%.
No filling missing days. Minimum 30 complete test days. SESOI=.01 USDT/day
per100; supporting evidence requires delta lower bound>SESOI, absolute earnings
lower bound>0, drawdown<=reference+.5USDT, stress positive and adequate sample.
Otherwise NOT_SUPPORTED or INCONCLUSIVE; never switch objective after results.
Even supportive retrospective numbers cannot authorize production: frozen current
survivors, uncertified receipts/execution, exposed test and full Truth Harness
FAIL remain explicit. Fresh financial canary and maximum actual population
validation still required by phase6. No model promotion or actual-fill training.

## Isolation and evidence

Own runtime feature-ablation/, atomic registration before fitting, reject changed
registration on resume. Pin source/spec/environment/data/model/report hashes.
Own Python3.11 research copy and pinned wheels (NumPy2.4.4, SciPy1.17.1,
CatBoost1.2.10, tzdata2025.3); no root/user site packages, keys, adapters or orders.
Ordinary app environment and live workers remain unchanged. Runtime never staged.

## Acceptance scenarios / test-first

FSA01: future suffix edits leave origin features/decision identical.
FSA02: chronological mature-only labels with 16-bar purge; gaps never imputed.
FSA03: known-zero flow distinguished from invalid/missing data.
FSA04: shared cash, entry debit, fees, stop-first and prior-peak trailing.
FSA05: missing selected execution invalidates; cannot pick another future-valid coin.
FSA06: daily reconciliation, complete DST calendar and no-trade days.
FSA07: paired comparisons align dates, preserve missingness, deterministic intervals.
FSA08: archive integrity and immutable registration fail closed.
FSA09: arms and source claims remain proxy/SIMULATION/research-only.
Tests must fail on intended absent APIs before implementation; preserve RED evidence.
Rollback: stop only own research process; delete nothing and leave live bot untouched.

Holding expiry clarification before implementation: both label and portfolio close at the sixteenth execution bar close (origin through origin+15*STEP). No seventeenth-bar open is read. Closed-bar feature cutoff is origin_ms, label_available_ms is origin+16*STEP. C0 uses this same simulator; it is a rule diagnostic, not certified execution parity with an older runner.


Before-fit review clarification: a weakening sale permits entry from the next bar only, identically in all arms. Daily loss/drawdown kill latches further entries for that day; existing positions remain protected and losses can exceed1USDT due gaps/costs. Reported maximum drawdown uses open/close MTM observations, not certified intrabar equity extrema.

Power limitation fixed before fitting: this first retrospective diagnostic has no certified prospective power/MDE calculation and no untouched test. Even when every numerical gate passes, label the result POSITIVE_PATTERN_POWER_UNCERTIFIED, never SUPPORTED or promotion eligible. Failed numerical gates remain NOT_SUPPORTED; too few/misaligned days remain INCONCLUSIVE. Four corrected intervals describe effects and do not certify the financial objective.

Technical repair v2 (before any fitted model/test outcome): first attempt stopped on CatBoost1.2.10 fit(feature_names=...) unsupported argument. Preserve failed attempt; use catboost.Pool(data,label,feature_names) then model.fit(Pool). Parameters, features, training rows, labels, periods, capital and financial rules unchanged. Add real pinned-library fit/save/reload contract test before code. New attempt ID FSA-1-BTCETH-20261029-v2, original failed attempt remains evidence.
