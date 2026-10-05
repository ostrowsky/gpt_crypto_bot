# Price / volatility overlay comparison

Date: 2026-10-05 Europe/Budapest. Offline diagnostic only. TH-01..TH-12 apply.
Full Truth Harness at registration: FAIL TH-11, canonical replay source mismatch.
The old portfolio artifact is not reused, repaired by rewriting hashes, or called
proof. No live config, trained model, position or notification is changed.

## Population and timing

Use the maximum recovered closed archive and extend it through the latest aligned
complete local-cache end, frozen before results. Verify archive input hashes and
every closed 15m/1h timestamp. Missing recent candles are retrieved from Binance
public klines with raw-response hashes. The recovered archive has 93 complete
of 105 requested symbols. Report all exclusions and the historical point-in-time
universe limitation. This fixed complete population is not certified live parity.
Do not select a smaller period or BTC/ETH/SOL-only cohort to obtain a winner.

Baseline is the current rule-only replay with unbound mutable learned scores
disabled inside the offline process. Freeze current sources/config hashes.
Preserve entry gates, exits, cooldown, replacement, cluster limits and ten slots.
Replacement options follow the established evaluator defaults when absent from
config: enabled=True and minimum delta=8, identical in baseline/direction arms.
Disable mutable temporal-event logs in all arms. This is not a reconstruction of
historical live models, agent-only policy, tick executions or original watchlists.
The established closed-bar fill convention is idealized; positive fees/slippage
are applied separately. Future day-top labels are evaluator-only.

## Registered hypotheses

All models forecast the next hour from closed 15m candles. This differs from the
minute-data demo: realized variance is the sum of FOUR future 15m squared
log-returns, a coarse proxy. No claim of minute-level RV precision is made.
Pooled Ridge predicts the next-hour cumulative log-return. HAR-style Ridge
predicts log next-hour realized variance from trailing 1h/4h/24h components.
Variance mean uses TRAIN-only exp-residual smearing, not the squared volatility
median. Fixed alpha=10; no search against historical portfolio outcomes.

Start with the entire archive warmup; then refit every 30 days using at most the
preceding 60 days, a fixed hourly TRAIN origin grid and all eligible symbols.
Require last training target close STRICTLY before the next inference block.
Scaling, coefficients and lognormal correction use only those training rows.
Predict every 15m origin through the original maximum evaluation end, including
origins whose future labels are not yet inside the archive. Evaluate forecast
loss only after all four future bars exist. Record folds, hashes and denominators.
Hypotheses are retrospective walk-forward OOS, not independently sealed evidence.

Four main arms:

- baseline: unchanged rule-only admission and unit position-size multiplier;
- volatility: same admission/exits, multiplier min(1, 0.01 / predicted hourly
  sigma), frozen at entry; no leverage and no volatility-based BUY prohibition;
- direction: retain a rule candidate only if predicted hourly log-return exceeds
  the exact roundtrip fee/slippage break-even threshold; preserve other gates;
- combined: direction admission plus the same volatility multiplier.

An additional EWMA sizing control uses the same 1% hourly risk target/cap, with
lambda=0.94 on past 15m squared returns and hourly variance=4*EWMA. Report exposure
beside drawdown so reduced capital deployment is not mistaken for forecast alpha.
The 1% target is an experiment parameter, not an optimized production risk budget.
Baseline/direction admission streams are rebuilt/simulated before sizing: a
direction filter acts on ALL rule candidates, not just baseline-selected trades.
Sizing does not feed back into rule admission in this experiment; cash/capacity
invariants are checked in an independent unified mark-to-market account.

## Accounting and conclusions

One account per arm, ten distinct symbol positions, starting capital 1, no
borrowing, allocation=min(cash, liquidation equity/10 * multiplier). Partial and
final exits precede same-time admissions, except own boundary entry/exit. Costs
apply at entry, partial/final exit and BTC buy-and-hold. Mark EVERY 15m close with
fresh prices; missing marks, unmatched exits, cash/slot violations fail the run.
Unit-size account must equal the canonical evaluator on focused fixtures.
Use positive current configured fee (at least 7.5bp per side) and 5bp slippage;
also report double-cost sensitivity without changing model/gate parameters.

Publish full-period after-cost return, BTC benchmark/alpha, max drawdown, average
gross exposure, costs, trade counts, daily leader capture and early/false BUY
counts with explicit denominators. Separately report model MAE, direction counts,
and variance QLIKE against past-RV/EWMA controls. No winning return guarantee.
Daily paired portfolio log-return block bootstrap (1/3-day blocks, 5000 draws,
Bonferroni for three main comparisons) describes uncertainty; already-seen
history, partial population, static costs and missing execution/PIT parity still
prevent production approval. A zero-trade cash arm is not evidence of alpha.

Drawdown peak starts at initial capital 1. Daily bootstrap also starts its first
complete day from capital 1, including fees on entries exactly at the evaluation
origin. An arithmetic repair may derive corrected statistics from unchanged
frozen curves, without repeating training/replay or modifying raw evidence.
Record the original result hash, derivation code hashes and changed fields;
keep original source hashes and raw results intact. No model/period/arm selection
is permitted as part of this correction.

Runtime inputs, predictions, weights, curves, reports and logs stay outside Git.
Source/spec/focused tests are reviewed, staged Harness checked, committed/pushed.
Indicator computation may use up to four separate processes and native NPZ
checkpoints. It must call the same indicator implementation, restore stable
symbol order before pooled training and match sequential computation. Interrupted
pre-outcome computation may reuse the SHA-verified frozen market with exactly the
registered population/bounds; performance changes must not select new outcomes.
The identical rule-only candidate function may run in independent processes per
symbol/timeframe, each with the same closed BTC context and symbol 15m/4h packs.
Merge candidates in original symbol/timeframe order. Verify every candidate field
against sequential computation on focused fixtures and full-archive spot checks.
Feature reuse requires identical market hashes, config/indicator/replay hashes,
complete symbol coverage and every native checkpoint hash. No boundary or model
parameter is selected during performance/resumption changes.
Trusted local candidate snapshots can be reused only with a receipt binding
their bytes, exact market bounds/hashes and every candidate-rule source hash.
Reject clock/count drift; this does not accept downloaded/untrusted pickle files.
Rollback: stop offline process; running bot is unchanged. Before any production
adoption require certified full-population policy/execution parity, independent
future shadow results, risk budgets, rollback and current Full Harness PASS.
