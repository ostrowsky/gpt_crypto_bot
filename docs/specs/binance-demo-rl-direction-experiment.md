# RDE-1: causal direction forecasts and sequential-control ablation

Date2026-10-29. SPEC -> RED tests -> CODE. Independent Demo app research only.
Parent: phase6, competitive GAP review2026-10-29, TH-01..12. No policy promotion.

## Question, scope and preregistration

F15/F60 compare probabilistic direction and return predictions. Raw probabilities
are evaluated, not claimed post-hoc calibrated. P compares shared
cash net daily equity PnL. R1 CatBoost, R2 GRU; R3 myopic actor-critic gamma0,
R4 same PPO architecture gamma1; R5 common entry signals with fixed/trailing/
learned exits. R3/R4 inherit SAME frozen GRU forecast heads, not independent
price predictions. Never interpret action probabilities as p_up.

Run RDE-1-MAX519-20261029 on every source-registered current-survivor USDT pair
in own all-usdt/history.sqlite, earliest available through1793197800000
(2026-10-28 14:30UTC). Read archive only, verify each compressed/source hash,
page/order/schema, count shortened/gapped data. No all-pairs common-start filter,
Source hashes use the collector's canonical UTF-8 JSON (`ensure_ascii=False`),
including non-ASCII symbol names. Escaped ASCII serialization is not equivalent.
no fabricated missing data, no dropping failed models/seeds. Archive universe
and historical receipt latency are UNCERTIFIED; this is SIMULATION.
Previous studies saw historical TEST: label RETROSPECTIVE_REUSED_HOLDOUT.
Never claim untouched confirmation or ACTUAL_DEMO training from this archive.

Use existing global calendar-duration70/15/15 boundaries: train ends
2024-01-24 18:30UTC, tune ends2025-06-11 16:30UTC; test ends archive cutoff.
Train-only sample and normalization; fixed models, no HPO. Tune may select one
of threshold values {0,.0005,.001,.002} by shared-cash net mean (lexical/tie
larger threshold); TEST never chooses threshold/checkpoint/seed.

Resource boundary: <=2CPU threads, symbolwise arrays/memmaps, no full55m Python
row lists. <=256 deterministic time-stratified mature training examples per
symbol across maximum train range; all eligible candidate origins in evaluation.
No global interpreter/trading-env mutations. Own research Python3.11.9 sibling
runtime with exact package/wheel manifest before fitting. Direct pins:
numpy2.4.4, catboost1.2.10, torch2.9.1+cpu, stable-baselines3 2.7.1,
gymnasium1.2.3; resolve/record exact transitive versions and distributions.

## Common candidate population and information

RDE01: A fixed causal screen scans all519 past-valid histories and selects ONE
candidate per15m origin. Score = absolute last4-bar logreturn / max(realized
32-return standard deviation,.0001). Require32 primitive rows (33price rows),
strictly closed, regular consecutive bars and valid OHLCV/taker fields. Zero
volume is valid but candidate requires last quote volume>0. Tie by symbol.
No future row, label, execution availability, model prediction or TEST outcome
may influence screen. This bounded top1 stream is explicit: result is not
direction skill on every symbol at every timestamp nor general RL allocation.

RDE02: Common sequence length32, channels: log close return; log high/low;
log close/open; first difference log1p quote volume; signed taker quote flow
(2*buy/quote-1, zero-volume=>0). All used row closes<origin. Normalize from
sampled TRAIN only, clip standardized inputs to[-10,10]. CatBoost receives
flattened same32x5 sequence; GRU receives same sequence, no symbol identity.
No new BTCETH context or external source in this first algorithm comparison.
Future suffix mutation leaves issued feature/screen/prediction unchanged.

RDE03: Targets are direct close logreturns from last known close to close at
horizon1/4 bars. Direction label = return>0; report class counts/base rates,
logloss/Brier/balanced accuracy and return error. Entire horizon contiguous and
mature at fit cutoff. Purge16bars (maximum holding horizon) around training/
validation boundaries; no labels crossing fit cutoff. Historical bar receipt
availability is an explicit assumed close-based proxy, not certified.
Forecast eligibility can exclude a missing label for scoring with counts, but
the trading candidate may NOT be removed because future data are missing.
Only15m forecast points are observable; no invented minute path.

## Shared-cash simulator and control information

RDE04: Simulation starts100USDT per independent train/tune/test segment, shared
cash, one spot position, no short/leverage, max entry debit30 including fee,
choices WAIT/HOLD=0, BUY10=1, BUY30=2, EXIT=3. Existing-position BUY is HOLD,
flat EXIT is WAIT; record invalid/veto counts. No same-bar exit/re-entry.
Costs: base10bp fee +5bp adverse market slip; stress20bp+10bp. TP limit at
target; stop/market fills adverse, gap stop worse open, stop first on same-bar
TP/SL. Fixed stop.5%, TP.5%, maxhold16bars. Trailing arm uses prior-bar peak,
activation.5%, trail.5%, replacing fixed TP. Learned R3/R4 and learned-exit
mode retain hard stop/maxhold without a fixed TP/trail, so EXIT is learnable.

Already-resting protection triggered at the execution OPEN precedes a
discretionary EXIT: SL/TRAIL for open<=stop, fixed TP for open>=target.
Subsequent bar high/low cannot preempt an EXIT completed at open; unresolved
intrabar protection retains stop-first ordering. A favorable gap never improves
a resting fixed TP beyond its target.
R4-R3 isolates credit assignment under identical learned-mode exits; R4-R2
is a combined-policy comparison. Fixed risk: interval peak drawdown/loss entry kill1USDT;
exits/protection continue. Never reset seed/risk at calendar midnight.
Float64 research arithmetic with tolerance; not native matching/lot/PIT filter
certification. Minimum entry5USDT; simulated fractional qty is disclosed.

RDE05: Observation freezes before resolving selected/held execution bar. Fill
at next-open bar after the last feature bar, proxy zero publication latency
plus adverse slip; NOT a current quote or actual exchange execution. Candidate
selection cannot see that open. Missing selected/held execution invalidates
the segment objective, no substitute token, no zero imputation. Previously
observed invalid/short/gapped histories simply block warmup causally.
Mark at completed close, include terminal open inventory and disclose it.

RDE06: R3/R4 observe SAME frozen encoder16 latent, supervised p_up/returns
for15/60m, candidate/held identity via market vectors (no numeric symbol ID),
cash/100, position MTM/100, entry return, age/16, prior peak return, risk margin,
remaining episode fraction. Flat uses candidate, holding uses held asset.
Reward=(next marked equity-prior equity)/100 with fees already in balances;
sum rewards*100 equals final equity-initial equity. No clipping/bonus/shaping.
R3 gamma0 and R4 gamma1 differ only credit assignment; not a reset bandit in
the portfolio task. Finite financial episodes terminated=True, terminal MTM
included; training interruptions/gaps never become successful terminal profit.
No automatic promotion, no exchange credentials in trainer/environment.

RDE07: Train PPO/myopic with seeds[11,23,37,53,71], MLP[32,32],131072steps
each, rollout512,batch128,epochs5,lr.0003,gae_lambda.95,clip.2,entropy.01,
vf.5,maxgrad.5,targetKL.03,CPU2threads. All seeds retained. Fixed TRAIN
episode512 consecutive bars sampled deterministically from valid TRAIN ranges.
Valid TRAIN episode starts are preregistered from TRAIN only: for every screen
candidate inside the512bar episode, its possible16bar holding execution window
must be complete/valid within TRAIN. Freeze start-set hash and coverage counts;
this training-only completeness selection never screens TUNE/TEST candidates.
Any unexpected invalid TRAIN transition raises immediately, preserves the error
and cannot produce an accepted policy artifact, terminal reward or effect verdict.
No TEST data in reset selection. Report training episode population/coverage
and finite-budget limitation; never call non-improvement proof against all RL.
Use full continuous tune/test account; no episodic test cash resets.

RDE08: CatBoost classifier+regressor each horizon: seed42,iterations150,
depth5,lr.05,L2=10,bootstrapNo,has_timeTrue,threads2,no early stop/writing.
GRU hidden16,one unidirectional layer,shared16 latent,two binary+two return
heads,8epochs,batch256,AdamW.001,weight_decay.001,gradclip1,seed42.
Returns scaled by TRAIN target std; BCE plus Huber, deterministic ordering.
Freeze encoder/scaler/heads/hash before control training. Full-history screen
and evaluation must include every source symbol, counts for short histories.

RDE09: R5 entry-signal generator is frozen R2 h60 expected-return threshold
chosen on TUNE. All exits see same immutable signal stream; actual admissions
may differ with holding/cash. Fixed/trailing policies and PPO exit compare
continuous portfolios. Isolated common-entry episode diagnostics, if any, are
separate and never summed into portfolio PnL. Learned exit cannot create entries.

Fixed implementation details registered before evaluation: R1/R2 primary
portfolio uses h60. Entry requires p_up>=.5 and expected logreturn greater
than twice base (fee+market slip) plus the tune threshold; BUY30. Holding
uses held-asset h60 return, EXIT when <=0, otherwise HOLD. The four thresholds
are compared on base TUNE only, then reused unchanged for stress and TEST.
R5 uses that frozen GRU signal; learned exit transfers each jointly trained
R4 policy, mapping its held action EXIT to exit and all others to HOLD. It is
an exit-transfer diagnostic, not a separately optimized exit-only learner.
Market-vector order is latent16,p15,p60,mu15,mu60. BTC buy/hold benchmark
spends30USDT at segment first open, retains70cash and fractional BTC, charges
base/stress entry costs, then marks closes without TP/SL or forced final sale;
its unmanaged risk is disclosed, so it is not an eligible guarded strategy.
Forecast sample charts use first TEST screen origin with both labels mature,
chosen without examining predictions/errors; plotted15/60m points do not
claim an intervening minute-by-minute price trajectory.

## Reporting and verification

RDE10: Daily Budapest MTM differences, DST-aware92/96/100 bars; all complete
days including zero-trade, partial/missing days explicitly excluded with
requested/observed denominators. Total interval and complete-day sum separated.
No selling seed or forced terminal closure to beautify result. Benchmarks cash
and same-start buy/hold; no trade revenue as profit. Research managed100USDT is
not the actual full Demo account or its USDC seed revaluation.

RDE11: Preregister six contrasts: R4-R3, R4-R2, R2-R1, learned-exit-fixed,
learned-exit-trailing, R4-cash. Paired7-calendar-day block bootstrap,5000draws,
seed42,99.1667% intervals (Bonferroni family95%). Aggregate RL per-day mean
across all five seeds AND publish each seed/range; never pick best seed.
SESOI.01USDT/day per100, minimum30completepaired days. Supporting exploratory
pattern requires delta lower>SESOI, positive absolute earnings lower, stress
net>0, maxDD<=reference+.5USDT. Otherwise NOT_SUPPORTED/INCONCLUSIVE as evidence
warrants. Power/PIT/receipt/execution parity remain uncertified; never promotion.

RDE12: Freeze registration before labels/models and save source/spec/data/
split/normalizer/model/encoder/seeds/cost/signal hashes. Changed inputs cannot
resume into same run; write stage/error/incomplete state.
The canonical SHA of the prepared manifest excluding its own `data_sha256`
must be recomputed before registration or loading; trusting its stored SHA and
individually editable member hashes is insufficient.
Frozen encoder and threshold bindings are immutable on resume. A policy loaded
from an existing directory must match the requested seed, gamma and fixed
requested/actual step budget; a sealed but different checkpoint is not an
acceptable substitute. Bind each accepted policy manifest SHA immutably.
Output tables,
equity and same-origin frozen forecast-vs-actual graphs per predictive model;
PPO charts actions/equity and explicitly shared GRU forecast. Missing/failed
model means incomplete, never silent fallback or replacing PPO by rules.

Mandatory tests before code: archive tamper/monotonicity/gaps/short histories;
future-prefix invariance, label maturity/purge, train-only scaler; candidate
future availability independence; shared-cash no double-spend, fees/reward MTM,
stop/gap/priorpeak, selected missing=>null, DST/zero days; same obs R3/R4,
frozen encoder, signal hash vs admission counts, seed completeness/tamper;
real tiny CPU GRU/CatBoost/PPO train-save-load integration in research env.

Rollout: research runtime only. Do not edit frozen trend/feature modules or
restart/stop existing pilot, history, boundary or public-flow processes; never
read credentials, submit orders, change pilot seed/expiry/limits or active model.
Rollback: stop only this research job, preserve all evidence; production entry
authority remains unchanged. Full Truth Harness FAIL TH-11 stays FAIL. Complete
focused/full tests, diff check/review, selective staged gate, commit and push.
Prospective validation requires later issued forecasts and mature outcomes;
this historical run alone cannot close that future dependency.
TRAIN512bar remaining fraction versus continuous evaluation's whole-segment
fraction is a registered horizon/distribution mismatch, limiting conclusions
about RL. R5 is transferred joint-PPO exit, not dedicated exit-policy training.

## Reproduction and verification evidence

Optional numerical/ML tests are explicitly discovered under
`apps/binance_demo_bot/tests/research/` with the isolated research Python;
minimal application discovery stays dependency-free. No skip/xfail substitute
for real training. Root runtime `rl-direction-experiment/` retains original
RED evidence for core, data, model, registration, runner, bootstrap and gap
fixes; focused GREEN, full application and actual CPU integration logs.
The vectorized page reader must match scalar row validation/counts and
preserve source/page hashes; this refactor changes speed, not eligibility.

To reproduce in the frozen sibling research interpreter (PowerShell, from
repository root):

```powershell
apps/binance_demo_bot/.runtime/rl-direction-python/python.exe -u -c "import sys; from pathlib import Path; sys.path.insert(0,str(Path('apps/binance_demo_bot/src').resolve())); from binance_demo_bot.rl_direction_runner import execute_study; execute_study('apps/binance_demo_bot/.runtime/rl-direction-experiment/prepared','apps/binance_demo_bot/.runtime/rl-direction-experiment/RDE-1-MAX519-20261029')"
```

First materialize `prepared/manifest.json` through `rl_direction_data.stream_archive`
with the immutable registered cutoff. Existing source/model/array/cache hashes
must match, otherwise the run is INCOMPLETE and requires a new registered attempt.
Runtime artifacts are evidence only and must never be committed or promoted.
