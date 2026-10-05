# Production tooling: causal capacity CatBoost pre-gate

Date: 2026-10-05
Status: baseline recovered; mechanical full harness PASS; maximum-archive ranker REJECTED

## Objective and sequence

Follow SCOUT_OPTIMIZATION_SPEC.md and the action-layer contracts in
continuous-improvement-control-plane.md. Restore the stale canonical 30-day
baseline by executing current replay, never editing old provenance hashes.
Then evaluate a new capacity discriminator on the maximum recovered archive,
including all available complete histories and disclosing missing symbols.
Earlier EV and structural rankers remain rejected; this is a new causal
candidate-versus-incumbent dataset, not a re-evaluation of their exposed test.

## Data and timing

Observe the existing replay capacity recorder without changing its decisions.
An existing maximum-period trace may replace recomputation only if its receipt,
every source hash, market manifest/input hash and full period match. Freeze its
registration, model descriptors, result and trade bundle in the new run. This
reuses a verified historical champion, not mutable current live model files.
Record this branch before training; preserve any aborted untrained registration.
Each group contains the rejected candidate and the lowest allocation-ranked
incumbent at that decision clock. Features use only closed 15-minute candles:
1/4/16-bar returns, trailing 16-bar return volatility, trailing volume ratio,
trailing range, position in trailing range, and candidate role. No future day
rank, trade outcome, exit fields, future volatility, mutable learned score, or
symbol identity is a feature. Require the exact past grid with finite positive
OHLC and nonnegative volume. No stale mark or future backfill.

Target: next five 15-minute closes (75 minutes) simple return. The candidate's
target subtracts 0.25 percentage points, representing one incumbent SELL plus
one candidate BUY at 7.5bps fee + 5bps slippage per side. The incumbent has zero
incremental turnover cost. This proxy is NOT full cash-account alpha. Require
all five exact future closes for both symbols; unfinished labels are counted.
Same-day Top-15 and remaining-move capture are evaluator-only diagnostics;
label availability includes the 22:00 Europe/Budapest daily cutoff. Exclude
incomplete daily objective populations from these mission diagnostics.

## Frozen experiment

Split entire local days chronologically by the maximum archive time span:
first 60% TRAIN, next 20% VAL, final 20% TEST. A row belongs to an earlier
cohort only if BOTH its return label and daily objective label mature strictly
before the next cohort starts. All repeated timestamp/pair groups stay together.
This historical archive has already been exposed in other investigations;
TEST is explicitly retrospective, not a sealed fresh confirmation.

CatBoostRanker: YetiRank, max 400 trees, depth 4, learning rate .03, L2 10,
seed 42, two threads, validation early stopping 40. Two candidates per group;
relevance is the relative return winner (ties both zero). Fixed default action:
replace iff candidate score > incumbent score; no TEST threshold tuning.
Controls: always keep incumbent and always replace. Neither is a forecast curve.
Validate native reload on fixed rows. Store sources, data/model hashes,
registration before training, frozen predictions and independently checkable
counts. No runtime/model artifacts are committed.

## Pre-gate and downstream boundary

Report group counts, independent complete day counts, candidate-win base rate,
action frequency, paired mean/median incremental return, direction decision
accuracy and same-day leader/capture changes. A 3-day moving block bootstrap,
5000 draws, seed 42 gives a 95% interval of paired daily return uplift versus
keep; preserve unequal event-count weighting by resampling daily sums/counts.
At least 100 mature TEST conflicts and 20 complete TEST days are required.
Advance only if mean incremental return >= .05pp, lower CI > 0, no lower
selected leader rate or selected leader remaining-move capture, and all
coverage/timing checks hold. Missing mission labels or partial historical
universe cannot pass a production gate. A failed adequate-data experiment is
REJECTED, insufficient evidence is INCONCLUSIVE, never an achievement.

A passed proxy pre-gate permits only a separately registered paired full-policy
ten-slot replay with unchanged entry/SELL/cluster/cooldown rules, complete
coverage, costs, named benchmark and non-inferiority guardrails. Before actual
promotion require new forward shadow, paper execution, canary, and immediate
rollback. MLflow/feature-store/execution-engine migrations come after proving
a concrete bottleneck and are not silently deployed by this offline study.

## Truth and verification

Apply TH-01..TH-12: explicit counts/provenance; proxy versus realized separation;
causal features and mature split boundaries; common cohorts/maximum archive;
missing data disclosed; rejected hypotheses retained; continuous after-cost
portfolio baseline; source/spec/test-only staged scope.
Tests cover future mutation invariance, exact grid/gaps, immature forward labels,
switching costs, purged daily split, finite inputs, ties and uncertainty counts.
An independent verifier reloads every TEST option, recomputes decision/mission
counts and return means, verifies chronological maturity and compares 32 fixed
TEST origins (64 options) against raw JSON prefixes and forward closes. Its
PASS is numerical verification, not certification of live population or fills.
Run focused regression tests, full Truth Harness and staged change harness.

## Reproduction

Use the isolated research dependencies before `files` on Python's module path;
embedded Python ignores PYTHONPATH. Keep OpenBLAS/OMP/MKL at one thread.
Run `files/run_capacity_catboost.py` with `--archive
.runtime/closed_grid_policy_replay/20261001_max_archive_v1`, `--trace-bundle
.runtime/paired_full_policy/20261004_live_hold_v1` and a NEW `--output` directory.
Then run `files/verify_capacity_catboost.py --run <output> --archive <archive>`.
The reused trace's source hashes must still match; future code drift requires
a new full maximum-period trace, not patching receipts.

The separate canonical baseline command is `files/replay_backtest.py --days 30
--end-at 2026-08-25T22:15:00Z --market-data-mode local-only --market-cache-dir
.runtime/signal_quality_cache --max-open-positions 10 --variant
replacement_block_non_losing --top-gainer-score-min 34 --objective-top-n 15
--no-baseline --portfolio-alpha-output <new runtime report> --json`.
This restores the established evaluator baseline, not historical live-fill parity.

Cache repair uses `files/prepare_capacity_baseline_cache.py`: complete series
come from SHA-bound recovery inputs, unavailable archive identities fall back
to existing local data with all source hashes and explicit missing counts.
No missing candle is imputed. Use a new cache directory and rerun the whole
baseline, because repaired BTC context can change decisions, not just valuation.

## Maximum-archive result (2026-10-05)

Run `.runtime/capacity_catboost_20261005_v2` reused the receipt-verified
`20261004_live_hold_v1` champion trace. Period: 2026-03-31 22:00 UTC through
2026-09-28 22:00 UTC (181 days after 10-day warmup). 93/105 complete symbols;
33,133 recorded conflicts, 1,754 repeated timestamp/pairs deduplicated,
31,379 usable pairs. TRAIN/VAL/TEST: 16,803 / 3,798 / 10,778; no future-grid
failure or cross-split immature label was observed. Daily mission labels are
unknown for 2,218 pairs, including 708 TEST pairs; no zero/success imputation.

CatBoost selected 16 trees on VAL. On 10,778 TEST conflicts over 33 observed
days it selected 27 replacements. Candidate is better after extra switching
cost in 4,150/10,778 pairs; model decision correct in 6,615/10,778, whereas
always keep is correct in 6,628/10,778. Mean 75-minute selection proxy return:
keep +0.178409%, model +0.170089%, always replace -0.195849%. Paired model-minus-
keep mean -0.008320pp; 3-observed-day block 95% CI [-0.030642,+0.000063]pp.
Observed conflict days are bootstrap units; this is not 33 independent trades.

On 10,070 mission-known TEST groups, selected daily leaders are 3,315 for model
versus 3,323 for keep; qualified early leader selections 1,174 versus 1,175.
Mean capture among selected leaders .329158 versus .328875. These are repeated
conflict selection diagnostics, NOT unique canonical early-capture pairs.
Mean uplift, positive lower CI and leader non-inferiority gates fail.
Verdict REJECTED; runtime_eligible=false; no separate promotion replay authorized
by this result. No threshold or tree count was retuned after viewing TEST.

Independent verification: all 21,556 TEST option predictions reproduced from
native weights; 31,379 label timing records checked; 64 options at 32 fixed
origins checked against raw JSON; selection returns and mission counts agree.

Initial current-source 30-day baseline failed with 78 missing BTC closes
(2,803/2,881 observed). TH-11 remains FAIL at this checkpoint. Cache recovery
produced 186/210 complete timeframe series (93/105 symbols), with explicit
partial/unavailable histories for the other 12; BTC grid repaired from actual
archived candles. Preserve both initial and repaired replay outputs. No forecast
model or simulated return is presented as achieved live improvement.

Final baseline recovery completed: `.runtime/reports/
canonical_portfolio_alpha_capacity_repaired_20261005.json`, profile
`replacement_block_non_losing`, 2026-07-26 22:15 through 2026-08-25 22:15 UTC.
2,881/2,881 benchmark and holding valuation clocks, zero contract violations;
1,810 trades, after-cost net -41.396963%, BTC +20.746843%, alpha -62.143806pp,
closed-grid drawdown 47.678618%. Canonical objective on 29 complete available-
population days: captured 360/435, early 152/435, precision 840/1,683.
This is the named replay profile, not certified current live champion parity;
availability-selected historical symbols, receipt-time/fill and agent parity
limitations remain. Better coverage is a measurement repair, not strategy uplift.
Final mechanical `truth_harness.py full`: PASS, 0 blocking, 0 warnings.
The initial FAIL artifact is retained; no provenance or result was patched.
The repaired frozen cache is `.runtime/capacity_baseline_cache_20261005_v2`.
Use it instead of the gapped `.runtime/signal_quality_cache` reproduction input.
Neither the full profile PASS nor rejection of one ranker approves production
promotion or an infrastructure migration. Follow-up hypotheses require their
own frozen registration and population-appropriate maximum-period proof.

## Cost attribution

Read-only scorecard/TCA: `files/report_capacity_experiment.py` writes a new
immutable runtime JSON/Markdown scorecard with input/source hashes. On the
maximum-archive control, 10,617/10,617 exits are known, no partial exit needs
simple-hurdle exclusion, median holding time is 90 minutes. Gross positive
movement: 4,501/10,617; gross move above .25% cost hurdle: 3,903/10,617;
above doubled .50% hurdle: 3,210/10,617. These unweighted simulated trade
counts are diagnostic; no claim of actual exchange execution or portfolio
profitability is derived from them. The read-only scorecard leaves orders and
policy untouched. Focused plus relevant regression suite: 52 tests PASS.

## Rollback

Stop the isolated runner. It never writes live config, model paths, positions,
orders, BUY/SELL gates, or a promotion ticket. Outputs go to a new immutable
runtime directory; previously published evidence is preserved.
