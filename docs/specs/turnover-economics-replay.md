# Entry economics and replacement turnover: maximum-period replay

Registered: 2026-10-05; completed 2026-10-06. All three hypotheses REJECTED.
Registered before policy outcomes. Offline only; production unchanged.
Parent: production-capacity-catboost.md; SCOUT_OPTIMIZATION_SPEC.md;
continuous-improvement-control-plane.md. TH-01..TH-12 apply.
Full Truth Harness at start: PASS; a mechanical PASS is not trading approval.

## Question and fixed hypotheses

Does less economically viable entry/replacement turnover explain losses, and
can it be reduced without losing early daily watchlist leaders?
Observe additive cash-account PnL, fees/slippage and holding time by entry mode,
exit reason, timeframe and chronological cohort before considering deployment.
Do not sum per-trade percentages or equate fewer trades/lower exposure with alpha.
Earlier capacity ranker remains REJECTED; no retuning of its exposed TEST.

Fixed arms (no threshold selection after viewing results):

- control: current rule-only candidate/replay path, existing ten-slot capacity,
  score floor, cluster gates, cooldown and all original SELL/protection rules;
- amplitude_cost: retain a candidate only when the observed last four CLOSED
  15m candles' high/low range is >= twice the exact roundtrip cost hurdle.
  At 7.5bp fee + 5bp slippage per side this is approximately .50%, calculated
  from multiplicative fills/fees. Same cutoff for 15m and 1h candidates.
  This is an observed amplitude screen, not a forecast of upside or direction.
- no_replacement: identical candidate stream, prohibit portfolio replacement;
- combined: both changes above. No additional exit hold/cooldown relaxation.

The supported replay profile is `score_replace_cluster` with unbound learned
scores and mutable temporal logs disabled. It is not the previously recovered
30-day `replacement_block_non_losing` profile or certified agent-only/live
champion. Do not compare their different-window/profile numbers as an uplift.

The amplitude gate requires the exact four-close past grid at the origin,
finite positive valid OHLC. Missing/stale history fails the candidate closed
with explicit counts; no interpolation or future backfill. Apply the filter
to ALL rule candidates, not only to baseline-selected trades. Simulate every
arm afresh with unchanged exits, occupancy and cooldown interactions.

## Maximum inputs and reproducibility

Use the longest complete receipt-bound recovered market in the existing
price/volatility experiment: Apr 1 through Oct 4 06:00 UTC (about 186 days after
10-day warmup), 93 of 105 requested symbols. Verify manifest/raw/feature hashes
and complete grids. Existing config/indicator/replay feature sources and ALL
candidate-rule kernel digests must match before native checkpoint reuse.
The old comparison helper changed only subsequent analysis/accounting; reuse
requires current generator/strategy/config equivalence and independent fixed
BTC 15m/ETH 1h spot regeneration, never blindly accepting changed rule sources.
Reject any source/input drift at publication. Freeze run registration and sources
before simulations. Runtime data/weights/reports remain outside Git.
If a newer fully recovered market exists, take its entire period; do not shorten
the maximum window for a favorable result. Partial current-cache tails alone
cannot replace the fully covered universe. Disclose historical universe/model,
agent-only policy, receive-time and fill parity limitations.

## Measurement

An independent event ledger uses one cash account, ten distinct symbols,
min(cash, liquidation equity/10) entry sizing and positive entry/exit costs.
Old exits precede admissions; own same-clock boundary exit follows its entry.
Handle partial exits as fractions of remaining quantity. Attribute entry budget,
raw-price PnL, fee, slippage and net PnL per trade. Sum of net PnL equals ending
cash minus initial cash, and net = raw-price PnL - fees - slippage. Cross-check
all ledger cash/fee/slippage totals and EVERY 15m equity point against the
canonical _simulate_account. Attribution is for after-cost allocated quantities;
it is not the separately compounded no-cost account. Do not call simulated
execution costs observed exchange fills.

Report canonical net return/BTC alpha, complete 15m drawdown, exposure, trade
count, replacements, turnover/cost attribution, daily Top-15 unique captured and
early pairs, precision with counts, MFE retention/giveback, missing populations.
Recompute 2x-cost sensitivity with unchanged trades and parameters; it is an
accounting stress, not a latency-aware execution simulation.

Frozen local-day boundaries at 60/80% of the full period label discovery,
validation and retrospective TEST. Policies do not fit/train; report all three
cohorts and full window. TEST portfolio return rebases each continuous account
at the same exact boundary (carried holdings disclosed). Exclude boundary days
from daily labels. Paired 3-calendar-day moving-block bootstrap of TEST daily
log-return differences, 5000 draws, seed 42, Bonferroni 3 comparisons (98.333%
individual intervals); no choosing block length or favorable date slice.

## Decision gate

At least 30 complete TEST days; all numerical cash/coverage checks pass.
Versus control: full alpha gain >=1pp, TEST return gain >=1pp, corrected paired
lower confidence bound >0, at least 10% lower turnover (trade count), no fewer
unique early or captured Top-15 pairs on full and TEST windows, precision no
worse by >.5pp and drawdown no worse by >1pp. Show exposure to expose inactivity.
Passing numerically permits only a new forward-shadow design; absent certified
full PIT population/live/execution parity, runtime_eligible remains false.
Reject adequate-data failed hypotheses, retain all arms and negative outcomes.
Incomplete evidence is INCONCLUSIVE, never an achievement or automatic promotion.

## Tests and engineering cycle

Future-mutation/prefix equivalence of amplitude filter, exact gap/invalid OHLC,
fee hurdle arithmetic, raw candidate preservation, disabled replacement option,
additive cash reconciliation, partial/boundary exits, initial-capital drawdown,
full grid and canonical equity equality, mission/return gate rejection and
bootstrap independence. Relevant replay/accounting regressions, full/staged
Truth Harness, diff review, commit and push source/spec/tests only.
Independent verification additionally reconstructs the amplitude decision on
every raw candidate from raw JSON OHLC, verifies all screened-arm entries are
eligible, checks every cash-attribution row and grouped sum, every equity clock,
recomputed TEST returns and initial-capital drawdown. Separate output/source
receipts bind scorecards, plots and unweighted exit diagnostics. Unknown exit
labels do not become zero-quality measurements.

## Rollback and production boundary

Stop offline runner; no active BUY/SELL/config/model/positions/orders changed.
Any later adoption requires certified maximum-period paired policy validation,
fresh shadow and paper/canary evidence, risk limits and immediate rollback flag.

## Completion evidence (2026-10-06)

Run `.runtime/turnover_economics_20261005_v1`: 186.333 evaluation days,
93/105 complete symbols, 303,532 original rule candidates, 276,500 amplitude-
eligible candidates, zero unknown past histories. Serial regeneration matches
2,690 BTC 15m and 692 ETH 1h candidates, including every decision field.
Every account has 17,889 fresh valuation points, positive equity and no
cash/slot violations or borrowing; each point and fee/slippage total matches canonical cash
accounting. The model-free policies are retrospective, not a sealed holdout.

| Arm | Full net % | TEST net % | Trades | Replacements | Early / 2,790 | Captured / 2,790 |
|---|---:|---:|---:|---:|---:|---:|
| control | -96.256673 | -44.809011 | 11,010 | 366 | 1,039 | 2,219 |
| amplitude_cost | -95.718912 | -42.777916 | 10,982 | 369 | 1,037 | 2,214 |
| no_replacement | -94.857080 | -46.114075 | 9,989 | 0 | 951 | 2,115 |
| combined | -94.872566 | -47.187727 | 9,951 | 0 | 946 | 2,110 |

38 complete TEST days and 570 TEST leader labels. TEST early/captured pairs:
control 253/480; amplitude 252/477; no replacement 244/460; combined 245/460.
Corrected 98.333% paired daily log-return intervals in bp: amplitude
[-16.972853,+46.179021], no replacement [-77.061872,+81.784408], combined
[-82.593171,+75.012184]. None has a positive corrected lower bound.
Trade reductions are 0.254%, 9.273% and 9.619%, below registered 10% effect.
Every candidate loses early/total capture on full and TEST populations. No
replacement also loses TEST return despite a smaller full-period loss.
All three numerical gates: REJECTED; no parameter/date/threshold retuning.

Control initial cash 10,000 USDT; after-cost net loss 9,625.667342 USDT,
raw-price PnL of actually allocated quantities -1,309.079945 USDT, fees
4,989.952634 USDT and slippage 3,326.634762 USDT. Cash identities reconcile.
This is additive allocated-quantity attribution, not the separately compounded
no-cost counterfactual. Amplitude screening reduces trades by only 28 and
INCREASES simulated fee/slippage spend to 8,738.383878 USDT as earlier outcomes
alter available equity and later allocations; trade count alone is not cost.
No replacement cost is 7,963.969785 USDT but raw-price loss worsens to
1,521.738245 USDT. Cheap turnover restriction cannot substitute for entry edge.

Largest control loss group: impulse_speed, 4,615 trades, raw-price PnL
-1,113.360540 USDT and net PnL -4,596.775545 USDT; this is future hypothesis
prioritization only, not permission to remove a mode after inspecting TEST.
All mode-level net contributions are negative. Full equity graph uses common
period/axes and an explicitly retrospective TEST band. Exit MFE/giveback
statistics remain unweighted gross-price diagnostics, separate from net cash.

Independent verification PASS: all 303,532 amplitude decisions recomputed from
raw JSON, all screened-arm admissions eligible; all four trade ledgers/grouped
cash totals and equity/TEST/drawdown/benchmark metrics checked. Result SHA256
`24dbd5b4a19d24a62ad39e8e968eaf2fb62191b67fb745a8118b4187dd452004`.
65 focused/relevant tests PASS; full Truth Harness PASS. Sources, tests and spec
only are committed; reports, native checkpoints and prices stay in runtime.
The own worker was stopped only after publication/independent verification
because it retained resources during interpreter teardown; cleanup receipt is
saved, no computation/model result is presented as interrupted or promoted.
Next research needs a separately registered entry/exit edge hypothesis, complete
live data/receipt parity and fresh forward evidence; these rejected generic
filters are not silently reintroduced with tuned thresholds.
