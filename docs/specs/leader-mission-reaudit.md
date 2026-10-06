# Leader mission re-audit

Registered 2026-10-06 following explicit user priority correction. Research-only.
TH-01..12. Objective: early discovery, daily leader coverage, BUY selection and
leader accompaniment until observable weakening of an upward trend. Account
returns/fees are not the winner criterion in this re-audit. No live change.

## Scope and sequence

Re-evaluate frozen tested hypotheses in original order: (1) capacity CatBoost,
(2) amplitude/turnover filters, (3) impulse CatBoost, (4) joint direction/amplitude,
(5) protected soft-SELL model, (6) order-flow execution. No retraining, new
threshold search or hindsight alteration of decisions. Maximum186-day available
93/105-symbol archive for full-policy arms; capacity remains181-day selection
proxy, not a fabricated portfolio replay. Previously untested sizing and standalone
price/volatility forecasts are not represented as verified admission policies.

Preserve all previous economic verdicts and artifacts. New mission-only results
use a separately registered evaluator, exposed retrospective TEST, and never
constitute sealed confirmation or production approval.

## Correct time/label contract

Raw candle timestamps are OPENS. Use candle close = t+900000 for every metric.
Daily labels: all exact15m bars from local00:00 open through22:00 CLOSE,
Top15 by final close/day first open, descending return then symbol ties to
match canonical leader ranking. Complete days and common available population
only; last partial day excluded. All ratios publish numerator/denominator.
First BUY per local-day/symbol through22:00 defines unique capture; every
eligible BUY defines trade precision. Also disclose unique-pair precision.
Early = remaining positive daily move fraction >=.35; fraction null if day move
nonpositive, clipped [0,1.5] for continuity. Recompute from RAW15m day prices and
trade decision entry_ts/price, never trust stored future annotations. Stored
_entry_day_metrics uses originating timeframe candle OPEN as entry clock and
OPEN-asof final index; it is misaligned with canonical22:00-close daily labels.
Record stored versus corrected early counts. This is an evaluator alignment
defect, not proof of training leakage; frozen score_replace_cluster variant does
not activate the shadow exit discriminator which can read the annotation.

Full available days plus the same existing TEST boundary1787781600000. Earlier
model cohorts were prequential; TEST is exposed retrospective. Direction/EV
models are unchanged. Mean/median remaining fraction and minutes to22:00 are
descriptive detection diagnostics, not independently optimized winner metrics.

## Leader accompaniment diagnostic (fixed before new exit results)

Evaluate each captured leader day/symbol's FIRST position. Follow exact15m
closes for24hours after entry, regardless of actual exit; tail with incomplete
follow-up remains unknown. Compute EMA20 recursively from archive start and
ATR14 as rolling mean true range from closed candles (explicit evaluator
definition, not inherited strategy ATR). No future normalization.

Operational trend-end marker: after running CLOSE peak rises >=1 entry-time ATR
above entry, require TWO consecutive closed bars with close at least2 current
ATR below that running peak AND declining EMA20. First second confirmation is
the detectable weakening clock. This is an operational definition, not knowledge
of the ultimate market top. Labels are evaluation-only, never model features.
Unconfirmed or never-established episodes are separate from incomplete data.
Hard risk exits can be appropriate before this marker; do not suppress them.

Report exit before/at-after confirmed marker; delay from marker in15m bars;
gross first-position retained movement relative to peak through marker,
including partial sale weighting; giveback in price-return pp (not cash PnL).
Retained movement is not capped, negative values remain visible. Denominator is
positive confirmed upswing; unknown or non-upward episodes are not zeros.
Report first exit's <=4hour later NEW CLOSE HIGH above pre-exit running peak,
requiring all exact bars, and remaining position fraction at marker; these
distinguish premature first-leg exits from full position liquidation. A later
new high does not retrospectively change the causal marker. Terminal archive
forced exits are separately counted; no positive exit-quality credit.

Use common captured leader pairs across each candidate/control to compare exits
without survivor-selection artifacts. Also publish all-captured descriptive
means with denominators. Capacity conflict options and execution quote tasks
do not become leader positions or comparable portfolio outcomes.

## Additional accompaniment completeness

After viewing preliminary first-position diagnostics, add a descriptive measure
of ALL re-entries in the same symbol between first BUY and the unchanged causal
marker: held-time fraction weighted by remaining partial-position fraction, and
whether any position remains at the marker. This supplements first-position
metrics; no model or acceptance threshold is changed. This addition is openly
post-inspection exploratory measurement, not a pre-holdout hypothesis. Include
overnight re-entries within the24h episode. Validate no overlapping symbol
intervals and never call first-position closure total abandonment of the leader.
The initial v1 evaluator failed in bootstrap array indexing before publishing
a result; preserve its partial outputs and rerun corrected v2, with a focused
bootstrap regression test. Trading decisions/models remain byte-identical.

## Mission verdict

Independent v2 scalar audit found ATR threshold roundoff at an exact WIF price
boundary. V2 exit diagnostics are NOT verified. V3 uses local-window ATR mean
instead of subtracting large cumulative sums and explicit1e-12*entry-price
tolerance at price thresholds/EMA-decline comparisons. This is numerical boundary
handling far below quoted ticks, not a changed2ATR/1ATR/two-bar rule. Preserve
v2 and failed verifier logs. No changes to decisions, models, entry labels or
cohorts; rerun complete evaluator and scalar audit. No exit claim from v1/v2.

BUY improvement: TEST early and captured counts not lower, trade precision not
lower, at least one strictly higher. SELL improvement: detection/coverage/
selection retained and common-pair retained movement improves, with no higher
first-exit-before-marker rate. Exact zero-tolerance noninferiority is a reporting
rule, not a calibrated economic utility. Separate mixed trade-offs and no effect.
Three-calendar-day paired block diagnostics,5000draws,seed42, familywise95%
across6full-policy alternative arms x3entry metrics, plus6exit comparisons.
Do not declare robust mission benefit if relevant lower interval includes zero
or fewer30complete TEST days. New causal marker remains exploratory and exposed.
No production adoption even from a favorable retrospective mission scorecard:
requires fresh actual decision/leader/exit shadow with unchanged rule and PIT
universe coverage. Missing receive-time/live parity remains explicit.

## Evidence and engineering

Verify pinned market/parent receipts, source snapshots and raw hashes. Rebuild
all daily leader labels, all first-BUY/precision records and per-leader episode
records, compare corrected results against old aligned coverage/precision counts.
Independent verifier reconstructs labels/counts without trusting frozen daily
annotations, and scalar-checks predetermined trend episodes and summary stats.
Tests:22:00-close boundary, midnight/1h annotation discrepancy, DST, future
prefix invariance, gaps/tail, partial exits, no-trend vs unknown, pair alignment,
denominator and zero-effect verdict. Publish common-period mission table and
graphs, preserved cohort receipts, old-versus-corrected early values. No cost or
net-return gate in the new report. Rollback: remove offline re-audit scripts.
Complete focused tests, diff review, full/staged Harness, commit/push only intended
source/spec/tests. Runtime reports/prices/weights remain uncommitted.

## Completed re-score, 2026-10-06

V3 result SHA2564a06e22818feffe7c9759ad95ec5a83bb84228a3b61bea2f6b19c0ec7119465c,
runtime `.runtime/leader_mission_reaudit_20261006_v3/`. Fixed decisions, raw93
complete symbols,186full days/2790leaders and38TEST days/570leaders. Entry and
SELL winner rules contain no account-return or fee requirement.

| Fixed arm | Corrected TEST early /570 | Captured /570 | Leader BUY / eligible BUY |
|---|---|---|---|
| control |247|480|1250/2470|
| amplitude |246|477|1254/2491|
| no replacement |240|460|1112/2309|
| combined |241|460|1115/2307|
| impulse CatBoost |217|437|1041/2336|
| joint direction/amplitude |218|453|1101/2392|
| soft SELL model |245|478|1248/2475|

All6fixed full-policy alternatives: MISSION_TRADEOFF_OR_WORSE. CatBoost entry
and joint early/coverage/precision matched corrected intervals lie belowzero.
Capacity:10070/10778 mission-known TEST conflicts, model leader3315vskeep3323,
early1174vs1175; NO_PROXY_LEADER_GAIN. Execution joins5/22020 actions and
changes none; this is not a mission policy improvement or full-period proof.

Original TEST early253/570 is superseded by corrected247/570; full1039 by1031.
32TEST and158full first-leader early classifications differ (net differences
partly cancel). Old metric bug remains documented, original bundles preserved;
score_replace_cluster trading decisions did not read these future annotations.
Corrected re-audit does not silently repair old runtime reports or enable the
shadow discriminator that can read them.

Control TEST:430confirmed episodes from480captured leader pairs;407first exits
before marker,317/470 later new-high outcomes within4h. Mean first-position
retention.079749 on430; including re-entries held-time.533776, any position at
marker104/430. These are operational accompaniment diagnostics, not proof that
every early exit is wrong: hard risk exits and repeated trend episodes matter.
Soft SELL:427confirmed,402beforemarker,meanretention.084468,held-time.545562,
any position106/427. Common same-entry confirmed388pairs:meanretention delta
.000810865 (ratio units), corrected CI[-.000009566637,.002322105125] includeszero;
first-before-marker count delta0. Different survivors cannot prove improvement.

Re-audit is completed, retrospective and not production promotion. Follow-up
priority is causal premature-exit and re-entry analysis on actual captured
leaders; a separately registered leader-target model is not tested by these
unchanged return-target models. No waiver, policy relaxation or live change.
