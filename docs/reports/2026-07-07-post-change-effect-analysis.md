# Post-change effect analysis - 2026-07-07

## Scope and confidence

This review covers final daily reports for 2026-07-01 through 2026-07-06 and
live telemetry through 2026-07-07. Confidence is mixed:

- Top-20 measurement has six complete daily observations but only 19 watchlist
  top-mover cases.
- Signal-quality coverage is partial (184/206 series on 2026-07-06).
- Entry events before 2026-07-06 contain restart/day-boundary duplicates, so raw
  BUY counts from 2026-07-01 through 2026-07-05 are not suitable for causal
  comparisons.
- Multi-day retention and per-bar alert deduplication have only one overnight
  observation.

## Executive result

The latest changes materially improved observability and data integrity, but
they have not yet produced a statistically supported improvement in trading
quality. Production BUY and SELL rules should not be relaxed from this sample.

## Top-20 capture

Across 2026-07-01 through 2026-07-06:

- watchlist Top-20 denominator: 19
- bought: 14/19 (73.68%)
- early captured: 12/19 (63.16%)
- blocked winners: 5

Daily capture was volatile: 33%, 100%, 100%, 100%, 50%, 100%. The latest day
was strong (4/4 bought and early), but denominator-weighted rolling metrics are
worse than the previous window:

- early capture: 60.0% vs 94.12%
- total capture: 75.0% vs 100.0%
- trend miss rate: 82.11% over the last seven daily reports

Decision: keep Top-20 measurement, but do not treat the 2026-07-06 result as a
policy win. Continue targeted blocked-winner replay.

## Entry and exit quality

The six signal-quality reports contain 1,330 evaluated BUY events and 387 closed
trades, but early reports include duplicated entry telemetry. The cleanest recent
day, 2026-07-06, still shows weak monetization:

- median entry capture ratio: 0.2981
- median closed-trade PnL: -0.745%
- median exit efficiency: -0.5484
- median giveback: 1.594%
- early exits: 9; late exits: 30
- false-positive rate: 23.08%
- trend miss rate: 85.94%

Across all six reports, false positives are 345/1,330 (25.94%), but this pooled
rate is diagnostic only because of duplicated pre-fix entries and partial candle
coverage.

Decision: continue exit-quality and entry-admission research. Do not deploy a
broader BUY gate or new SELL rule without maximum-period replay.

## Data integrity changes

Blocked-learning deduplication is confirmed effective. Duplicate ratio per
candle fell from 11.44x on 2026-07-01 and 2.16x on 2026-07-02 to exactly 1.00x
from 2026-07-03 onward.

Entry duplication also normalized: raw/unique-candle entries were 510/106 on
2026-07-03 and 279/55 on 2026-07-04, but 62/62 on 2026-07-06. Historical daily
reports containing the duplicated events must not be used as clean training or
causal evaluation samples without deduplication.

Decision: mark pre-2026-07-06 entry-event metrics as contaminated and rebuild
future training/evaluation datasets from candle-deduplicated events.

## Observable tail shadow

The live shadow selector has 17 candidates and 43 completed horizon labels:

| Horizon | n | Avg partial delta | Median | Positive |
| --- | ---: | ---: | ---: | ---: |
| T+2 | 15 | +0.1545% | +0.1285% | 53.3% |
| T+5 | 14 | +0.1476% | +0.3301% | 78.6% |
| T+10 | 14 | +0.5628% | +0.5699% | 85.7% |

This is promising directional evidence for retaining a partial tail, especially
at T+10, but n=14 is too small for production. The stale canonical replay shown
in the morning report must not override these fresh live labels.

Decision: continue shadow collection and run a maximum-period replay after the
live sample is large enough; keep production SELL unchanged.

## Multi-day position retention

After the 2026-07-06 restart, eight positions were restored before the local day
boundary. No `stale intraday position` cleanup occurred after midnight. Positions
were subsequently closed only by normal ATR, EMA, weakening-trend, or time-exit
rules. This confirms the calendar-day deletion bug is removed.

There is not yet evidence that the continuation rule can retain a profitable
trend for several days. Current positions are all recent, and the longest
observed post-change exit in this slice was measured in hours, not days.

Decision: operational fix accepted; trading effect remains unproven. Measure
multi-day MFE retention, realized PnL, and giveback over at least seven full days
and validate the continuation policy on maximum available history.

## Strong blocked-signal alerts

Since the final alert changes, Telegram successfully delivered:

- 33 production BUY messages
- 27 strong score-gate alerts
- 6 cluster-cap alerts
- 34 SELL messages

Per-bar deduplication allows a new alert on a new candle while suppressing
minute-by-minute repeats on the same candle. Repeated alerts were concentrated in
symbols that remained strong across bars (for example POL, TIA, ETH, and ORDI).

Decision: alert visibility is improved. Outcome labels are still required before
changing production admission based on these alerts.

## Ranker training

The ranker is fresh with 15,853 rows, but it is not ready as a hard admission
gate:

- test top-1 target return: 0.0908% vs baseline 0.1587% (delta -0.0679%)
- test top-1 win rate delta: -0.28 percentage points
- teacher top-gainer-rate delta: -9.10 percentage points
- teacher capture-ratio delta: -5.07 percentage points
- top-3 return improves by +0.0273% and drawdown improves, showing ranking value
  may exist at broader selection depth

Decision: keep the ranker as shadow/bonus context, not a hard blocker. Investigate
top-3 portfolio ranking and target calibration on a maximum-period replay.

## Reporting and delivery

The 2026-07-07 morning report was generated and delivered successfully
(`last_telegram_sent_count=1`, no error). Exit summaries now use closed trades,
and rolling Top-N rates are denominator-weighted.

Research sections in the report remain stale (tail selector, entry admission,
blocker reward, and portfolio replacement). Their displayed decisions are not
fresh evidence.

## Priority actions

1. Rebuild contaminated pre-fix entry telemetry with candle-level deduplication.
2. Run maximum-period exit-tail replay using the fresh T+10 hypothesis.
3. Run maximum-period blocked-winner replay for score gate and cluster-cap alerts,
   preserving production thresholds until positive reward is demonstrated.
4. Collect at least seven full days of multi-day retention outcomes, then replay
   the continuation policy against baseline time exits.
5. Evaluate ranker top-3 selection and recalibrate targets; do not enable hard
   ranker blocking.
6. Repair missing candle coverage for the two affected TRADING series before
   using daily miss-rate changes as a production gate.
