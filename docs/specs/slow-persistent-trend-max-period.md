# Slow Persistent Trend: Maximum-Period Validation

Status: pre-registered research validation; no production trading effect
Owner: bot research loop
Registered: 2026-09-03, before the first result-bearing replay

## Question and terminal scope

The hypothesis is that a separate detector based on a 12-36 hour recovery and
persistence window can identify slow, multi-session rises such as the observed
TRXUSDT move earlier than the daily-reset top-gainer gate.

This experiment may end in `eligible_for_shadow`, `rejected`, or
`inconclusive`. It cannot approve a Telegram BUY/WATCH notification, portfolio
entry, score-gate relaxation, or any other production change. That would still
require replay on the bot's actual candidate population and a unified
portfolio comparison after fees/slippage (TH-06 and TH-11).

## Frozen population and timing contract

- Population: every symbol in the current configured watchlist for which
  Binance Spot provides sufficient closed 1h USDT klines.
- Maximum period: from `2017-08-17T00:00:00Z` (or the symbol's listing, if
  later) through the last fully closed 1h candle at execution time.
- Decision time: close time of candle `i`. Every feature uses data with index
  `<= i`.
- Simulated entry: open of candle `i+1`; no close-to-close same-bar entry.
- Round-trip cost: 20 basis points, split equally between entry and exit.
- Outcomes: net close return at 12h, 24h and 36h; net MFE/MAE over 24h.
- Useful label: net 24h return `>= +0.50%`, net 24h MFE `>= +1.00%`, and net
  24h MAE `> -1.00%`.
- Deduplication: emit only on a false-to-true transition and impose a 24h
  per-symbol cooldown.
- Warm-up: at least 200 closed 1h bars.

The useful label is a research proxy for a monetisable slow rise, not the bot's
top-mover objective and not realised portfolio PnL.

## Frozen broad precursor and variants

All percentages below are percentage points, not fractions. Common broad
precursor requirements are: close above EMA7, EMA7 above EMA25, positive
three-hour EMA7 slope, positive MACD histogram, RSI in `[45, 75]`, and price
above the rolling 12h low. The broad precursor is the named base-rate
population and uses the same transition/cooldown rule.

The following variants are all registered before inspecting replay results:

| Requirement | `balanced_v1` | `strict_v1` | `low_vol_persistence_v1` |
|---|---:|---:|---:|
| return from rolling 12h low | 0.35-2.50% | 0.50-2.00% | 0.30-2.00% |
| 12h close return, minimum | +0.15% | +0.25% | 0.00% |
| 24h close return, minimum | -0.50% | 0.00% | -1.00% |
| EMA7 slope over 3h, minimum | +0.06% | +0.10% | +0.04% |
| EMA25 slope over 6h, minimum | 0.00% | +0.02% | 0.00% |
| RSI14 | 52-70 | 55-68 | 50-70 |
| ADX14, minimum | 18 | 20 | 18 |
| ADX change over 3h, minimum | 0 | 0 | -1 |
| volume / 20h mean, minimum | 0.50 | 0.70 | 0.40 |
| close above EMA25, maximum | 1.50% | 1.20% | 1.20% |
| MACD histogram vs 3h ago | rising | rising | rising |
| non-negative returns in last 12h | 7 | 8 | 7 |
| ATR14 / close, maximum | none | none | 1.00% |

No threshold may be edited after a result is observed in this experiment. A
different threshold set is a new hypothesis and needs a new untouched holdout.

## Chronological selection and holdout

The global requested interval is split 60% train, 20% validation and 20%
untouched holdout by UTC time. A 48h embargo is removed on both sides of every
boundary, which exceeds the 36h maximum label horizon.

The variant is selected using validation only, among variants with at least
100 validation labels. Selection order is: useful precision, mean net 24h
return, then sample count. The holdout is evaluated once after selection. If
no variant has 100 validation labels, the result is `inconclusive`.

## Pre-registered shadow acceptance gates

Every condition must pass:

1. At least 95% of the configured watchlist has sufficient history.
2. At least 100 validation and 200 holdout labels.
3. Holdout spans at least 180 calendar days.
4. Validation and holdout mean and median net 24h returns are positive.
5. Holdout useful precision is at least 30%, at least 5 percentage points
   above its broad-precursor base rate, and lift is at least 1.25x.
6. Holdout useful precision is no more than 10 percentage points below
   validation.
7. Holdout 10th-percentile net 24h return is at least -2.00%.
8. Holdout alert pressure is at most 5 signals per calendar day.
9. As a non-labelled incident diagnostic, the selected variant must first
   identify TRXUSDT no later than `2026-09-02T10:00:00Z` (12:00 Europe/Budapest)
   inside the registered incident window beginning `2026-09-02T04:00:00Z`.

Failure with adequate coverage/sample is `rejected`; missing coverage/sample
is `inconclusive`. Passing is only `eligible_for_shadow`.

## Evidence and safety

The audit artifact must record symbol coverage and errors, exact actual
period, split boundaries, denominators and numerators, base rates and lift,
cost assumptions, selected-variant rule, specification SHA-256, incident
timestamps, failed checks, and all three variant results. Runtime caches and
reports stay under `.runtime/` and are never committed.

Rollback is deletion/disablement of the research job; the experiment has no
runtime import, configuration flag, Telegram path, order path, or portfolio
side effect. Any future shadow implementation needs an explicit default-off
switch, telemetry, rate guardrail, and expiry/review date. Production remains
unchanged regardless of this audit outcome.
