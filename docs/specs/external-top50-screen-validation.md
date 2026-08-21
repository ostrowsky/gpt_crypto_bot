# External Binance Top-50 Screen Validation

Date: 2026-08-21
Status: maximum-period validation complete; `screen_v1` rejected; production unchanged

## Purpose

Validate the external Binance Spot screen used to rank watchlist symbols that
are outside the rolling-24h market Top-50 at observation time but may enter the
Top-50 by 23:00 Europe/Budapest. The replay is independent of bot signals,
positions, gates, models, and reports.

## Frozen prediction contract

- Universe: Binance Spot symbols quoted in USDT with valid market bars at all
  required timestamps. Historical availability is inferred from raw Binance
  bars; the report must disclose that historical `exchangeInfo` snapshots are
  unavailable and therefore delisted-symbol coverage may be incomplete.
- Watchlist: the versioned `files/watchlist.json` loaded at replay time. The
  hash and symbol count are evidence fields.
- Observation time: 12:15 Europe/Budapest. With `1h` source bars, only the last
  bar closed before the observation instant is visible.
- Target time: 23:00 Europe/Budapest on the same local day.
- Current rank: rolling-24h return ending at the causal observation price.
- Target label: rolling-24h return ending at the target price. Target labels
  must never enter features or ranking.
- Candidate population: valid watchlist symbols whose observation-time market
  rank is greater than 50.
- Selection size: top 10 candidates; top-1 is reported separately.

## Frozen screen variants

1. `current_rank`: order candidates by observation-time rolling-24h return.
2. `static_target`: order by the return that would remain at target time if the
   observation price did not change.
3. `screen_v1`: the formula used by the 2026-08-21 external screen:

   `static_target + 0.20*ret_1h + 0.05*ret_3h + clamp(0.35*(vol_accel-1), -0.5, 1.0)`

   `vol_accel` is the most recent closed one-hour quote volume divided by the
   mean of the preceding four closed hours.

Constants are frozen before reading replay outcomes and may not be retuned on
the same history.

## Causality and coverage

- Every feature timestamp must be strictly earlier than or equal to the last
  bar close before observation time.
- The target return uses the target close and is attached only after ranking.
- A local day is eligible only when at least 200 market symbols and at least 50
  watchlist symbols have every required price and feature.
- Missing days or symbols are coverage loss, not misses or successes.
- Source files are immutable runtime evidence and are never committed.
- Overlapping cache files are deduplicated by `(symbol, open_time)`.

## Decision metrics

For every variant disclose:

- eligible days and selected predictions;
- Top-1 hits/days and Wilson 95% interval;
- Top-10 hits/selections (precision), denominator and Wilson 95% interval;
- target entrant recall: selected hits / all watchlist symbols that entered the
  target Top-50 from outside it at observation time;
- candidate base rate and precision lift over that base rate;
- paired daily precision delta versus `current_rank`, with a deterministic
  day-bootstrap 95% interval;
- full-period and recent-period results.

`screen_v1` is supported only if it improves precision or recall over both
baselines and its paired interval excludes zero. Otherwise its verdict is
`REJECTED` or `INCONCLUSIVE`, never `validated`.

## Data collection

The validator may reuse raw Binance `1h` kline caches and may fetch a missing
tail from the public `/api/v3/klines` endpoint. New downloads are written only
under `.runtime/external_top50_history`. Reports include requested/covered
timestamps, symbols, rows, source paths, and content hashes.

Private account endpoints, API keys, bot runtime state, positions, and model
outputs are out of scope.

## Truth Harness mapping

- TH-01: every rate has numerator, denominator, base rate, lift, and interval.
- TH-02/03: raw market features and forward labels remain separated.
- TH-04: formula is frozen; full and chronological recent slices are reported.
- TH-05/06: paired variants use the same daily universe and candidate set.
- TH-08: negative and inconclusive verdicts are persisted in the runtime report.
- TH-10: freshness, coverage, survivorship limitations, and uncertainty remain
  explicit.
- TH-12: specification, focused tests, runtime evidence, and staged review are
  required; generated market data and reports are not committed.

## Production boundary

This work cannot change BUY, SELL, portfolio, alert, or bot ranking behavior.
A favorable historical result authorizes only a separately specified forward
shadow collector. It does not authorize production promotion.

## Maximum-period result

The frozen replay covered 2025-06-22 through 2026-08-20: 425 eligible local
days out of 427 requested. Per-day market coverage was 359--484 symbols and
watchlist coverage was 90--100 symbols. The public Binance tail refresh fetched
682,312 `1h` rows for 484/484 currently active Spot USDT symbols with zero
request failures. The combined used-source content hash was
`e843c8d4e9ab9d5e9d281149dfe71d76b716e9502aa9d79f20bd1165690ee1ae`.

Full-period results:

| Variant | Top-1 hits | Top-10 precision | Entrant recall | Base-rate lift |
|---|---:|---:|---:|---:|
| current rank | 88/425 = 20.71% | 648/4250 = 15.25% | 648/2336 = 27.74% | 2.42x |
| static target | 205/425 = 48.24% | 1037/4250 = 24.40% | 1037/2336 = 44.39% | 3.87x |
| screen v1 | 197/425 = 46.35% | 1020/4250 = 24.00% | 1020/2336 = 43.66% | 3.81x |

`static_target` improved paired daily Top-10 precision over `current_rank` by
9.15pp, day-bootstrap 95% interval `[+7.98pp, +10.33pp]`. The added
momentum/volume terms in `screen_v1` reduced full-period precision versus
`static_target` by 0.40pp; its interval `[-0.99pp, +0.16pp]` includes zero.
Therefore the added terms are not validated and `screen_v1` is rejected.

On the latest 60 eligible days, `screen_v1` was numerically ahead of
`static_target` by 0.50pp Top-10 precision (20.17% versus 19.67%) and by three
entrant hits, but the paired interval `[-1.17pp, +2.17pp]` includes zero. This
slice is `INCONCLUSIVE` and cannot reverse the maximum-period rejection.

The truthful conclusion is narrower: shifting the 24-hour denominator to the
target cutoff is a strong ranking baseline; the ad-hoc momentum and volume
bonuses do not add proven value. Current AAVE evidence may remain consistent
with the static baseline, but its 2026-08-21 target label is not mature until
23:00 and must stay pending.
