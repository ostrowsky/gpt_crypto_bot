# Full Watchlist Rotating Monitor

Status: shipped under guarded forward canary
Last updated: 2026-08-13

## Problem

The 15-minute auto-reanalysis replaces `state.hot_coins` with confirmed coins
plus symbols already showing a signal. A quiet watchlist member can therefore
disappear from dense polling before its trend begins. Discovery and wake-up
scans reduce this blind spot but do not remove it; the main monitor still does
not observe the whole objective population.

## Behavioral contract

When `MONITOR_FULL_WATCHLIST=True`:

1. `_update_hot_coins` keeps exactly one report for every configured watchlist
   symbol, preferring the newest scan report and retaining an existing report
   when a transient scan omits a symbol.
2. Open positions outside the current watchlist remain monitored on their
   original timeframe.
3. The monitor polls at most `MAX_POLL_PER_CYCLE` symbols per cycle unless the
   number of open positions itself exceeds the cap.
4. Every open position is polled every cycle. Remaining capacity rotates
   deterministically through non-position symbols; no symbol may starve.
5. The monitor remains active even when the latest scan has zero `in_play`
   symbols, because signal discovery is now part of monitoring.
6. This change does not modify BUY, SELL, score, chase, cooldown, replacement,
   or portfolio gates.

## Rollback

Set `MONITOR_FULL_WATCHLIST=False`. The previous shortlist construction and
start/stop behavior are restored without a code rollback. Set
`MAX_POLL_PER_CYCLE=0` only for an explicit unbounded diagnostic run.

## Acceptance and replay gate

Before enabling the default:

- focused tests prove full-watchlist inclusion, deduplication, rollback,
  bounded rotation, no starvation, and open-position priority;
- a deterministic coverage replay uses all available final critic days and
  proves that every historical watchlist-top symbol is scheduled within one
  full rotation;
- the maximum feasible all-watchlist trading replay of the unchanged policy is
  recorded as the safety baseline. Because the old shortlist membership was
  not persisted, this replay cannot claim a causal recall uplift; that delta is
  `UNKNOWN` until forward canary evidence exists;
- cycle duration and Binance error rate remain within operational limits.

Maximum-period structural result recorded on 2026-08-13:

- canonical local final-critic period: 2026-04-01 through 2026-08-12,
  117 available days;
- historical final watchlist-top scheduling: 263/263 (100%);
- current watchlist: 103 symbols, cap 45, full sweep in 3 cycles / at most
  180 seconds at the configured 60-second poll interval;
- causal recall/PnL uplift: `UNKNOWN`, because historical shortlist membership
  was not persisted.

Trading replay note:

- the existing all-watchlist replay fetches one non-paginated candle request
  per symbol/timeframe; a 134-day run (the full 2026-04-01..2026-08-12 span)
  returned zero candidates and is rejected as invalid evidence;
- the largest nominally fetchable 15m window is seven days (672 of Binance's
  1000-candle response limit), but the run also returned zero candidates in the
  current external market window and is rejected as invalid evidence;
- no trade/PnL safety claim is made from either run. The structural replay plus
  guarded forward canary are the only evidence for this observation-layer
  release; they cannot prove a recall or profitability uplift.

## Forward canary

For at least seven complete live days compare, with explicit denominators:

- watchlist symbols scheduled and successfully fetched;
- full-sweep latency p50/p95/max;
- first structural observation and first BUY time for final watchlist top
  movers;
- `watchlist_top_bought`, `early_captures`, false-positive BUY rate;
- monitor cycle duration, timeout rate, and HTTP error rate.

Rollback if p95 full-sweep latency exceeds one closed 15-minute bar, fetch error
rate materially worsens, open-position polling is delayed, or early capture
worsens without a compensating quality improvement.

## Truth Harness mapping

- TH-05: forward comparisons use complete days and comparable denominators.
- TH-06: replay population is the configured watchlist plus historical critic
  top symbols over the maximum locally available period.
- TH-07: feature flag, bounded load, rollback, and canary are explicit.
- TH-09/TH-12: config, code, spec, tests, and staged scope must agree.
