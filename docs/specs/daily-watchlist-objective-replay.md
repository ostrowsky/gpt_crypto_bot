# Daily Watchlist Objective Replay

Date: 2026-08-21
Status: implemented; corrected maximum-period production control complete

## Problem

The portfolio replay currently ranks symbols by start-to-end return over the
whole requested window and calls the resulting set `final_top_n`. On a 30-day
run this produces one denominator of 15 symbols instead of daily watchlist
leader pairs. The reported average capture is also aggregated over all trades,
not over captured daily leaders. Both measures can look saturated while the
bot still misses its actual mission.

## Objective contract

The canonical replay objective is `same_day_watchlist_top_at_22_v1`:

- timezone: `Europe/Budapest`;
- one label population per complete local calendar day;
- label return: first observable 15m day-open price to the last 15m close
  observable at or before 22:00 local;
- population: the replay watchlist with complete prices for that day;
- leaders: the highest `objective_top_n` returns within that population;
- decision cutoff: only entries at or before 22:00 can capture that day's
  leader;
- boundary days are excluded unless both local midnight and the objective
  cutoff are inside the frozen replay window.

Future daily rank is an evaluator label only. Candidate ordering, gates,
replacement, and exits must not read it.

## Required metrics

Every rate carries counts:

- `label_pair_count`: complete `(local_day, symbol)` leader labels;
- `captured_pair_count / label_pair_count`: first eligible BUY for the pair;
- `early_pair_count / label_pair_count`: captured pair whose first-entry
  `capture_ratio_at_entry >= 0.35`;
- `objective_trade_count / eligible_trade_count`: trade precision on complete
  days before the cutoff;
- average and median capture and lead time over captured leader pairs only;
- eligible-day count, per-day available-symbol counts, minimum coverage, and
  excluded boundary-day count.

No labels means `null`/`unknown`, never `0%`. Unique symbols may be reported as
diagnostics but cannot replace day-symbol denominators.

## Acceptance criteria

1. Unit tests prove daily ranking, pair denominators, first-entry selection,
   early-capture counts, precision, and boundary exclusion.
2. Existing replay variants receive identical daily objective labels and no
   label is exposed to their decision functions.
3. Maximum-available 30-day, full-watchlist production control is rerun with
   the corrected objective and canonical ten-slot alpha.
4. Previous whole-window `15/15` claims are marked superseded; they cannot be
   used to promote a policy.
5. Ranking hypotheses are not evaluated until this corrected baseline is
   decision-grade.

## Truth Harness mapping

- TH-01: explicit daily pair and trade numerators/denominators.
- TH-02: evaluator labels remain separate from causal decisions.
- TH-03/04: labels use later cutoff prices only for evaluation.
- TH-05/06: complete comparable days and actual full watchlist.
- TH-08: superseded metric claims remain documented.
- TH-10: missing and partial coverage is visible.
- TH-11: profitability remains canonical portfolio alpha after costs.
- TH-12: spec, focused tests, maximum-period replay, staged review.

## Rollback

This is a replay measurement correction, not a live trading change. Rollback
reverts the objective-report schema, but the old whole-window metric must then
be labelled diagnostic and cannot become a production gate.

## Maximum-period corrected baseline

Frozen requested window: `2026-07-21T09:00:00Z` through
`2026-08-20T09:00:00Z`; 105 requested symbols; `15m` and `1h`; Top-15 per
complete local day; ten slots; fee 7.5 bps and slippage 5 bps per side. The
price-stream hash is
`8908020dffb48f2338b85f328f568ce1bee86adccc3c509f2943809e3606002c`.

- complete eligible days: `29`; excluded partial boundary days: `2`;
- daily leader denominator: `435 = 29 x 15` pairs;
- captured leaders: `353/435 = 81.15%`;
- early leaders (`capture_ratio >= 0.35`): `132/435 = 30.34%`;
- objective trades / eligible trades: `828/1509 = 54.87%` precision;
- captured-pair capture ratio: `n=343`, average `28.37%`, median `21.43%`;
- captured-pair lead time: `n=353`, average `651.2m`, median `615m`;
- available daily population: minimum `91` of `105` requested symbols;
- canonical portfolio: decision-grade, net `-47.2765%`, BTC-relative alpha
  `-55.1296pp`, maximum drawdown `51.0277%`.

The historical early-capture target of 25% is exceeded on this frozen window,
but this is not a blanket health claim: daily symbol availability varies and
the portfolio economic guardrail remains poor. The old whole-window result of
`15/15` and its all-trade average capture are superseded and must not be used
for promotion. The next ranking replay must use the new 435-pair objective and
must report both numerator and denominator.
