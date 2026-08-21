# Static-Target Top-50 Forward Shadow

Date: 2026-08-21
Status: implementation approved; shadow-only

## Purpose

Collect forward, immutable evidence for the replay-validated `static_target`
ranking baseline. The component predicts which watchlist symbols currently
outside the Binance Spot rolling-24h Top-50 will be in that Top-50 by 23:00
Europe/Budapest.

This component is research measurement. It cannot emit BUY/SELL/WATCH
messages, alter portfolio admission, change a score, or train a model.

## Historical authorization

The maximum-available causal replay in
`docs/specs/external-top50-screen-validation.md` covered 425 eligible days from
2025-06-22 through 2026-08-20. `static_target` improved Top-10 precision over
`current_rank` from 648/4250 (15.25%) to 1037/4250 (24.40%), a paired daily
delta of +9.15pp with 95% bootstrap interval [+7.98pp, +10.33pp]. The replay
authorizes forward shadow collection only. The extra momentum/volume formula
was rejected and is not implemented here.

## Frozen forward contract

- Universe: Binance Spot symbols with `status=TRADING`, `quoteAsset=USDT`, and
  `isSpotTradingAllowed=true` in exchangeInfo fetched during observation.
- Watchlist: the exact bytes of `files/watchlist.json` at observation; SHA-256,
  count, and normalized symbols are stored.
- Observation slot: 12:15 Europe/Budapest, with a 30-minute grace window.
- Source: Binance public `1h` klines. Only bars closed at or before 12:15 are
  features. A process starting after the grace window records a missed slot;
  it must not reconstruct a prediction from later data.
- Target slot: 23:00 Europe/Budapest. Labels may be fetched no earlier than
  23:05.
- `current_return`: observation close / close at observation minus 24h - 1.
- `static_target_return`: observation close / close at target minus 24h - 1.
- Candidate population: valid watchlist symbols with observation-time market
  rank greater than 50.
- Shadow selection: first 10 candidates ordered by descending
  `static_target_return`, then symbol. A paired control selects 10 by descending
  `current_return`, then symbol.
- Target label: target close / stored target-minus-24h close - 1. The target
  label is absent from the observation artifact and attached only in a separate
  final artifact.

## Evidence artifacts

Runtime-only files under `.runtime/reports`:

- `static_target_top50_shadow_YYYY-MM-DD_observation.json`: immutable prediction,
  complete market reference rows, feature timestamps, formula/version hashes,
  watchlist and exchangeInfo provenance, coverage, paired selections.
- `static_target_top50_shadow_YYYY-MM-DD_missed.json`: observation slot missed;
  never counted as a miss or success.
- `static_target_top50_shadow_YYYY-MM-DD_final.json`: post-target labels and
  numerator/denominator metrics for both variants.
- `static_target_top50_shadow_scorecard_latest.json`: aggregate forward evidence,
  including scheduled/observed/final/eligible denominators and uncertainty.

Observation files are create-once. Finalization verifies their schema and
contract hash. Missing or malformed source data produces `partial`/`unknown`,
not zero performance.

## Forward decision metrics

For `static_target` and `current_rank`, report:

- Top-1 hits / eligible days with Wilson 95% interval;
- Top-10 hits / selections (precision) with Wilson 95% interval;
- selected entrant hits / all eligible watchlist entrants (recall);
- candidate base rate and precision lift;
- paired daily Top-10 precision delta with deterministic bootstrap interval;
- calendar coverage: observed slots / scheduled slots and eligible finals /
  scheduled slots.

An entrant is a watchlist symbol outside market Top-50 at observation that is
inside market Top-50 at target. A day is eligible only when observation and
target each cover at least 200 market symbols and the observation covers at
least 50 watchlist symbols.

## Promotion gate

The scorecard may return `ELIGIBLE_FOR_SEPARATE_PRODUCTION_REVIEW` only when:

1. at least 30 eligible forward days exist;
2. at least 90% of scheduled calendar slots have immutable observations and at
   least 85% have eligible final labels;
3. the paired 95% bootstrap lower bound for Top-10 precision delta versus
   `current_rank` is greater than zero;
4. `static_target` precision is above the disclosed candidate base rate;
5. all eligible artifacts have the frozen contract hash and no feature/label
   timing violation.

Any unmet evidence condition is `COLLECTING`, `INCONCLUSIVE`, or `FAIL`, never
promotion. Passing this gate does not change production: a separate spec,
maximum-period portfolio-aware replay, alert-noise/precision constraints,
rollback switch, and explicit operator decision remain mandatory.

## Operations and rollback

- `STATIC_TARGET_TOP50_SHADOW_ENABLED=False` stops new collection immediately.
- The first scheduled day is 2026-08-22, the first full observation day after
  this implementation. The component must not report a pre-deployment slot as
  downtime.
- The headless worker checks the schedule once per minute.
- A restart may finalize an existing observation after 23:05, but may never
  backfill an observation after its grace window.
- API errors remain visible in worker status and artifact coverage.
- No Telegram notification is emitted by this phase.

## Forward canary

For this alert product, the canary is the first 30 eligible days of parallel
shadow collection. It has zero user traffic: no Telegram message, BUY, SELL,
WATCH, portfolio mutation, or production-ranker input. Load is bounded to one
observation fetch and one label fetch per active symbol per day with concurrency
8. The operator reviews calendar coverage, API failures, parity, precision,
recall, base-rate lift, and paired uncertainty after the minimum cohort matures.
There is no automatic escalation from this canary. Any user-visible alert
canary or trading use requires a separate spec and explicit approval.

## Truth Harness mapping

- TH-01: all ratios retain numerators, denominators, base rate, lift, and
  uncertainty.
- TH-02/03: shadow evidence is separate from production; observations and
  forward labels are separate immutable artifacts with timestamps.
- TH-04/06: the formula was frozen by maximum-period causal replay; forward
  evidence is chronological and uses the actual watchlist candidate population.
- TH-05: paired variants use identical daily populations; missing/downtime days
  are disclosed and excluded from performance denominators.
- TH-07: collection has a config rollback; trading promotion requires a new
  review and cannot be automatic.
- TH-08: negative/inconclusive scorecards persist.
- TH-10: freshness, coverage, sample size, and uncertainty are gate inputs.
- TH-12: spec, focused tests, staged review, and runtime-only evidence are
  required.
