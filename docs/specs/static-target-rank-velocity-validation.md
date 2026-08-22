# Static-Target Rank-Velocity Validation

Date: 2026-08-22
Status: maximum-period validation complete; hypothesis rejected; production unchanged

## Purpose

Test one causal WATCH-ranking hypothesis without changing Telegram, BUY, SELL,
replacement, sizing, or the production portfolio:

> Among watchlist symbols outside the Binance Spot Top-50 at 12:15
> Europe/Budapest, a modest three-hour improvement in cross-sectional market
> rank adds useful information to the already supported `static_target`
> baseline and improves same-day 23:00 Top-50 Top-10 precision.

This experiment is deliberately narrow. Earlier maximum-period evidence showed
that adding ad-hoc one-hour/three-hour return and volume bonuses to
`static_target` reduced precision. This protocol therefore tests one new
cross-sectional feature family and preserves a negative or inconclusive result.

## Frozen population and clocks

- Source: cached Binance Spot `1h` klines, deduplicated by `(symbol, open_time)`.
- Historical universe: symbols with valid bars at all required feature and
  label timestamps. Historical `exchangeInfo` snapshots are unavailable, so
  delisted-symbol survivorship remains an explicit limitation.
- Watchlist: versioned `files/watchlist.json`, with its SHA-256 recorded.
- Observation: 12:15 Europe/Budapest; only the last fully closed `1h` bar is
  visible.
- Target: 23:00 Europe/Budapest on the same local day.
- Candidate population: valid watchlist symbols whose observation-time rolling
  24-hour market rank is greater than 50.
- Selection: ten candidates per eligible day; Top-1 is reported separately.
- Eligibility: at least 200 market symbols, 50 watchlist symbols, ten
  candidates, and complete causal feature/label bars.
- Maximum period: all eligible local days in the available cache. No favorable
  subperiod may replace the maximum-period result.
- Chronological holdout: the latest 90 eligible days. Earlier days are reported
  as development context only; the frozen formula is not fitted on either
  slice.

Missing days and symbols are coverage loss, never successes or misses.

## Feature timing and definitions

At observation cutoff `t`:

- `current_return(t) = close(t) / close(t-24h) - 1`;
- `static_target_return(t) = close(t) / close(target-24h) - 1`;
- `prior_current_return(t-3h) = close(t-3h) / close(t-27h) - 1`;
- `current_market_rank` is the descending rank of `current_return(t)` across
  the eligible market universe;
- `prior_market_rank` is the descending rank of
  `prior_current_return(t-3h)` across that same eligible universe;
- `rank_velocity_3h = prior_market_rank - current_market_rank`, so positive
  values mean that the symbol moved toward rank 1.

The 23:00 target price and target rank are labels. They must not influence any
feature, percentile, candidate filter, or selection.

## Frozen policies

1. `static_target`: descending `static_target_return`, then symbol.
2. `static_target_rank_velocity_v1`:
   - calculate deterministic candidate-population percentile ranks in `[0, 1]`
     for `static_target_return` and `rank_velocity_3h`;
   - `score = static_target_percentile + 0.25 * rank_velocity_percentile`;
   - order by descending score, then descending `static_target_return`, then
     symbol.

The `0.25` coefficient and three-hour horizon are frozen before outcome
computation. This task does not scan alternate horizons, weights, thresholds,
or formulas. A rejected result may be revisited only by a new hypothesis and a
new untouched evidence window.

## Metrics and decision rule

For both policies and every reported slice disclose:

- Top-1 hits / eligible days and Wilson 95% interval;
- Top-10 hits / selections, precision, and Wilson 95% interval;
- target-entrant hits / entrants and recall;
- candidate hits / candidate population, base rate, and precision lift;
- paired daily Top-10 precision delta with a deterministic day-bootstrap 95%
  interval;
- paired-day standard deviation, effective sample size, and 95%/80%-power MDE;
- requested/eligible days, symbol coverage, hashes, and limitations.

The result is `SUPPORTED_FOR_FORWARD_SHADOW_ONLY` only when all conditions hold:

1. at least 30 eligible maximum-period days and 90 holdout days;
2. maximum-period paired precision delta is at least `+2pp`;
3. maximum-period paired bootstrap lower bound is above zero;
4. holdout paired precision delta is non-negative;
5. candidate entrant recall is not lower than `static_target` on the maximum
   period or holdout;
6. candidate precision lift over the disclosed base rate is above `1.0`;
7. maximum-period 95%/80%-power MDE does not exceed the registered `2pp`
   minimum practical effect;
8. independent verification exactly reproduces the raw-snapshot result;
9. coverage and causality checks have no violation.

If the point estimate is below `+2pp` or recall is worse, the verdict is
`REJECTED`. If direction is favorable but uncertainty, power, or coverage is
insufficient, the verdict is `INCONCLUSIVE`. Invalid hashes, timing, or verifier
disagreement produce `INVALID_RESULT`.

## Evidence separation and independent verification

This is an engineering-lane, pre-admission validator for a new feature family.
The code, focused tests, and validator had to exist before the experiment could
be considered for the durable validation queue. Because the pre-registered
maximum-period result is rejected, no new capability or validator binding is
added to the durable loop and no attempt is advanced toward forward shadow.
Had the candidate passed, registration and a new orchestrator-owned durable
attempt would still have been required before any forward collection.

The orchestrator writes an immutable normalized snapshot before validation.
The validator reads only that snapshot and the frozen contract. A separate
verifier must not import validator aggregation code; it independently rebuilds
candidate percentiles, selections, denominators, metrics, bootstrap interval,
and hashes from the raw snapshot.

Runtime snapshots and reports live under `.runtime/` and are not committed.
The tracked specification records the terminal maximum-period result, including
negative and inconclusive evidence.

## Production boundary and rollback

This validation has `production_effect=none_research_only`. Even a supported
result authorizes only a separately specified immutable forward shadow. It does
not authorize production ranking, Telegram messages, BUY/SELL gates, portfolio
replacement, model training, or claims of realized trading improvement.

Rollback is deletion of the research modules and untracked runtime evidence.
No production migration exists.

## Truth Harness mapping

- TH-01: numerators, denominators, base rate, lift, and intervals are explicit.
- TH-02/03: causal features are separated from 23:00 labels.
- TH-04: the latest 90 eligible days form a chronological holdout.
- TH-05/06: paired policies share the actual candidate population and maximum
  available period.
- TH-07: no production effect; any later use requires a rollback switch and
  forward shadow.
- TH-08: rejected and inconclusive evidence remains in this specification.
- TH-10: power, freshness, coverage, and survivorship limitations are visible.
- TH-11: this WATCH experiment does not claim portfolio profitability.
- TH-12: focused tests, independent verification, staged review, commit, and
  push are required.

## Acceptance tests

1. Changing a future target price changes labels but not any ranking feature.
2. `rank_velocity_3h` uses market ranks at `t-3h` and `t`, not future ranks.
3. Candidate percentile scoring is deterministic, bounded, and matches the
   frozen `0.25` formula.
4. Missing feature bars remove coverage rather than becoming a miss.
5. Metrics expose reconstructable denominators, base rate, lift, uncertainty,
   and chronological holdout.
6. The verifier rejects a modified aggregate or mismatched snapshot hash.
7. The maximum-period run persists one terminal verdict and leaves production
   behavior unchanged.

## Result

The frozen protocol was executed once over the maximum locally available
period `2025-06-21 .. 2026-08-21`. There were 425 eligible days out of 427
requested (`99.53%` calendar coverage), 37,070 candidates, and 2,334 target
entrants. Market/cache provenance had 503 symbols with rows, 925 used files,
zero malformed files, and content hash
`e843c8d4e9ab9d5e9d281149dfe71d76b716e9502aa9d79f20bd1165690ee1ae`.

Maximum-period result:

| Policy | Top-1 | Top-10 precision | Entrant recall | Base-rate lift |
|---|---:|---:|---:|---:|
| `static_target` | 204/425 = 48.00% | 1,036/4,250 = 24.38% | 1,036/2,334 = 44.39% | 3.87x |
| `static_target_rank_velocity_v1` | 146/425 = 34.35% | 955/4,250 = 22.47% | 955/2,334 = 40.92% | 3.57x |

The candidate changed paired daily Top-10 precision by `-1.91pp`; the
day-bootstrap 95% interval was `[-2.68pp, -1.15pp]`. Full-period MDE was
`1.10pp`, below the registered `2pp` SESOI, so the negative result was
adequately powered for that practical effect.

The latest 90 eligible days independently agreed with the direction:

| Policy | Top-1 | Top-10 precision | Entrant recall |
|---|---:|---:|---:|
| `static_target` | 38/90 = 42.22% | 186/900 = 20.67% | 186/414 = 44.93% |
| `static_target_rank_velocity_v1` | 27/90 = 30.00% | 172/900 = 19.11% | 172/414 = 41.55% |

Holdout delta was `-1.56pp`, bootstrap 95% interval
`[-3.00pp, -0.11pp]`. The candidate failed the registered effect, recall,
holdout, and interval gates. Verdict: **`REJECTED`**. It must not enter forward
shadow or production under this hypothesis ID.

The paired baseline differs slightly from the earlier static-target study
(1,036 rather than 1,037 hits) because this experiment requires complete
`t-3h` and `t-27h` feature bars and therefore freezes its own common candidate
cohort. Cross-study numerators must not be mixed; the within-study comparison
above is paired on identical days and candidates.

The independent verifier recomputed the snapshot and returned
`VERIFIED_RESULT` with no errors. Snapshot SHA-256:
`ff4846f346fc74e3e0d00ada85a1c9bf8b4a4c11c03bf3d60c96cd3f02b09420`;
contract SHA-256:
`85965ddb99a7b1177fc20610c5a35d6c7484f7286e1c21d6b01f71f28d6a6320`.
Runtime snapshot, contract, traces, and report remain uncommitted under
`.runtime/`.
