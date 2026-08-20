# Objective Leader Admission Replay

Date: 2026-08-21
Status: maximum-period replay complete; all variants rejected; production policy unchanged

## Problem

The bot frequently observes strong watchlist leaders but cannot admit them when
the ten-slot portfolio is full or the generic RSI chase guard treats strong
continuation as late overheat. Broad replacement, static agent rescue, and a
plain RSI 82 relaxation did not prove mission improvement.

## Objective

Improve causal early capture of final watchlist Top-15 movers. The primary
metrics are average and median `capture_ratio_at_entry` and
`lead_time_to_final_top_min`, always reported with Top-15 numerator/denominator.
Recall is secondary when it is saturated. Guardrails are trade precision,
false-positive activity, canonical ten-slot portfolio alpha after costs,
turnover, and maximum drawdown.

## Causal hypotheses

### H1: strong continuation chase

Keep the hard `daily_range > 25%` guard. Permit the RSI-only 76-82 region only
for `strong_trend` or `impulse_speed` candidates with all information available
at the decision candle:

- candidate score >= 150;
- ADX >= 20;
- volume multiple >= 1.2;
- RSI <= 82;
- daily range <= 25%.

### H2: objective slot reserve

Reserve one of ten slots from ordinary candidates. The reserved slot is
available to a causal leader profile only:

- top-gainer replay score >= 80;
- intraday change >= 1%;
- daily range <= 12%;
- RSI <= 76;
- ADX >= 20;
- volume multiple >= 1.2.

The profile does not use final Top-15 membership, later returns, or rolling-24h
rank reconstructed after the fact.

### H3: combined

Apply H1 and H2 together to measure interaction rather than adding their
individual deltas.

## Replay variants

- `chase_guard_strong_continuation`
- `objective_slot_reserve`
- `objective_leader_combined`

All variants retain current score gates, cooldowns, exits, costs, symbol
deduplication, and ten-slot capital accounting. They exist only in the replay
module and are disabled by default.

## Acceptance gate

A variant advances only if the maximum-available full-watchlist replay is
decision-grade and:

1. improves average capture by at least 1.0 percentage point or median capture
   by at least 2.0 points;
2. does not worsen median lead time;
3. trade precision is no worse by more than 0.5 percentage point;
4. maximum drawdown is no worse by more than 1.0 point;
5. canonical net alpha after costs is no worse by more than 1.0 point;
6. no coverage or open-position contract violation is present.

Recall without numerator/denominator and activity without precision cannot
support promotion. An incomplete result is `INCONCLUSIVE`, not a pass.

## Canary and rollback

There is no live canary in this change. If a later decision-grade replay and
independent forward cohort pass, rollout must be shadow-first with an expiring
flag and per-symbol decision traces. Rollback disables that flag and restores
the current chase and capacity policy. No production configuration is changed
by these replay variants.

## Truth Harness mapping

- TH-01/02: report counts, denominators, mission metrics, and canonical alpha.
- TH-03/04: closed-candle inputs and frozen chronological window only.
- TH-05/06: identical window, universe, costs, capacity, and coverage.
- TH-07: research-only variants; later shadow flag and rollback required.
- TH-08: persist negative and inconclusive decisions in this spec.
- TH-11: use canonical unified ten-slot alpha, never diagnostic PnL sum.
- TH-12: spec, focused tests, maximum-period replay, staged Harness.

## Maximum-period result

Frozen requested window: `2026-07-21T09:00:00Z` through
`2026-08-20T09:00:00Z`; 105 symbols; `15m` and `1h`; ten slots; Top-15
objective; 7.5 bps fee plus 5 bps slippage per side. Every paired report used
price-stream hash
`8908020dffb48f2338b85f328f568ce1bee86adccc3c509f2943809e3606002c`.

| Variant | Captured | Trade precision | Avg/median capture | Median lead | Net after costs | Max DD | Reserved admit/skip |
|---|---:|---:|---:|---:|---:|---:|---:|
| production control | 15/15 | 25.21% | 16.38% / 0.00% | 570m | -47.28% | 51.03% | 0 / 0 |
| strong continuation chase | 15/15 | 25.15% | 16.28% / 0.00% | 570m | -47.51% | 50.81% | 0 / 0 |
| objective slot reserve | 15/15 | 25.60% | 16.48% / 0.00% | 570m | -43.41% | 46.62% | 1 / 2,064 |
| combined | 15/15 | 25.58% | 16.41% / 0.00% | 570m | -43.59% | 46.64% | 2 / 2,144 |

All four canonical reports are decision-grade with complete valuation coverage
and no contract violations. A boundary-accounting defect found during this
validation was fixed: a trade opened and liquidated on the final replay
timestamp is now booked entry-before-exit for that trade, still pays both-side
costs, and no longer becomes a phantom open position. The mission result is:

- strong continuation reduced average capture by 0.10 percentage point;
- slot reserve improved average capture by only 0.10 point, below the frozen
  1.0-point gate, and median capture/lead did not improve;
- combined improved average capture by only 0.03 point and did not improve
  median capture or lead;
- reserved capacity mostly suppressed ordinary entries and was consumed by a
  qualifying leader only one or two times. Lower drawdown therefore reflects
  lower activity, not demonstrated leader capture.

Decision: reject all three profiles and keep production unchanged. Do not
retune these thresholds on the same window. The next hypothesis must improve
leader selection among simultaneous candidates rather than merely holding a
slot empty or weakening an overheat threshold.
