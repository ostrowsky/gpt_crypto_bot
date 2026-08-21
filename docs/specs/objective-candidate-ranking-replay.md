# Objective Candidate Ranking Replay

Date: 2026-08-21
Status: maximum-period validation complete; both profiles rejected; production unchanged

## Problem

The corrected daily objective shows `132/435` early captures (`30.34%`) but a
captured-pair median capture of only `21.43%`. Candidate ordering still includes
an ML-ranker whose historical rows have no verified policy/label provenance.
When several admissible candidates compete for scarce slots, the replay does
not expose whether a blocked candidate later outperformed the incumbent.

## Frozen causal hypotheses

### H1: structural allocation rank

Keep all production gates and exits. Rank simultaneous candidates and compare
replacement scores with the existing top-gainer formula recomputed from the
current candle while forcing both unverified ML-ranker inputs to zero. This
tests whether the legacy ranker distorts scarce-slot ordering.

Replay variant: `objective_rank_structural`.

### H2: extension-efficient structural rank

Start with H1, then favor observable strength before extension:

- quality bonus: `min(12, max(0, ADX-20)*0.5)` plus
  `min(10, max(0, vol_x-1.2)*4)`;
- extension penalty: `min(18, max(0, daily_range-6)*1.5)` plus
  `min(10, max(0, RSI-68))`.

Replay variant: `objective_rank_extension_efficiency`.

The constants are frozen before the maximum-period replay. They may not be
retuned on the same window.

## Capacity regret audit

Whenever the portfolio is full and an otherwise processed candidate cannot
replace a position, record only causal event fields: timestamp, candidate,
lowest allocation-ranked incumbent, and their decision-time scores. After the
simulation has completed, an isolated evaluator attaches:

- 5x15m forward return for candidate and incumbent;
- candidate-minus-incumbent return delta;
- whether the candidate, incumbent, or both belong to that day's Top-15.

The report must disclose event and mature-label denominators, candidate-win
count/rate, average/median delta, and missed-objective pair count. Future labels
must not affect ordering, admission, replacement, or exits.

## Acceptance gate

Against the corrected production control on the same frozen 30-day snapshot:

1. daily objective and canonical portfolio reports are decision-grade;
2. early capture improves by at least 5 of 435 pairs (at least 1.0pp), or
   captured-pair median capture improves by at least 2.0pp;
3. captured-pair recall is no worse by more than 2 pairs;
4. trade precision is no worse by more than 0.5pp;
5. canonical alpha and maximum drawdown are each no worse by more than 1.0pp;
6. no coverage, capacity, or open-position contract violation occurs.

If regret labels are immature or a mission gate fails, the result is rejected
or inconclusive, never promoted because of lower activity alone.

## Production boundary

Both variants and regret events exist only in replay. No live sorting,
replacement, BUY, SELL, configuration, or model payload changes in this work.
A passing replay would authorize only a separate shadow-first feature with an
expiring rollback flag and forward per-decision traces.

## Truth Harness mapping

- TH-01: daily pair and regret numerators/denominators are explicit.
- TH-02/03: legacy ML evidence is excluded from variant ordering; future labels
  are post-simulation audit data only.
- TH-05/06: same daily objective, window, universe, costs, and candidate path.
- TH-07/08: replay-only variants; negative results persist.
- TH-10/11: maturity, coverage, canonical alpha, and drawdown stay visible.
- TH-12: spec, tests, maximum-period replay, staged review.

## Maximum-period result

Frozen window: `2026-07-21T09:00:00Z` through
`2026-08-20T09:00:00Z`; 105 requested symbols; 29 complete local days; 435
daily Top-15 labels; `15m` and `1h`; ten slots; 7.5 bps fee and 5 bps
slippage per side. All reports used price hash
`8908020dffb48f2338b85f328f568ce1bee86adccc3c509f2943809e3606002c`
and source hash
`b95f07500771a9922baf802e25119231f982eeb02134add43bf1632b9126cc44`.

| Variant | Captured | Early | Precision | Pair capture avg / median | Net after costs | Alpha | Max DD |
|---|---:|---:|---:|---:|---:|---:|---:|
| production control | 353/435 | 132/435 | 828/1509 = 54.87% | 28.37% / 21.43% | -47.28% | -55.13pp | 51.03% |
| structural rank | 353/435 | 132/435 | 828/1509 = 54.87% | 28.37% / 21.43% | -47.28% | -55.13pp | 51.03% |
| extension-efficient rank | 352/435 | 133/435 | 827/1542 = 53.63% | 28.16% / 20.00% | -47.83% | -55.69pp | 51.37% |

H1 produced no changed portfolio decision or metric on this candidate
population. H2 added only one early pair, below the frozen five-pair gate,
lost one captured pair, reduced precision by 1.24pp, and reduced median capture
by 1.43pp. Both are rejected. No thresholds may be retuned on this window.

Capacity regret remained measurable and fully mature:

| Variant | Events / mature | Candidate wins | Ret5 delta avg / median | Missed objective events / unique pairs |
|---|---:|---:|---:|---:|
| control | 1667 / 1667 | 952 = 57.11% | +0.1496pp / +0.1361pp | 242 / 79 |
| structural | 1667 / 1667 | 952 = 57.11% | +0.1496pp / +0.1361pp | 242 / 79 |
| extension-efficient | 1645 / 1645 | 799 = 48.57% | +0.0239pp / -0.0180pp | 213 / 79 |

This is evidence of ranking opportunity, not a deployable rule: the control
candidate wins only 57% of conflicts and the audit label is future data. The
next priority is a dedicated capacity-conflict dataset with causal feature
timestamps, day-grouped chronological train/holdout splits, and explicit
provenance. Until that discriminator passes a pre-gate and a separate paired
portfolio replay, production ordering remains unchanged.
