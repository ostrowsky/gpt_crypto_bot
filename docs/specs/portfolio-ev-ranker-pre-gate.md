# Portfolio EV/Opportunity Ranker Pre-Gate

Date: 2026-08-03
Status: research-only chronological pre-gate

## Problem

The ten-slot portfolio often has several candidates competing at the same
decision time. Ranking by a learned EV/opportunity score could improve capital
allocation, but the current bot objective is earlier capture of same-day
watchlist top movers, not return optimization in isolation.

## Hypothesis

Among candidates observed at the same causal decision timestamp, the frozen
candidate ranker can improve forward return and downside without reducing the
teacher top-gainer rate or capture ratio relative to the current candidate-score
ordering.

## Pre-Gate

Use only the chronological test split already produced by the maximum-history
candidate-ranker training run. Evaluate top-1, top-3, and top-5 competitions.
The pre-gate passes only when every required slice has:

- at least 100 eligible competitions;
- average target-return uplift of at least `+0.05pp`;
- no lower win rate;
- no worse average drawdown;
- no lower teacher top-gainer rate;
- no lower teacher capture ratio.

This deliberately treats the north-star metrics as mandatory rather than
allowing a return-only ranker to displace early top-mover candidates.

## Promotion Boundary

Passing authorizes a separate paired ten-slot portfolio replay with fees,
slippage, exits, turnover, and capacity. Failing rejects the current ranker for
capacity allocation and avoids that expensive replay. Either result leaves
production ranking, BUY gates, replacement, and the shadow training loop
unchanged.

## Maximum-History Result — 2026-08-03

The frozen CatBoost report contained `22,163` labeled rows split
chronologically into `15,514` train, `3,324` validation, and `3,325` untouched
test rows. The test split contained `559` grouped candidate competitions.

The current ranker failed every north-star slice:

| Slice | Return delta | Win-rate delta | Drawdown delta | Top-gainer-rate delta | Capture delta |
|---|---:|---:|---:|---:|---:|
| top-1 | `+0.0009pp` | `-1.61pp` | `-0.3252pp` | `-5.11pp` | `-1.70pp` |
| top-3 | `-0.1091pp` | `+0.34pp` | `-0.1607pp` | `-3.33pp` | `-1.11pp` |
| top-5 | `-0.0376pp` | `+0.71pp` | `-0.1375pp` | `-2.59pp` | `-0.87pp` |

The lower predicted downside does not compensate for lost top-mover capture,
and top-3/top-5 also lose forward return. Decision:
`reject_current_ranker_for_capacity_ranking`. Do not spend a full ten-slot
replay on this frozen model and do not enable it for production allocation.
