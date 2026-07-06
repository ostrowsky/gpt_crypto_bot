# Binance Top-20 Priority Hypothesis Validation - 2026-06-29

Status: research-only. No production BUY or SELL policy changes.

## Rule

Every strategy hypothesis is evaluated on the maximum feasible local period.
Short windows are diagnostics only and cannot promote a production change.

## Top-20 Objective

- `TOP_GAINER_CRITIC_TOP_N = 20`
- `SIGNAL_QUALITY_EVALUATOR_TOP_MOVERS_N = 20`
- daily Binance Top-20 backfill: `2026-05-30` through `2026-06-28`
- exchange symbols loaded: `429/429`
- informative daily Binance Top-20/watchlist label days: `27`
- labeled day-symbols: `86`

The portfolio replay engine uses its established maximum `30d` window and
`--objective-top-n 20`. Its objective rank is calculated over the replay window,
not per local day. Daily admission and blocker audits therefore use the separate
Binance Top-20 backfill rather than the portfolio objective labels.

## Exit Quality

Maximum available case period:

- reports loaded: `44`
- closed trades: `1735`
- visible cases: `1381` (partial case coverage)
- median exit efficiency: `-0.5000`
- median giveback: `1.7278%`
- early exits: `287`
- post-exit continuation cases: `651`

Decision: exit monetization remains the first research priority.

## Case-Level Exit Replays

Coverage: `945` cases, `788` eligible, `461` labeled, `327` missing/pending.

| Hypothesis | Maximum-period result | Decision |
|---|---:|---|
| Hold 5 bars after weak sell | avg delta `+0.3664%`, worse `41.21%` | reject broad hold |
| Partial 50%, hold 5 | avg delta `+0.1832%`, worse `41.21%` | research only |
| EMA trailing tail | avg delta `+0.1026%`, worse `50.76%` | reject |
| `non_ema_positive_giveback` selector | test delta `+0.2114%`, worse `5.80%`, FP allow `0%` | advance shadow-only |

## Portfolio Replay

Window: `2026-05-30` through `2026-06-29`, full watchlist, `15m + 1h`,
score floor `34`, objective Top-20.

| Variant | Trades | Total PnL | Avg PnL | Win | Precision | Exit eff | Giveback | Decision |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `score_replace` | 1464 | -519.97% | -0.3552% | 35.31% | 29.99% | -0.5000 | 1.3122% | baseline |
| `suspicious_exit_reentry` | 1527 | -538.93% | -0.3529% | 36.41% | 30.45% | -0.5000 | 1.3482% | reject for production |
| `partial_profit_take` | 1460 | -498.02% | -0.3411% | 36.44% | 29.93% | -0.5000 | 1.3393% | mixed, research only |
| cooldown `2` | 1656 | -578.13% | -0.3491% | 35.75% | 30.37% | -0.4952 | 1.3747% | not confirmed |
| `chase_guard_off` | 1447 | -637.18% | -0.4403% | 35.59% | 29.65% | -0.5556 | 1.3110% | reject |
| chase RSI `82` | 1461 | -560.70% | -0.3838% | 35.93% | 30.18% | -0.5000 | 1.3055% | reject |

All variants retain `100%` window-level Top-20 symbol capture, so capture alone
does not distinguish them. Absolute replay PnL remains negative for every policy.

## Entry Admission

- daily Top-20 label days: `27`
- labels: `86`
- blocked events: `383266`
- decision: `no_positive_shadow_reward`

The best non-empty cooldown slice had `2` candidates, `1` Top-20 and `1` false,
but rescued no winner and produced net reward `-1%`. Do not expand BUY gates.

## Blockers

The causal reward proxy found protected false candidates far more often than
harmful Top-20 misses. Notable harmful cases still require narrow diagnostics:

| Blocker | Cases | Top-20 | Harmful | Harm | Decision |
|---|---:|---:|---:|---:|---|
| `chase_guard` | 98 | 24 | 2 | 20.00% | keep; global relax rejected |
| `open_cluster_cap` | 661 | 51 | 5 | 53.53% | targeted discriminator only |
| `top_gainer_score_gate` | 1539 | 43 | 5 | 53.53% | targeted discriminator only |
| `agent_mode_disabled` | 1168 | 50 | 7 | 57.21% | shadow/targeted replay only |

The reward table assigns a fixed `1%` protection credit to each non-Top-20
candidate. It is a triage proxy, not a tradable PnL estimate.

## Priority Decision

1. Advance `non_ema_positive_giveback` in shadow only and collect live labels.
2. Refine partial exit only inside that observable selector; broad partial is not approved.
3. Keep production re-entry disabled; Top-20 total PnL regressed.
4. Keep BUY admission gates unchanged.
5. Keep chase guard unchanged; test score/cluster/agent harmful cases with narrow discriminators.
6. Treat cooldown `2` as unresolved: opportunity harm falls, but total PnL and giveback worsen.
