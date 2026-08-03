# Portfolio Replacement Behavior Replay

Date: 2026-06-02
Last revalidated: 2026-07-17
Status: `replacement_block_non_losing` enabled in production after 30d/14d replay and live shadow validation

## Problem

The executed replacement shadow reward report shows that current portfolio
rotation is often harmful, especially when replacing positions that are not yet
losing.

## Goal

Add replay-only variants that test causal replacement restrictions:

- `replacement_block_non_losing`: do not replace a position whose current PnL is
  non-negative;
- `replacement_block_leader_delta_lt_10`: do not replace unless candidate leader
  edge is at least 10 points;
- `replacement_block_non_losing_unless_delta20`: do not replace a non-losing
  position unless candidate leader edge is at least 20 points.

## Guardrails

- Do not change live `market_signal_agent.py` replacement behavior before the
  maximum-period and fresh stability gates pass.
- Do not use future top-mover labels in the replay gate.
- Compare against the current `score_replace` behavior before promotion.
- If fresh Binance candle replay is unavailable, report the replay as blocked
  and keep the event-log shadow simulation as hypothesis prioritization only.
- Production flag must default OFF. Shadow logging may run while live behavior
  remains unchanged.
- Candidate and control variants in a revalidation must share one frozen candle
  cache, symbol universe, time window, feature computation, and final-top
  denominator. Historical downloads are concurrent but simulation order stays
  deterministic. `--compare-variant score_replace` provides this paired run.

## Promotion Gate

A replacement restriction may advance toward production only if replay shows:

- no material degradation in watchlist top capture / early capture;
- lower replacement churn or better replacement PnL;
- no unacceptable increase in skipped winners under a full portfolio;
- stable result over more than one replay window.

## Implementation Gate

After replay evidence on 2026-06-02, the next allowed implementation step was:

- add `AGENT_REPLACEMENT_BLOCK_NON_LOSING_ENABLED = False`;
- add `AGENT_REPLACEMENT_BLOCK_NON_LOSING_SHADOW = True`;
- log `replacement_policy_shadow` events for candidates that would be blocked;
- do not block live replacements unless the enable flag is explicitly turned on.

## 2026-07-17 Production Gate

The shadow cohort reached `153` `replacement_policy_shadow` observations. The
executed-replacement reward report also grew from `269` to `463` closed incoming
outcomes. In that expanded causal diagnostic, replacing a non-losing position
remained harmful on average (`-0.261622%`, median `-0.401%`, `n=275`).

The policy was then replayed against `score_replace` over the maximum feasible
portfolio window and a fresh stability window. Both runs used the same current
`105`-symbol watchlist, `15m + 1h` decisions with `4h` context, portfolio size
`10`, replacement leader delta `0`, Top-20 objective, and score floor `34`.

| Window | Metric | `score_replace` | `block_non_losing` | Delta |
|---|---:|---:|---:|---:|
| 30d | total PnL | `-301.0831%` | `-295.7474%` | `+5.3357%` |
| 30d | average PnL | `-0.2081%` | `-0.2068%` | `+0.0013%` |
| 30d | Top-20 capture | `100%` | `100%` | `0.0pp` |
| 30d | trade precision | `27.23%` | `27.27%` | `+0.04pp` |
| 30d | improved / worsened replacements | `30 / 16` | `38 / 0` | all harmful replacements removed |
| 14d | total PnL | `-174.4783%` | `-154.4616%` | `+20.0167%` |
| 14d | average PnL | `-0.2652%` | `-0.2351%` | `+0.0301%` |
| 14d | Top-20 capture | `95%` | `95%` | `0.0pp` |
| 14d | trade precision | `31.16%` | `31.66%` | `+0.50pp` |
| 14d | improved / worsened replacements | `7 / 9` | `14 / 0` | all harmful replacements removed |

The candidate slightly reduced win rate (`-0.74pp` on 30d, `-0.09pp` on 14d)
and slightly increased average giveback (`+0.0144%` and `+0.0208%`). Those
trade-offs are accepted because both replay windows improved total and average
PnL, preserved Top-20 capture, did not reduce precision, and removed all
replay-classified harmful replacements.

Decision:

- set `AGENT_REPLACEMENT_BLOCK_NON_LOSING_ENABLED = True`;
- keep `AGENT_REPLACEMENT_BLOCK_NON_LOSING_SHADOW = True` for post-deploy audit;
- keep losing-position replacement behavior unchanged;
- do not generalize this result into broader leader-delta or cluster rules.

## 2026-08-03 Retention Revalidation

The guard and the former `score_replace` behavior were rerun on one frozen
candle/feature cache for each window, using the current 103-symbol watchlist,
`15m + 1h` decisions with `4h` context, portfolio size 10, score floor 34,
replacement delta 0, and Top-20 objective.

| Window | Metric | guard | `score_replace` | Guard delta |
|---|---:|---:|---:|---:|
| 30d | total PnL | `-203.2424%` | `-214.0077%` | `+10.7653%` |
| 30d | average PnL | `-0.1234%` | `-0.1299%` | `+0.0065%` |
| 30d | Top-20 capture | `95%` | `95%` | `0.0pp` |
| 30d | win rate | `38.7%` | `39.6%` | `-0.9pp` |
| 14d | total PnL | `-98.3883%` | `-90.6671%` | `-7.7212%` |
| 14d | average PnL | `-0.1242%` | `-0.1158%` | `-0.0084%` |
| 14d | Top-20 capture | `95%` | `95%` | `0.0pp` |
| 14d | win rate | `38.0%` | `39.2%` | `-1.2pp` |

The maximum window supports the guard, while the fresh stability window
reverses both PnL metrics. Raw same-horizon candidate/protected pairs therefore
remain a warning, not sufficient rollback evidence. The retention gate is
conflicted: keep the current production guard unchanged, do not broaden it,
and require another independent stability window before either confirmation or
rollback.
