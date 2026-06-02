# Portfolio Replacement Behavior Replay

Date: 2026-06-02
Status: research-only replay variants; no live rotation changes

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

- Do not change live `market_signal_agent.py` replacement behavior.
- Do not use future top-mover labels in the replay gate.
- Compare against the current `score_replace` behavior before promotion.
- If fresh Binance candle replay is unavailable, report the replay as blocked
  and keep the event-log shadow simulation as hypothesis prioritization only.
- Production flag must default OFF. Shadow logging may run while live behavior
  remains unchanged.

## Promotion Gate

A replacement restriction may advance toward production only if replay shows:

- no material degradation in watchlist top capture / early capture;
- lower replacement churn or better replacement PnL;
- no unacceptable increase in skipped winners under a full portfolio;
- stable result over more than one replay window.

## Implementation Gate

After replay evidence on 2026-06-02, the next allowed implementation step is:

- add `AGENT_REPLACEMENT_BLOCK_NON_LOSING_ENABLED = False`;
- add `AGENT_REPLACEMENT_BLOCK_NON_LOSING_SHADOW = True`;
- log `replacement_policy_shadow` events for candidates that would be blocked;
- do not block live replacements unless the enable flag is explicitly turned on.
