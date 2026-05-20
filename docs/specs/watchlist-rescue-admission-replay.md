# Watchlist Rescue Admission Replay

Status: research-only implementation  
Owner: Codex  
Date: 2026-05-21

## Problem

The bot often sees watchlist coins before or during a move, but the early
structure is blocked by conservative production gates such as
`top_gainer_score_gate`, `agent_mode_disabled`, `agent_leader_filter`, or
post-exit cooldown. Broad rescue rules were previously rejected because repeated
early blocks also occur on many non-winners.

## Goal

Find a narrow, causal, replay-tested admission profile for coins already in the
active watchlist. The profile should improve early watchlist mover capture
without broadly relaxing BUY rules.

## Inputs

- `files/critic_dataset.jsonl` for causal candidate features and short-horizon
  labels;
- embedded `teacher.final` fields when available for final watchlist-top
  outcome;
- optional focus symbols for operator case review.

## Candidate Profiles

Research profiles are evaluated as shadow selectors only:

- `entry_ok_trend`: existing trend/strong-trend structure that reached
  `entry_ok`;
- `alignment_structural`: alignment state with positive slope, positive MACD,
  controlled RSI, and bounded daily range;
- `surge_followthrough`: surge state with follow-through and range cap;
- `near_miss_momentum`: trend/impulse candidates with enough slope, volume,
  ADX, and non-overheated RSI;
- `watchlist_mover_rescue_v1`: stricter combined structural selector;
- `watchlist_mover_rescue_strict`: higher-confidence version of the combined
  selector.

## Metrics

Report both all-window and chronological holdout metrics:

- selected candidate count;
- unique day/symbol count;
- ret_5 precision;
- ret_5 >= 1% rate;
- average ret_3/ret_5/ret_10;
- final watchlist-top precision when teacher labels exist;
- selected missed final-top candidates;
- focus-symbol examples.

## Promotion Gate

No live behavior is authorized by this script. A profile can only advance to a
real behavior replay if holdout results show:

- support >= 20 selected candidates;
- ret_5 precision >= 55%;
- average ret_5 positive after conservative fees/slippage allowance;
- no obvious collapse in final watchlist-top precision;
- concrete examples include missed or late watchlist winners.

Even then, production adoption requires replay through the normal bot entry,
portfolio, exit, fee, cooldown, and replacement path.

