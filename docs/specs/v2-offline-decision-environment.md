# V2 Offline Decision Environment

Status: research-only  
Last updated: 2026-05-18

## Purpose

Create the first reproducible environment where v2 policies can be evaluated as
sequential decision makers rather than as isolated BUY-mode classifiers.

This package turns calibrated belief trajectories plus canonical bars into
deterministic symbol-day episodes with:

- explicit legal actions;
- position state;
- named reward components;
- deterministic replay of the same market path for different policies.

## Why Now

The belief filter is now calibrated enough to use as research input:

- balanced default: `self_bias=0.85`, `temperature=0.75`;
- OOS macro F1: `0.319`;
- `emerging_move` recall: `0.388`;
- `reversal` recall: `0.409`.

The next bottleneck is no longer state estimation alone. It is whether actions chosen
from those beliefs improve the bot's objective over full trajectories.

## First Slice

### In scope

- symbol-day episodes built from canonical bars and filtered belief rows;
- local one-symbol position lifecycle:
  - flat;
  - long;
- legal-action checks;
- deterministic action stepping;
- reward decomposition using the existing v2 reward contract;
- portfolio-cap hook via `max_open_positions`, even though the first tests use a
  single-symbol episode.

### Out of scope

- live trading imports;
- learned policy fitting;
- order-book fills / latency;
- multi-symbol allocator;
- slippage model beyond future explicit extension;
- promotion into production.

## Episode Contract

Each frame contains:

- canonical bar;
- hindsight lifecycle label;
- filtered belief;
- belief prediction.

The environment exposes:

- current frame;
- current position count;
- legal actions;
- deterministic `step(action)`.

## Reward Semantics

The first implementation intentionally reuses the named reward decomposition from
`files/v2/reward.py`:

- opening on a qualifying winner can earn early-capture reward;
- holding through `confirmed_trend` / `mature_trend` can earn trend-hold reward;
- selling realizes PnL and MFE-retention reward;
- late entry, giveback, churn, and false-buy penalties remain explicit.

This is not yet the final reward function. It is the first transparent environment
where reward-shaping mistakes can be inspected before any RL is introduced.

## Acceptance Criteria

1. Same episode + same actions -> same result.
2. Illegal actions are rejected explicitly.
3. Opening a position respects `max_open_positions`.
4. Reward remains decomposed, not hidden inside one scalar.
5. Tests cover:
   - legal-action surface;
   - open / hold / sell path;
   - deterministic episode reset.
6. No production module imports the environment.

## Next Gate

After this package:

1. build fixed policy baselines over the environment;
2. compare lifecycle-oracle, belief-driven, and always-flat policies;
3. only then add richer multi-symbol portfolio episodes and offline RL.
