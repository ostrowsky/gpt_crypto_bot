# Belief-State V2 Architecture

Status: research-only  
Last updated: 2026-05-17

## Purpose

Start a clean research core for the next architecture of the bot without disturbing the
current production system.

The old line of work searches for better BUY modes and static gates. The v2 line instead
models trading as sequential decision-making under partial observability:

1. infer a belief over hidden **market-environment** states;
2. infer a belief over hidden symbol lifecycle states;
3. choose an action conditioned on both beliefs plus portfolio state;
4. learn from intraday and end-of-day rewards aligned with the bot objective.

## Problem

No universal early-trend mode exists for all coins and all market regimes. Early trend
formation, noisy rebounds, isolated pumps, broad market beta, trend continuation, and
exhaustion can be observationally similar at the first bars.

The right problem is therefore not:

> "Which universal BUY signal detects every early trend?"

It is:

> "Given noisy observations and changing market regimes, what hidden state is most
> plausible now, and which action has the best expected long-term utility?"

## Design Principles

1. **Greenfield core inside the existing repository**
   - reuse data, evaluators, reports, operational tooling, and objective metrics;
   - do not reuse the legacy mode/gate cascade as the new decision model.
2. **Research-only isolation**
   - v2 modules are not imported by the live trading path;
   - no BUY/SELL behavior changes in this phase.
3. **Environment belief before policy**
   - the bot is not playing one stationary game;
   - first infer what kind of market environment is currently "playing against it";
   - then condition symbol-level policy on that environment belief.
4. **Objective-aligned rewards**
   - rewards must reflect early top-mover capture, hold quality, exit quality, fees,
     and portfolio opportunity cost, not only short-horizon PnL.
5. **Offline before online**
   - sequence datasets, walk-forward validation, and shadow recommendations precede
     any live promotion.

## Initial Module Boundary

```text
files/v2/
  state.py       # hidden state vocabularies and graph contracts
  belief.py      # belief-state container and deterministic update helpers
  reward.py      # objective-aligned action reward decomposition
  dataset.py     # sequence / transition data contracts
  __init__.py
```

Later modules, intentionally not built yet:

```text
files/v2/
  inference/     # HMM / Bayesian / particle-filter implementations
  env/           # offline RL environment
  policy/        # offline RL / contextual policy
  replay/        # v2-specific counterfactual simulator
```

## Initial State Model

### Symbol lifecycle states

- `noise`
- `emerging_move`
- `confirmed_trend`
- `mature_trend`
- `exhaustion`
- `reversal`

### Market-environment states

- `continuation_favorable`
- `mixed_rotation`
- `noise_dominant`
- `risk_off_decay`

These are policy-oriented latent states: they describe what kinds of actions the
environment tends to reward, not merely whether BTC is green or red.

### Actions

- `ignore`
- `watch`
- `elevate_priority`
- `reserve_slot`
- `open_small`
- `open_full`
- `hold`
- `tighten_exit`
- `reduce`
- `sell`

## Reward Contract

The first research reward decomposition includes:

- `early_capture_reward`
- `trend_hold_reward`
- `realized_pnl_reward`
- `mfe_retention_reward`
- `false_buy_penalty`
- `late_entry_penalty`
- `giveback_penalty`
- `churn_penalty`
- `blocked_winner_penalty`

The contract is decomposed on purpose so later training reports can show *why* an agent
won or lost reward instead of hiding the answer inside one scalar.

## Scope Of This First Package

### In scope

- architecture spec;
- pure enums / graph contracts;
- pure belief-state container;
- pure reward decomposition helper;
- typed dataset records for future sequence building;
- unit tests.

### Out of scope

- fitting HMM parameters;
- MCMC sampling;
- RL training;
- production imports;
- automatic replacement of legacy BUY/SELL logic.

## Acceptance Criteria

1. `files/v2/` exists as an importable research-only package.
2. Symbol and market state graphs are explicit and validated.
3. Belief distributions can be normalized, queried, and safely updated.
4. Reward calculation is decomposed into named components.
5. Dataset contracts can represent observations, actions, and transitions.
6. Unit tests pass.
7. No live code imports the v2 package.

## Verification

- `python -m unittest test_v2_core.py`
- repo grep showing no production import of `v2`

## Target Decision Form

```text
action = pi(market_environment_belief, symbol_belief, portfolio_state)
```

not:

```text
action = pi(symbol_state)
```

This is analogous to an agent that adapts its game plan to the inferred style or
strength of an opponent rather than using the same strategy against everyone.

## Next Gate

After this package:

1. build a sequence dataset from existing event logs and OHLCV;
2. label hindsight lifecycle transitions;
3. evaluate whether symbol lifecycle states can be reconstructed out-of-sample;
4. build the first market-environment observability audit before introducing RL.

## Rollback / Safety

- delete `files/v2/` and this spec with no effect on production;
- no config switch is required because the package is inert until explicitly imported.
