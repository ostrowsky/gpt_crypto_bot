# Belief-State V2 Roadmap

Last updated: 2026-05-17

## North Star

Replace the search for a universal BUY mode with a research architecture that:

1. maintains a belief over hidden symbol / market states;
2. learns how states evolve over time;
3. selects actions under uncertainty;
4. optimizes the bot's real objective:
   - earlier same-day top-mover capture;
   - better use of the unified 10-slot portfolio;
   - better MFE retention near trend exhaustion.

## Current Decision

`v2` is a **greenfield research core inside the existing repository**:

- old production bot remains operational;
- old telemetry, evaluator, reports, and objective metrics are reused;
- new decision logic is built separately under `files/v2/`;
- no production promotion before offline validation and shadow evidence.

## Roadmap

| Phase | Package | Status | Why it exists | Exit gate |
|---|---|---:|---|---|
| 0 | Belief-state architecture core | done | define states, belief, rewards, dataset contracts | package inert, tests pass |
| 1 | Sequence dataset builder | done | turn legacy rows into ordered sequences/transitions | coverage is measurable |
| 2 | Sequence coverage audit | done | determine whether existing history is fit for modeling | coverage limitations explicit |
| 3 | Canonical market history contract | done | define clean continuous OHLCV abstraction | continuity is explicit |
| 4 | Canonical history source/store | done | create the actual v2 source for continuous history | repeatable slices with provenance |
| 5 | History coverage backfill plan | done | fill enough continuous history for learning | training-grade windows defined |
| 6 | Canonical history population adapter | done | actually populate 60d canonical windows | adapter validated on real source |
| 7 | Full-history population run | done | fill the store for watchlist / 60d target | 95/105 symbols valid, 90.48% |
| 8 | Hindsight lifecycle labeling | now | label hidden-state proxies after the fact | labels pass audit |
| 9 | State reconstruction baseline | pending | test whether latent lifecycle can be recovered OOS | baseline beats naive states |
| 10 | Belief update / filtering | pending | move from labels to live-like belief trajectories | calibrated belief quality |
| 11 | Offline decision environment | pending | expose actions/rewards under portfolio constraints | reproducible offline episodes |
| 12 | Policy baselines + offline RL | pending | compare rule policy, contextual policy, RL | walk-forward uplift |
| 13 | Unified runtime integration | pending | ensure any v2 worker starts from the same BAT and reports health | one-command stack startup |
| 14 | Shadow policy | pending | recommend actions without trading | live-shadow evidence |
| 15 | Promotion protocol | pending | define replacement of legacy core safely | explicit go/no-go gate |

## What We Have Learned So Far

1. A universal early-trend mode is the wrong abstraction.
2. The current local `ml_dataset` is not a valid primary sequence source for `v2`:
   - `15m`: 1,292 observations, 0 usable transitions;
   - current rows are useful diagnostics, not canonical history.
3. Therefore the critical path is now:

```text
continuous history -> hindsight labels -> state reconstruction -> belief -> policy
```

not:

```text
new BUY mode -> more replay tuning
```

## Progress Snapshot

| Dimension | Current state |
|---|---|
| Architecture | defined |
| Production isolation | preserved |
| Sequence contract | defined |
| Existing history quality | measured, insufficient for 15m state modeling |
| Canonical OHLCV contract | defined |
| Canonical continuous history source | implemented as local research store |
| Canonical 60d history coverage | passed minimum gate on 95/105 symbols |
| Lifecycle labels | not started |
| HMM / Bayesian inference | not started |
| RL | intentionally not started |
| Unified runtime integration | required before live-shadow workers |

## Runtime Rule

Offline-only v2 scripts may exist independently while the architecture is still in
research. The moment v2 gains a mandatory long-running worker, the single operator
entrypoint must remain:

```text
restart_full_stack.bat
```

Any such worker must ship with:

- its own background launcher;
- its own status check;
- integration into `restart_full_stack.bat`;
- verification from the clean release worktree.
