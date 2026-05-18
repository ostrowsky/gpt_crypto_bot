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
| 8 | Shadow signal observer (expedite) | done | expose provisional v2 lifecycle signals for immediate operator observation | tomorrow's shadow stream is inspectable |
| 9 | Shadow explainability | done | answer quickly why a v2 signal did or did not occur | fast why/why-not lookup works |
| 10 | Shadow daily summary | done | replace raw alert noise with one daily operator view | concise end-of-day visibility |
| 11 | Hindsight lifecycle labeling | done baseline | label hidden-state proxies after the fact | labels pass audit |
| 12 | Lifecycle label sensitivity audit | done | test robustness of teacher labels before training | stable grid evidence |
| 13 | Soft teacher confidence | done | represent teacher uncertainty instead of binary truth | confidence-weighted labels |
| 14 | State reconstruction baseline | done baseline | test whether latent lifecycle can be recovered OOS | macro-F1 beats naive baseline |
| 15 | Belief update / filtering | done v1 | move from labels to live-like belief trajectories | macro-F1 and emerging recall beat isolated baseline |
| 16 | Belief calibration audit | done | tune transition / emission balance before policy learning | balanced filter selected without late-state collapse |
| 17 | Offline decision environment | in progress | expose actions/rewards under portfolio constraints | reproducible offline episodes |
| 18 | Policy baselines + offline RL | in progress | compare rule policy, contextual policy, RL | baseline gap understood before learned policy |
| 19 | Unified runtime integration | done for first worker | ensure any v2 worker starts from the same BAT and reports health | one-command stack startup |
| 20 | Learned shadow policy | pending | replace provisional observer with modeled recommendations | live-shadow evidence |
| 21 | Promotion protocol | pending | define replacement of legacy core safely | explicit go/no-go gate |

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
| Provisional shadow observer | running, shadow-only |
| Fast why/why-not lookup | implemented from compact decision trace |
| Lifecycle labels | baseline + confidence layer available |
| State reconstruction | nearest-centroid OOS macro-F1 0.288 > majority 0.141; emerging recall still weak at 0.285 |
| Belief filtering | calibrated research default selected: self-bias `0.85`, temperature `0.75`; macro-F1 `0.319`, emerging recall `0.388`, reversal recall `0.409` |
| HMM / Bayesian inference | first deterministic filter calibrated; next step is decision-environment construction |
| Offline decision environment | first single-symbol deterministic episode scaffold under construction |
| Policy baselines | first OOS anchors done: oracle `+5867`, naive belief policy `-1864`; action calibration now required before RL |
| Belief-to-action bridge | threshold audit complete: best threshold reward `-554`; useful but still insufficient before RL |
| Policy-gap audit | complete: dominant remaining defect is admission (`1529` noise entries vs `373` emerging entries) |
| Entry admission layer | first v1-enriched dataset built; temporal reuse is useful, exact structural joins are sparse (`0.96%`) |
| V1 structural projection | complete: projected block covers `100%` of OOS rows; useful for mature phases, weak alone for `emerging_move` vs `noise` |
| Admission baseline comparison | first audit complete: projected-v1 improves recall-preserving admission; temporal hard gate collapses early recall |
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
