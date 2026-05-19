# Belief-State V2 Roadmap

Last updated: 2026-05-19

## North Star

Replace the search for a universal BUY mode with a research architecture that:

1. maintains a belief over hidden market-environment states;
2. maintains a belief over hidden symbol lifecycle states;
3. learns how both evolve over time;
4. selects actions under uncertainty conditioned on both beliefs;
5. optimizes the bot's real objective:
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
| 22 | Market-environment belief model | in progress | infer which external game the bot is currently playing | separability before classifier |

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
| Architecture | upgraded: symbol belief + market-environment belief + adaptive policy |
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
| V1 structural projection | complete: projected block covers `100%` of OOS rows; useful for mature phases, not sufficient alone for `emerging_move` vs `noise` |
| Admission baseline comparison | first audit complete: projected-v1 modestly improves recall-preserving admission; temporal hard gate collapses early recall |
| Admission reward replay | first replay complete: combined admission improves reward `-554 -> -384` while preserving `371/373` emerging entries |
| Residual gap decomposition | complete: admission remains weak, but largest next reward lever is exit monetization |
| Exit-quality baselines | first audit complete: earlier transparent exits all lose to base; need better exhaustion discrimination |
| Exhaustion discrimination | complete: interpretable separation exists, but it is not enough by itself for a winning sell rule |
| Exhaustion-aware exit baselines | complete: all four manual hypotheses lost to `base_sell_0_70`; static threshold exits rejected |
| Temporal exit baselines | complete first pass: `mature_decay_late_rise` wins OOS (`+122.90` vs base reward); robustness gate next |
| Temporal exit robustness | complete: local 3x3 grid is positive in `9/9` cells; center `+122.90`, worst `+91.86` |
| Temporal exit window stability | complete: wins `3/4` windows, aggregate `+152.49`, but latest window loses `-100.29` |
| Temporal exit failure-slice audit | complete: latest losing window is noise-dominant / weak-structure, with lower RSI, lower range, below-EMA mean |
| Market-environment taxonomy | defined: policy-oriented hidden states, not simple bullish/bearish labels |
| Market-environment separability | complete first pass: policy-favorable vs unfavorable days are separable enough for a classifier baseline, with small-sample caution |
| Market-environment switched policy | complete first pass: oracle switch strongly wins, first causal prefix classifier loses to both fixed policies |
| Market-environment belief v1 | complete first pass, rejected: rolling belief still loses to both fixed policies |
| Market-environment target design | complete: day labels disagree with future-horizon truth in `35-47%` of samples; use 1h/2h targets next |
| Market-environment horizon-belief diagnostic | complete: 1h/2h targets are better balanced, but current causal classifier is below majority baseline |
| Market observation feature audit | complete: richer features improve 1h to near-majority but still fail the +3pp switched-replay gate |
| Market-environment edge target audit | complete, rejected: no-edge labels still below majority baseline |
| RL | intentionally not started |
| Unified runtime integration | required before live-shadow workers |

## Latest Backtest-Gated Findings

### Exhaustion-aware exit hypotheses

The first four transparent exit hypotheses were replayed on the same OOS sample
(`1710` episodes / `163305` bars) under the same fixed admission layer:

| Profile | Reward delta vs base | Result |
|---|---:|---|
| `late_mass_rsi_weak` | `-203.097729` | reject |
| `late_mass_ema_loss` | `-178.395335` | reject |
| `exhaustion_belief_combo` | `-146.799013` | reject |
| `consensus_exhaustion` | `-271.917323` | reject |

All four reduced average giveback but worsened total reward. This confirms that
the next exit step is **not** more manual threshold tuning. The exit track should
move to temporal / supervised modeling while keeping `base_sell_0_70` as the
control profile.

### Temporal exit hypotheses

The next package replayed short-window trajectory hypotheses on the same OOS
sample:

| Profile | Reward delta vs base | Result |
|---|---:|---|
| `late_mass_acceleration` | `-225.321371` | reject |
| `mature_decay_late_rise` | `+122.901657` | advance |
| `rsi_ema_decay` | `-158.311309` | reject |
| `consensus_temporal` | `-168.251254` | reject |

The winning profile improves total reward by trading less and reducing noisy
entries, not by reducing average giveback. This makes the next gate a
**robustness test around the winning thresholds**, not immediate promotion.

### Temporal exit robustness

The first local sensitivity grid around `mature_decay_late_rise` stayed positive
in all `9 / 9` cells:

- center uplift: `+122.901657`
- best uplift: `+152.487919`
- worst uplift: `+91.861713`

The signal is therefore locally robust rather than a one-cell threshold accident.
The next mandatory gate is time-window / regime stability before any promotion
claim.

### Temporal exit window stability

The leading candidate remains positive on aggregate OOS and wins `3 / 4`
chronological windows, but the latest window (`2026-05-13 -> 2026-05-17`) loses
`-100.286171`.

This is good enough to preserve the candidate, but not enough to promote it.
The next exit step is a failure-slice / regime audit focused on why the latest
window reverses the uplift.

### Temporal exit failure slice

The losing latest window differs coherently from the winning windows:

- noise share `81.55%` vs `70.15%`;
- mature-trend share `4.70%` vs `8.53%`;
- RSI `46.54` vs `50.94`;
- daily range `2.23%` vs `3.12%`;
- mean price-vs-EMA20 already negative.

The candidate still reduces false buys there, but no longer improves giveback and
loses `-82.21` on realized-PnL delta. This supports a new, narrower hypothesis:
the temporal exit family is useful only when market structure is strong enough
to support real mature trends.

## Architecture Reframe

The v2 agent should behave less like a detector with one universal strategy and
more like a game-playing agent that models its opponent.

Its target decision form is:

```text
policy(action | market_environment_belief, symbol_belief, portfolio_state)
```

not:

```text
policy(action | symbol_state)
```

The next architecture track is therefore a **market-environment belief model**:
infer what kind of external game the bot is currently playing, then condition
entry / hold / exit policy on that belief.

### Market-environment separability

The first day-level audit found `14` favorable and `4` unfavorable days and
returned `separable_candidate`.

Important nuance: unfavorable days are not simply "weak markets". Some have
higher visible trend strength than favorable days. That reinforces the correct
target: classify **policy-favorability of the environment**, not a human label
such as bullish / bearish or strong / weak.

### Market-environment switched policy

The full contour now separates two questions cleanly:

- `oracle_switched`: `+498.49` vs fixed base and `+346.00` vs fixed candidate;
- `causal_prefix_switched`: `-64.54` vs fixed base and `-217.03` vs fixed candidate.

Conclusion: adaptation is valuable, but the first causal environment classifier
is not yet competent enough to drive it. The next bottleneck is environment
inference quality, not another hand-written policy family.

### Market-environment belief v1

The first rolling belief implementation also loses:

- `belief_switched = -460.66`
- `-76.18` vs fixed base
- `-228.67` vs fixed candidate

It abstains too long, becomes confident too late, and suffers from prior inertia
under a tiny / imbalanced day-level teacher set. The next step is not another
policy branch, but an environment-belief diagnostic audit.

### Market-environment target design

The current day-level ground truth is too coarse for intraday belief:

- `1h` horizon disagreement: `35.75%`;
- `2h` horizon disagreement: `36.31%`;
- `rest_of_day` disagreement: `46.93%`.

The next environment belief model should train/evaluate against rolling
future-horizon policy advantage, not whole-day labels.

### Market-environment horizon-belief diagnostic

The `1h` / `2h` horizon targets improve sample balance, but current features and
nearest-centroid inference still fail:

- `1h` accuracy `0.5367` vs majority `0.6089`;
- `2h` accuracy `0.5314` vs majority `0.6369`.

So the next bottleneck is the market observation layer, not another switched
policy replay.

### Market observation feature audit

Richer prefix/recent/delta/breadth features improved the `1h` classifier from
`0.5367` to `0.6102` and reduced wrong-confident errors, but it still does not
beat majority by the required margin. `2h` remains below majority.

Therefore no switched replay should be run from this model yet.

### Market-environment edge target audit

The no-edge target hypothesis was tested across `0.5`, `1.0`, `2.0`, `5.0`, and
`10.0` reward-delta thresholds for both `1h` and `2h` horizons.

Result: rejected. The best actionable classifier remained below majority:

- `1h @ 0.5`: accuracy `0.5833` vs majority `0.7239`;
- `2h @ 0.5`: accuracy `0.6039` vs majority `0.6835`.

Higher thresholds increase candidate-edge precision but reduce coverage and make
the class balance easier for the majority baseline. The next bottleneck is not
label reshaping; it is stronger causal market observations.

### Market breadth observation store

The first canonical OHLCV breadth store covered `99` tracked symbols and all
`179` evaluation anchors.

Result: useful but insufficient. `existing + breadth` improved `1h` accuracy to
`0.6215` vs majority `0.6089`, but this is only `+1.25pp`, below the required
`+3pp` gate. `2h` did not improve.

Decision: keep breadth features as a causal observation primitive, but do not use
them yet for switched policy or RL. The next gate should test feature selection /
regularization and conditional breadth use.

### V1 market-structure feature audit

The audit confirmed that v1 already has substantial reusable measurement history:
`6279` `ml_dataset` rows from `2024-07-22` through `2026-05-18`, with `49`
market-structure features at near/full coverage.

The full broad feature sets remain noisy, but compact selected subsets pass the
causal diagnostic gate:

- `1h`: `market_ret4_positive_share` + `prefix_projected_leader_score_trend`
  reaches `0.6497` accuracy vs `0.6089` majority (`+4.08pp`);
- `2h`: `market_btc_ret4_pct` + `market_volume_gt_mean20_share` reaches `0.6800`
  accuracy vs `0.6369` majority (`+4.31pp`).

This validates reusing v1 as an observation source, not copying v1 policy logic.
The next mandatory gate is selected-feature switched replay.

### Selected-feature market switch replay

The selected features passed plain classification gates, but failed reward gates:

- `1h`: switched `+820.61` vs base, but fixed candidate `+881.23`; reject;
- `2h`: switched `-27.29` vs base while fixed candidate is `+86.58`; reject.

This is an important correction: accuracy is not enough for policy selection.
Wrong candidate-favorable predictions are asymmetric and can carry very large
losses. The next selector must be reward-weighted / downside-aware before any
further switched replay claims.

### Reward-weighted market selector

A downside-aware selector was tested over the selected market-structure features.
The result splits by horizon:

- `1h`: best selector still loses to fixed candidate (`+821.60` vs `+881.23`);
- `2h`: `knn_k5_down3.0_thr0.0` wins strongly: `+800.88` vs base and
  `+714.29` vs fixed candidate.

This validates the important design shift: policy selection must be optimized for
reward-weighted mistakes, not plain accuracy. It also suggests the external
market-environment belief should initially operate on a `2h` policy horizon.

Next gate: full offline environment replay for the `2h` reward-weighted selector.

### Reward-weighted selector full offline replay

The full environment replay rejected the selector despite the earlier horizon
replay win:

- fixed base: `-384.48`;
- fixed candidate: `-231.99`;
- reward-weighted market switch: `-451.30`.

This is a useful failure. It shows the current selector is not yet aligned with
path-dependent entry / hold / sell lifecycle. It should not be exposed as an
operator-facing Telegram signal. At most it can be logged as research telemetry
with an explicit `offline_full_replay_failed` gate.

### Selector failure decomposition

The full-replay failure is now localized. The largest loss versus fixed candidate
is `candidate_suppressed_hold`: `-851.23` reward over `2222` frames. The largest
action-pair loss is `switch:sell|candidate:hold` at `-878.59`.

This means the market switch is too coarse. It should not choose between whole
policies while a position is open. The next package should split the problem:

```text
flat-state entry admission selector
position-aware exit/hold selector
```

The immediate next research target is the position-aware exit/hold selector.

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
