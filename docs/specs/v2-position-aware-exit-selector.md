# V2 Position-Aware Exit Selector

Status: research-only  
Last updated: 2026-05-19

## Purpose

The selector failure decomposition showed that the broad market switch loses
mostly by suppressing useful candidate holds:

```text
candidate_suppressed_hold = -851.227112
switch:sell | candidate:hold = -878.585191
```

This spec tests a narrower selector that acts only while a position is already
open. It does not control flat-state entry admission.

## Hypothesis

A position-aware selector can keep the fixed candidate's useful hold behavior
while selectively taking base exits only when the position context shows high
downside risk.

The decision form is:

```text
if flat:
    use fixed candidate entry policy
if in position:
    choose candidate hold/exit unless a local exit-risk gate says base is safer
```

## Candidate Inputs

Use only causal frame-local information:

- current symbol belief:
  - late mass = exhaustion + reversal;
  - mature mass;
  - emerging + confirmed mass;
- v1 projected structure:
  - price vs EMA20;
  - RSI;
  - ADX;
  - leader score;
  - daily range;
- temporal deltas:
  - late mass delta 3 bars;
  - mature delta 3 bars;
  - RSI delta 3 bars;
  - price-vs-EMA20 delta 3 bars.

## Replay Protocol

Compare:

- fixed base;
- fixed temporal candidate;
- position-aware selector variants.

The selector must keep fixed candidate behavior while flat. It may override the
candidate action while in position only when the base action is `SELL` and risk
conditions are met.

## Acceptance Criteria

A selector advances only if:

- total reward beats fixed candidate;
- trade count does not exceed fixed candidate by more than 10%;
- average giveback does not materially worsen;
- the result is explainable by risk buckets, not a single accidental threshold.

Passing this spec still does not authorize live BUY / SELL behavior.


## Replay Result

Command:

```powershell
.\pyembed\python.exe files\audit_v2_position_aware_exit_selector.py
```

| Policy | Total reward | Delta vs fixed candidate | Trades | Avg giveback penalty |
|---|---:|---:|---:|---:|
| fixed candidate | `-231.993036` | `0.000000` | `1957` | `-2.119644` |
| `base_override_ema_break` | `-254.690952` | `-22.697916` | `1984` | `-2.094932` |
| `base_override_strict_break` | `-260.626587` | `-28.633551` | `1966` | `-2.116599` |
| `base_override_decay` | `-270.549175` | `-38.556139` | `1974` | `-2.109146` |
| `base_override_late_accel` | `-429.314552` | `-197.321516` | `2180` | `-1.974126` |
| `base_override_late_mass` | `-436.760144` | `-204.767108` | `2182` | `-1.974588` |

## Interpretation

The narrower position-aware override is much less harmful than the broad market
switch, but still does not beat the fixed temporal candidate. The best profile
improves average giveback slightly, but loses total reward.

This means manual exit-risk thresholds are not enough. The next target should be
an action-level advantage dataset for position states:

```text
when in position, what is the realized advantage of SELL now vs HOLD / temporal candidate?
```

## Decision

Rejected for promotion.

Do not expose as Telegram signal. Continue research with action-level advantage
labels rather than another hand-written threshold profile.
