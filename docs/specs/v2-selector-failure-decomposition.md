# V2 Selector Failure Decomposition

Status: research-only  
Last updated: 2026-05-19

## Purpose

The reward-weighted market selector won the isolated `2h` horizon replay but lost
inside the full offline decision environment. This spec decomposes that failure
before any new selector or RL work.

The goal is to answer:

```text
Why did a horizon-winning selector become a full-rollout loser?
```

## Questions

The audit must separate at least these loss sources:

- no market selector choice available yet;
- stale selector choice carried forward too long;
- candidate policy suppressed while flat;
- candidate policy suppressed while already in position;
- candidate policy enabled when base would have avoided loss;
- action mismatch on open / hold / sell;
- state-specific losses by true lifecycle state.

## Protocol

Use the same offline contour as
`v2-reward-weighted-market-selector-offline-replay.md`:

- fixed base policy;
- fixed temporal candidate policy;
- reward-weighted market switch.

For each aligned symbol-day frame:

1. record latest selector choice and its age;
2. compare switch action / reward with fixed candidate;
3. compare switch action / reward with fixed base;
4. aggregate deltas by source bucket;
5. keep compact worst examples for inspection.

## Output

The report must include:

- summary totals;
- decomposition vs fixed candidate;
- decomposition vs fixed base;
- stale-age buckets;
- lifecycle-state buckets;
- action-pair buckets;
- top negative examples.

## Acceptance Criteria

This audit is diagnostic-only. It does not authorize Telegram or live behavior.

It passes if it identifies a dominant failure class that can drive the next
research package.


## Audit Result

Command:

```powershell
.\pyembed\python.exe files\audit_v2_selector_failure_decomposition.py
```

### Summary

The dominant failure is not stale market selection. The selector loses because it
uses the base branch in moments where the fixed candidate would have continued to
hold.

Worst buckets versus fixed candidate:

| Bucket | Reward delta | Count |
|---|---:|---:|
| `candidate_suppressed_hold` | `-851.227112` | `2222` |
| `no_selector_choice` | `-56.724432` | `28690` |
| `candidate_suppressed_sell` | `-36.685301` | `198` |
| `action_mismatch_candidate_in_position_only_or_partial_candidate` | `-25.647868` | `1394` |

Worst action-pair bucket:

| Action pair | Reward delta |
|---|---:|
| `switch:sell|candidate:hold` | `-878.585191` |

Lifecycle state losses versus fixed candidate:

| State | Reward delta |
|---|---:|
| `noise` | `-138.136052` |
| `confirmed_trend` | `-60.122075` |
| `emerging_move` | `-22.327078` |
| `reversal` | `-0.186664` |
| `exhaustion` | `+0.464847` |
| `mature_trend` | `+0.996658` |

Staleness is not the main explanation versus fixed candidate:

| Age bucket | Reward delta |
|---|---:|
| `age_0_1` | `-419.364463` |
| `age_2_4` | `-166.470045` |
| `age_5_8` | `+423.248576` |
| `no_choice` | `-56.724432` |

### Interpretation

The market switch is too coarse. It chooses between whole policies, so when it
selects `base`, it can prematurely sell positions where the temporal candidate
would hold. That converts a useful horizon-level market signal into a bad
path-dependent policy switch.

The failure is mostly **exit/hold suppression**, not entry admission and not stale
anchor carry.

## Decision

The next research package should split the selector into separate responsibilities:

```text
entry admission selector != exit/hold selector
```

The immediate next target is a **position-aware exit selector** that only decides
whether to apply temporal exits while a position is already open. It should not
control flat-state entry admission.

## Telegram Implication

Do not expose this selector as a real-time Telegram signal. If surfaced at all,
it must be diagnostic-only and explicitly say:

```text
gate: full_replay_failed
reason: suppresses candidate hold behavior
```
