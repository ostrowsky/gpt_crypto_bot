# V2 Lifecycle Label Sensitivity Audit

Status: research-only  
Last updated: 2026-05-17

## Purpose

Test whether the first hindsight lifecycle labels are stable enough to become teacher
targets for state reconstruction.

## Why This Gate Exists

If small threshold changes completely rewrite the teacher labels, then a later HMM or RL
agent would be trained on an arbitrary taxonomy rather than a durable market structure.

## Grid

The first audit varies the most decision-critical thresholds:

| Parameter | Values |
|---|---|
| minimum favorable move | `3.0`, `4.0`, `5.0` |
| confirmed move | `1.5`, `2.0`, `2.5` |
| exhaustion giveback ratio | `0.30`, `0.35`, `0.40` |

All other v1 thresholds remain fixed.

## Required Outputs

- row count and qualifying-day count per variant;
- per-state class balance;
- invalid transition count;
- deltas versus baseline `4.0 / 2.0 / 0.35`;
- compact ranking of stable vs unstable dimensions.

## Acceptance Criteria

1. Every variant preserves valid state-graph transitions.
2. The baseline is not an isolated outlier.
3. Class balance changes are explainable and monotonic with thresholds.
4. The audit is complete before any state-reconstruction training.

