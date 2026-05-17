# V2 Hindsight Lifecycle Labeling

Status: research-only planning  
Last updated: 2026-05-17

## Purpose

Define the hindsight labeling contract for symbol lifecycle states before any attempt to
train a state estimator.

## Problem

The v2 store now contains enough continuous history for a first experiment:

- 95 valid symbols;
- 60-day continuous history;
- `15m` and `1h`.

The next risk is to label lifecycle states in a way that merely encodes the desired answer
after the fact or leaks future information into later live-like inference.

Before writing labeling code, we need an explicit contract for what the hindsight labels
mean and how they will be separated from features available at decision time.

## Labeling Principle

Labels are **teacher targets**, not live features.

They may use future information to describe what truly happened, but any later model that
predicts them must be trained only from observations available up to time `t`.

## Initial Lifecycle Labels

For each symbol/day trajectory:

- `noise`
  - no sustained favorable move followed;
- `emerging_move`
  - earliest segment before the future sustained move is fully confirmed;
- `confirmed_trend`
  - sustained favorable move became objectively established;
- `mature_trend`
  - the move continued after confirmation with meaningful additional MFE;
- `exhaustion`
  - post-MFE region where continuation weakens and reversal risk rises;
- `reversal`
  - move has failed or materially retraced after exhaustion.

## Candidate Quantitative Anchors

The first implementation spec must define exact thresholds for:

- minimum same-day favorable move;
- minimum persistence duration;
- confirmed-trend threshold;
- MFE peak detection;
- exhaustion / giveback threshold;
- reset conditions after reversal.

These thresholds must be reported and versioned, not hidden in code.

## Leakage Rule

- label generation may look forward;
- feature generation for future estimators may not;
- every dataset row must preserve both:
  - `label_time`;
  - `observation_cutoff_time`.

## Acceptance Criteria For The Later Implementation

1. Label definitions are deterministic and versioned.
2. Every label can be explained from the future path.
3. A separate audit report shows class balance and transition counts.
4. No lifecycle-label code is used by production trading.
5. State reconstruction experiments must compare against naive baselines.

## Next Gate

Implement the first labeling algorithm and audit:

- label balance;
- transition matrix;
- example trajectories;
- sensitivity to threshold choices.

## Rollback / Safety

- planning only today;
- no production behavior change.

