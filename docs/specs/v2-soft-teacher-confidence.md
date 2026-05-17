# V2 Soft Teacher Confidence

Status: research-only  
Last updated: 2026-05-17

## Purpose

Replace the fragile idea of a single binary teacher truth with a confidence-weighted
teacher layer for future state reconstruction.

## Why This Exists

The first sensitivity audit showed:

- lifecycle graph structure is stable;
- the exact count of qualifying trend-days depends strongly on whether the minimum
  favorable move threshold is `3%`, `4%`, or `5%`.

Therefore the next model should not treat every hindsight label as equally certain.

## V1 Confidence Contract

Confidence is attached to each labeled bar and is composed from day-level evidence:

| Component | Meaning |
|---|---|
| `move_strength_score` | how far same-day MFE sits between weak and strong trend thresholds |
| `confirmation_score` | how early the day reached confirmed-trend territory |
| `persistence_score` | how long the path remained active after confirmation |
| `clean_path_score` | penalty when exhaustion / reversal arrives too quickly |

Final confidence is clipped to `[0, 1]`.

## Intended Use

- future reconstruction loss weighting;
- curriculum learning from high-confidence to low-confidence examples;
- audit of ambiguous vs strong teacher examples.

## Not Intended Yet

- no live trading;
- no automatic change to shadow observer;
- no policy reward weighting until reconstruction is validated.

## Acceptance Criteria

1. Every lifecycle label gains deterministic `teacher_confidence`.
2. Non-qualifying `noise` rows retain low confidence rather than fake certainty.
3. Strong clean trend examples score above weak borderline examples.
4. Report shows confidence distribution by lifecycle state and day bucket.
5. Unit tests cover monotonicity and boundedness.

