# V2 V1 Structural Feature Projection

Status: research-only  
Last updated: 2026-05-18

## Purpose

Project the mature *language* of v1 structural admission onto every v2 OOS frame,
instead of depending only on sparse historical rows where v1 happened to log a
candidate.

## Why This Exists

The first entry-admission dataset showed:

- exact v1 structural joins: `0.958%` of OOS rows;
- prior structural scout evidence: `48.224%` of OOS rows.

That is enough to reuse v1 temporal evidence, but not enough to train or compare a
mainline admission layer with v1 structural context.

## Projection Contract

The projection is **not** a replay of the full v1 BUY cascade. It is an additive,
research-only feature block computed from canonical v2 bars using v1-inspired
formulas:

- `projected_today_change_pct`;
- `projected_forecast_proxy_pct`;
- `projected_candidate_score_trend`;
- `projected_candidate_score_impulse_speed`;
- `projected_leader_score_trend`;
- raw structural inputs already shared with v1:
  - slope;
  - ADX;
  - RSI;
  - `vol_x`;
  - daily range;
  - price versus EMA20.

## Guardrails

1. Do not call the projected score a production v1 score.
2. Do not treat the projection as a BUY gate.
3. Do not use future bars.
4. Keep exact historical v1 joins separately; projected and observed-v1 features
   must not be silently conflated.

## Acceptance Criteria

1. Projection coverage is close to all valid OOS rows.
2. Projection uses only features available at the current frame.
3. Output reports projected feature averages by true lifecycle state.
4. Dataset keeps both:
   - `v1_structural` exact historical join;
   - `v1_projected_structural` dense projected block.

## Next Gate

Compare admission baselines:

1. belief-only;
2. projected-v1-only;
3. belief plus projected-v1;
4. later, belief plus projected-v1 plus temporal evidence.

## First Audit Result

The projected feature block reached full OOS coverage:

- rows with exact observed v1 structural join: `1565` (`0.958%`);
- rows with projected v1 structural block: `163305` (`100%`).

Selected projected means by true lifecycle state:

| State | Forecast proxy | Leader score trend |
|---|---:|---:|
| `noise` | `0.469` | `2.959` |
| `emerging_move` | `0.610` | `3.389` |
| `confirmed_trend` | `1.555` | `8.143` |
| `mature_trend` | `3.954` | `21.360` |
| `exhaustion` | `2.981` | `19.234` |

Interpretation:

- v1-style strength features are reusable densely;
- they separate established trend phases well;
- they only weakly separate `emerging_move` from `noise`;
- admission still needs belief dynamics and temporal evidence, not just projected
  v1 strength scores.
