# V2 Market-Environment Taxonomy

Status: research-only  
Last updated: 2026-05-18

## Purpose

Define the first hidden-state vocabulary for the external environment that the
bot is playing against.

## First Taxonomy

| State | Meaning for policy |
|---|---|
| `continuation_favorable` | broad enough structure exists that early trends often continue and trend-hold / temporal exit logic can monetize |
| `mixed_rotation` | opportunities exist, but leadership is narrow or rotating; selection quality matters more than raw aggression |
| `noise_dominant` | most apparent starts decay into noise; admission should be stricter and trend-following policies should be discounted |
| `risk_off_decay` | downside / reversal pressure dominates; preservation and avoidance matter more than capture |

## Important Distinction

These are latent policy-relevant states, not merely BTC up/down or green/red
breadth. The future classifier should estimate which policy family is currently
likely to receive reward.

## Candidate Observations

- universe-level noise / emerging / confirmed / mature / exhaustion shares;
- mean projected leader score;
- mean projected forecast proxy;
- mean RSI;
- mean ADX;
- mean daily range;
- mean price-vs-EMA20.

## First Research Labels

Before a learned latent model exists, use weak policy-favorability labels:

- `candidate_favorable`: temporal exit candidate beats control for the day;
- `candidate_unfavorable`: it loses to control for the day.

These are diagnostic teacher signals, not final environment labels.

## Next Gate

Run a separability audit to test whether market-day observations contain enough
signal to justify a first environment classifier baseline.
