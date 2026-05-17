# Agent Mode Rescue Replay

Status: research-only  
Last updated: 2026-05-17

## Purpose

Evaluate whether a narrow replay-only rescue path can recover missed same-day top movers
that the market agent currently blocks with `agent_mode_disabled`, without weakening the
production agent entry contract.

## Problem

Recent trend-start funnel reports show that `agent_mode_disabled` is the dominant first
loss point among missed winners. The current agent intentionally allows only:

- `trend`
- `strong_trend`
- `impulse_speed`
- `4h_leader_watch`

This protects precision, but it also means `breakout` and `retest` candidates can never
be promoted by the agent even when the later outcome proves that the symbol became a
same-day top mover.

## Hypothesis

A **narrow** rescue path for otherwise disabled `breakout` / `retest` candidates may
improve early winner capture if it is restricted to structurally strong candidates only.

The first replay profile is intentionally conservative:

- mode is `breakout` or `retest`;
- timeframe is `15m`;
- `top_gainer_score >= 34`;
- `ADX >= 22`;
- `vol_x >= 2.0`;
- `daily_range <= 8.0%`;
- `intraday_change_pct >= 1.0%`.

These conditions are only a candle-replay proxy for the stronger live idea
(`first structural alert -> wake-up -> candidate rescue`). The wake-up dependency is not
yet encoded in historical replay, so this first profile must not be treated as production
evidence by itself.

## Scope

### In scope

- Add replay variant `agent_allowed` to model current agent mode restrictions.
- Add replay variant `agent_mode_rescue` to admit only the narrow disabled-mode profile above.
- Compare both variants with the same score gate, portfolio cap, fees, and objective metrics.

### Out of scope

- Any production change to `AGENT_ALLOWED_MODES`.
- Any live BUY bypass.
- Any broad relaxation of score, ADX, volume, or watch-alert rules.

## Metrics

Primary:

- `capture_rate`
- `trade_precision`
- `pnl_total`
- `capture_ratio_at_entry`
- `lead_time_to_final_top_min`

Secondary:

- `false_positive` proxy via non-objective symbols
- `giveback_pct`
- `exit_efficiency`

## Acceptance Rule

The rescue profile may advance only if, on valid replay windows:

1. `capture_rate` improves versus `agent_allowed`;
2. `trade_precision` does not materially degrade;
3. `pnl_total` does not degrade;
4. earlier-capture metrics improve or remain neutral;
5. the effect persists beyond one short window.

If it improves recall but hurts PnL/precision, it remains rejected or diagnostic-only.

## Rollback / Safety

- Replay-only code path.
- No config change to live agent behavior.
- No production promotion without a later spec update, replay evidence, and explicit approval.

## First Profile Result

Executed on 2026-05-17 with:

```text
--top-gainer-score-min 34 --max-open-positions 10 --replace-min-delta 0 --objective-top-n 15
```

| Window | Variant | Trades | PnL total | Trade precision | Capture rate | Avg capture ratio at entry |
|---|---:|---:|---:|---:|---:|---:|
| 3d | `agent_allowed` | 58 | -22.4467% | 0.3276 | 0.4667 | 0.1903 |
| 3d | `agent_mode_rescue` | 62 | -23.8278% | 0.3065 | 0.4667 | 0.1853 |
| 7d | `agent_allowed` | 232 | -66.9239% | 0.3147 | 0.9333 | 0.1817 |
| 7d | `agent_mode_rescue` | 238 | -65.9412% | 0.3151 | 0.9333 | 0.1845 |
| 14d | `agent_allowed` | 622 | -60.2015% | 0.2363 | 1.0000 | 0.2280 |
| 14d | `agent_mode_rescue` | 649 | -71.5689% | 0.2388 | 1.0000 | 0.2381 |

Decision:

- reject the first rescue profile for promotion;
- it adds trades but does not improve capture on any tested window;
- on 14d it materially worsens total PnL despite slightly better entry timing;
- the next profile must include stronger temporal evidence than a static candle proxy,
  ideally an explicit historical wake-up / structural sequence rather than broader gate relaxation.
