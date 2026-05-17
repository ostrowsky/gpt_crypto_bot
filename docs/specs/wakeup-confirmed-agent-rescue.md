# Wake-up Confirmed Agent Rescue

Status: research-only  
Last updated: 2026-05-17

## Purpose

Test whether the market agent should ever rescue an otherwise disabled candidate only
after a meaningful temporal sequence:

1. an earlier structural scout alert;
2. a later `wake_up_1m_light15_v1` alert;
3. then a disabled-mode replay candidate on the same symbol/timeframe.

## Hypothesis

The failed static rescue profile suggests that `agent_mode_disabled` cannot be fixed with
one-candle thresholds alone. If there is useful signal in missed winners, it should be
stronger after the market has shown both a slow structural alert and a later fast wake-up.

## Replay Contract

Variant: `agent_wakeup_rescue`

The variant admits:

- every mode already allowed by `AGENT_ALLOWED_MODES`;
- otherwise disabled `breakout` / `retest` candidates only when:
  - same symbol/timeframe had a non-wake-up `scout_shadow` event first;
  - then a `wake_up_1m_light15_v1` event;
  - both happened before the replay candidate;
  - the structural alert is not older than 24 hours before the wake-up;
  - the wake-up is not older than 6 hours;
  - the candidate also satisfies the narrow static safety profile from
    `docs/specs/agent-mode-rescue-replay.md`.

## Important Limitation

This is a hybrid replay:

- candles are replayed historically;
- structural/wake-up sequence is read from logged `files/bot_events.jsonl`.

Earlier windows without wake-up logging are **coverage-limited**, not negative evidence.

## Metrics

Primary:

- `capture_rate`
- `trade_precision`
- `pnl_total`
- `capture_ratio_at_entry`
- `lead_time_to_final_top_min`

Diagnostic:

- loaded structural events;
- loaded wake-up events;
- temporal rescue admissions.

## Acceptance Rule

Advance only if, on windows with sufficient event coverage:

1. capture improves, or materially earlier capture improves with no capture loss;
2. PnL does not degrade;
3. precision does not materially degrade;
4. rescue admissions are non-zero, so the test is informative.

## Rollback / Safety

- Replay-only.
- No change to `AGENT_ALLOWED_MODES`.
- No live BUY effect.

## First Profile Result

Executed on 2026-05-17 with:

```text
--top-gainer-score-min 34 --max-open-positions 10 --replace-min-delta 0 --objective-top-n 15
```

Wake-up coverage in the local logs currently exists only for:

- 2026-05-15: 10 rows
- 2026-05-16: 72 rows
- 2026-05-17: 200 rows

| Window | Variant | Trades | PnL total | Trade precision | Capture rate | Avg capture ratio at entry | Rescue admissions |
|---|---:|---:|---:|---:|---:|---:|---:|
| 3d | `agent_allowed` | 58 | -22.4467% | 0.3276 | 0.4667 | 0.1903 | n/a |
| 3d | `agent_wakeup_rescue` | 59 | -26.4181% | 0.3390 | 0.5333 | 0.1849 | 10 |
| 7d | `agent_allowed` | 232 | -66.9239% | 0.3147 | 0.9333 | 0.1817 | n/a |
| 7d | `agent_wakeup_rescue` | 230 | -60.3753% | 0.3130 | 0.9333 | 0.1860 | 10 |
| 14d | `agent_allowed` | 622 | -60.2015% | 0.2363 | 1.0000 | 0.2280 | n/a |
| 14d | `agent_wakeup_rescue` | 630 | -58.7279% | 0.2333 | 1.0000 | 0.2258 | 10 |

Decision:

- keep the profile research-only;
- unlike the static rescue profile, the temporal profile is informative and sometimes improves
  capture or PnL;
- it still fails the promotion bar because:
  - 3d capture improves but PnL worsens;
  - 7d/14d PnL improves but capture does not;
  - the current wake-up coverage is only three calendar days, so evidence is too thin.

Next gate:

- collect a longer wake-up history;
- evaluate only on coverage-valid windows;
- inspect the 10 admitted rescue events individually before trying a second temporal profile.
