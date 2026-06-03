# Chase Guard Behavior Replay — 2026-06-03

Status: research-only replay result; no live BUY gate changes

## Trigger

The 2026-06-02 daily learning report flagged `chase_guard` with
`passed_harm_gate`, so it advanced to targeted behavior replay. The guard was not
changed in production.

## Hypotheses Tested

Baseline:

- `score_replace`

Replay-only variants:

- `chase_guard_off`: disable chase guard entirely;
- `chase_guard_rsi_off`: keep daily-range hard guard, disable RSI-overheat subguard;
- `chase_guard_rsi_82`: keep daily-range hard guard, raise RSI max from 76 to 82.

Common parameters:

- `--days 14`
- full watchlist: 105 symbols
- `--top-gainer-score-min 34`
- `--max-open-positions 10`
- `--replace-min-delta 0`
- `--objective-top-n 15`

## Results

| variant | trades | PnL | ΔPnL vs baseline | avg PnL | win rate | ret3 | ret5 | ret10 | repl W/I | skipped full | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `score_replace` | 682 | -157.4089 | +0.0000 | -0.2308 | 0.3798 | 0.0514 | 0.2033 | 0.7997 | 7/13 | 1317 | baseline |
| `chase_guard_off` | 668 | -185.2582 | -27.8493 | -0.2773 | 0.3802 | -0.0328 | 0.1817 | 0.6771 | 11/11 | 1368 | reject |
| `chase_guard_rsi_off` | 682 | -193.5249 | -36.1160 | -0.2838 | 0.3666 | 0.0077 | 0.1482 | 0.6589 | 11/12 | 1353 | reject |
| `chase_guard_rsi_82` | 682 | -193.2837 | -35.8748 | -0.2834 | 0.3666 | 0.0104 | 0.1483 | 0.6589 | 11/12 | 1353 | reject |

`repl W/I` means `replacements_worsened / replacements_improved`.

## Decision

Reject chase guard relaxations. The blocker reward table correctly found some
harmful chase-guard cases, but broad or RSI-only relaxation creates larger
portfolio-level damage in replay.

## Next Step

Do not relax chase guard. Continue with the next priority:

1. exit failure slice search / exit monetization;
2. daily report confidence calibration;
3. replacement shadow monitoring.
