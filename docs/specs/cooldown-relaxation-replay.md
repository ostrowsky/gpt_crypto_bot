# Post-exit Cooldown Relaxation Replay

Date: 2026-07-17
Status: `COOLDOWN_BARS=2` enabled after maximum-period and stability replay

## Goal

Reduce missed continuation entries after an exit without sacrificing the
Top-20 objective, entry precision, or portfolio PnL.

## Scope

- Compare `COOLDOWN_BARS=8` with `COOLDOWN_BARS=2`.
- Hold the current production replacement policy
  (`replacement_block_non_losing`) and every other replay parameter constant.
- Use the current `103`-symbol watchlist, `15m + 1h` decisions with `4h`
  context, portfolio size `10`, replacement delta `0`, score floor `34`, and
  Top-20 objective.
- Do not change `AGENT_MAIN_EXIT_COOLDOWN_BARS`; cross-layer market-agent
  re-entry requires a separate replay.

## Acceptance Gate

The relaxation may ship only if both the maximum `30d` window and a fresh `14d`
stability window improve total PnL, average PnL, win rate, trade precision, and
cooldown harm while preserving Top-20 capture.

## Results

| Window | Metric | 8 bars | 2 bars | Delta |
|---|---:|---:|---:|---:|
| 30d | total PnL | `-294.0253%` | `-180.7267%` | `+113.2986%` |
| 30d | average PnL | `-0.2121%` | `-0.1137%` | `+0.0984%` |
| 30d | win rate | `37.09%` | `38.14%` | `+1.05pp` |
| 30d | trade precision | `27.78%` | `29.14%` | `+1.36pp` |
| 30d | Top-20 capture | `100%` | `100%` | `0.0pp` |
| 30d | cooldown harm | `1896.7100%` | `646.1536%` | `-1250.5564%` |
| 14d | total PnL | `-149.8721%` | `-87.2356%` | `+62.6365%` |
| 14d | average PnL | `-0.2316%` | `-0.1180%` | `+0.1136%` |
| 14d | win rate | `37.09%` | `38.84%` | `+1.75pp` |
| 14d | trade precision | `33.85%` | `36.13%` | `+2.28pp` |
| 14d | Top-20 capture | `100%` | `100%` | `0.0pp` |
| 14d | cooldown harm | `863.6801%` | `293.3765%` | `-570.3036%` |

Average giveback increased by `0.0746pp` on 30d and `0.0185pp` on 14d. The
14d median giveback improved by `0.0205pp`. Portfolio-full skips increased by
`793` and `309`, respectively. These trade-offs are accepted because both
windows pass every promotion metric and capture is unchanged.

## Decision

- Set the production base `COOLDOWN_BARS` to `2`, so behavior does not silently
  revert to `8` when a feedback file ages past 48 hours.
- Keep feedback auto-apply limited to the same replay-confirmed value.
- Require both maximum-period and stability evidence in the feedback gate.
- Leave `AGENT_MAIN_EXIT_COOLDOWN_BARS=8` unchanged.
