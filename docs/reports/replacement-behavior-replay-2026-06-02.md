# Portfolio Replacement Behavior Replay — 2026-06-02

Status: research-only replay result; no live rotation changes

## Hypothesis

Current portfolio replacement/rotation hurts when it replaces a position that is not losing at the rotation moment. Test a causal replay-only restriction:

- `replacement_block_non_losing`: skip replacement when replaced position current PnL is `>= 0%`.

## Replay Setup

Baseline variant:

- `score_replace`

Candidate variant:

- `replacement_block_non_losing`

Common parameters:

- `--top-gainer-score-min 34`
- `--max-open-positions 10`
- `--replace-min-delta 0`
- `--objective-top-n 15`

## Results

| window | symbols | score PnL | block PnL | ΔPnL | score repl W/I | block repl W/I | policy skipped | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| 7d full | 105 | -182.2320 | -178.5214 | +3.7106 | 2/7 | 0/7 | 10 | positive |
| 14d full | 105 | -162.1942 | -144.8777 | +17.3165 | 7/15 | 0/15 | 20 | positive |
| 14d focused replacement symbols | 35 | -126.3802 | -122.8881 | +3.4921 | 5/3 | 0/2 | 11 | positive |

`repl W/I` means `replacements_worsened / replacements_improved`.

## Interpretation

The candidate consistently reduced harmful replacement exits:

- 7d full: worsened replacements `2 -> 0`;
- 14d full: worsened replacements `7 -> 0`;
- 14d focused: worsened replacements `5 -> 0`.

The total PnL remained negative for both variants, so this is not a complete strategy fix. It is a targeted improvement to portfolio rotation quality.

## Decision

Advance `replacement_block_non_losing` to the next gate:

1. Add a production feature flag, default OFF.
2. Run shadow/live logging for replacement candidates that would be blocked.
3. Only enable live after one more daily report confirms no degradation in watchlist capture / early capture and no rare-winner loss.

## Guardrail

Do not enable this rule as an unconditional production change from this report alone. This replay supports the hypothesis, but rotation policy still needs live-shadow observability around skipped winners.
