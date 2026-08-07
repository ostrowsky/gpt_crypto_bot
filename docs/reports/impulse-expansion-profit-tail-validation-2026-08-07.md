# Impulse Expansion Profit-Tail Validation — 2026-08-07

## Decision

Reject all tested profiles. Do not change production SELL behavior.

## C98USDT incident

The latest position entered at `0.01525` and exited at `0.01583` (`+3.803%`)
after four 15-minute bars because of a soft RSI-divergence warning. The chart
price later shown by the operator was `0.01637`, another `+3.41%` after the
exit and `+7.34%` from that entry. At the preceding observation the move still
had strong expansion evidence: ADX about `69`, EMA20 slope about `+2.7%`, and a
positive MACD histogram.

This was a real premature exit, but C98 also demonstrated why a universal hold
is unsafe. An earlier C98 RSI-WEAK exit at `0.01428` was followed by a negative
T+2 continuation, while later re-entry restored exposure. The defect is
selection between continuation and exhaustion, not the existence of the WEAK
exit itself.

## Tested hypothesis family

A causal 11-component expansion score used only fully closed decision candles:

- ADX level and three-bar change;
- EMA20 slope and acceleration;
- MACD histogram normalized by ATR;
- Kaufman efficiency ratio;
- `+DM/-DM` directional ratio;
- Donchian-channel position;
- Chaikin Money Flow;
- EMA distance in ATR and EMA20/EMA50 alignment.

The primary policy sold `75%`, retained a protected `25%` tail, locked `90%`
of already earned PnL, and used an `ATR x 1.8` trailing exit. Seven frozen
sensitivities varied the expansion floor, decay exit, tail fraction, profit
lock, and ATR multiplier. A `5 bps` change penalty was charged to the retained
tail.

## Coverage

| Item | Result |
|---|---:|
| Eligible event rows | 314 |
| Causally labeled | 301 |
| Missing / rejected by data-quality gate | 13 |
| Labeled interval | 2026-05-03 — 2026-08-06 |
| UTC-day split | 42 train / 14 validation / 14 holdout days |
| Baseline PnL, average / median | +1.2539% / +0.7330% |

The latest C98 exit on August 7 is not included in the completed T+10 replay
because its future candle horizon was not yet present in the local cache. It is
retained as forward evidence, not backfilled into model selection.

## Results

| Profile | Validation avg / median | Holdout avg / median | Holdout worse-rate | Gate |
|---|---:|---:|---:|---:|
| score 8, tail 25%, lock 90%, ATR 1.8 | -0.1053 / -0.1654pp | +0.0127 / -0.0823pp | 64.71% | fail |
| score 7 | -0.1027 / -0.1610pp | -0.0497 / -0.1294pp | 66.67% | fail |
| score 9 | -0.1220 / -0.1654pp | -0.0551 / -0.1294pp | 66.67% | fail |
| no decay exit | -0.1053 / -0.1654pp | +0.0127 / -0.0823pp | 64.71% | fail |
| tail 50% | -0.2106 / -0.3309pp | +0.0255 / -0.1645pp | 64.71% | fail |
| lock 75% | -0.1511 / -0.1957pp | -0.1488 / -0.1765pp | 58.82% | fail |
| ATR 1.4 | -0.0873 / -0.1434pp | +0.0127 / -0.0823pp | 64.71% | fail |
| ATR 2.2 | -0.1053 / -0.1654pp | +0.0127 / -0.0823pp | 64.71% | fail |

Across all `100` primary-selected cases, net delta averaged `-0.0829pp`, median
was `-0.1772pp`, and `82.0%` were worse than the existing exit. Median profit
retention was `94.335%`, so the protection limited damage but did not create an
edge.

## Interpretation and next priority

Strong trend readings identify both continuation and climactic exhaustion.
Static combinations of the new indicators therefore do not solve the selection
problem. More hand-tuned thresholds on the same target would be holdout fitting.

The next valid hypothesis is a learned causal discriminator trained on
universal post-exit advantage labels for all WEAK exits. Its feature set should
add trade-path context (`bars_held`, unrealized PnL, MFE, giveback, entry mode,
and recent score trajectory) to the market-structure indicators tested here.
It remains research-only until chronological event-level gates and a full
portfolio replay both pass.
