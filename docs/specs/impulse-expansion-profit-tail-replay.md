# Impulse Expansion Profit-Tail Replay

Date: 2026-08-07
Status: complete/rejected; no live SELL change

## Problem

C98USDT repeatedly exited profitable `15m/impulse_speed` positions on soft
`WEAK:` warnings while the larger impulse continued. Broad WEAK suppression and
static RSI/MTF filters were already rejected because they harmed the majority of
historical exits. The new hypothesis must therefore distinguish active trend
expansion from ordinary profitable exhaustion and preserve the already realized
profit when it is wrong.

## Causal expansion indicators

All values use the last fully closed decision candle:

- ADX level and three-bar ADX delta;
- EMA20 slope and three-bar slope delta;
- MACD histogram normalized by ATR;
- ten-bar Kaufman-style price efficiency ratio;
- fourteen-bar `+DM/-DM` directional ratio;
- twenty-bar Donchian position;
- twenty-bar Chaikin Money Flow;
- distance above EMA20 measured in ATR;
- EMA20/EMA50 alignment.

The expansion score has eleven binary components:

1. `ADX >= 40`;
2. `ADX delta(3) >= 0`;
3. `EMA20 slope >= +0.50%`;
4. slope delta(3) is non-negative;
5. MACD histogram / ATR is positive;
6. efficiency ratio(10) is at least `0.45`;
7. `+DM/-DM` ratio is at least `1.8`;
8. Donchian position(20) is at least `0.80`;
9. CMF(20) is non-negative;
10. close is at least `0.75 ATR` above EMA20;
11. EMA20 is at or above EMA50.

## Frozen policy family

The primary policy applies only to profitable (`>= +1%`) `15m/impulse_speed`
soft-WEAK exits with expansion score at least `8`:

- sell `75%` at the existing SELL price;
- keep a `25%` protected tail;
- initialize the tail floor at `90%` of the PnL already earned at SELL;
- trail the highest close by `ATR x 1.8`;
- ignore additional soft-WEAK warnings;
- preserve all hard exits;
- exit after two consecutive closed bars with expansion score below `4`;
- force-close after ten additional bars.

Frozen sensitivity/ablation profiles:

- expansion score floors `7` and `9`;
- no expansion-decay exit;
- tail fraction `50%`;
- profit lock `75%`;
- ATR multipliers `1.4` and `2.2`.

No thresholds may be added after viewing validation/holdout results in this
experiment.

## Dataset and validation

- Source: all valid `event=exit`, `15m/impulse_speed`, soft `WEAK:` exits in
  `files/bot_events.jsonl`.
- Price paths: local signal-quality candle cache.
- Decision close must match logged exit price within `0.25%`.
- Indicators are recomputed from cached candles.
- Whole UTC days are split chronologically `60%/20%/20%`.
- A conservative `5 bps` policy-change penalty is charged on the retained tail.

## Promotion gate

A profile may advance to full portfolio replay only when:

- validation and holdout each contain at least ten selected exits;
- validation and holdout average net delta are positive;
- validation and holdout median net delta are non-negative;
- holdout worse-rate is at most `45%`;
- holdout p10 net delta is no worse than `-0.50pp`;
- the result is not carried by one reason bucket or one symbol.

Passing this gate does not authorize production. Portfolio replay must then
include fees/slippage, turnover, position-slot occupation, drawdown, top-mover
capture, and re-entry churn.

## Result

The maximum locally labelable period contained `314` eligible events. Candle
paths were reconstructed for `301` (`2026-05-03` through `2026-08-06`); `13`
were unavailable because of cache overlap, future-candle, or decision-price
quality gates.

All eight frozen profiles failed. For the primary profile:

- selected exits: `100`;
- all-period net delta: average `-0.0829pp`, median `-0.1772pp`;
- all-period worse-rate: `82.0%`;
- validation: `n=16`, average `-0.1053pp`, median `-0.1654pp`;
- holdout: `n=17`, average `+0.0127pp`, median `-0.0823pp`, worse-rate
  `64.71%`, p10 `-0.3458pp`;
- median retained baseline profit: `94.335%`.

The best holdout mean is not promotable because validation is negative and the
majority of both validation and holdout cases are worse. Raising the expansion
score, changing tail size, changing the profit lock, changing the ATR trail, or
removing the decay exit did not repair the result. Production SELL behavior
therefore remains unchanged.

The next allowed branch is a causal learned selector over universal post-exit
advantage labels, with trade-lifecycle context and chronological validation. It
must not reuse these hand-tuned score thresholds as a production rule.
