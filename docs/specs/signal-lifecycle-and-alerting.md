# Signal lifecycle and alerting

## Objective

The bot must surface strong market opportunities without weakening replay-gated
production BUY rules, retain positions while their trend remains healthy, and
measure entry and exit quality without contaminating closed-trade metrics.

## Entry and alerts

- Production BUY requires the configured top-gainer live-score gate.
- A blocked candidate with candidate score at least 100, live score at least 28,
  and score deficit no greater than 6 emits a `STRONG SIGNAL` watch alert.
- Watch alerts never open a position.
- Identical alerts are deduplicated on the same candle. A new candle may emit a
  new alert when the opportunity remains strong or the market regime changes.
- Portfolio and cluster-cap blocks may emit explanatory alerts, but portfolio
  capacity is the authoritative limit.
- Binance top-mover measurement uses Top-20.

## Position lifetime

- Crossing a local calendar-day boundary must not remove a position.
- A time limit is deferred while price is above EMA20, EMA slope is non-negative,
  RSI is at least 50, and MACD histogram is non-negative.
- ATR, EMA, fast-loss, portfolio-replacement, and weakening-trend exits remain
  active. Multi-day retention is not an unconditional hold.
- Restored portfolios may still be trimmed when they exceed configured capacity
  or contain a disabled alignment entry.

## Learning and reporting

- Early exits under one hour and post-exit continuation are persisted as labels.
- Blocked-learning labels are emitted once per candle.
- Observable partial-tail candidates remain shadow-only and receive T+2/T+5/T+10
  labels.
- Exit efficiency, giveback, and PnL summaries use closed trades only.
- Rolling Top-N rates are weighted by their actual daily denominator.
- A failed Telegram delivery releases its persisted slot claim so it can retry.

## Replay gate

Every production policy hypothesis must be tested over the maximum available
history. The 44-day, 105-symbol replay rejected lowering the global top-gainer
gate from 34 to 33: total PnL changed from -530.90% to -562.07%, while objective
precision and recall were unchanged. Therefore this change expands watch alerts
only; it does not lower the production BUY threshold.

## Acceptance tests

- A restored prior-day trend position survives startup and the monitoring loop.
- Time exit waits only while all continuation conditions pass.
- A blocked strong signal sends once on one candle and may send again on a new
  candle.
- Telegram slot claims can be released after failed delivery.
- Signal-quality exit metrics exclude open positions.
- Rolling Top-N metrics are denominator-weighted.
