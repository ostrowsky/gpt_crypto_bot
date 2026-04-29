# Scout Optimization Spec

Last updated: 2026-04-29 19:15 Europe/Budapest

## Objective
The bot is optimized for timely capture of same-day watchlist top gainers while keeping a single unified portfolio of the 10 most promising positions. Every production algorithm change must be backed by replay/backtest evidence before it is enabled.

## Primary Metrics
- `watchlist_top_bought`: count of watchlist top-N gainers that received a BUY.
- `early_captures`: bought top gainers with capture ratio >= 0.35.
- `false_positive_buys`: bought symbols that did not finish in watchlist top-N.
- `blocked_winners`: top gainers that were not bought but had explicit block events.
- `missed_reason_counts`: top missed symbols grouped by normalized blocker code.
- `blocked_reason_harm`: blocker-level missed opportunity from first block to day close.
- `trade_precision`: replay trades whose symbol is in final top-N divided by all replay trades.
- `capture_rate`: replay captured final top-N symbols divided by final top-N symbols.

## Data Sources
- Bot events: `files/bot_events.jsonl`.
- Market-agent events: `files/agent_events.jsonl`.
- Critic reports: `.runtime/reports/top_gainer_critic_*`.
- Replay engine: `files/replay_backtest.py`.
- Config gates: `files/config.py`.

## Current Production Decisions
- `ALIGNMENT_BUY_ENABLED=False`. Alignment remains context/diagnostic only. Replay on 2026-04-25..2026-04-29 rejected enabling it because total PnL worsened from -36.36% to -51.48%.
- `TOP_GAINER_SCORE_GATE_MIN_SCORE=34.0`. Replay confirmed this stronger gate preserves capture while cutting low-quality trades:
  - 4d: trades 375 -> 88, total PnL -36.36% -> -3.01%, capture@15 stayed 1.0.
  - 7d: trades 638 -> 176, total PnL -40.99% -> -7.59%, trade precision 0.2257 -> 0.2898, capture@15 stayed 1.0.
- Top-gainer critic must include both bot and agent logs, not only main bot events.

## Backtest Gate For Future Changes
1. Form one narrow hypothesis from critic metrics.
2. Run baseline vs variant replay on at least 4 days and a wider 7-day window when feasible.
3. Apply only if capture is not worse and at least one quality metric improves materially: total PnL, average PnL, precision, false positives, or blocked-winner reduction.
4. If results are mixed, keep the change disabled or replay-only.
5. After applying, run unit tests, compile touched modules, and restart bot/RL worker/market agent.

## Near-Term Plan
1. Day 1: Watch the first full final critic report with the new score gate. Compare false positives, blocked winners, and missed reasons against 2026-04-28 and 2026-04-29 midday.
2. Day 1-2: Build a small daily score-gate sweep report for thresholds 30/34/38, using identical replay data, so threshold tuning is evidence-driven instead of manual.
3. Day 2: Investigate `agent_mode_disabled` and `accuracy_gate` only for missed top gainers, not all blocks. Any relaxation must preserve replay precision and PnL.
4. Day 3: Add a portfolio opportunity-cost metric: rank current 10 positions against current watchlist top movers and report replacement candidates with expected benefit.
5. Day 3-4: Test replacement policy variants in replay: replace weakest non-top-gainer slot only when candidate is watchlist top15 and score delta clears a conservative threshold.
