# Scout Optimization Spec

Last updated: 2026-05-01 18:31 Europe/Budapest

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

## Metric Quality Requirements
- Entry timing must be measured directly, not inferred from count-only capture:
  - `capture_ratio_at_entry = remaining_move_after_entry / full_day_move`.
  - `entry_day_range_percentile = (entry_price - day_low) / (day_high - day_low)`.
  - `lead_time_to_final_top = day_close_time - first_alert_time`.
- Top-mover objective metrics must be split by action type:
  - BUY precision/recall affects portfolio risk and PnL.
  - WATCH precision/recall affects operator noise and should not be mixed with BUY quality.
  - Block diagnostics affect learning only and may be intentionally low precision.
- Exit quality must be measured with profit retention:
  - `exit_efficiency = realized_pnl / max_favorable_excursion` for winning trades.
  - `giveback_pct = max_favorable_excursion - realized_pnl`.
  - `reversal_delay_bars` from first weakening marker to actual exit.
- Cooldown quality must be measured as `cooldown_harm`: later valid candidates in the same symbol/day that would have improved PnL or top-mover capture.

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
  - 30d replay ending 2026-05-01 16:03 UTC: score-gated production variant had 896 trades, +180.14% total PnL, 46.65% win rate, top15 recall 100%; baseline had 2575 trades, +29.99% total PnL, 37.13% win rate, top15 recall 100%. Keep the gate.
- Top-gainer critic must include both bot and agent logs, not only main bot events.
- Watch-only soft-block alerts are enabled, but they never open positions. Backtests rejected lowering the BUY score gate:
  - 4d current gate: 73 trades, +27.48% PnL, trade precision 45.21%, capture 100%.
  - 4d global 30: 134 trades, -7.73% PnL, precision 32.09%.
  - 7d current gate: 169 trades, +19.63% PnL, precision 33.14%, capture 100%.
  - 7d `impulse_speed=30`: 193 trades, -18.31% PnL; `impulse_speed=26`: 242 trades, -47.97% PnL.
  - Accepted alert-only rules from 2026-04-27..2026-04-29 sweeps: main `impulse_speed` score 30..34, agent RSI<=75 with vol_x>=3, agent disabled `impulse` with ADX>=15/RSI<=75/vol_x>=2, and agent daily range block <=20%.
- Agent entry quality gates were tightened on 2026-04-30 after 7d event-replay diagnostics:
  - `trend` now requires `AGENT_TREND_MIN_ADX=35.0`; single-filter replay improved total PnL by +27.12% with positive train/test deltas.
  - `4h_leader_watch` now requires strength confirmation in addition to the original 4h/local reclaim gates: `today_change >= 10.0%` and `vol_x >= 3.0`. The tested combo `trend ADX>=35 + 4h vol_x>=3 + 4h today>=10` improved replay by +59.47% with positive train/test deltas.
  - `strong_trend` remains unchanged because diagnostics showed it was the only clearly positive mode over the 7d event sample.

## 2026-05-01 Complex Mode Audit
Contradictions found:
- The objective says "early stable trend capture", but the production BUY layer correctly optimizes PnL/precision with score and chase gates. This can make valid-looking stable trends appear as "no signal". Do not solve this by weakening BUY gates; split BUY, WATCH, and diagnostic metrics.
- `ALIGNMENT` and `TREND_SURGE` detect smooth/early structure, but `ALIGNMENT_BUY_ENABLED=False` and the current top-gainer watch alert only covers `impulse_speed`. That is intentional after negative replay, but it creates a visibility gap for slow stable trends.
- The market agent uses symbol cooldown until the next local day after exits, while the main bot uses bar-count cooldown. This protects against churn but can block re-entry in continuing trends. It needs explicit `cooldown_harm` measurement before relaxation.
- 4h context is a ranking bonus and `4h_leader_watch` trigger with strict local confirmation, not a standalone BUY timeframe. This prevents late 4h chasing but means screenshots of 4h trends may not correspond to a live BUY.
- Unified portfolio display deduplicates for ranking, but main and agent can still hold the same symbol internally. Risk and exposure metrics should use symbol-level unified exposure, not raw file counts.

Backtested hypotheses:
- H1 stable trend watch-only mode, 30 complete local days, watchlist top15 objective:
  - Best broad MTF version (`stable_mtf_score26`) covered 59.78% of top15 pairs and caught 38.89% early, but precision was only 25.89% with 34.63 alerts/day.
  - Strict variants still produced 19-31 alerts/day with precision 27-31%.
  - Decision: reject as user-facing alert; keep as a possible offline diagnostic metric only.
- H2 score-gate near-miss watch expansion, 30 complete local days:
  - `all_30_34`: 1170 alerts, precision 27.26%, coverage 70.89%, 39.0 alerts/day.
  - `impulse_speed_33_34`: precision 42.52%, but only 12.0% coverage, weak median capture ratio, and later entries.
  - Quality-filtered variants were either still noisy or empty.
  - Decision: do not broaden live top-gainer watch alerts beyond the already accepted narrow rules.
- H3 exit/re-entry relaxation:
  - 7d weak-exit hold policies worsened total PnL versus current.
  - Profitable weak-exit re-entry brackets were negative after fees.
  - Decision: do not relax exits/cooldown without a new, narrower replay.

Implementation decision:
- No production runtime algorithm change is approved from the 2026-05-01 audit.
- Measurement-first implementation is approved before any new BUY/EXIT optimization.

## 2026-05-05 Metric-Only Instrumentation
Implemented metric-only fields in replay, top-gainer critic, and teacher payloads. No production BUY, EXIT, portfolio, or cooldown rule was changed.

Added fields:
- `capture_ratio_at_entry`: remaining same-day move at entry divided by full same-day move, clipped to 0..1.5. Higher means earlier/better capture.
- `lead_time_to_final_top_min`: minutes from entry to the local objective cutoff, currently 22:00 Europe/Budapest.
- `exit_efficiency`: realized PnL divided by max favorable excursion after entry. Values below 1 show profit left on the table.
- `giveback_pct`: max favorable excursion minus realized PnL.
- `cooldown_harm_pct`: positive forward move from a cooldown-blocked candidate that the bot could not re-enter.

Implemented in:
- `files/replay_backtest.py`: per-trade fields, aggregate summaries, examples, and run-level cooldown skip/harm counters.
- `files/top_gainer_critic.py`: daily top-gainer rows now show entry lead/capture, exit efficiency/giveback, and first cooldown harm.
- `files/critic_dataset.py`: teacher payload now exports the new metric-only fields for later supervised analysis.
- `files/bot.py`: version badge now uses static `BUILD_APPLIED_AT` application time.

ICP incident classification:
- Main bot sent `ICPUSDT` BUY on 2026-05-05 12:21 UTC / 14:21 Europe/Budapest at 2.500.
- Main bot exited on 2026-05-05 12:49 UTC / 14:49 Europe/Budapest after 2 bars at +0.68%, reason `rsi_divergence_momentum_weakens`.
- Later `ICPUSDT` candidates were blocked by `top_gainer_score_gate` in the main bot and by `symbol_cooldown` in the market agent.
- This is not a missing-signal case; it is an exit/cooldown continuation case. Future changes must prove lower `giveback_pct` and lower `cooldown_harm_pct` without hurting 30d PnL/capture.

## 2026-05-05 Signal Quality Evaluator Skill
Implemented repo-local skill `skills/signal-quality-evaluator`.

Role:
- Post-factum evaluator only. It reads already emitted BUY/SELL events and historical candles.
- It does not generate new signals, does not change live gates, and does not train models directly.
- It turns chart disputes into metrics: late entry, early exit, late exit, false-positive BUY, missed trend, trend capture, exit efficiency, and giveback.
- Automatic launch is handled by `files/rl_headless_worker.py`: once per local day for the previous day, default `00:15 Europe/Budapest`.
- Outputs are saved as `.runtime/reports/signal_quality_YYYY-MM-DD_final.json` and `.runtime/reports/signal_quality_YYYY-MM-DD_final.txt`; Telegram summary is controlled by `SIGNAL_QUALITY_EVALUATOR_TELEGRAM_REPORTS_ENABLED`.

Use it before optimization:
- Run evaluator for a day, 7d window, symbol, timeframe, or bot/agent source.
- Form one narrow hypothesis from the error class.
- Validate that hypothesis with `files/replay_backtest.py` before changing production.

Architecture decision:
- Keep scout/monitor/market-agent as live components.
- Keep replay/backtest as the production gate.
- The evaluator can gradually absorb overlapping critic diagnostics, but only after preserving the daily Telegram report contract.

## Backtest Gate For Future Changes
1. Form one narrow hypothesis from critic metrics.
2. Run baseline vs variant replay on at least 4 days and a wider 7-day window when feasible.
3. Apply only if capture is not worse and at least one quality metric improves materially: total PnL, average PnL, precision, false positives, or blocked-winner reduction.
4. If results are mixed, keep the change disabled or replay-only.
5. After applying, run unit tests, compile touched modules, and restart bot/RL worker/market agent.

## Near-Term Plan
1. Day 1: Done for requested metric-only fields: `capture_ratio_at_entry`, `lead_time_to_final_top_min`, `exit_efficiency`, `giveback_pct`, and `cooldown_harm_pct`.
2. Day 1-2: Build a daily trend-start audit table that lists first structural alert, first BUY, first block, and final top15 status per symbol/day.
3. Day 2: Re-test only narrow BUY bypasses that are conditioned on positive trend-start metrics and low cooldown harm; reject any variant that lowers 30d PnL or raises alerts/day materially.
4. Day 3: Add unified symbol-exposure diagnostics so duplicate main/agent positions are visible as one risk unit.
5. Day 3-4: Revisit exit optimization using `exit_efficiency` and `giveback_pct`, not weak-exit count alone.
