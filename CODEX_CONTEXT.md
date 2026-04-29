# CODEX_CONTEXT

Last updated: 2026-04-29 19:15 Europe/Budapest

## Purpose
This file is the compact source of truth for Codex work on `D:\Projects\gpt_crypto_bot`.
Read this file first. Only read extra files when this file does not answer the question.

## Workspace Rules
- Work in `D:\Projects\gpt_crypto_bot`.
- Do not treat `.codex\worktrees\...` as the primary project state.
- Prefer local project artifacts and generated reports over raw repo-wide scans.
- When adding new durable context, update this file briefly instead of repeating long explanations in chat.

## Token Saving Rules
1. Read `CODEX_CONTEXT.md` first.
2. Prefer compact reports in `.runtime\reports\` over raw `jsonl` logs.
3. Prefer point queries over full-file reads.
4. Summarize new findings here in 3-8 bullets, not long prose.
5. Keep this file concise; store stable facts, decisions, paths, and current operating rules only.
6. Do not re-read large files if a latest summary artifact already exists.

## Project Summary
Live intraday crypto trading bot with:
- Telegram bot execution and monitoring
- Signal generation, blocking, entry/exit tracking
- Market agent
- RL headless worker and retraining pipeline
- Dataset collection for ML / critic / top-gainer analysis
- Daily and midday reporting

## Primary Sources Of Truth

### Runtime data
- Bot events: `D:\Projects\gpt_crypto_bot\files\bot_events.jsonl`
- Agent events: `D:\Projects\gpt_crypto_bot\files\agent_events.jsonl`
- Open positions: `D:\Projects\gpt_crypto_bot\files\positions.json`
- Watchlist: `D:\Projects\gpt_crypto_bot\files\watchlist.json`
- Agent positions: `D:\Projects\gpt_crypto_bot\files\agent_positions.json`

### Compact report artifacts
- Top gainer critic reports: `D:\Projects\gpt_crypto_bot\.runtime\reports\top_gainer_critic_*`
- RL daily reports: `D:\Projects\gpt_crypto_bot\.runtime\reports\rl_daily_*.json`
- Latest RL training report: `D:\Projects\gpt_crypto_bot\.runtime\reports\rl_train_latest.json`
- Watchlist top gainer goal latest: `D:\Projects\gpt_crypto_bot\.runtime\reports\watchlist_top_gainer_goal_latest.json`
- Top gainer history: `D:\Projects\gpt_crypto_bot\.runtime\reports\top_gainer_critic_history.jsonl`

### Health / status files
- RL worker status: `D:\Projects\gpt_crypto_bot\.runtime\rl_worker_status.json`
- Root process helpers:
  - `D:\Projects\gpt_crypto_bot\bot_status.ps1`
  - `D:\Projects\gpt_crypto_bot\rl_worker_status.ps1`
  - `D:\Projects\gpt_crypto_bot\market_agent_status.ps1`

## Current Automation State
As of 2026-04-18, all Codex automations tied to this project are paused to stop background token spend:
- `daily-24-00-audit`
- `midnight-crypto-report`
- `rl-train-watch`
- `signal-and-rl-watch`
- `signal-watch`
- `top-movers-midnight-audit`
- `top-movers-noon-audit`
- `top-movers-noon-audit-2`

## Existing Useful Report Logic
- `D:\Projects\gpt_crypto_bot\files\report_top_movers.py`
  - Computes top movers, compares with portfolio and bot events.
  - Current weakness: direct Binance HTTP + raw event scan.
- `D:\Projects\gpt_crypto_bot\files\report_rl_daily.py`
  - Already prefers `rl_train_latest.json` and other compact runtime artifacts.
- `D:\Projects\gpt_crypto_bot\files\top_gainer_critic.py`
  - Produces compact midday/final critic snapshots suitable for reuse.

## MCP Plan For Minimum Token Spend

### Goal
Move all heavy reading, filtering, counting, and joining into MCP or local code.
Use LLM only for the final short explanation.

### Recommended MCP server
Create a project MCP server, for example `crypto_bot_context`, that reads local files and returns compact JSON.

### Phase 1: highest-value tools
1. `get_project_context()`
   - Returns the current stable summary from this file.
   - Use before any repo scan.
2. `get_top_movers_audit(date, phase)`
   - Returns:
     - top movers
     - captured movers
     - missed movers
     - extra portfolio positions
     - capture rate
     - blocker counts
   - Prefer latest `top_gainer_critic_*` report if available.
   - Fall back to local recompute only when needed.
3. `get_signal_summary(date)`
   - Returns:
     - entries by symbol
     - blocked counts
     - top blocker reasons
     - forwards / exits / cooldown counts
   - Never return raw `bot_events.jsonl` unless explicitly asked.
4. `get_portfolio_snapshot()`
   - Returns open positions, count, exposure summary, updated_at.
5. `get_rl_summary()`
   - Returns:
     - worker heartbeat
     - latest training run metadata
     - rows_total
     - top1_delta
     - latest report path
6. `get_runtime_health()`
   - Returns bot / RL / market-agent liveness and stale-heartbeat flags.

### Phase 2: delta-oriented tools
7. `get_changes_since(ts)`
   - Returns only changed signals, positions, and reports since timestamp.
8. `get_blocker_reasons(date, top_n=10)`
   - Returns normalized blocker reason counts.
9. `get_watchlist_snapshot()`
   - Returns symbol count and checksum, not the full file by default.

### Phase 3: write-side helpers
10. `update_codex_context(section, bullets)`
    - Writes concise durable updates into this file.
11. `record_audit_result(date, phase, summary)`
    - Stores compact daily audit summaries for later retrieval without log re-scan.

## MCP Response Design Rules
- Default response target: under 1-2 KB JSON.
- Return counts, top N, timestamps, and paths.
- Do not return full logs by default.
- Include `source_paths` and `generated_at` in each response.
- Include `used_cached_report=true/false` so the caller knows whether recompute happened.

## Preferred Read Order For Future Work
1. `D:\Projects\gpt_crypto_bot\CODEX_CONTEXT.md`
2. Relevant compact file in `.runtime\reports\`
3. Specific status JSON
4. Narrow raw-file query
5. Full raw-file read only if unavoidable

## Immediate Implementation Priorities
1. Build `get_project_context`, `get_top_movers_audit`, `get_signal_summary`, `get_rl_summary`.
2. Repoint future automations and ad hoc audits to MCP tools instead of raw file reads.
3. Deduplicate the duplicate noon audit before any automation is re-enabled.
4. Treat this file as the durable handoff note for important decisions and current operating state.

## Current Durable Notes
- `restart_full_stack.bat` already exists at `D:\Projects\gpt_crypto_bot\restart_full_stack.bat`.
- Compact report artifacts exist and should be preferred over direct Binance/API recomputation whenever fresh enough.
- Duplicate noon audit exists (`top-movers-noon-audit` and `top-movers-noon-audit-2`); keep only one before re-enabling schedules.

## Latest Update
- 2026-04-18 20:14 Europe/Budapest
- All project-linked Codex automations are paused to stop background token spend.
- `CODEX_CONTEXT.md` is now the first file to read before any broader repo inspection.
- MCP priority order is: project context, top movers audit, signal summary, RL summary.
- Minimal MCP scaffold added:
  - `D:\Projects\gpt_crypto_bot\mcp_server.py`
  - `D:\Projects\gpt_crypto_bot\mcp_context_tools.py`
  - `D:\Projects\gpt_crypto_bot\MCP_CONTEXT_README.md`
- Working Python environment for MCP is `D:\Projects\gpt_crypto_bot\files\.venv\Scripts\python.exe`.
- Root venv `D:\Projects\gpt_crypto_bot\.venv` is stale/broken and should not be used until rebuilt.
- Added MCP delta/snapshot tools:
  - `changes_since(ts)`
  - `write_daily_signal_snapshot(day_str)`
- 2026-04-18 20:46 Europe/Budapest
- `files\market_signal_agent.py` was tightened to the top-gainer objective:
  - max 2 positions
  - distinct movement-pattern clusters
  - stronger day-change / forecast / leader-score filters
  - same-day symbol cooldown after exit
- Fixed runtime blockers in the market agent caused by stub helper signatures and an incompatible `check_exit_conditions(...)` call.
- After a clean restart, stale agent portfolio was pruned from 16 positions to 0; current noisy backlog is cleared.
- Current intent for the market agent:
  - avoid back-to-back churny BUY/SELL alerts
  - prefer only the strongest intraday leader candidates
  - hold at most 2 differentiated symbols
- 2026-04-19 11:42 Europe/Budapest
- Restored market-agent BUY alert text to the previous user-facing design (`🟢 СИГНАЛ ПОКУПКИ ...`) without reverting the stricter top-gainer filters.
- Restored fast 15m forecast display for agent alerts to `T+2/T+5/T+7`.
- Restored Telegram quick menu reply buttons in `files\bot.py`: `📋 Открыть меню` and `🙈 Скрыть меню`.
- Fixed `start_market_agent_bg.ps1` so restarts kill an existing `market_signal_agent.py` process even when the PID file has `python_pid=0`.
- 2026-04-19 11:49 Europe/Budapest
- Main monitor portfolio had stale positions from prior days (`MDTUSDT`, `ORDIUSDT`, `PYRUSDT`) even though `MAX_OPEN_POSITIONS=2`.
- Added startup/live trimming in `files\monitor.py`: drop non-current-local-day positions and enforce max restored positions.
- After restart, `files\positions.json` was cleared to `{}`.
- Positions UI now shows signal mode plus entry slope/ADX so trend context is visible.
- Live check: ARUSDT 15m was confirmed with active `trend`; TRXUSDT was confirmed but no current entry because RSI/volume filter blocked; CRVUSDT was not confirmed and 24h change was negative.
- Follow-up one-off poll confirmed ARUSDT entry and wrote it to `files\positions.json`; bot restart restored 1 current position: `ARUSDT`.
- Fixed `files\botlog.py` so `log_entry(..., ml_proba=...)` no longer fails.
- 2026-04-19 11:56 Europe/Budapest
- Training assessment:
  - latest RL/ranker training is stale: `rl_train_latest.json` generated `2026-04-08T09:04:17Z`, run 126.
  - current `critic_dataset.jsonl` has 7234 rows, but `rows_last_24h=0`; ML dataset keeps growing.
  - latest test top1 ranker result is negative (`ret5_delta=-0.4365`, `win_rate_delta=-0.0588`), despite positive historical shadow top1 delta.
- Critic assessment:
  - final top-gainer capture was 0% on 2026-04-14 through 2026-04-18.
  - latest goal report 2026-04-18: recall@cutoff 0/15, blockers `blocked_portfolio=8`, `no_signal=7`.
- Priority: make top-gainer goal the training target, restore critic-row collection, and gate entries by daily top-gainer score rather than generic forward win-rate.
- 2026-04-19 12:10 Europe/Budapest
- Disabled Telegram report spam from RL/report workers:
  - `RL_TELEGRAM_REPORTS_ENABLED=False`
  - `TOP_GAINER_CRITIC_TELEGRAM_REPORTS_ENABLED=False`
  - `WATCHLIST_TOP_GAINER_GOAL_TELEGRAM_REPORTS_ENABLED=False`
  - `RL_TRAIN_TELEGRAM_REPORTS_ENABLED=False`
- Stopped all `gpt_crypto_bot` RL/report worker processes (`rl_headless_worker.py`, `headless_loop.ps1`, top-gainer report jobs).
- Disabled Windows Scheduled Tasks:
  - `CryptoBot_DailyLearning_EOD`
  - `CryptoBot_IntradaySnapshot`
  - `GPT Crypto RL Daily Report`
- Codex project automations remain paused; live bot and market agent remain running.

- 2026-04-19 17:40 Europe/Budapest
- Added main-bot `TOP_GAINER_CHASE_GUARD`: blocks extreme late top-gainer chases while preserving normal leader logic.
- Guard defaults: modes `trend/strong_trend/impulse_speed/impulse`, TF `15m/1h`, block `daily_range > 25%` or `RSI > 76` with `daily_range >= 8%`.
- Replay validation: focused DOGS/BLUR/AR/TRX/CRV/AMP 3d improved slightly (`pnl_total -2.5351 -> -2.4385`); watchlist 1d unchanged; DOGS/TRX pass, BLUR 41.66% range blocks.
- Fixed `strategy.check_exit_conditions` compatibility with monitor/replay keyword args (`mode`, `bars_elapsed`, `tf`).
- BTC 2026-04-19 afternoon: no alert because main bot portfolio was full and agent rejected BTC as non-top-gainer (`day change 0.25% < 1.25%`, later `alignment` mode disabled).

- 2026-04-19 17:53 Europe/Budapest
- Corrected portfolio semantics: total main capacity is `MAX_OPEN_POSITIONS=10`; `MAX_POSITIONS_PER_GROUP=2` limits similar coin groups.
- Added/enabled main signal pattern caps: max 2 per 15m short-bounce, 15m impulse, 15m momentum, 15m alignment, 1h retest, 1h momentum, 1h alignment.
- Corrected market-agent capacity to `AGENT_MAX_POSITIONS=10`; agent caps similar pattern clusters and coin groups at 2 each.
- Restarted bot and market agent after config/code changes; bot restored 2 positions under 10-slot capacity.

- 2026-04-19 18:09 Europe/Budapest
- Fixed Telegram menu build badge: `BUILD_ID=menu_build_v3`, `BUILD_DATE` now uses current process start date instead of stale `2026-04-13`.
- Made callback buttons more responsive: callback answer is non-blocking, `edit_message_text` has a short timeout and falls back to `send_message`, callback handler runs with `block=False`.
- Restarted Telegram bot after changes.

- 2026-04-19 18:22 Europe/Budapest
- Disabled `alignment` as a buy trigger by default: `ALIGNMENT_BUY_ENABLED=False`.
- Main monitor, strategy analysis, and replay now treat alignment as diagnostic context unless explicitly re-enabled.
- Verified CHZUSDT 15m and ARUSDT 1h: `alignment_ok=true`, but `buy_signal_after_fix=false`.
- Tests passed and Telegram bot restarted.

- 2026-04-19 18:23 Europe/Budapest
- Added restored-position prune for alignment entries when `ALIGNMENT_BUY_ENABLED=False`; old CHZ/SUSHI/AR-style alignment positions are dropped on bot startup.
- Bot restarted after prune patch.

- 2026-04-19 18:29 Europe/Budapest
- Menu button logs showed the Telegram bot process was overloaded by startup/background work, especially in-process `DataCollector` and repeated `ml_dataset fill_pending_from_data` warnings.
- Added `BOT_ENABLE_DATA_COLLECTOR=False`, `BOT_STARTUP_AUTO_SCAN_ENABLED=False`, and set `AUTO_REANALYZE_SEC=0` so the Telegram process stays interactive; manual analysis remains available from the menu.
- Added explicit `MENU CALLBACK` and `MENU TEXT` logging in `files\bot.py`.
- Restarted bot: PID `17792`; log confirms `DataCollector disabled in Telegram bot process`, `Startup auto scan disabled`, regular `getUpdates`, and no new `fill_pending` spam.

- 2026-04-19 23:50 Europe/Budapest
- Restored market-agent SELL alert design in `files\market_signal_agent.py`: no `AGENT SELL` header, old icon layout with exit price, reason, entry change, prediction accuracy, and bars held.
- Market agent restarted; real Python PID `5976`, status heartbeat fresh, first cycle completed with 0 restored/open positions.
- Portfolio vs watchlist top-gainer objective near midnight: current main portfolio captures 2/10 watchlist top-10 movers (`BLURUSDT`, `JASMYUSDT`); `BNTUSDT` is rank ~12, `AMPUSDT` and `ADAUSDT` are off-objective.

- 2026-04-20 00:13 Europe/Budapest
- Tested proposed `TOP_GAINER_OBJECTIVE_GATE` for the main defect (TA-score selects non-top-gainer slots).
- Broad gate improved trade precision/PnL but reduced top-gainer capture from 1.0 to 0.4 on the focused 3d replay, so it is not production-safe.
- Narrow retest/breakout variants preserved capture but reduced PnL/win-rate; not enabled.
- Added disabled instrumentation/config plus replay objective metrics for future A/B, but live behavior remains unchanged: `TOP_GAINER_OBJECTIVE_GATE_ENABLED=False`.
- Conclusion: fix should be explicit top-gainer ranker/replacement policy, not hard pre-entry filters over successful signal modes.

- 2026-04-20 10:15 Europe/Budapest
- Objective metrics snapshot for user request (watchlist top-gainers, precision/recall):
  - Official goal history 2026-04-13..2026-04-18: micro precision 6/16 = 37.5%, micro recall 6/90 = 6.7%; only 2026-04-13 had alerts/captures (precision 37.5%, recall 40.0%).
  - 2026-04-19 full day from Binance klines + event logs: combined main+agent top15 precision 9/15 = 60.0%, recall 9/15 = 60.0%; top10 precision 7/15 = 46.7%, recall 7/10 = 70.0%.
  - 2026-04-19 split: main top15 precision 6/10 = 60.0%, recall 6/15 = 40.0%; agent top15 precision 5/7 = 71.4%, recall 5/15 = 33.3%.
  - 2026-04-20 partial to 10:15 local: main had no buys; agent/combined top15 precision 3/4 = 75.0%, recall 3/15 = 20.0%.
- Interpretation: recent precision improved, especially in agent, but recall remains unstable; main bot silence/slot selection is now the biggest objective risk.

- 2026-04-20 proposal
- Current defect fix should be a top-gainer candidate ranker + portfolio replacement policy, not another hard signal gate.
- Desired behavior: keep total capacity 10, but every new/restored position must compete by `top_gainer_score`; replace weak/off-objective slots when a stronger same-day top-gainer candidate appears.
- Score inputs should include intraday change from local midnight, rank among watchlist, acceleration over recent 15m bars, volume expansion, ADX/slope confirmation, pattern cluster diversity penalty, and late-chase/range risk penalty.
- Acceptance gate: only buy if score clears dynamic threshold or candidate is already watchlist top15; only keep max 2 per correlated pattern cluster.
- This targets current issue directly: main bot has low recall or holds lower-objective slots while agent catches cleaner leaders.

- 2026-04-20 10:45 Europe/Budapest
- Added replay-only A/B support in `files/replay_backtest.py`: `--variant baseline|score|score_replace|score_replace_cluster`, `--top-gainer-score-min`, `--objective-top-n`, symbol precision, recall, score/cluster skip stats, and UTF-8 stdout for JSON/text output.
- Full watchlist 3d replay, top15, max positions 10, `top_gainer_score_min=18`:
  - Baseline: 280 trades, pnl_total -109.72%, win_rate 29.6%, symbol precision 16.9%, recall 100%.
  - Variant A score only: 213 trades, pnl_total -63.41%, win_rate 33.3%, symbol precision 18.1%, recall 100%.
  - Variant B score+replacement: 211 trades, pnl_total -65.86%, win_rate 34.1%, symbol precision 18.1%, recall 100%, replacements 0.
  - Variant C score+replacement+cluster: 209 trades, pnl_total -64.41%, win_rate 34.0%, symbol precision 17.9%, recall 100%, cluster skips 8.
- Top10 replay with same settings: all variants kept recall 100%; Variant C had best pnl_total (-64.15% vs baseline -102.29%) and best/near-best trade precision (18.0% vs baseline 18.2%, with 74 fewer trades).
- Stress replay with `top_gainer_score_min=0`: replacement triggers only 2-3 times and does not improve results; score threshold is doing the useful work.
- Conclusion: do not deploy replacement first. Deploy/tune score gate + pattern cluster cap first; keep replacement disabled or conservative until it proves value in a fuller portfolio stress test.

- 2026-04-20 10:55 Europe/Budapest
- Investigated user report: ACHUSDT/CRVUSDT BUY alerts at 10:45 were missing from the menu Positions view.
- Root cause: alerts were market-agent positions stored in `files/agent_positions.json`; Telegram menu read only main `state.positions`/`files/positions.json`, which was stale.
- Changed `files/bot.py` so the menu count and Positions view dynamically include `agent_positions.json` and display agent positions under a separate `Market agent` section with leader/today fields.
- Restarted Telegram bot; restart pruned stale main positions from `positions.json` to `{}`. Restart script also stopped market-agent, so market-agent was restarted; status shows 2 restored/open agent positions.

- 2026-04-20 10:58 Europe/Budapest
- Fixed mojibake in the new Positions/Market agent menu block (`рџ...`, `Р’С...`, `вЏ...`).
- `files/bot.py` now uses proper UTF-8 strings for the combined positions header, agent labels, entry line, bars, target/leader line, and truncation suffix.
- Verified `py_compile bot.py`; restarted Telegram bot and restarted market-agent again because the bot stop script kills all project Python processes.

- 2026-04-20 11:05 Europe/Budapest
- Reviewed CRVUSDT 15m trend BUY quality after user screenshot.
- CRV was objective-relevant (watchlist rank ~7/98 intraday, +1.5% local-day change), but entry quality was weak/late: it followed a spike to 0.2306 and entered around 0.2284/0.2285 after pullback; user alert showed catch-up drift -0.52%.
- Agent logs repeatedly blocked CRV before entry because `best accuracy 50.0 < 55.0`, then entered at 08:45Z with trend metrics; position state one bar later had mark 0.2279 and leader_score down to 16.3.
- Recommendation: treat this as a valid watch/top-gainer candidate but bad BUY timing. Add guard for trend catch-up entries: if catch-up drift is negative beyond ~0.25-0.35% or current/previous 15m closes are declining after spike, downgrade to WATCH or require reclaim of prior candle high/positive close before BUY.

- 2026-04-20 11:15 Europe/Budapest
- Backtested the proposed 15m trend catch-up negative-drift hard block over full watchlist recent Binance 15m history (~1000 bars/symbol).
- Reconstructed catch-up cases as: current bar has no active signal, previous bar had `trend`, current entry would catch up one bar later.
- Total reconstructed 15m trend catch-ups: 1558; all-negative T+3/T+5/T+10: 562 (36.1%); T+5<=0: 857 (55.0%).
- Hard block `drift <= -0.25%`: would block 476 signals, avoid 140 all-negative signals (239 T+5 losers), but also block 336 signals with at least one positive horizon; blocked set all-negative rate was only 29.4%, lower than kept set 39.0%.
- Hard block `drift <= -0.35%`: would block 364, avoid 103 all-negative (178 T+5 losers), but lose 261 with at least one positive horizon; blocked set all-negative rate 28.3%.
- Conclusion: simple negative-drift hard block is not production-safe and would likely reduce recall. Better: use drift as a score penalty/reconfirmation trigger, e.g. require reclaim/green close only when drift is negative AND volume/leader score/top-gainer rank is weak.

- 2026-04-20 12:35 Europe/Budapest
- Checked user question why no BONKUSDT signal despite apparent trend.
- BONK is in watchlist and being scanned; agent repeatedly blocked it as `agent mode disabled: retest` around 10:17-10:29Z.
- Current 15m tech: price 0.00000610, EMA20 0.0000060646, slope +0.329%, ADX 33.1, RSI 59.7, but vol_x only 0.43; no active entry/breakout/surge/impulse/alignment, retest false at checked bar; prior agent blocks saw retest mode with vol_x ~1.05.
- Current 1h tech: price above EMA20 but below EMA50, ADX 16.4, vol_x 0.68; no valid 1h signal.
- Intraday watchlist rank ~26/98 at +1.50%, outside top15; current open agent positions CRV/POL have stronger leader/top-gainer profile.
- Conclusion: no BONK signal is expected under current filters; to catch this setup, need separate low-volume retest/watch logic, not trend BUY.

- 2026-04-20 12:58 Europe/Budapest
- User noted BONKUSDT 4h shows 3 green candles. Verified config and current Binance data.
- Bot/agent currently scan only `TIMEFRAMES=["15m","1h"]`; `AGENT_ALLOWED_TIMEFRAMES=("15m","1h")`, so 4h cannot directly create a BUY.
- On closed Binance 4h candles at check time: last3 closed were red, green, green; the third green on screenshot is likely the forming current candle, not a closed confirmation.
- 4h indicators remain not bullish enough for entry: price ~0.00000608 below EMA20 ~0.000006124, near/below EMA50 ~0.000006080, EMA20 slope -0.84%, RSI ~48, vol_x ~0.66, MACD hist negative.
- Conclusion: 4h is useful as a macro watch context but does not justify current BUY. If desired, add 4h context score/watch label, but only buy after 15m/1h confirmation with volume/reclaim.

- 2026-04-20 23:40 Europe/Budapest
- User requested that all future history of learning-progress metrics and scout/market-agent work be persisted in `D:\Projects\gpt_crypto_bot\CODEX_CONTEXT.md` (user called it `CODEX_CONTENT`; treat `CODEX_CONTEXT.md` as the project context file).
- Standing context rule: append concise snapshots of objective precision/recall, critic/scout decisions, A/B replay results, blocker reasons, and production changes here before final responses, so future analysis can read this file instead of reconstructing from chat.
- Added/verified 4h context score in production logic:
  - `files\config.py`: `FOUR_H_CONTEXT_SCORE_ENABLED=True`, weight `1.0`, leader weight `0.8`, clamp `-6..+8`.
  - `files\market_signal_agent.py` and `files\monitor.py`: compute 4h score from last closed 4h candle context using price vs EMA20, EMA20 vs EMA50, EMA20 slope, up to 3 green closed candles, MACD hist, RSI, and volume. It adjusts candidate ranking only; it does not add 4h as a direct BUY timeframe.
  - `files\bot.py`: positions/ranker line can display stored `4h` context score/label.
  - `files\replay_backtest.py`: fixed replay support to fetch/cache `4h`, pass `cache_4h` into `simulate_portfolio`, and include `BAR_MS["4h"]`.
- Verification:
  - `py_compile config.py market_signal_agent.py monitor.py replay_backtest.py bot.py` passed in `D:\Projects\gpt_crypto_bot\files`.
  - Short replay sanity check: `BONKUSDT CRVUSDT ACHUSDT`, 1 day, 15m/1h, max positions 10, variant `score_replace`, top-gainer min score 18: 5 trades, pnl_total +0.0300%, win_rate 60.0%, objective precision 100.0%, recall 100.0%, score skips 24, cluster skips 0.
  - Live 4h context snapshot: BONKUSDT +5.85 `4h_bull_context`, CRVUSDT +5.26 `4h_bull_context`, ACHUSDT +8.00 `4h_bull_context`.
- Restarted bot and market-agent after verification:
  - Bot PID 25804, startup logs show DataCollector/startup scan disabled and polling started.
  - Market-agent real Python PID 27480; status heartbeat fresh at 21:40:43Z, first cycle completed with 2 restored/open agent positions.

- 2026-04-20 23:52 Europe/Budapest
- User reported portfolio still always contained 2 positions despite intended capacity 10.
- Root cause: `AGENT_MAX_POSITIONS` was already 10, but `AGENT_MAX_POSITIONS_PER_MODE_CLUSTER=2` capped the market-agent because allowed modes (`trend`, `strong_trend`, `impulse_speed`) all map to coarse `momentum` cluster.
- Changed `files\config.py`: `AGENT_MAX_POSITIONS_PER_MODE_CLUSTER` from 2 to 10. Kept `AGENT_MAX_POSITIONS=10` and `AGENT_MAX_POSITIONS_PER_GROUP=2`, so the agent can fill up to 10 leader/top-mover candidates while still limiting same sector/group exposure.
- Changed `files\market_signal_agent.py` prune message from hard-coded "keep top 2" to dynamic `keep top {max_positions}`.
- Verification:
  - `py_compile config.py market_signal_agent.py` passed.
  - One-shot market-agent run after stopping old agent processes restored 2 positions and opened 4 more; `agent_positions.json` now has 6 positions.
  - Background market-agent restarted cleanly with real Python PID 2928; status heartbeat 21:52:00Z, `n_open_positions=6`.
  - Current ranked agent positions: EGLD leader 33.636, AR 31.8881, AVAX 31.07, 1INCH 29.5081, WIF 19.5614, RUNE 11.1332.

- 2026-04-20 23:58 Europe/Budapest
- User requested a Git PR.
- Used `$github:yeet` workflow. Because the worktree is highly mixed, staged only tracked functional-slice files: `files\bot.py`, `files\config.py`, `files\market_signal_agent.py`, `files\monitor.py`, `files\replay_backtest.py`.
- Created branch `codex/top-mover-portfolio-cap` and commit `df6fb37 Fix top mover portfolio capacity`; pushed to `origin/codex/top-mover-portfolio-cap`.
- PR creation via `gh pr create` is blocked because GitHub CLI is not authenticated in this environment (`gh auth status` says not logged in). Git push succeeded via existing git credentials.
- PR compare URL for manual opening: `https://github.com/ostrowsky/gpt_crypto_bot/pull/new/codex/top-mover-portfolio-cap`.
- Validation before commit: `py_compile config.py market_signal_agent.py monitor.py replay_backtest.py bot.py` passed.

- 2026-04-22 09:40 Europe/Budapest
- Checked why no ICPUSDT signal after user screenshot.
- ICPUSDT is in watchlist and not in current open positions; `agent_positions.json` currently has 10/10 open positions.
- Current open agent portfolio leaders: APE 42.93, BONK 36.91, C98 36.14, CRV 35.79, WIF 32.51, KSM 32.21, BCH 31.97, NEAR 29.66, BNT 29.52, XTZ 28.58.
- ICP 15m current closed-bar diagnostics: price 2.535, EMA20 2.51469, EMA50 2.49152, slope +0.389%, ADX 43.5, RSI 66.9, vol_x 0.38, daily_range 5.19%, MACD hist -0.00039988. Blocks: entry false because volume 0.38x < 1.10x; breakout false because no flat breakout; retest false because RSI 66.9 >= 65; surge false because slope not accelerating; impulse false because r1 +0.04% < 1.5%; alignment false because MACD hist <= 0.
- ICP 1h diagnostics: price 2.53, EMA20 2.47916, EMA50 2.47147, slope +0.957%, ADX 17.4, RSI 67.8, vol_x 1.00, MACD hist +0.008964. Blocks: entry false because ADX 17.4 < 22; surge false because vol_x 1.00 < 1.5; only alignment true, but `ALIGNMENT_BUY_ENABLED=False`, so no BUY candidate. 1h leader_raw only 17.86, below current weakest open leader 28.58.
- ICP 4h context score is bullish (+6.07 `4h_bull_context`), but 4h is only ranking context and not a standalone trigger.
- Conclusion: no ICP BUY is expected under current rules. The main improvement needed is portfolio replacement scanning even when 10/10 full, because current market-agent skips new candidate scans when portfolio is full.

- 2026-04-22 10:08 Europe/Budapest
- Implemented full-portfolio replacement scanning in `files\market_signal_agent.py`.
- Behavior change: `_run_cycle` now always scans the full watchlist/timeframes (`n_scanned=210` on current config), even when the agent portfolio is full. Full portfolios are no longer blind to new leaders.
- Added ranking helpers: candidate/position rank tuple = leader_score, today_change_pct, forecast_return_pct, 4h_context_score, ADX, vol_x. Replacement checks the weakest replaceable open position and confirms the candidate fits group/cluster caps after removing that weakest slot.
- Added replacement execution: exits weakest position with reason `portfolio replacement: NEW leader X > OLD leader Y`, sets cooldown for replaced symbol until next local day, then opens the stronger candidate.
- Added config in `files\config.py`: `AGENT_REPLACEMENT_ENABLED=True`, `AGENT_REPLACEMENT_MIN_LEADER_DELTA=0.0`, `AGENT_MAX_REPLACEMENTS_PER_CYCLE=10`.
- Verification:
  - `py_compile config.py market_signal_agent.py` passed.
  - Stopped old market-agent process, ran `market_signal_agent.py --once --log-level INFO`, and restarted background market-agent.
  - Live one-shot and restarted background cycle both scanned `210` symbol/timeframe pairs even with active positions; current status heartbeat `2026-04-22T08:06:57Z`, `n_open_positions=8`.
  - In that live check two existing positions exited by ordinary exit logic before replacement; no new valid candidate passed entry filters in that cycle, so no replacement fired.
- Git: committed and pushed to existing PR branch `codex/top-mover-portfolio-cap`, commit `684205d Add full-portfolio replacement scan`.

- 2026-04-22 22:55 Europe/Budapest
- Checked why no STRKUSDT signal after user screenshot showing strong 4h trend.
- STRKUSDT is in watchlist and not in current open positions. Current agent portfolio is 5/10; market-agent status shows active scan cycle.
- No recent STRK event in the last 1000 `agent_events.jsonl` entries because STRK did not become a valid candidate after basic 15m/1h signal checks.
- STRK 15m current closed-bar diagnostics: price 0.0438, EMA20 0.0426795, EMA50 0.0418257, slope +0.616%, ADX 41.4, RSI 65.9, vol_x 0.73, daily_range 18.70%, MACD hist +0.00000757. Blocks: entry false because volume 0.73x < 1.10x; breakout false because flat range 3.3% > 2.0%; retest false because RSI 65.9 >= 65; surge false because vol_x 0.73 < 1.5; impulse false because r1 +0.92% < 1.5; alignment false because MACD hist not positive for 3 consecutive bars.
- STRK 1h diagnostics: price 0.0426, EMA20 0.0410043, EMA50 0.0390567, slope +2.977%, ADX 47.8, RSI 62.8, vol_x 0.88, daily_range 26.41%, MACD hist +0.0001016. Blocks: entry false because volume 0.88x < 1.10x; breakout false because range 10.9% > 2.0%; retest false because no EMA20 touch in 5 bars; surge false because MACD hist not growing; impulse false because r1 -0.70% < 1.5%; alignment false because daily_range 26.4% > 18.0% (late).
- STRK 4h diagnostics: price 0.0426, EMA20 0.0381506, EMA50 0.0364448, slope +5.777%, ADX 34.9, RSI 72.4, vol_x 4.84, daily_range 33.12%, MACD hist +0.0006308; 4h context score +8.00 `4h_bull_context`.
- Conclusion: no STRK BUY is expected under current rules. 4h is ranking context only; 15m/1h filters treated the move as low-volume locally / already extended. Potential next improvement: explicit 4h-leader watch/replacement mode that can promote very strong 4h trend only after a fresh 15m/1h reclaim/pullback confirmation, not direct 4h BUY.
- 2026-04-22 23:05 Europe/Budapest
- Implemented explicit `4h_leader_watch` mode for market-agent candidates.
- Purpose: catch symbols like STRKUSDT where 4h trend is very strong but normal 15m/1h pattern gates reject the local candle as low-volume/extended.
- Guardrails:
  - Still scans only `15m`/`1h` for entries; 4h alone is not a direct BUY trigger.
  - Requires 4h context score >= 7, local-day change >= 4%, price > EMA20 > EMA50, positive MACD hist, slope >= 0.35%, ADX >= 30, RSI 50-78, daily range <= 35%, and controlled local reclaim/pullback.
  - Allows same-day symbol cooldown bypass only for `4h_leader_watch` with leader score >= 55; same-bar re-entry after exit remains blocked.
  - Optimized 4h API usage: 4h context is fetched only after a cheap 15m/1h prefilter or for an already-classic candidate.
- STRK verification:
  - Point check: STRKUSDT 15m returned `4h_leader_watch`, price 0.0436, leader_score 68.3908, 4h_context_score 8.0, cooldown_bypass=True.
  - Live agent entry logged: STRKUSDT 1h `4h_leader_watch`, price 0.0436, daily_change +17.2043%, forecast proxy 12.0%, trail ATRx2.8.
  - Agent status after restart: fresh heartbeat, cycle finished, `n_scanned=210`, open positions=3.
- Verification: `.venv\Scripts\python.exe -m py_compile config.py market_signal_agent.py bot.py` passed.

- 2026-04-29 19:15 Europe/Budapest
- User asked to evaluate scout/critic learning progress, apply only backtest-confirmed improvements, and restart the bot.
- Added top-gainer critic measurement upgrades in `files\top_gainer_critic.py`:
  - Reads both `files\bot_events.jsonl` and `files\agent_events.jsonl`.
  - Adds `blocked_winner_count`, `missed_reason_counts`, `blocked_winner_reason_counts`, `watchlist_blocked_reason_counts`, `blocked_reason_harm`, per-symbol block windows, source attribution, and opportunity from first block.
  - Normalizes blocker reasons including `portfolio_full`, `open_cluster_cap`, `symbol_cooldown`, `top_gainer_score_gate`, `mtf_correction`, `accuracy_gate`, `agent_mode_disabled`, `agent_replacement_filter`, and `chase_guard`.
- Updated top-gainer Telegram summary in `files\rl_headless_worker.py` to include blocked winners, missed reasons, and top blocker; final-only daily reporting remains controlled by config.
- Historical critic reruns:
  - 2026-04-28 final: watchlist top bought 14/15, early 6/15, blocked winners 1; missed AEVO was mainly `accuracy_gate`.
  - 2026-04-29 midday: watchlist top bought 13/15, early 12/15, blocked winners 2; MEME/FIL were mainly `agent_mode_disabled: alignment`.
- Backtest-gated decisions:
  - Rejected enabling `ALIGNMENT_BUY_ENABLED`: 4d replay worsened total PnL from -36.36% to -51.48% with unchanged capture, so alignment stays diagnostic-only.
  - Accepted stronger top-gainer score gate. 4d replay `TOP_GAINER_SCORE_GATE_MIN_SCORE=34`: trades 375 -> 88, total PnL -36.36% -> -3.01%, capture@15 stayed 1.0. 7d replay: trades 638 -> 176, total PnL -40.99% -> -7.59%, precision 0.2257 -> 0.2898, capture@15 stayed 1.0.
- Production change: `files\config.py` now sets `TOP_GAINER_SCORE_GATE_MIN_SCORE = 34.0`; `ALIGNMENT_BUY_ENABLED` remains `False`.
- Verification: `py_compile config.py top_gainer_critic.py rl_headless_worker.py` passed; `python -m unittest test_top_gainer_critic test_rl_headless_worker` ran 8 tests OK.
- Restart status after change:
  - Bot running PID 21236.
  - RL worker running PID 7456.
  - Market agent running actual Python PID 14252; status heartbeat fresh and first cycle completed.
