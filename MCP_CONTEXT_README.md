# MCP Context Server

This repository now includes a minimal MCP server for compact local context reads:

- Server entrypoint: `D:\Projects\gpt_crypto_bot\mcp_server.py`
- Logic module: `D:\Projects\gpt_crypto_bot\mcp_context_tools.py`
- Primary context file: `D:\Projects\gpt_crypto_bot\CODEX_CONTEXT.md`

## Goal
Keep token usage low by returning compact JSON summaries instead of raw logs or broad file scans.

## Implemented tools
- `project_context(max_lines=80)`
- `top_movers_audit(day_str=None, phase="midday", top_n=10)`
- `signal_summary(day_str=None, top_n=10)`
- `portfolio_snapshot()`
- `rl_summary()`
- `runtime_health()`
- `changes_since(ts, top_n=10)`
- `write_daily_signal_snapshot(day_str=None, top_n=10)`
- `codex_context_append(section, bullets)`

## Current design
- Prefer `.runtime\reports\top_gainer_critic_*` over direct Binance/API reads.
- Prefer `rl_train_latest.json` and `rl_worker_status.json` for RL summary.
- Read raw `bot_events.jsonl` only inside `signal_summary`, and return compact aggregates.
- Use `changes_since(ts)` for delta reads instead of re-reading entire day context.
- Use `write_daily_signal_snapshot()` to persist compact daily summaries for later MCP reads.
- Treat `CODEX_CONTEXT.md` as the durable handoff note.

## Example run
```bash
python D:\Projects\gpt_crypto_bot\mcp_server.py
```

## Known follow-ups
- Normalize mojibake blocker reasons during aggregation.
- Add narrow regression tests for report fallback behavior and JSON serializability.
