# Local Context MCP

Status: proposed for local tooling  
Last updated: 2026-05-17

## Purpose

Provide compact, local, read-mostly context snapshots for the bot so operators and coding
assistants do not need to repeatedly scan broad logs, raw event streams, or large report folders.

## Scope

- Expose a small local MCP server backed by cached repository/runtime files.
- Prefer compact JSON summaries over raw file dumps.
- Reuse existing reports and state files whenever possible instead of calling exchanges directly.
- Keep write operations narrow and explicit: daily summary snapshot generation and durable
  append-only context notes.

## Non-Goals

- No trading decisions.
- No BUY/SELL mutation.
- No model retraining.
- No direct production gate changes.
- No replacement for the existing evaluator/replay tools.

## Initial Tools

- `project_context(max_lines=80)`
- `top_movers_audit(day_str=None, phase="midday", top_n=10)`
- `signal_summary(day_str=None, top_n=10)`
- `portfolio_snapshot()`
- `rl_summary()`
- `runtime_health()`
- `changes_since(ts, top_n=10)`
- `write_daily_signal_snapshot(day_str=None, top_n=10)`
- `codex_context_append(section, bullets)`

## Acceptance Criteria

1. The server starts locally without exchange/API access.
2. Read tools return JSON-serializable payloads from local files only.
3. Missing files degrade to concise empty/error payloads rather than crashing.
4. The package documents file sources and intended operator usage.
5. `mcp` and `tzdata` dependencies are declared because the package depends on both
   the MCP server runtime and local timezone handling on Windows.

## Verification

- `python -m py_compile mcp_context_tools.py mcp_server.py`
- Import both modules under the repository runtime interpreter.
- Call representative pure helper functions against the local repository state.

## Promotion / Rollback

This package is local tooling only. If it becomes noisy or unnecessary, rollback is simply
removing the server/docs/dependencies; no trading behavior should change.
