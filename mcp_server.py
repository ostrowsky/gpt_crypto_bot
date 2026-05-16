from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from mcp_context_tools import (
    get_portfolio_snapshot,
    get_project_context,
    get_changes_since,
    get_rl_summary,
    get_runtime_health,
    get_signal_summary,
    get_top_movers_audit,
    update_codex_context,
    write_signal_snapshot,
)


mcp = FastMCP("crypto-bot-context")


@mcp.tool()
def project_context(max_lines: int = 80) -> dict:
    """Return the compact CODEX context for this repository."""

    return get_project_context(max_lines=max_lines)


@mcp.tool()
def top_movers_audit(day_str: str | None = None, phase: str = "midday", top_n: int = 10) -> dict:
    """Return a compact top movers audit using the cached critic report when available."""

    return get_top_movers_audit(day_str=day_str, phase=phase, top_n=top_n)


@mcp.tool()
def signal_summary(day_str: str | None = None, top_n: int = 10) -> dict:
    """Return compact signal, entry, and blocker summaries for the selected local day."""

    return get_signal_summary(day_str=day_str, top_n=top_n)


@mcp.tool()
def portfolio_snapshot() -> dict:
    """Return the current portfolio symbol list and file freshness."""

    return get_portfolio_snapshot()


@mcp.tool()
def rl_summary() -> dict:
    """Return a compact RL worker and latest training summary."""

    return get_rl_summary()


@mcp.tool()
def runtime_health() -> dict:
    """Return a small runtime health snapshot for bot, agent, portfolio, and RL status."""

    return get_runtime_health()


@mcp.tool()
def changes_since(ts: str, top_n: int = 10) -> dict:
    """Return compact bot, agent, and report changes since the provided ISO timestamp."""

    return get_changes_since(ts=ts, top_n=top_n)


@mcp.tool()
def write_daily_signal_snapshot(day_str: str | None = None, top_n: int = 10) -> dict:
    """Persist a compact daily signal summary snapshot into .runtime/reports."""

    return write_signal_snapshot(day_str=day_str, top_n=top_n)


@mcp.tool()
def codex_context_append(section: str, bullets: list[str]) -> dict:
    """Append concise durable notes to CODEX_CONTEXT.md."""

    return update_codex_context(section=section, bullets=bullets)


if __name__ == "__main__":
    mcp.run()
