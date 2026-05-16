from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import date, datetime, time, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


ROOT = Path(__file__).resolve().parent
FILES_DIR = ROOT / "files"
RUNTIME_DIR = ROOT / ".runtime"
REPORTS_DIR = RUNTIME_DIR / "reports"

def _resolve_local_tz() -> ZoneInfo | timezone:
    try:
        return ZoneInfo("Europe/Budapest")
    except ZoneInfoNotFoundError:
        return timezone.utc


LOCAL_TZ = _resolve_local_tz()

CODEX_CONTEXT_FILE = ROOT / "CODEX_CONTEXT.md"
BOT_EVENTS_FILE = FILES_DIR / "bot_events.jsonl"
POSITIONS_FILE = FILES_DIR / "positions.json"
WATCHLIST_FILE = FILES_DIR / "watchlist.json"
AGENT_EVENTS_FILE = FILES_DIR / "agent_events.jsonl"
RL_STATUS_FILE = RUNTIME_DIR / "rl_worker_status.json"
RL_TRAIN_LATEST_FILE = REPORTS_DIR / "rl_train_latest.json"


def _now_local_iso() -> str:
    return datetime.now(LOCAL_TZ).isoformat(timespec="seconds")


def _load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _iter_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def _parse_utc_ts(raw: str | None) -> datetime | None:
    if not raw:
        return None
    raw = raw.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(raw)
    except ValueError:
        return None


def _local_day_bounds(day_str: str | None) -> tuple[datetime, datetime, date]:
    target_day = datetime.now(LOCAL_TZ).date() if not day_str else datetime.strptime(day_str, "%Y-%m-%d").date()
    start_local = datetime.combine(target_day, time.min, tzinfo=LOCAL_TZ)
    end_local = datetime.combine(target_day, time.max, tzinfo=LOCAL_TZ)
    return start_local, end_local, target_day


def _latest_report(pattern: str) -> Path | None:
    matches = sorted(REPORTS_DIR.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return matches[0] if matches else None


def _resolve_top_gainer_report(day_str: str | None, phase: str) -> Path | None:
    if day_str:
        explicit = REPORTS_DIR / f"top_gainer_critic_{day_str}_{phase}.json"
        if explicit.exists():
            return explicit
    return _latest_report(f"top_gainer_critic_*_{phase}.json")


def _top_reason_counts(items: list[dict[str, Any]], reason_key: str = "reason") -> list[dict[str, Any]]:
    counts = Counter(str(item.get(reason_key) or "").strip() for item in items if item.get(reason_key))
    return [{"reason": reason, "count": count} for reason, count in counts.most_common(10)]


def _parse_any_ts(raw: str | None) -> datetime | None:
    if not raw:
        return None
    try:
        return datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    except ValueError:
        return None


def get_project_context(max_lines: int = 80) -> dict[str, Any]:
    text = CODEX_CONTEXT_FILE.read_text(encoding="utf-8", errors="ignore") if CODEX_CONTEXT_FILE.exists() else ""
    lines = [line.rstrip() for line in text.splitlines()]
    if max_lines > 0:
        lines = lines[:max_lines]
    return {
        "generated_at": _now_local_iso(),
        "source_paths": [str(CODEX_CONTEXT_FILE)],
        "context_excerpt": "\n".join(lines),
        "line_count": len(lines),
    }


def get_portfolio_snapshot() -> dict[str, Any]:
    payload = _load_json(POSITIONS_FILE, {})
    positions = payload if isinstance(payload, dict) else {}
    symbols = sorted(positions.keys())
    return {
        "generated_at": _now_local_iso(),
        "source_paths": [str(POSITIONS_FILE)],
        "positions_count": len(symbols),
        "symbols": symbols,
        "file_mtime": datetime.fromtimestamp(POSITIONS_FILE.stat().st_mtime, tz=LOCAL_TZ).isoformat(timespec="seconds")
        if POSITIONS_FILE.exists()
        else None,
    }


def get_top_movers_audit(day_str: str | None = None, phase: str = "midday", top_n: int = 10) -> dict[str, Any]:
    report_path = _resolve_top_gainer_report(day_str, phase)
    if not report_path:
        return {
            "generated_at": _now_local_iso(),
            "used_cached_report": False,
            "error": "No top gainer critic report found.",
            "source_paths": [],
        }
    report = _load_json(report_path, {})
    summary = report.get("summary") or {}
    watchlist_top = list((report.get("watchlist_top_gainers") or [])[:top_n])
    top_symbols = [str(item.get("symbol")) for item in watchlist_top if item.get("symbol")]
    portfolio = get_portfolio_snapshot()
    portfolio_symbols = set(portfolio.get("symbols") or [])
    captured = [sym for sym in top_symbols if sym in portfolio_symbols]
    missed = [sym for sym in top_symbols if sym not in portfolio_symbols]
    extra_positions = [sym for sym in portfolio.get("symbols") or [] if sym not in set(top_symbols)]
    blocker_counts = _top_reason_counts([item for item in watchlist_top if item.get("reason")])
    trimmed = []
    for item in watchlist_top:
        trimmed.append(
            {
                "symbol": item.get("symbol"),
                "day_change_pct": item.get("day_change_pct"),
                "status": item.get("status"),
                "reason": item.get("reason"),
                "entries_count": item.get("entries_count"),
                "blocked_count": item.get("blocked_count"),
                "first_entry_time": item.get("first_entry_time"),
                "latest_exit_time": item.get("latest_exit_time"),
            }
        )
    return {
        "generated_at": _now_local_iso(),
        "used_cached_report": True,
        "source_paths": [str(report_path), str(POSITIONS_FILE)],
        "target_day_local": report.get("target_day_local"),
        "phase": report.get("phase", phase),
        "capture_rate_pct": round((len(captured) / len(top_symbols) * 100.0), 2) if top_symbols else 0.0,
        "summary": {
            "watchlist_top_count": min(top_n, int(summary.get("watchlist_top_count", len(top_symbols)) or len(top_symbols))),
            "watchlist_top_bought": summary.get("watchlist_top_bought"),
            "watchlist_top_missed": summary.get("watchlist_top_missed"),
            "bot_unique_buys": summary.get("bot_unique_buys"),
            "bot_false_positive_buys": summary.get("bot_false_positive_buys"),
        },
        "captured": captured,
        "missed": missed,
        "extra_portfolio_positions": extra_positions,
        "blocker_counts": blocker_counts,
        "top_movers": trimmed,
    }


def get_signal_summary(day_str: str | None = None, top_n: int = 10) -> dict[str, Any]:
    start_local, end_local, target_day = _local_day_bounds(day_str)
    events = _iter_jsonl(BOT_EVENTS_FILE)
    entries_by_symbol: Counter[str] = Counter()
    event_counts: Counter[str] = Counter()
    blocked_reasons: Counter[str] = Counter()
    blocked_by_symbol: Counter[str] = Counter()
    recent_entries: list[dict[str, Any]] = []

    for event in events:
        ts = _parse_utc_ts(str(event.get("ts") or ""))
        sym = str(event.get("sym") or "").strip()
        event_name = str(event.get("event") or "").strip()
        if not ts or ts.tzinfo is None:
            continue
        ts_local = ts.astimezone(LOCAL_TZ)
        if not (start_local <= ts_local <= end_local):
            continue
        event_counts[event_name] += 1
        if event_name == "entry" and sym:
            entries_by_symbol[sym] += 1
            recent_entries.append(
                {
                    "symbol": sym,
                    "time_local": ts_local.strftime("%H:%M"),
                    "mode": event.get("mode"),
                    "price": event.get("price"),
                }
            )
        if event_name == "blocked":
            reason = str(event.get("reason") or event.get("reason_code") or "").strip()
            if reason:
                blocked_reasons[reason] += 1
            if sym:
                blocked_by_symbol[sym] += 1

    recent_entries = recent_entries[-top_n:]
    return {
        "generated_at": _now_local_iso(),
        "used_cached_report": False,
        "source_paths": [str(BOT_EVENTS_FILE)],
        "target_day_local": target_day.isoformat(),
        "event_counts": dict(event_counts),
        "entries_by_symbol": dict(entries_by_symbol.most_common(top_n)),
        "blocked_reason_counts": [{"reason": r, "count": c} for r, c in blocked_reasons.most_common(top_n)],
        "blocked_symbols": [{"symbol": s, "count": c} for s, c in blocked_by_symbol.most_common(top_n)],
        "recent_entries": recent_entries,
    }


def write_signal_snapshot(day_str: str | None = None, top_n: int = 10) -> dict[str, Any]:
    summary = get_signal_summary(day_str=day_str, top_n=top_n)
    target_day = str(summary.get("target_day_local") or datetime.now(LOCAL_TZ).date().isoformat())
    out_path = REPORTS_DIR / f"signal_summary_{target_day}.json"
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "generated_at": _now_local_iso(),
        "source_paths": [str(BOT_EVENTS_FILE), str(out_path)],
        "target_day_local": target_day,
        "snapshot_path": str(out_path),
        "event_counts": summary.get("event_counts", {}),
    }


def get_rl_summary() -> dict[str, Any]:
    worker_status = _load_json(RL_STATUS_FILE, {})
    latest = _load_json(RL_TRAIN_LATEST_FILE, {})
    training = worker_status.get("training") or {}
    worker = worker_status.get("worker") or {}
    top_gainer = latest.get("top_gainer_critic") or {}
    return {
        "generated_at": _now_local_iso(),
        "used_cached_report": True,
        "source_paths": [str(RL_STATUS_FILE), str(RL_TRAIN_LATEST_FILE)],
        "worker": {
            "started_at": worker.get("started_at"),
            "last_heartbeat": worker.get("last_heartbeat"),
            "mode": worker.get("mode"),
        },
        "training": {
            "run_index": latest.get("training_run_index") or training.get("runs_total"),
            "model_name": latest.get("model_name") or training.get("last_model_name"),
            "rows_total": latest.get("rows_total") or training.get("last_rows_total"),
            "top1_delta": latest.get("top1_delta") or training.get("last_top1_delta"),
            "generated_at_utc": latest.get("generated_at_utc") or training.get("last_finished_at"),
        },
        "top_gainer_critic": {
            "last_phase": top_gainer.get("last_phase"),
            "last_target_day_local": top_gainer.get("last_target_day_local"),
            "last_capture_rate_pct": top_gainer.get("last_capture_rate_pct"),
            "last_report_json": top_gainer.get("last_report_json"),
        },
    }


def get_runtime_health() -> dict[str, Any]:
    rl = get_rl_summary()
    agent_rows = _iter_jsonl(AGENT_EVENTS_FILE)
    latest_agent = agent_rows[-1] if agent_rows else {}
    portfolio = get_portfolio_snapshot()
    return {
        "generated_at": _now_local_iso(),
        "source_paths": [str(AGENT_EVENTS_FILE), str(POSITIONS_FILE), str(RL_STATUS_FILE)],
        "portfolio_positions": portfolio.get("positions_count"),
        "agent_last_event": {
            "ts": latest_agent.get("ts"),
            "event": latest_agent.get("event"),
            "n_entries": latest_agent.get("n_entries"),
            "n_open_positions": latest_agent.get("n_open_positions"),
        },
        "rl_worker_last_heartbeat": ((rl.get("worker") or {}).get("last_heartbeat")),
        "rl_training_last_run": ((rl.get("training") or {}).get("generated_at_utc")),
    }


def get_changes_since(ts: str, top_n: int = 10) -> dict[str, Any]:
    since = _parse_any_ts(ts)
    if since is None:
        return {
            "generated_at": _now_local_iso(),
            "error": f"Invalid timestamp: {ts}",
            "source_paths": [],
        }
    if since.tzinfo is None:
        since = since.replace(tzinfo=timezone.utc)

    bot_events = []
    for item in _iter_jsonl(BOT_EVENTS_FILE):
        item_ts = _parse_any_ts(str(item.get("ts") or ""))
        if item_ts and item_ts >= since:
            bot_events.append(item)

    agent_events = []
    for item in _iter_jsonl(AGENT_EVENTS_FILE):
        item_ts = _parse_any_ts(str(item.get("ts") or ""))
        if item_ts and item_ts >= since:
            agent_events.append(item)

    report_changes = []
    if REPORTS_DIR.exists():
        for path in REPORTS_DIR.glob("*.json"):
            modified = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
            if modified >= since:
                report_changes.append(
                    {
                        "path": str(path),
                        "modified_at": modified.isoformat(timespec="seconds"),
                    }
                )
    report_changes.sort(key=lambda item: item["modified_at"], reverse=True)

    bot_event_counts = Counter(str(item.get("event") or "") for item in bot_events)
    bot_symbol_counts = Counter(str(item.get("sym") or "") for item in bot_events if item.get("sym"))
    agent_event_counts = Counter(str(item.get("event") or "") for item in agent_events)

    return {
        "generated_at": _now_local_iso(),
        "source_paths": [str(BOT_EVENTS_FILE), str(AGENT_EVENTS_FILE), str(REPORTS_DIR)],
        "since": since.isoformat(timespec="seconds"),
        "bot_events_total": len(bot_events),
        "bot_event_counts": dict(bot_event_counts),
        "bot_top_symbols": [{"symbol": sym, "count": count} for sym, count in bot_symbol_counts.most_common(top_n)],
        "agent_events_total": len(agent_events),
        "agent_event_counts": dict(agent_event_counts),
        "changed_reports": report_changes[:top_n],
    }


def update_codex_context(section: str, bullets: list[str]) -> dict[str, Any]:
    existing = CODEX_CONTEXT_FILE.read_text(encoding="utf-8", errors="ignore") if CODEX_CONTEXT_FILE.exists() else ""
    bullet_lines = "\n".join(f"- {line}" for line in bullets)
    snippet = f"\n\n## {section}\n{bullet_lines}\n"
    CODEX_CONTEXT_FILE.write_text(existing.rstrip() + snippet, encoding="utf-8")
    return {
        "generated_at": _now_local_iso(),
        "source_paths": [str(CODEX_CONTEXT_FILE)],
        "updated_section": section,
        "bullet_count": len(bullets),
    }
