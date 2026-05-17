from __future__ import annotations

import json
from collections import Counter
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = ROOT.parent
EVENTS_FILE = ROOT / "v2_shadow_events.jsonl"
REPORT_DIR = WORKSPACE_ROOT / ".runtime" / "reports"


def _local_day(raw: str, timezone_name: str) -> str:
    return datetime.fromisoformat(raw.replace("Z", "+00:00")).astimezone(ZoneInfo(timezone_name)).date().isoformat()


def _is_bootstrap(row: dict[str, Any]) -> bool:
    return bool(row.get("bootstrap") is True or ("previous_state" in row and row.get("previous_state") is None))


def build_summary(
    target_day: date,
    events_file: Path = EVENTS_FILE,
    *,
    timezone_name: str = "Europe/Budapest",
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    if events_file.exists():
        for line in events_file.read_text(encoding="utf-8", errors="ignore").splitlines():
            try:
                row = json.loads(line)
            except Exception:
                continue
            if _is_bootstrap(row):
                continue
            if _local_day(str(row.get("ts") or ""), timezone_name) != target_day.isoformat():
                continue
            rows.append(row)

    upside_rows = [row for row in rows if row.get("state") in {"emerging_move", "confirmed_trend"}]
    confirmed_rows = [row for row in rows if row.get("state") == "confirmed_trend"]
    deescalations = [row for row in rows if row.get("state") == "noise"]
    state_counts = Counter(str(row.get("state") or "") for row in rows)
    symbol_latest: dict[str, dict[str, Any]] = {}
    for row in rows:
        symbol_latest[str(row.get("sym") or "")] = row

    upside_symbols = sorted({str(row.get("sym") or "") for row in upside_rows if row.get("sym")})
    confirmation_ratio = round(len(confirmed_rows) / len(upside_rows), 4) if upside_rows else 0.0
    latest_upside = [row for row in symbol_latest.values() if row.get("state") in {"emerging_move", "confirmed_trend"}]
    latest_upside.sort(key=lambda row: str(row.get("ts") or ""), reverse=True)
    return {
        "target_day_local": target_day.isoformat(),
        "timezone": timezone_name,
        "events_total": len(rows),
        "upside_discovery_events": len(upside_rows),
        "confirmed_trend_events": len(confirmed_rows),
        "unique_upside_symbols": len(upside_symbols),
        "deescalation_to_noise_events": len(deescalations),
        "confirmation_ratio": confirmation_ratio,
        "state_counts": dict(state_counts),
        "upside_symbols": upside_symbols,
        "latest_upside": latest_upside[:10],
    }


def render_text(summary: dict[str, Any]) -> str:
    symbols = ", ".join(summary.get("upside_symbols") or []) or "none"
    return (
        f"V2 shadow daily summary — {summary['target_day_local']} ({summary['timezone']})\n"
        f"events={summary['events_total']} | upside={summary['upside_discovery_events']} | "
        f"confirmed={summary['confirmed_trend_events']} | unique={summary['unique_upside_symbols']}\n"
        f"confirmation_ratio={summary['confirmation_ratio']:.2f} | "
        f"deescalations_to_noise={summary['deescalation_to_noise_events']}\n"
        f"upside_symbols: {symbols}"
    )


def save_report(summary: dict[str, Any]) -> dict[str, str]:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"v2_shadow_daily_{summary['target_day_local']}"
    json_path = REPORT_DIR / f"{stem}.json"
    txt_path = REPORT_DIR / f"{stem}.txt"
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    txt_path.write_text(render_text(summary), encoding="utf-8")
    return {"json": str(json_path), "txt": str(txt_path)}
