from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable


DEFAULT_REPORTS_DIR = Path(__file__).resolve().parent.parent / ".runtime" / "reports"


def normalize_symbol(symbol: str) -> str:
    return symbol.strip().upper().replace("/", "")


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def find_latest_critic_report(reports_dir: Path, *, phase: str | None = None) -> Path | None:
    candidates: list[Path] = []
    for path in reports_dir.glob("top_gainer_critic_*.json"):
        if phase and not path.name.endswith(f"_{phase}.json"):
            continue
        if path.name.endswith("_history.json"):
            continue
        candidates.append(path)
    if not candidates:
        return None
    return max(candidates, key=lambda p: (p.stat().st_mtime, p.name))


def build_focus_audit(report_path: Path, symbols: Iterable[str]) -> dict[str, Any]:
    payload = _load_json(report_path)
    wanted = [normalize_symbol(s) for s in symbols if normalize_symbol(s)]
    rows_by_symbol = {
        normalize_symbol(str(row.get("symbol") or "")): row
        for row in payload.get("watchlist_top_gainers", [])
        if isinstance(row, dict)
    }
    focus_rows: list[dict[str, Any]] = []
    for symbol in wanted:
        row = rows_by_symbol.get(symbol)
        if not row:
            focus_rows.append({
                "symbol": symbol,
                "status": "not_in_latest_top_report",
                "note": "symbol is absent from latest top-gainer critic report",
            })
            continue
        reason_counts = row.get("blocked_reason_counts") or {}
        if not isinstance(reason_counts, dict):
            reason_counts = {}
        dominant_reason = None
        if reason_counts:
            dominant_reason = max(reason_counts.items(), key=lambda kv: int(kv[1] or 0))[0]
        focus_rows.append({
            "symbol": symbol,
            "status": row.get("status"),
            "day_change_pct": row.get("day_change_pct"),
            "entries_count": row.get("entries_count"),
            "blocked_count": row.get("blocked_count"),
            "dominant_block_reason": dominant_reason,
            "first_block_time": row.get("first_block_time"),
            "first_block_reason_code": row.get("first_block_reason_code"),
            "first_block_price": row.get("first_block_price"),
            "latest_block_time": row.get("last_block_time"),
            "latest_block_reason_code": row.get("latest_block_reason_code"),
            "latest_block_reason": row.get("latest_block_reason"),
            "missed_reason_code": row.get("missed_reason_code"),
            "first_entry_time": row.get("first_entry_time"),
            "first_entry_mode": row.get("first_entry_mode"),
            "first_entry_price": row.get("first_entry_price"),
            "capture_ratio_at_entry": row.get("capture_ratio_at_entry"),
            "latest_exit_time": row.get("latest_exit_time"),
            "latest_exit_pnl_pct": row.get("latest_exit_pnl_pct"),
            "exit_efficiency": row.get("exit_efficiency"),
            "giveback_pct": row.get("giveback_pct"),
            "opportunity_no_entry_pct": row.get("opportunity_no_entry_pct"),
            "opportunity_from_first_block_pct": row.get("opportunity_from_first_block_pct"),
            "blocked_reason_counts": reason_counts,
        })
    return {
        "source_report": str(report_path),
        "target_day_local": payload.get("target_day_local"),
        "phase": payload.get("phase"),
        "summary": payload.get("summary") or {},
        "focus_symbols": focus_rows,
    }


def render_text(audit: dict[str, Any]) -> str:
    lines = [
        f"Blocked-winner focus audit — {audit.get('target_day_local') or '?'} {audit.get('phase') or ''}".rstrip(),
        f"source: {audit.get('source_report')}",
        "",
    ]
    for row in audit.get("focus_symbols", []):
        symbol = row.get("symbol")
        status = row.get("status")
        if status == "not_in_latest_top_report":
            lines.append(f"- {symbol}: not in latest critic top report")
            continue
        lines.append(
            f"- {symbol}: {status}, day {row.get('day_change_pct')}%, "
            f"entries {row.get('entries_count')}, blocks {row.get('blocked_count')}"
        )
        if row.get("first_entry_time"):
            lines.append(
                f"  BUY: {row.get('first_entry_time')} {row.get('first_entry_mode')} "
                f"price={row.get('first_entry_price')} capture={row.get('capture_ratio_at_entry')}"
            )
        else:
            lines.append(
                f"  LOST: first block {row.get('first_block_time')} "
                f"{row.get('first_block_reason_code')} price={row.get('first_block_price')}; "
                f"latest={row.get('latest_block_reason_code')} ({row.get('latest_block_reason')})"
            )
        if row.get("latest_exit_time"):
            lines.append(
                f"  EXIT: {row.get('latest_exit_time')} pnl={row.get('latest_exit_pnl_pct')}% "
                f"eff={row.get('exit_efficiency')} giveback={row.get('giveback_pct')}%"
            )
        if row.get("dominant_block_reason"):
            lines.append(f"  dominant blocker: {row.get('dominant_block_reason')} {row.get('blocked_reason_counts')}")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a compact focus audit from the latest top-gainer critic report.")
    parser.add_argument("--reports-dir", type=Path, default=DEFAULT_REPORTS_DIR)
    parser.add_argument("--phase", default=None, help="Optional critic phase, e.g. midday or final.")
    parser.add_argument("--symbols", required=True, help="Comma-separated symbols, e.g. STRKUSDT,TIAUSDT")
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-txt", type=Path, default=None)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    report_path = find_latest_critic_report(args.reports_dir, phase=args.phase)
    if not report_path:
        raise SystemExit(f"no top_gainer_critic report found in {args.reports_dir}")
    audit = build_focus_audit(report_path, args.symbols.split(","))
    text = render_text(audit)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.output_txt:
        args.output_txt.parent.mkdir(parents=True, exist_ok=True)
        args.output_txt.write_text(text, encoding="utf-8")
    if args.json:
        print(json.dumps(audit, ensure_ascii=False, indent=2))
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
