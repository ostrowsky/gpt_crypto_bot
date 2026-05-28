from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = ROOT.parent
REPORT_DIR = WORKSPACE_ROOT / ".runtime" / "reports"
HISTORY_FILE = REPORT_DIR / "top_gainer_critic_history.jsonl"
DENOMINATOR = "exchange_top_filtered_to_watchlist"


def recompute_report(report: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    before = dict(report.get("summary") or {})
    exchange_top = list(report.get("exchange_top_gainers") or [])
    old_watchlist_top = list(report.get("watchlist_top_gainers") or [])
    watchlist_universe_top = list(report.get("watchlist_universe_top_gainers") or old_watchlist_top)
    filtered = [row for row in exchange_top if bool(row.get("in_watchlist"))]

    bought_top = [row for row in filtered if row.get("status") == "bought"]
    missed_top = [row for row in filtered if row.get("status") != "bought"]
    blocked_winners = [row for row in missed_top if int(row.get("blocked_count") or 0) > 0]
    early_captured = [row for row in bought_top if float(row.get("capture_ratio") or 0.0) >= 0.35]

    missed_reason_counts = Counter(str(row.get("missed_reason_code") or row.get("status") or "unknown") for row in missed_top)
    missed_reason_symbols: dict[str, list[str]] = {}
    for row in missed_top:
        reason = str(row.get("missed_reason_code") or row.get("status") or "unknown")
        missed_reason_symbols.setdefault(reason, []).append(str(row.get("symbol") or ""))

    watchlist_blocked_reason_counts: Counter = Counter()
    for row in filtered:
        for reason, count in (row.get("blocked_reason_counts") or {}).items():
            watchlist_blocked_reason_counts[str(reason)] += int(count or 0)

    blocked_winner_reason_counts: Counter = Counter()
    for row in blocked_winners:
        for reason, count in (row.get("blocked_reason_counts") or {}).items():
            blocked_winner_reason_counts[str(reason)] += int(count or 0)

    summary = dict(before)
    summary.update(
        {
            "exchange_top_count": len(exchange_top),
            "exchange_top_in_watchlist": len(filtered),
            "watchlist_top_count": len(filtered),
            "watchlist_top_denominator": DENOMINATOR,
            "watchlist_universe_top_count": len(watchlist_universe_top),
            "watchlist_top_bought": len(bought_top),
            "watchlist_top_early_captured": len(early_captured),
            "watchlist_top_missed": len(missed_top),
            "watchlist_top_capture_rate_pct": _pct(len(bought_top), len(filtered)),
            "watchlist_top_early_capture_rate_pct": _pct(len(early_captured), len(filtered)),
            "blocked_winner_count": len(blocked_winners),
            "blocked_winner_symbols": [str(row.get("symbol") or "") for row in blocked_winners],
            "missed_reason_counts": dict(missed_reason_counts),
            "watchlist_blocked_reason_counts": dict(watchlist_blocked_reason_counts),
            "blocked_winner_reason_counts": dict(blocked_winner_reason_counts),
        }
    )
    report["summary"] = summary
    report["watchlist_top_gainers"] = filtered
    report["watchlist_universe_top_gainers"] = watchlist_universe_top
    report["missed_reason_symbols"] = missed_reason_symbols
    report["blocked_reason_harm"] = _blocked_reason_harm(filtered)
    report["why_no_signal_examples"] = [
        {
            "symbol": row.get("symbol"),
            "missed_reason_code": row.get("missed_reason_code"),
            "blocked_reason_counts": row.get("blocked_reason_counts"),
            "first_block_time": row.get("first_block_time"),
            "last_block_time": row.get("last_block_time"),
            "latest_block_reason_code": row.get("latest_block_reason_code"),
            "latest_block_reason": row.get("latest_block_reason"),
            "latest_block_gate": row.get("latest_block_gate"),
        }
        for row in blocked_winners[:10]
    ]
    after = dict(summary)
    return report, {"before": _compact(before), "after": _compact(after)}


def recompute_reports(reports_dir: Path = REPORT_DIR, *, write: bool = False) -> dict[str, Any]:
    changed = []
    for path in sorted(reports_dir.glob("top_gainer_critic_*.json")):
        if path.name.endswith("_history.json"):
            continue
        try:
            report = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
        except Exception:
            continue
        if not isinstance(report, dict) or not report.get("exchange_top_gainers"):
            continue
        old_payload = json.dumps(report, ensure_ascii=False, sort_keys=True)
        report, delta = recompute_report(report)
        new_payload = json.dumps(report, ensure_ascii=False, indent=2)
        if json.dumps(report, ensure_ascii=False, sort_keys=True) != old_payload:
            changed.append({"path": str(path), **delta})
            if write:
                path.write_text(new_payload, encoding="utf-8")
                txt_path = path.with_suffix(".txt")
                if txt_path.exists():
                    txt_path.write_text(_render_minimal_text(report), encoding="utf-8")
    history = _rewrite_history(reports_dir, write=write)
    return {"write": write, "reports_changed": len(changed), "changed": changed, "history": history}


def _rewrite_history(reports_dir: Path, *, write: bool) -> dict[str, Any]:
    history = reports_dir / "top_gainer_critic_history.jsonl"
    if not history.exists():
        return {"exists": False, "rows_changed": 0}
    summaries: dict[tuple[str, str], dict[str, Any]] = {}
    for path in reports_dir.glob("top_gainer_critic_*.json"):
        try:
            report = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
        except Exception:
            continue
        day = str(report.get("target_day_local") or "")
        phase = str(report.get("phase") or "")
        if day and phase:
            summaries[(day, phase)] = dict(report.get("summary") or {})
    rows = []
    changed = 0
    for line in history.read_text(encoding="utf-8", errors="ignore").splitlines():
        try:
            row = json.loads(line)
        except Exception:
            rows.append(line)
            continue
        key = (str(row.get("target_day_local") or ""), str(row.get("phase") or ""))
        if key in summaries and row.get("summary") != summaries[key]:
            row["summary"] = summaries[key]
            changed += 1
        rows.append(json.dumps(row, ensure_ascii=False))
    if write and changed:
        history.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return {"exists": True, "rows_changed": changed}


def _blocked_reason_harm(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for row in rows:
        if row.get("status") == "bought" or int(row.get("blocked_count") or 0) <= 0:
            continue
        opportunity = _float(row.get("opportunity_from_first_block_pct"))
        if opportunity is None:
            opportunity = _float(row.get("opportunity_no_entry_pct")) or 0.0
        for reason, count in (row.get("blocked_reason_counts") or {}).items():
            item = grouped.setdefault(str(reason), {"reason_code": str(reason), "missed_symbols": [], "blocked_events": 0, "missed_opportunity_pct": 0.0})
            item["missed_symbols"].append(str(row.get("symbol") or ""))
            item["blocked_events"] += int(count or 0)
            item["missed_opportunity_pct"] += float(opportunity)
    out = []
    for item in grouped.values():
        symbols = sorted(set(item["missed_symbols"]))
        out.append({
            "reason_code": item["reason_code"],
            "missed_symbols": symbols,
            "missed_symbols_count": len(symbols),
            "blocked_events": item["blocked_events"],
            "missed_opportunity_pct": round(float(item["missed_opportunity_pct"]), 4),
        })
    out.sort(key=lambda item: (item["missed_symbols_count"], item["blocked_events"]), reverse=True)
    return out


def _render_minimal_text(report: dict[str, Any]) -> str:
    summary = report.get("summary") or {}
    lines = [
        f"Top-gainer critic for {report.get('target_day_local')} ({report.get('phase')})",
        "",
        "Summary:",
        f"  exchange top in watchlist: {summary.get('exchange_top_in_watchlist')}/{summary.get('exchange_top_count')}",
        f"  watchlist top bought: {summary.get('watchlist_top_bought')}/{summary.get('watchlist_top_count')} ({float(summary.get('watchlist_top_capture_rate_pct') or 0):.1f}%)",
        f"  denominator: {summary.get('watchlist_top_denominator')}; watchlist-universe diagnostic top={summary.get('watchlist_universe_top_count')}",
        f"  early captures: {summary.get('watchlist_top_early_captured')}/{summary.get('watchlist_top_count')} ({float(summary.get('watchlist_top_early_capture_rate_pct') or 0):.1f}%)",
        f"  blocked winners: {summary.get('blocked_winner_count')}",
        "",
        "Watchlist-filtered exchange top gainers and bot reaction:",
    ]
    for idx, row in enumerate(report.get("watchlist_top_gainers") or [], start=1):
        lines.append(f"{idx}. {row.get('symbol')} {float(row.get('day_change_pct') or 0):+.2f}% status={row.get('status')}")
    return "\n".join(lines) + "\n"


def _compact(summary: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "exchange_top_count",
        "exchange_top_in_watchlist",
        "watchlist_top_count",
        "watchlist_top_bought",
        "watchlist_top_early_captured",
        "watchlist_top_missed",
        "watchlist_top_capture_rate_pct",
        "watchlist_top_early_capture_rate_pct",
        "blocked_winner_count",
    ]
    return {key: summary.get(key) for key in keys}


def _pct(num: int, den: int) -> float:
    return round(100.0 * num / den, 2) if den else 0.0


def _float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        value = float(value)
        return value if value == value else None
    except Exception:
        return None


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    parser = argparse.ArgumentParser()
    parser.add_argument("--reports-dir", type=Path, default=REPORT_DIR)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    result = recompute_reports(args.reports_dir, write=args.write)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
