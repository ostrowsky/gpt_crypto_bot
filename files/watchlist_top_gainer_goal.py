from __future__ import annotations

import argparse
import json
import os
import threading
from collections import Counter
from datetime import date, datetime, time, timezone
from pathlib import Path
from statistics import median
from typing import Any, Dict, Iterable, List, Optional, Sequence
from zoneinfo import ZoneInfo

import ml_dataset
import top_gainer_critic


ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = ROOT.parent
RUNTIME_DIR = WORKSPACE_ROOT / ".runtime"
REPORT_DIR = RUNTIME_DIR / "reports"
HISTORY_FILE = REPORT_DIR / "watchlist_top_gainer_goal_history.jsonl"
BOT_EVENTS_FILE = ROOT / "bot_events.jsonl"

DEFAULT_CUTOFF_HOUR = 22
DEFAULT_CHECKPOINT_HOURS = (1, 4, 8, 12, 18, 22)
DEFAULT_PRECISION_TOP_NS = (5, 10, 20)
DEFAULT_TZ = top_gainer_critic.DEFAULT_TZ


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the watchlist-wide daily goal report for symbols that must reach top gainers by a local cutoff."
    )
    parser.add_argument("--date", required=True, help="Local date in YYYY-MM-DD.")
    parser.add_argument("--cutoff-hour", type=int, default=DEFAULT_CUTOFF_HOUR)
    parser.add_argument("--top", type=int, default=top_gainer_critic.DEFAULT_TOP_N)
    parser.add_argument("--min-quote-volume", type=float, default=top_gainer_critic.DEFAULT_MIN_QUOTE_VOLUME)
    parser.add_argument("--timezone", default=DEFAULT_TZ)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(rec, dict):
            rows.append(rec)
    return rows


def _parse_utc_ts(raw: Any) -> datetime | None:
    if not raw:
        return None
    text = str(raw).strip()
    if not text:
        return None
    for fmt in ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S.%fZ"):
        try:
            return datetime.strptime(text, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


def _record_local_dt(rec: Dict[str, Any], tz: ZoneInfo) -> datetime | None:
    ts = _parse_utc_ts(rec.get("ts_signal"))
    if ts is None:
        bar_ts = rec.get("bar_ts")
        try:
            bar_ts = int(bar_ts)
        except Exception:
            return None
        ts = datetime.fromtimestamp(bar_ts / 1000, tz=timezone.utc)
    return ts.astimezone(tz)


def _local_window(target_day: date, cutoff_hour: int, tz: ZoneInfo) -> tuple[datetime, datetime]:
    start_local = datetime.combine(target_day, time.min, tzinfo=tz)
    end_local = datetime.combine(target_day, time(hour=int(cutoff_hour)), tzinfo=tz)
    return start_local, end_local


def _teacher_key(cutoff_hour: int) -> str:
    return f"watchlist_top_gainer_{int(cutoff_hour):02d}h"


def _normalize_hours(hours: Sequence[int], cutoff_hour: int) -> List[int]:
    normalized = {
        int(hour)
        for hour in hours
        if 0 < int(hour) <= int(cutoff_hour)
    }
    normalized.add(int(cutoff_hour))
    return sorted(normalized)


def _normalize_top_ns(values: Sequence[int]) -> List[int]:
    out = {max(1, int(v)) for v in values}
    return sorted(out)


def _entry_local_dt(target_day: date, hhmm: Any, tz: ZoneInfo) -> datetime | None:
    text = str(hhmm or "").strip()
    if not text:
        return None
    try:
        parts = text.split(":")
        hour = int(parts[0])
        minute = int(parts[1]) if len(parts) > 1 else 0
    except Exception:
        return None
    return datetime.combine(target_day, time(hour=hour, minute=minute), tzinfo=tz)


def _load_first_unique_entries(
    *,
    start_local: datetime,
    end_local: datetime,
    tz: ZoneInfo,
    watchlist: set[str],
) -> List[Dict[str, Any]]:
    first_by_symbol: Dict[str, Dict[str, Any]] = {}
    for rec in _iter_jsonl(BOT_EVENTS_FILE):
        if rec.get("event") != "entry":
            continue
        sym = str(rec.get("sym", "")).strip()
        if not sym:
            continue
        if watchlist and sym not in watchlist:
            continue
        ts = _parse_utc_ts(rec.get("ts"))
        if ts is None:
            continue
        ts_local = ts.astimezone(tz)
        if not (start_local <= ts_local <= end_local):
            continue
        current = first_by_symbol.get(sym)
        if current is None or ts_local < current["ts_local"]:
            first_by_symbol[sym] = {
                "symbol": sym,
                "ts_local": ts_local,
                "time_local": ts_local.strftime("%H:%M"),
                "mode": rec.get("mode"),
            }
    return sorted(first_by_symbol.values(), key=lambda item: (item["ts_local"], item["symbol"]))


def annotate_ml_dataset_teacher(report: Dict[str, Any]) -> Dict[str, Any]:
    if not ml_dataset.ML_FILE.exists():
        return {
            "teacher_key": "",
            "rows_scanned": 0,
            "rows_in_window": 0,
            "rows_annotated": 0,
            "positive_rows": 0,
            "negative_rows": 0,
            "positive_symbols_with_rows": 0,
            "positive_symbols_missing_rows": [],
            "mandatory_positive_coverage_pct": 0.0,
        }

    target_day_text = str(report.get("target_day_local", "") or "").strip()
    if not target_day_text:
        raise ValueError("Goal report missing target_day_local")
    settings = report.get("settings") or {}
    timezone_name = str(settings.get("timezone", DEFAULT_TZ) or DEFAULT_TZ)
    cutoff_hour = int(settings.get("cutoff_hour_local", DEFAULT_CUTOFF_HOUR) or DEFAULT_CUTOFF_HOUR)
    top_n = int(settings.get("top_n", top_gainer_critic.DEFAULT_TOP_N) or top_gainer_critic.DEFAULT_TOP_N)
    target_day = date.fromisoformat(target_day_text)
    tz = ZoneInfo(timezone_name)
    start_local, end_local = _local_window(target_day, cutoff_hour, tz)
    teacher_key = _teacher_key(cutoff_hour)
    watchlist = top_gainer_critic._load_watchlist()

    watchlist_map: Dict[str, Dict[str, Any]] = {}
    watchlist_rank_map: Dict[str, int] = {}
    for idx, item in enumerate(report.get("watchlist_top_gainers") or [], start=1):
        if not isinstance(item, dict):
            continue
        sym = str(item.get("symbol", "")).strip()
        if not sym:
            continue
        watchlist_map[sym] = item
        watchlist_rank_map[sym] = idx

    rows_scanned = 0
    rows_in_window = 0
    positive_rows = 0
    negative_rows = 0
    symbols_in_window: set[str] = set()
    positive_symbols_with_rows: set[str] = set()

    def _payload_for_row(rec: Dict[str, Any]) -> tuple[str, Dict[str, Any]] | None:
        local_dt = _record_local_dt(rec, tz)
        if local_dt is None or not (start_local <= local_dt <= end_local):
            return None
        sym = str(rec.get("sym", "")).strip()
        if not sym:
            return None
        if watchlist and sym not in watchlist:
            return None
        summary = watchlist_map.get(sym)
        minutes_to_cutoff = max(0, int((end_local - local_dt).total_seconds() // 60))
        payload = {
            "goal": "watchlist_top_gainer_by_cutoff",
            "target_day_local": target_day_text,
            "timezone": timezone_name,
            "cutoff_hour_local": cutoff_hour,
            "top_n": top_n,
            "watchlist_top_gainer": summary is not None,
            "watchlist_top_rank": watchlist_rank_map.get(sym),
            "day_change_pct": None if summary is None else summary.get("day_change_pct"),
            "quote_volume_24h": None if summary is None else summary.get("quote_volume_24h"),
            "minutes_to_cutoff": minutes_to_cutoff,
        }
        return sym, payload

    def _preview_mutate(rec: Dict[str, Any]) -> bool:
        nonlocal rows_scanned, rows_in_window, positive_rows, negative_rows
        rows_scanned += 1
        built = _payload_for_row(rec)
        if built is None:
            return False
        sym, payload = built
        rows_in_window += 1
        symbols_in_window.add(sym)
        if payload["watchlist_top_gainer"]:
            positive_rows += 1
            positive_symbols_with_rows.add(sym)
        else:
            negative_rows += 1
        teacher = rec.get("teacher") or {}
        return teacher.get(teacher_key) != payload

    _, maybe_changed, maybe_bad_rows = ml_dataset._collect_mutated_lines(_preview_mutate)
    rows_annotated = 0
    if maybe_changed or maybe_bad_rows:
        def _commit_mutate(rec: Dict[str, Any]) -> bool:
            nonlocal rows_annotated
            built = _payload_for_row(rec)
            if built is None:
                return False
            _, payload = built
            teacher = rec.setdefault("teacher", {})
            if teacher.get(teacher_key) == payload:
                return False
            teacher[teacher_key] = payload
            rows_annotated += 1
            return True

        with ml_dataset._dataset_io_lock():
            updated, changed, had_bad_rows = ml_dataset._collect_mutated_lines(_commit_mutate)
            if changed or had_bad_rows:
                ml_dataset.ML_FILE.parent.mkdir(parents=True, exist_ok=True)
                tmp = ml_dataset.ML_FILE.with_name(
                    f"{ml_dataset.ML_FILE.name}.{os.getpid()}.{threading.get_ident()}.tmp"
                )
                tmp.write_text("\n".join(updated) + "\n", encoding="utf-8")
                ml_dataset._atomic_replace_with_retry(tmp, ml_dataset.ML_FILE)

    missing_positive_symbols = sorted(sym for sym in watchlist_map if sym not in positive_symbols_with_rows)
    mandatory_positive_coverage_pct = (
        round(len(positive_symbols_with_rows) / len(watchlist_map) * 100.0, 2)
        if watchlist_map
        else 0.0
    )
    return {
        "teacher_key": teacher_key,
        "rows_scanned": rows_scanned,
        "rows_in_window": rows_in_window,
        "rows_annotated": rows_annotated,
        "positive_rows": positive_rows,
        "negative_rows": negative_rows,
        "symbols_in_window": len(symbols_in_window),
        "positive_symbols_with_rows": len(positive_symbols_with_rows),
        "positive_symbols_missing_rows": missing_positive_symbols,
        "mandatory_positive_coverage_pct": mandatory_positive_coverage_pct,
    }


def build_goal_report(
    *,
    target_day: date,
    cutoff_hour: int = DEFAULT_CUTOFF_HOUR,
    timezone_name: str = DEFAULT_TZ,
    top_n: int = top_gainer_critic.DEFAULT_TOP_N,
    min_quote_volume: float = top_gainer_critic.DEFAULT_MIN_QUOTE_VOLUME,
    checkpoint_hours: Sequence[int] = DEFAULT_CHECKPOINT_HOURS,
    precision_top_ns: Sequence[int] = DEFAULT_PRECISION_TOP_NS,
    day_performance: Optional[List[top_gainer_critic.DayPerformance]] = None,
) -> Dict[str, Any]:
    tz = ZoneInfo(timezone_name)
    start_local, cutoff_local = _local_window(target_day, cutoff_hour, tz)
    base_report = top_gainer_critic.build_report(
        target_day=target_day,
        phase="final",
        timezone_name=timezone_name,
        top_n=top_n,
        min_quote_volume=min_quote_volume,
        day_performance=day_performance,
        cutoff_hour=cutoff_hour,
    )
    annotation = annotate_ml_dataset_teacher(base_report)
    watchlist_top = list(base_report.get("watchlist_top_gainers") or [])
    watchlist_top_set = {str(item.get("symbol", "")).strip() for item in watchlist_top}
    watchlist_top_count = len(watchlist_top)
    checkpoint_list = _normalize_hours(checkpoint_hours, cutoff_hour)
    precision_list = _normalize_top_ns(precision_top_ns)

    checkpoint_metrics: List[Dict[str, Any]] = []
    lead_times_min: List[int] = []
    for checkpoint_hour in checkpoint_list:
        checkpoint_dt = datetime.combine(target_day, time(hour=checkpoint_hour), tzinfo=tz)
        captured = 0
        for item in watchlist_top:
            entry_dt = _entry_local_dt(target_day, item.get("first_entry_time"), tz)
            if entry_dt is not None and entry_dt <= checkpoint_dt:
                captured += 1
                if checkpoint_hour == cutoff_hour:
                    lead_times_min.append(max(0, int((cutoff_local - entry_dt).total_seconds() // 60)))
        checkpoint_metrics.append(
            {
                "hour_local": checkpoint_hour,
                "captured": captured,
                "total": watchlist_top_count,
                "recall_pct": round((captured / watchlist_top_count * 100.0), 2) if watchlist_top_count else 0.0,
            }
        )

    watchlist = top_gainer_critic._load_watchlist()
    first_entries = _load_first_unique_entries(
        start_local=start_local,
        end_local=cutoff_local,
        tz=tz,
        watchlist=watchlist,
    )
    precision_metrics: List[Dict[str, Any]] = []
    for top_k in precision_list:
        sample = first_entries[:top_k]
        hits = sum(1 for item in sample if item["symbol"] in watchlist_top_set)
        precision_metrics.append(
            {
                "top_n": top_k,
                "alerts_considered": len(sample),
                "hits": hits,
                "precision_pct": round((hits / len(sample) * 100.0), 2) if sample else 0.0,
                "symbols": [item["symbol"] for item in sample],
            }
        )

    missed = [item for item in watchlist_top if item.get("status") != "bought"]
    miss_status_counts = Counter(str(item.get("status", "")) for item in missed)
    miss_reason_counts = Counter(
        str(item.get("reason") or item.get("status") or "")
        for item in missed
    )
    summary = {
        "watchlist_top_count": watchlist_top_count,
        "watchlist_top_bought": int(base_report.get("summary", {}).get("watchlist_top_bought", 0)),
        "watchlist_top_missed": int(base_report.get("summary", {}).get("watchlist_top_missed", 0)),
        "recall_at_cutoff_pct": round(
            float(base_report.get("summary", {}).get("watchlist_top_capture_rate_pct", 0.0)),
            2,
        ),
        "median_lead_time_min": None if not lead_times_min else int(median(lead_times_min)),
        "bot_unique_buys": int(base_report.get("summary", {}).get("bot_unique_buys", 0)),
        "bot_false_positive_buys": int(base_report.get("summary", {}).get("bot_false_positive_buys", 0)),
        "mandatory_positive_coverage_pct": annotation["mandatory_positive_coverage_pct"],
        "teacher_rows_in_window": annotation["rows_in_window"],
        "teacher_rows_annotated": annotation["rows_annotated"],
    }
    return {
        "goal_name": "watchlist_top_gainer_by_cutoff",
        "target_day_local": target_day.isoformat(),
        "window_local": {
            "start": start_local.strftime("%Y-%m-%d %H:%M:%S %Z"),
            "end": cutoff_local.strftime("%Y-%m-%d %H:%M:%S %Z"),
        },
        "settings": {
            "timezone": timezone_name,
            "top_n": top_n,
            "min_quote_volume_24h": min_quote_volume,
            "cutoff_hour_local": cutoff_hour,
            "checkpoint_hours_local": checkpoint_list,
            "precision_first_ns": precision_list,
        },
        "summary": summary,
        "early_recall_checkpoints": checkpoint_metrics,
        "precision_first_n": precision_metrics,
        "miss_status_breakdown": dict(miss_status_counts.most_common()),
        "miss_reason_breakdown": dict(miss_reason_counts.most_common(10)),
        "teacher_annotation": annotation,
        "watchlist_top_gainers": watchlist_top,
        "bot_false_positive_symbols": list(base_report.get("bot_false_positive_symbols") or []),
    }


def render_text(report: Dict[str, Any]) -> str:
    summary = report.get("summary") or {}
    settings = report.get("settings") or {}
    lines = [
        f"Watchlist top-gainer goal for {report.get('target_day_local')} cutoff {int(settings.get('cutoff_hour_local', DEFAULT_CUTOFF_HOUR)):02d}:00",
        f"Window: {report['window_local']['start']} -> {report['window_local']['end']}",
        "",
        "Summary:",
        f"  recall@cutoff: {summary.get('watchlist_top_bought')}/{summary.get('watchlist_top_count')} ({summary.get('recall_at_cutoff_pct')}%)",
        f"  median lead time: {summary.get('median_lead_time_min')} min",
        f"  false-positive buys: {summary.get('bot_false_positive_buys')}/{summary.get('bot_unique_buys')}",
        f"  mandatory positive coverage: {summary.get('mandatory_positive_coverage_pct')}%",
        "",
        "Early recall checkpoints:",
    ]
    for item in report.get("early_recall_checkpoints") or []:
        lines.append(
            f"  {int(item.get('hour_local', 0)):02d}:00 -> {item.get('captured')}/{item.get('total')} ({item.get('recall_pct')}%)"
        )
    lines.append("")
    lines.append("Precision first N alerts:")
    for item in report.get("precision_first_n") or []:
        lines.append(
            f"  first {item.get('top_n')}: {item.get('hits')}/{item.get('alerts_considered')} ({item.get('precision_pct')}%)"
        )
    miss_status = report.get("miss_status_breakdown") or {}
    if miss_status:
        lines.append("")
        lines.append("Miss status breakdown:")
        lines.append("  " + ", ".join(f"{key}={value}" for key, value in miss_status.items()))
    missing_rows = (report.get("teacher_annotation") or {}).get("positive_symbols_missing_rows") or []
    if missing_rows:
        lines.append("")
        lines.append("Positive symbols missing ml_dataset rows:")
        lines.append("  " + ", ".join(str(x) for x in missing_rows[:15]))
    return "\n".join(lines)


def save_report(report: Dict[str, Any]) -> Dict[str, str]:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    cutoff_hour = int((report.get("settings") or {}).get("cutoff_hour_local", DEFAULT_CUTOFF_HOUR))
    base = f"watchlist_top_gainer_goal_{report['target_day_local']}_{cutoff_hour:02d}h"
    json_path = REPORT_DIR / f"{base}.json"
    txt_path = REPORT_DIR / f"{base}.txt"
    latest_json = REPORT_DIR / "watchlist_top_gainer_goal_latest.json"
    latest_txt = REPORT_DIR / "watchlist_top_gainer_goal_latest.txt"
    json_payload = json.dumps(report, ensure_ascii=False, indent=2)
    txt_payload = render_text(report)
    json_path.write_text(json_payload, encoding="utf-8")
    txt_path.write_text(txt_payload, encoding="utf-8")
    latest_json.write_text(json_payload, encoding="utf-8")
    latest_txt.write_text(txt_payload, encoding="utf-8")
    with HISTORY_FILE.open("a", encoding="utf-8") as fh:
        history_row = {
            "ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "target_day_local": report["target_day_local"],
            "cutoff_hour_local": cutoff_hour,
            "summary": report.get("summary", {}),
            "files": {
                "json": str(json_path),
                "txt": str(txt_path),
                "latest_json": str(latest_json),
                "latest_txt": str(latest_txt),
            },
        }
        fh.write(json.dumps(history_row, ensure_ascii=False) + "\n")
    return {
        "json": str(json_path),
        "txt": str(txt_path),
        "latest_json": str(latest_json),
        "latest_txt": str(latest_txt),
    }


def run_goal_report(
    *,
    target_day: date,
    cutoff_hour: int = DEFAULT_CUTOFF_HOUR,
    timezone_name: str = DEFAULT_TZ,
    top_n: int = top_gainer_critic.DEFAULT_TOP_N,
    min_quote_volume: float = top_gainer_critic.DEFAULT_MIN_QUOTE_VOLUME,
    checkpoint_hours: Sequence[int] = DEFAULT_CHECKPOINT_HOURS,
    precision_top_ns: Sequence[int] = DEFAULT_PRECISION_TOP_NS,
    day_performance: Optional[List[top_gainer_critic.DayPerformance]] = None,
) -> Dict[str, Any]:
    report = build_goal_report(
        target_day=target_day,
        cutoff_hour=cutoff_hour,
        timezone_name=timezone_name,
        top_n=top_n,
        min_quote_volume=min_quote_volume,
        checkpoint_hours=checkpoint_hours,
        precision_top_ns=precision_top_ns,
        day_performance=day_performance,
    )
    report["files"] = save_report(report)
    return report


def main() -> int:
    args = parse_args()
    report = run_goal_report(
        target_day=date.fromisoformat(args.date),
        cutoff_hour=args.cutoff_hour,
        timezone_name=args.timezone,
        top_n=args.top,
        min_quote_volume=args.min_quote_volume,
    )
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(render_text(report))
        print("")
        print(f"JSON report saved to: {report['files']['json']}")
        print(f"Text report saved to: {report['files']['txt']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
