from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = ROOT.parent
RUNTIME_DIR = WORKSPACE_ROOT / ".runtime"
REPORT_DIR = RUNTIME_DIR / "reports"

BOT_EVENTS_FILE = ROOT / "bot_events.jsonl"
CRITIC_FILE = ROOT / "critic_dataset.jsonl"
ML_FILE = ROOT / "ml_dataset.jsonl"
RL_STATUS_FILE = RUNTIME_DIR / "rl_worker_status.json"
TRAIN_REPORT_FILE = ROOT / "ml_candidate_ranker_report.json"
SHADOW_REPORT_FILE = ROOT / "ml_candidate_ranker_shadow_report.json"

LOCAL_TZ = ZoneInfo("Europe/Budapest")


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


def _parse_utc_ts(raw: str | None) -> datetime | None:
    if not raw:
        return None
    for fmt in ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S.%fZ"):
        try:
            return datetime.strptime(raw, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


def _local_day_bounds(target_day: date) -> tuple[datetime, datetime]:
    start_local = datetime.combine(target_day, datetime.min.time(), tzinfo=LOCAL_TZ)
    end_local = start_local + timedelta(days=1)
    return start_local.astimezone(timezone.utc), end_local.astimezone(timezone.utc)


def _count_rows_in_day(path: Path, start_utc: datetime, end_utc: datetime) -> int:
    count = 0
    for rec in _iter_jsonl(path):
        ts = _parse_utc_ts(rec.get("ts_signal")) or _parse_utc_ts(rec.get("ts"))
        if ts and start_utc <= ts < end_utc:
            count += 1
    return count


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def build_report(
    target_day: date,
    *,
    bot_events_file: Path = BOT_EVENTS_FILE,
    critic_file: Path = CRITIC_FILE,
    ml_file: Path = ML_FILE,
    rl_status_file: Path = RL_STATUS_FILE,
    train_report_file: Path = TRAIN_REPORT_FILE,
    shadow_report_file: Path = SHADOW_REPORT_FILE,
) -> Dict[str, Any]:
    start_utc, end_utc = _local_day_bounds(target_day)
    bot_rows = list(_iter_jsonl(bot_events_file))
    day_events = []
    for rec in bot_rows:
        ts = _parse_utc_ts(rec.get("ts"))
        if ts and start_utc <= ts < end_utc:
            day_events.append(rec)

    ranker_shadow = [r for r in day_events if r.get("event") == "ranker_shadow"]
    bot_take_ranker_skip = [
        r for r in ranker_shadow
        if r.get("bot_action") == "take" and not bool(r.get("ranker_take"))
    ]
    bot_blocked_ranker_take = [
        r for r in ranker_shadow
        if r.get("bot_action") == "blocked" and bool(r.get("ranker_take"))
    ]

    worst_take_by_proba = sorted(
        bot_take_ranker_skip,
        key=lambda r: (
            float(r.get("ranker_proba", 0.0)),
            -float(r.get("candidate_score", 0.0)),
        ),
    )[:10]
    missed_by_proba = sorted(
        bot_blocked_ranker_take,
        key=lambda r: (
            -float(r.get("ranker_proba", 0.0)),
            -float(r.get("candidate_score", 0.0)),
        ),
    )[:10]

    blocked_counts = Counter(str(r.get("reason_code", r.get("signal_type", ""))) for r in bot_blocked_ranker_take)
    mode_counts = Counter(str(r.get("mode", "")) for r in ranker_shadow)
    symbol_counts = Counter(str(r.get("sym", "")) for r in ranker_shadow)

    report = {
        "target_day_local": target_day.isoformat(),
        "generated_at_local": datetime.now(LOCAL_TZ).strftime("%Y-%m-%d %H:%M:%S %Z"),
        "datasets": {
            "critic_rows_total": sum(1 for _ in _iter_jsonl(critic_file)),
            "critic_rows_day": _count_rows_in_day(critic_file, start_utc, end_utc),
            "ml_rows_total": sum(1 for _ in _iter_jsonl(ml_file)),
            "ml_rows_day": _count_rows_in_day(ml_file, start_utc, end_utc),
        },
        "worker_status": _load_json(rl_status_file),
        "train_report": _load_json(train_report_file),
        "shadow_report": _load_json(shadow_report_file),
        "ranker_shadow": {
            "events_total": len(ranker_shadow),
            "bot_take_ranker_skip": len(bot_take_ranker_skip),
            "bot_blocked_ranker_take": len(bot_blocked_ranker_take),
            "top_symbols": symbol_counts.most_common(10),
            "top_modes": mode_counts.most_common(10),
            "blocked_reason_counts": blocked_counts.most_common(10),
            "worst_bot_takes": worst_take_by_proba,
            "missed_bot_blocks": missed_by_proba,
        },
    }
    return report


def render_text(report: Dict[str, Any]) -> str:
    ds = report["datasets"]
    rs = report["ranker_shadow"]
    train = report.get("train_report", {})
    lines = [
        f"RL daily report for {report['target_day_local']}",
        f"Generated: {report['generated_at_local']}",
        "",
        "Datasets:",
        f"  critic_dataset: +{ds['critic_rows_day']} rows today, total {ds['critic_rows_total']}",
        f"  ml_dataset: +{ds['ml_rows_day']} rows today, total {ds['ml_rows_total']}",
        "",
        "Training:",
        f"  chosen_model: {train.get('chosen_model', '') or 'n/a'}",
        f"  train_rows: {train.get('train_rows', 0)} val_rows: {train.get('val_rows', 0)} test_rows: {train.get('test_rows', 0)}",
        f"  test_ret5_delta: {train.get('improvement_delta', {}).get('ret5_avg_delta', 'n/a')}",
        f"  test_win_rate_delta: {train.get('improvement_delta', {}).get('win_rate_delta', 'n/a')}",
        "",
        "Shadow disagreements:",
        f"  total: {rs['events_total']}",
        f"  bot_take vs ranker_skip: {rs['bot_take_ranker_skip']}",
        f"  bot_blocked vs ranker_take: {rs['bot_blocked_ranker_take']}",
    ]
    if rs["top_symbols"]:
        lines.append("  top symbols: " + ", ".join(f"{sym}={cnt}" for sym, cnt in rs["top_symbols"]))
    if rs["blocked_reason_counts"]:
        lines.append("  blocked reasons: " + ", ".join(f"{reason}={cnt}" for reason, cnt in rs["blocked_reason_counts"]))

    if rs["worst_bot_takes"]:
        lines.append("")
        lines.append("Worst bot takes (ranker wanted skip):")
        for rec in rs["worst_bot_takes"][:5]:
            lines.append(
                "  "
                + f"{rec.get('sym')} {rec.get('tf')} {rec.get('mode')} "
                + f"score={rec.get('candidate_score')} proba={rec.get('ranker_proba')} reason={rec.get('reason')}"
            )

    if rs["missed_bot_blocks"]:
        lines.append("")
        lines.append("Missed bot blocks (ranker wanted take):")
        for rec in rs["missed_bot_blocks"][:5]:
            lines.append(
                "  "
                + f"{rec.get('sym')} {rec.get('tf')} {rec.get('mode')} "
                + f"score={rec.get('candidate_score')} proba={rec.get('ranker_proba')} reason={rec.get('reason')}"
            )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a daily RL status report from local datasets and shadow-ranker disagreements.")
    parser.add_argument("--date", default="", help="Local date in YYYY-MM-DD.")
    parser.add_argument("--previous-day", action="store_true", help="Report the previous local day.")
    args = parser.parse_args()

    if args.date:
        target_day = date.fromisoformat(args.date)
    else:
        target_day = datetime.now(LOCAL_TZ).date()
        if args.previous_day:
            target_day = target_day - timedelta(days=1)

    report = build_report(target_day)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = REPORT_DIR / f"rl_daily_{target_day.isoformat()}.json"
    txt_path = REPORT_DIR / f"rl_daily_{target_day.isoformat()}.txt"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    txt_path.write_text(render_text(report), encoding="utf-8")
    print(render_text(report))
    print("")
    print(f"JSON report saved to: {json_path}")
    print(f"Text report saved to: {txt_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
