from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / ".runtime" / "reports"
DEFAULT_TOP_N = 15
DEFAULT_HIGH_GIVEBACK_PCT = 1.0
DEFAULT_WEAK_EXIT_EFFICIENCY = 0.35


@dataclass(frozen=True)
class ExitAuditConfig:
    days: int = 14
    top_n: int = DEFAULT_TOP_N
    high_giveback_pct: float = DEFAULT_HIGH_GIVEBACK_PCT
    weak_exit_efficiency: float = DEFAULT_WEAK_EXIT_EFFICIENCY


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _num(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if out != out:  # NaN
        return None
    return out


def _int(value: Any) -> int | None:
    number = _num(value)
    if number is None:
        return None
    return int(number)


def _median(values: Iterable[Any]) -> float | None:
    vals = [_num(v) for v in values]
    nums = [v for v in vals if v is not None]
    if not nums:
        return None
    return round(float(median(nums)), 4)


def _avg(values: Iterable[Any]) -> float | None:
    nums = [v for v in (_num(v) for v in values) if v is not None]
    if not nums:
        return None
    return round(sum(nums) / len(nums), 4)


def _metric_median(summary: dict[str, Any], key: str) -> float | None:
    metric = summary.get(key) or {}
    if not isinstance(metric, dict):
        return None
    return _num(metric.get("median"))


def _metric_avg(summary: dict[str, Any], key: str) -> float | None:
    metric = summary.get(key) or {}
    if not isinstance(metric, dict):
        return None
    return _num(metric.get("avg"))


def _day_from_path(path: Path) -> str | None:
    match = re.search(r"(20\d\d-\d\d-\d\d)", path.name)
    return match.group(1) if match else None


def _report_paths(reports_dir: Path, days: int) -> list[Path]:
    paths = sorted(reports_dir.glob("signal_quality_*_final.json"), key=lambda p: (_day_from_path(p) or "", p.name))
    if days > 0:
        return paths[-days:]
    return paths


def _rounded_key_num(value: Any) -> float | None:
    number = _num(value)
    if number is None:
        return None
    return round(number, 6)


def _case_key(day: str, row: dict[str, Any]) -> tuple[Any, ...]:
    # Open/marked-down positions can be emitted repeatedly with entry_ts drifting by
    # a minute while all economics are identical. Collapse those operational
    # artifacts so the audit counts positions, not repeated snapshots. Closed
    # trades keep entry/exit timestamps, which preserves real re-entries.
    exit_ts = row.get("exit_ts")
    if not exit_ts:
        return (
            day,
            row.get("source"),
            row.get("sym"),
            row.get("tf"),
            row.get("mode"),
            "open_or_unpaired",
            _rounded_key_num(row.get("entry_price")),
            _rounded_key_num(row.get("exit_price")),
            _rounded_key_num(row.get("pnl_pct")),
            _rounded_key_num(row.get("max_favorable_pct")),
        )
    return (
        day,
        row.get("source"),
        row.get("sym"),
        row.get("tf"),
        row.get("entry_ts"),
        exit_ts,
        row.get("mode"),
    )


def _normalize_reason(reason: Any) -> str:
    text = str(reason or "").strip()
    if not text:
        return "unknown"
    lowered = text.lower()
    if "atr" in lowered and "trail" in lowered:
        return "atr_trail"
    if "ema20" in lowered or "ema" in lowered:
        return "ema_break"
    if "stop" in lowered or "стоп" in lowered:
        return "stop_loss_or_trail"
    if "profit" in lowered or "lock" in lowered:
        return "profit_lock"
    if "timeout" in lowered or "max hold" in lowered:
        return "time_exit"
    if len(text) > 80:
        return text[:77] + "..."
    return text


def _case_tags(row: dict[str, Any], *, top_n: int, high_giveback_pct: float, weak_exit_efficiency: float) -> list[str]:
    tags: list[str] = []
    pnl = _num(row.get("pnl_pct"))
    mfe = _num(row.get("max_favorable_pct"))
    future = _num(row.get("future_favorable_pct"))
    eff = _num(row.get("exit_efficiency"))
    giveback = _num(row.get("giveback_pct"))
    top_rank = _int((row.get("trend") or {}).get("top_mover_rank")) if isinstance(row.get("trend"), dict) else None
    exit_timing = str(row.get("exit_timing") or "").lower()

    weak_or_bad = (pnl is not None and pnl <= 0.0) or (eff is not None and eff < weak_exit_efficiency) or (giveback is not None and giveback >= high_giveback_pct)
    if top_rank is not None and 1 <= top_rank <= top_n and weak_or_bad:
        tags.append("top_mover_exit_failure")
    if pnl is not None and pnl <= 0.0 and (mfe or 0.0) > 0.0:
        tags.append("negative_after_mfe")
    if giveback is not None and giveback >= high_giveback_pct:
        tags.append("high_giveback")
    if exit_timing == "late":
        tags.append("late_exit")
    if exit_timing == "early":
        tags.append("early_exit")
    if exit_timing == "open" and pnl is not None and pnl < 0.0:
        tags.append("open_marked_down")
    if future is not None and mfe is not None and future > mfe:
        tags.append("post_exit_continuation")
    return tags


def _case_opportunity_loss(row: dict[str, Any]) -> float:
    pnl = _num(row.get("pnl_pct")) or 0.0
    mfe = _num(row.get("max_favorable_pct")) or 0.0
    future = _num(row.get("future_favorable_pct")) or 0.0
    giveback = _num(row.get("giveback_pct")) or 0.0
    # Conservative case ranking: how much favorable movement was not monetized.
    return round(max(mfe, future, 0.0) - pnl + max(giveback, 0.0), 4)


def _compact_case(day: str, row: dict[str, Any], cfg: ExitAuditConfig) -> dict[str, Any]:
    trend = row.get("trend") if isinstance(row.get("trend"), dict) else {}
    tags = _case_tags(
        row,
        top_n=cfg.top_n,
        high_giveback_pct=cfg.high_giveback_pct,
        weak_exit_efficiency=cfg.weak_exit_efficiency,
    )
    return {
        "day": day,
        "sym": row.get("sym"),
        "tf": row.get("tf"),
        "source": row.get("source"),
        "mode": row.get("mode"),
        "entry_ts": row.get("entry_ts"),
        "exit_ts": row.get("exit_ts"),
        "exit_timing": row.get("exit_timing"),
        "entry_timing": row.get("entry_timing"),
        "pnl_pct": _num(row.get("pnl_pct")),
        "max_favorable_pct": _num(row.get("max_favorable_pct")),
        "future_favorable_pct": _num(row.get("future_favorable_pct")),
        "exit_efficiency": _num(row.get("exit_efficiency")),
        "giveback_pct": _num(row.get("giveback_pct")),
        "capture_ratio_at_entry": _num(row.get("capture_ratio_at_entry")),
        "realized_capture_ratio": _num(row.get("realized_capture_ratio")),
        "top_mover_rank": _int(trend.get("top_mover_rank")),
        "top_mover_change_pct": _num(trend.get("top_mover_change_pct")),
        "exit_reason_bucket": _normalize_reason(row.get("exit_reason")),
        "tags": tags,
        "opportunity_loss_pct": _case_opportunity_loss(row),
    }


def _load_reports(reports_dir: Path, cfg: ExitAuditConfig) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    day_rows: list[dict[str, Any]] = []
    cases: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()

    for path in _report_paths(reports_dir, cfg.days):
        day = _day_from_path(path)
        if not day:
            continue
        data = _read_json(path)
        summary = data.get("summary") or {}
        coverage = data.get("coverage") or {}
        visible_case_rows = 0
        for bucket in ("late_entries", "early_exits", "false_positive_buys", "trades"):
            rows = data.get(bucket) or []
            if not isinstance(rows, list):
                continue
            for row in rows:
                if not isinstance(row, dict):
                    continue
                key = _case_key(day, row)
                if key in seen:
                    continue
                seen.add(key)
                visible_case_rows += 1
                cases.append(_compact_case(day, row, cfg))

        closed = int(summary.get("closed_trades") or 0)
        case_coverage = "full" if data.get("trades") and len(data.get("trades") or []) >= closed else "partial"
        day_rows.append(
            {
                "day": day,
                "coverage_status": coverage.get("status"),
                "coverage_reasons": coverage.get("reasons") or [],
                "closed_trades": closed,
                "paired_trades": coverage.get("paired_trades"),
                "visible_case_rows": visible_case_rows,
                "case_coverage_status": case_coverage,
                "early_exits": int(summary.get("early_exits") or 0),
                "late_exits": int(summary.get("late_exits") or 0),
                "false_positive_buys": int(summary.get("false_positive_buys") or 0),
                "late_entries": int(summary.get("late_entries") or 0),
                "exit_efficiency_median": _metric_median(summary, "exit_efficiency"),
                "exit_efficiency_avg": _metric_avg(summary, "exit_efficiency"),
                "giveback_pct_median": _metric_median(summary, "giveback_pct"),
                "giveback_pct_avg": _metric_avg(summary, "giveback_pct"),
                "pnl_pct_median": _metric_median(summary, "pnl_pct"),
                "pnl_pct_avg": _metric_avg(summary, "pnl_pct"),
                "realized_capture_ratio_median": _metric_median(summary, "realized_capture_ratio"),
            }
        )
    return day_rows, cases


def _counter_by(cases: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    counts = Counter(str(case.get(key) or "unknown") for case in cases)
    return [{key: name, "count": count} for name, count in counts.most_common()]


def _tag_counts(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts: Counter[str] = Counter()
    for case in cases:
        for tag in case.get("tags") or []:
            counts[tag] += 1
    return [{"tag": tag, "count": count} for tag, count in counts.most_common()]


def _cases_with_tag(cases: list[dict[str, Any]], tag: str) -> list[dict[str, Any]]:
    return [case for case in cases if tag in (case.get("tags") or [])]


def build(days: int = 14, *, reports_dir: Path = REPORTS, top_n: int = DEFAULT_TOP_N, high_giveback_pct: float = DEFAULT_HIGH_GIVEBACK_PCT, weak_exit_efficiency: float = DEFAULT_WEAK_EXIT_EFFICIENCY) -> dict[str, Any]:
    cfg = ExitAuditConfig(
        days=days,
        top_n=top_n,
        high_giveback_pct=high_giveback_pct,
        weak_exit_efficiency=weak_exit_efficiency,
    )
    rows, cases = _load_reports(reports_dir, cfg)
    worst_cases = sorted(cases, key=lambda case: case.get("opportunity_loss_pct") or 0.0, reverse=True)
    top_failures = sorted(_cases_with_tag(cases, "top_mover_exit_failure"), key=lambda case: case.get("opportunity_loss_pct") or 0.0, reverse=True)
    negative_after_mfe = _cases_with_tag(cases, "negative_after_mfe")
    high_giveback = _cases_with_tag(cases, "high_giveback")
    partial_days = [r["day"] for r in rows if r.get("coverage_status") != "complete" or r.get("case_coverage_status") != "full"]

    closed_total = sum(int(r.get("closed_trades") or 0) for r in rows)
    case_rows_loaded = len(cases)
    case_coverage_status = "full" if rows and all(r.get("case_coverage_status") == "full" for r in rows) else "partial"
    if not rows:
        case_coverage_status = "empty"

    summary = {
        "days_loaded": len(rows),
        "closed_trades_total": closed_total,
        "case_rows_loaded": case_rows_loaded,
        "case_coverage_status": case_coverage_status,
        "partial_days": partial_days,
        "early_exits_total": sum(int(r.get("early_exits") or 0) for r in rows),
        "late_exits_total": sum(int(r.get("late_exits") or 0) for r in rows),
        "false_positive_buys_total": sum(int(r.get("false_positive_buys") or 0) for r in rows),
        "late_entries_total": sum(int(r.get("late_entries") or 0) for r in rows),
        "exit_efficiency_median": _median(r.get("exit_efficiency_median") for r in rows),
        "giveback_pct_median": _median(r.get("giveback_pct_median") for r in rows),
        "pnl_pct_median": _median(r.get("pnl_pct_median") for r in rows),
        "realized_capture_ratio_median": _median(r.get("realized_capture_ratio_median") for r in rows),
        "negative_exit_case_count": len([c for c in cases if (_num(c.get("pnl_pct")) or 0.0) <= 0.0]),
        "negative_after_mfe_count": len(negative_after_mfe),
        "high_giveback_case_count": len(high_giveback),
        "top_mover_exit_failure_count": len(top_failures),
        "post_exit_continuation_count": len(_cases_with_tag(cases, "post_exit_continuation")),
        "avg_visible_opportunity_loss_pct": _avg(c.get("opportunity_loss_pct") for c in cases),
        "median_visible_opportunity_loss_pct": _median(c.get("opportunity_loss_pct") for c in cases),
    }

    if not rows:
        status = "empty"
        recommendation = "Нет signal_quality reports: сначала запустить evaluator."
    elif case_coverage_status != "full":
        status = "partial"
        recommendation = "Использовать summary для тренда, но для case-level выводов перегенерировать signal_quality с --include-trades."
    else:
        status = "complete"
        recommendation = "Можно выбирать exit-гипотезы для replay по worst_cases/top_mover_exit_failures."

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "status": status,
        "config": {
            "days": days,
            "top_n": top_n,
            "high_giveback_pct": high_giveback_pct,
            "weak_exit_efficiency": weak_exit_efficiency,
            "reports_dir": str(reports_dir),
        },
        "summary": summary,
        "daily": rows,
        "tag_counts": _tag_counts(cases),
        "exit_reason_buckets": _counter_by(cases, "exit_reason_bucket")[:20],
        "source_buckets": _counter_by(cases, "source"),
        "mode_buckets": _counter_by(cases, "mode")[:20],
        "worst_cases": worst_cases[:25],
        "top_mover_exit_failures": top_failures[:25],
        "negative_after_mfe_examples": sorted(negative_after_mfe, key=lambda case: case.get("opportunity_loss_pct") or 0.0, reverse=True)[:25],
        "recommendation": recommendation,
    }


def format_text(report: dict[str, Any]) -> str:
    summary = report.get("summary") or {}
    lines = [
        "Exit-quality auditor",
        f"status: {report.get('status')}",
        f"days loaded: {summary.get('days_loaded')} | closed trades: {summary.get('closed_trades_total')} | visible cases: {summary.get('case_rows_loaded')} ({summary.get('case_coverage_status')})",
        f"exit_eff median: {summary.get('exit_efficiency_median')} | giveback median: {summary.get('giveback_pct_median')} | pnl median: {summary.get('pnl_pct_median')}",
        f"late exits: {summary.get('late_exits_total')} | early exits: {summary.get('early_exits_total')} | top-mover exit failures: {summary.get('top_mover_exit_failure_count')}",
        f"negative-after-MFE: {summary.get('negative_after_mfe_count')} | high-giveback: {summary.get('high_giveback_case_count')} | post-exit continuation: {summary.get('post_exit_continuation_count')}",
    ]
    partial_days = summary.get("partial_days") or []
    if partial_days:
        lines.append("partial days: " + ", ".join(partial_days[:12]) + (" ..." if len(partial_days) > 12 else ""))
    lines.append("")
    lines.append("Worst visible cases:")
    for case in (report.get("worst_cases") or [])[:10]:
        lines.append(
            f"- {case.get('day')} {case.get('sym')} {case.get('tf')} {case.get('source')}/{case.get('mode')} "
            f"pnl={case.get('pnl_pct')} mfe={case.get('max_favorable_pct')} future={case.get('future_favorable_pct')} "
            f"eff={case.get('exit_efficiency')} giveback={case.get('giveback_pct')} loss={case.get('opportunity_loss_pct')} "
            f"tags={','.join(case.get('tags') or [])} reason={case.get('exit_reason_bucket')}"
        )
    lines.append("")
    lines.append("Top-mover exit failures:")
    for case in (report.get("top_mover_exit_failures") or [])[:10]:
        lines.append(
            f"- {case.get('day')} rank#{case.get('top_mover_rank')} {case.get('sym')} {case.get('tf')} "
            f"pnl={case.get('pnl_pct')} eff={case.get('exit_efficiency')} giveback={case.get('giveback_pct')} reason={case.get('exit_reason_bucket')}"
        )
    lines.append("")
    lines.append("Recommendation: " + str(report.get("recommendation")))
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit exit monetization quality from signal-quality reports.")
    parser.add_argument("--days", type=int, default=14, help="Number of latest daily final reports to load; <=0 means all.")
    parser.add_argument("--reports-dir", default=str(REPORTS))
    parser.add_argument("--top-n", type=int, default=DEFAULT_TOP_N)
    parser.add_argument("--high-giveback-pct", type=float, default=DEFAULT_HIGH_GIVEBACK_PCT)
    parser.add_argument("--weak-exit-efficiency", type=float, default=DEFAULT_WEAK_EXIT_EFFICIENCY)
    parser.add_argument("--output")
    parser.add_argument("--text-output")
    parser.add_argument("--json", action="store_true", help="Print JSON instead of compact text.")
    args = parser.parse_args()

    payload = build(
        args.days,
        reports_dir=Path(args.reports_dir),
        top_n=args.top_n,
        high_giveback_pct=args.high_giveback_pct,
        weak_exit_efficiency=args.weak_exit_efficiency,
    )
    out_dir = REPORTS
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.output:
        json_path = Path(args.output)
    else:
        json_path = out_dir / "exit_quality_audit_latest.json"
    if args.text_output:
        text_path = Path(args.text_output)
    else:
        text_path = out_dir / "exit_quality_audit_latest.txt"

    json_text = json.dumps(payload, ensure_ascii=False, indent=2)
    text = format_text(payload)
    json_path.write_text(json_text, encoding="utf-8")
    text_path.write_text(text, encoding="utf-8")

    stdout_encoding = sys.stdout.encoding or "utf-8"
    if args.json:
        print(json_text.encode(stdout_encoding, errors="replace").decode(stdout_encoding, errors="replace"))
    else:
        print(text.encode(stdout_encoding, errors="replace").decode(stdout_encoding, errors="replace"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
