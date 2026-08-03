from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from statistics import mean, median
from typing import Any

import report_portfolio_replacement_shadow_reward as replacement_report
import research_event_cohort_store as cohort_store


ROOT = Path(__file__).resolve().parent.parent
FILES = ROOT / "files"
REPORTS = ROOT / ".runtime" / "reports"
DEFAULT_OUTPUT = REPORTS / "targeted_replacement_pre_gate_latest.json"
DEFAULT_TEXT_OUTPUT = REPORTS / "targeted_replacement_pre_gate_latest.txt"


@dataclass(frozen=True)
class TargetedReplacementConfig:
    leader_delta_thresholds: tuple[float, ...] = (0.0, 5.0, 10.0, 15.0, 20.0)
    train_fraction: float = 0.70
    purge_days: int = 1
    recent_days: int = 14
    min_train_allowed: int = 30
    min_holdout_allowed: int = 10
    min_recent_allowed: int = 10
    min_avg_delta_pct: float = 0.10
    min_median_delta_pct: float = 0.0
    min_positive_rate_pct: float = 50.0
    max_regret_rate_pct: float = 25.0


def build_report(
    files_dir: Path = FILES,
    reports_dir: Path = REPORTS,
    cfg: TargetedReplacementConfig = TargetedReplacementConfig(),
) -> dict[str, Any]:
    cohort_db = reports_dir.parent / "research_event_cohorts.sqlite3"
    events, cohort_sync = cohort_store.load_trade_events(files_dir=files_dir, db_path=cohort_db)
    labels = replacement_report._load_watchlist_labels(reports_dir)
    rows = replacement_report._replacement_rows(events, labels, replacement_report.ReplacementConfig())
    closed = [row for row in rows if _num(row.get("replacement_delta_pct")) is not None]
    return build_from_rows(closed, cfg=cfg, cohort_sync=cohort_sync)


def build_from_rows(
    rows: list[dict[str, Any]],
    *,
    cfg: TargetedReplacementConfig = TargetedReplacementConfig(),
    cohort_sync: dict[str, Any] | None = None,
) -> dict[str, Any]:
    ordered = sorted(rows, key=lambda row: (str(row.get("day") or ""), str(row.get("ts") or "")))
    train, holdout, split = _chronological_day_split(ordered, cfg.train_fraction, cfg.purge_days)
    recent = _recent_rows(ordered, cfg.recent_days)
    candidates = []
    for threshold in cfg.leader_delta_thresholds:
        train_stats = _evaluate(train, threshold, cfg, cfg.min_train_allowed)
        candidates.append({"leader_delta_min": threshold, "train": train_stats})
    candidates.sort(key=_train_rank_key, reverse=True)
    selected = candidates[0] if candidates else None
    if selected:
        threshold = float(selected["leader_delta_min"])
        selected["holdout"] = _evaluate(holdout, threshold, cfg, cfg.min_holdout_allowed)
        selected["recent"] = _evaluate(recent, threshold, cfg, cfg.min_recent_allowed)
    passed = bool(
        selected
        and selected["train"]["passed"]
        and selected["holdout"]["passed"]
        and selected["recent"]["passed"]
    )
    decision = (
        "advance_to_paired_ten_slot_replacement_replay"
        if passed
        else "reject_targeted_losing_incumbent_replacement"
    )
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "status": "complete" if ordered else "partial",
        "decision": decision,
        "config": asdict(cfg),
        "coverage": {
            "closed_replacements": len(ordered),
            "train_rows": len(train),
            "holdout_rows": len(holdout),
            "recent_rows": len(recent),
            "split": split,
            "cohort_store": cohort_sync or {},
        },
        "baseline": {
            "all": _unfiltered_stats(ordered),
            "train": _unfiltered_stats(train),
            "holdout": _unfiltered_stats(holdout),
            "recent": _unfiltered_stats(recent),
        },
        "train_candidates": candidates,
        "selected": selected or {},
    }


def _chronological_day_split(
    rows: list[dict[str, Any]],
    train_fraction: float,
    purge_days: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    days = sorted({str(row.get("day") or "") for row in rows if str(row.get("day") or "")})
    if len(days) < 5:
        return [], [], {"days_total": len(days), "train_days": [], "holdout_days": [], "purged_days": days}
    cut = max(2, min(len(days) - 2, int(len(days) * train_fraction)))
    purge_days = max(0, int(purge_days))
    train_days = days[: max(1, cut - purge_days)]
    holdout_days = days[min(len(days), cut + purge_days) :]
    train_set = set(train_days)
    holdout_set = set(holdout_days)
    return (
        [row for row in rows if str(row.get("day") or "") in train_set],
        [row for row in rows if str(row.get("day") or "") in holdout_set],
        {
            "days_total": len(days),
            "train_days": train_days,
            "holdout_days": holdout_days,
            "purged_days": [day for day in days if day not in train_set | holdout_set],
        },
    )


def _recent_rows(rows: list[dict[str, Any]], recent_days: int) -> list[dict[str, Any]]:
    parsed = [(_parse_day(row.get("day")), row) for row in rows]
    dated = [(day, row) for day, row in parsed if day is not None]
    if not dated:
        return []
    cutoff = max(day for day, _ in dated) - timedelta(days=max(1, recent_days) - 1)
    return [row for day, row in dated if day >= cutoff]


def _parse_day(value: Any) -> date | None:
    try:
        return date.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None


def _evaluate(
    rows: list[dict[str, Any]],
    threshold: float,
    cfg: TargetedReplacementConfig,
    min_allowed: int,
) -> dict[str, Any]:
    allowed = [
        row
        for row in rows
        if _num(row.get("replaced_exit_pnl_pct")) < 0.0
        and _num(row.get("leader_delta")) >= threshold
    ]
    blocked = [row for row in rows if row not in allowed]
    values = [_num(row.get("replacement_delta_pct")) for row in allowed]
    blocked_values = [_num(row.get("replacement_delta_pct")) for row in blocked]
    positive_rate = _positive_rate(values)
    regret_count = sum(1 for value in blocked_values if value > 0.0)
    regret_rate = round(regret_count / len(blocked) * 100.0, 2) if blocked else 0.0
    avg = _avg(values)
    med = _median(values)
    checks = {
        "support": len(allowed) >= min_allowed,
        "avg_delta": avg is not None and avg > cfg.min_avg_delta_pct,
        "median_delta": med is not None and med >= cfg.min_median_delta_pct,
        "positive_rate": positive_rate >= cfg.min_positive_rate_pct,
        "regret_rate": regret_rate <= cfg.max_regret_rate_pct,
    }
    return {
        "rows": len(rows),
        "allowed_count": len(allowed),
        "blocked_count": len(blocked),
        "avg_delta_pct": avg,
        "median_delta_pct": med,
        "total_delta_pct": round(sum(values), 6),
        "positive_rate_pct": positive_rate,
        "blocked_regret_count": regret_count,
        "blocked_regret_rate_pct": regret_rate,
        "checks": checks,
        "passed": all(checks.values()),
    }


def _unfiltered_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    values = [_num(row.get("replacement_delta_pct")) for row in rows]
    return {
        "count": len(values),
        "avg_delta_pct": _avg(values),
        "median_delta_pct": _median(values),
        "total_delta_pct": round(sum(values), 6),
        "positive_rate_pct": _positive_rate(values),
    }


def _train_rank_key(candidate: dict[str, Any]) -> tuple[int, float, float, float]:
    stats = candidate.get("train") or {}
    return (
        int(bool(stats.get("passed"))),
        float(stats.get("total_delta_pct") or 0.0),
        float(stats.get("avg_delta_pct") or -999.0),
        -float(stats.get("blocked_regret_rate_pct") or 100.0),
    )


def _num(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _avg(values: list[float]) -> float | None:
    return round(mean(values), 6) if values else None


def _median(values: list[float]) -> float | None:
    return round(median(values), 6) if values else None


def _positive_rate(values: list[float]) -> float:
    return round(sum(1 for value in values if value > 0.0) / len(values) * 100.0, 2) if values else 0.0


def render_text(report: dict[str, Any]) -> str:
    selected = report.get("selected") or {}
    lines = [
        "Targeted losing-incumbent replacement pre-gate",
        f"status: {report.get('status')}",
        f"decision: {report.get('decision')}",
        f"coverage: {report.get('coverage', {}).get('closed_replacements')} closed replacements",
        f"selected leader delta: {selected.get('leader_delta_min', 'n/a')}",
    ]
    for split_name in ("train", "holdout", "recent"):
        stats = selected.get(split_name) or {}
        lines.append(
            f"{split_name}: n={stats.get('allowed_count')} avg={stats.get('avg_delta_pct')} "
            f"median={stats.get('median_delta_pct')} positive={stats.get('positive_rate_pct')}% "
            f"regret={stats.get('blocked_regret_rate_pct')}% passed={stats.get('passed')}"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Causal targeted replacement pre-gate")
    parser.add_argument("--files-dir", type=Path, default=FILES)
    parser.add_argument("--reports-dir", type=Path, default=REPORTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--text-output", type=Path, default=DEFAULT_TEXT_OUTPUT)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()
    report = build_report(args.files_dir, args.reports_dir)
    text = render_text(report)
    if not args.no_save:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        args.text_output.write_text(text, encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2) if args.json else text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
