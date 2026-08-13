from __future__ import annotations

import argparse
import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from statistics import mean, median
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
REPORTS = ROOT / ".runtime" / "reports"
EARLY_CAPTURE_TARGET_PCT = 25.0
MIN_COMPARISON_DAYS_PER_WINDOW = 3
MIN_COMPARISON_TOP_DENOMINATOR = 10


def _load_json(path: Path) -> dict | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return None
    return value if isinstance(value, dict) else None


def _valid(values: list[float | None]) -> list[float]:
    return [float(v) for v in values if v is not None]


def _complete_days(days: int, end_day: date | None = None) -> list[date]:
    end = end_day or (datetime.now().date() - timedelta(days=1))
    return [end - timedelta(days=i) for i in range(days - 1, -1, -1)]


def _pct(num: float, den: float) -> float | None:
    return round((num / den) * 100.0, 2) if den > 0 else None


def _avg(values: list[float | None]) -> float | None:
    cleaned = _valid(values)
    return round(mean(cleaned), 4) if cleaned else None


def _med(values: list[float | None]) -> float | None:
    cleaned = _valid(values)
    return round(median(cleaned), 4) if cleaned else None


def _num(value: float | None, default: float = 0.0) -> float:
    return default if value is None else float(value)


def _load_day_report(prefix: str, day: str, suffix: str) -> dict | None:
    """Load one canonical report per target day, ignoring manual duplicates."""
    return _load_json(REPORTS / f"{prefix}{day}{suffix}")


def _critic_complete(report: dict | None) -> bool:
    if not report or str(report.get("phase") or "") != "final":
        return False
    return "watchlist_top_count" in (report.get("summary") or {})


def _goal_complete(report: dict | None) -> bool:
    summary = (report or {}).get("summary") or {}
    return bool(report) and "watchlist_top_count" in summary and "watchlist_top_bought" in summary


def _quality_complete(report: dict | None) -> bool:
    coverage = (report or {}).get("coverage") or {}
    return bool(report) and str(coverage.get("status") or "").lower() in {"ok", "complete"}


def _coverage_status(critic: dict | None, goal: dict | None, quality: dict | None) -> tuple[str, list[str]]:
    reasons: list[str] = []
    if not critic:
        reasons.append("critic_missing")
    elif not _critic_complete(critic):
        reasons.append("critic_not_final_or_denominator_unknown")
    if not goal:
        reasons.append("goal_missing")
    elif not _goal_complete(goal):
        reasons.append("goal_denominator_unknown")
    if not quality:
        reasons.append("quality_missing")
    elif not _quality_complete(quality):
        quality_status = str(((quality.get("coverage") or {}).get("status") or "unknown"))
        reasons.append(f"quality_coverage_{quality_status}")
    if not reasons:
        return "complete", []
    if any(reason.endswith("_missing") for reason in reasons):
        return "missing", reasons
    return "partial", reasons


def _ratio_evidence(numerator: float, denominator: float) -> dict[str, float | None]:
    return {
        "numerator": numerator,
        "denominator": denominator,
        "rate_pct": _pct(numerator, denominator),
    }


def _aggregate_scout(records: list[dict[str, Any]]) -> dict[str, Any]:
    summaries = [(row["critic"].get("summary") or {}) for row in records]
    top_bought = sum(_num(s.get("watchlist_top_bought")) for s in summaries)
    top_count = sum(_num(s.get("watchlist_top_count")) for s in summaries)
    early = sum(_num(s.get("watchlist_top_early_captured")) for s in summaries)
    false_positive = sum(_num(s.get("bot_false_positive_buys")) for s in summaries)
    buys = sum(_num(s.get("bot_unique_buys")) for s in summaries)
    return {
        "days_loaded": len(records),
        "denominator_definition": "exchange_top_n_filtered_to_configured_watchlist",
        "watchlist_top_bought": _ratio_evidence(top_bought, top_count),
        "early_capture": _ratio_evidence(early, top_count),
        "false_positive_buys": _ratio_evidence(false_positive, buys),
        # Backward-compatible aliases. The evidence-bearing objects above are
        # canonical and prevent a ratio from losing its denominator.
        "watchlist_top_bought_avg": _avg([s.get("watchlist_top_bought") for s in summaries]),
        "watchlist_top_count_avg": _avg([s.get("watchlist_top_count") for s in summaries]),
        "watchlist_top_bought_rate_pct": _pct(top_bought, top_count),
        "early_capture_rate_pct": _pct(early, top_count),
        "false_positive_buys_total": int(false_positive),
    }


def _aggregate_goal(records: list[dict[str, Any]]) -> dict[str, Any]:
    summaries = [(row["goal"].get("summary") or {}) for row in records]
    bought = sum(_num(s.get("watchlist_top_bought")) for s in summaries)
    top_count = sum(_num(s.get("watchlist_top_count")) for s in summaries)
    false_positive = sum(_num(s.get("bot_false_positive_buys")) for s in summaries)
    buys = sum(_num(s.get("bot_unique_buys")) for s in summaries)
    return {
        "days_loaded": len(records),
        "recall_at_cutoff": _ratio_evidence(bought, top_count),
        "false_positive_buys": _ratio_evidence(false_positive, buys),
        "recall_at_cutoff_avg_pct": _pct(bought, top_count),
        "median_lead_time_min": _med([s.get("median_lead_time_min") for s in summaries]),
        "positive_coverage_avg_pct": _avg([s.get("mandatory_positive_coverage_pct") for s in summaries]),
        "false_positive_buys_total": int(false_positive),
    }


def _aggregate_quality(records: list[dict[str, Any]]) -> dict[str, Any]:
    summaries = [(row["quality"].get("summary") or {}) for row in records]
    missed = sum(_num(s.get("missed_trends")) for s in summaries)
    trends = sum(_num(s.get("trend_episodes_total")) for s in summaries)
    false_positive = sum(_num(s.get("false_positive_buys")) for s in summaries)
    buys = sum(_num(s.get("buys_total")) for s in summaries)
    return {
        "days_loaded": len(records),
        "miss_rate": _ratio_evidence(missed, trends),
        "false_positive_rate": _ratio_evidence(false_positive, buys),
        "miss_rate_avg_pct": _pct(missed, trends),
        "false_positive_rate_avg_pct": _pct(false_positive, buys),
        "median_capture_ratio": _med([
            s.get("capture_ratio_at_entry", {}).get("median")
            for s in summaries if s.get("capture_ratio_at_entry", {}).get("median") is not None
        ]),
        "median_exit_efficiency": _med([
            s.get("exit_efficiency", {}).get("median")
            for s in summaries if s.get("exit_efficiency", {}).get("median") is not None
        ]),
        "median_giveback_pct": _med([
            s.get("giveback_pct", {}).get("median")
            for s in summaries if s.get("giveback_pct", {}).get("median") is not None
        ]),
        "aggregate_method": "pooled_count_ratios_and_median_of_daily_medians",
    }


def _comparison(
    objective_eligible: list[dict[str, Any]],
    quality_eligible: list[dict[str, Any]],
) -> dict[str, Any]:
    window_days = min(7, len(objective_eligible) // 2)
    if window_days < MIN_COMPARISON_DAYS_PER_WINDOW:
        return {"status": "insufficient_data", "window_days": window_days}
    previous = objective_eligible[-2 * window_days:-window_days]
    latest = objective_eligible[-window_days:]
    prev_scout = _aggregate_scout(previous)
    curr_scout = _aggregate_scout(latest)
    prev_early = prev_scout["early_capture_rate_pct"]
    curr_early = curr_scout["early_capture_rate_pct"]
    prev_capture = prev_scout["watchlist_top_bought_rate_pct"]
    curr_capture = curr_scout["watchlist_top_bought_rate_pct"]
    quality_by_day = {row["day"]: row for row in quality_eligible}
    comparable_days = [row["day"] for row in [*previous, *latest]]
    quality_complete = all(day in quality_by_day for day in comparable_days)
    if quality_complete:
        prev_quality = _aggregate_quality([quality_by_day[row["day"]] for row in previous])
        curr_quality = _aggregate_quality([quality_by_day[row["day"]] for row in latest])
    else:
        prev_quality = _aggregate_quality([])
        curr_quality = _aggregate_quality([])
    prev_fp = prev_quality["false_positive_rate_avg_pct"]
    curr_fp = curr_quality["false_positive_rate_avg_pct"]
    denominator = curr_scout["early_capture"]["denominator"]
    if denominator < MIN_COMPARISON_TOP_DENOMINATOR:
        status = "small_denominator"
    elif not quality_complete:
        status = "quality_guardrails_incomplete"
    else:
        status = "comparable"
    return {
        "status": status,
        "window_days": window_days,
        "previous_days": [row["day"] for row in previous],
        "latest_days": [row["day"] for row in latest],
        "latest_top_denominator": denominator,
        "quality_complete_for_compared_days": quality_complete,
        "early_capture_delta_pp": None if None in (prev_early, curr_early) else round(curr_early - prev_early, 2),
        "capture_delta_pp": None if None in (prev_capture, curr_capture) else round(curr_capture - prev_capture, 2),
        "false_positive_delta_pp": None if None in (prev_fp, curr_fp) else round(curr_fp - prev_fp, 2),
        "exit_efficiency_delta": None if None in (
            prev_quality["median_exit_efficiency"], curr_quality["median_exit_efficiency"]
        ) else round(curr_quality["median_exit_efficiency"] - prev_quality["median_exit_efficiency"], 4),
    }


def _verdict(scout: dict[str, Any], quality: dict[str, Any], comparison: dict[str, Any]) -> dict[str, Any]:
    early = scout.get("early_capture_rate_pct")
    denominator = _num((scout.get("early_capture") or {}).get("denominator"))
    confidence = "low" if denominator < 30 else ("medium" if denominator < 100 else "high")
    if early is None:
        blocker = "objective denominator is unknown"
    elif not quality.get("days_loaded"):
        blocker = "signal-quality guardrails are incomplete"
    elif early < EARLY_CAPTURE_TARGET_PCT:
        blocker = "early top-mover capture remains below target"
    elif (quality.get("median_exit_efficiency") or 0.0) < 0:
        blocker = "exit monetization is negative"
    else:
        blocker = "false-positive and portfolio quality require monitoring"

    if comparison.get("status") != "comparable":
        toward_goal = "inconclusive"
        reason = f"comparison_{comparison.get('status', 'unknown')}"
    else:
        early_delta = comparison.get("early_capture_delta_pp")
        capture_delta = comparison.get("capture_delta_pp")
        fp_delta = comparison.get("false_positive_delta_pp")
        exit_delta = comparison.get("exit_efficiency_delta")
        if (
            early_delta is not None and early_delta >= 5.0
            and (capture_delta is None or capture_delta >= -2.0)
            and (fp_delta is None or fp_delta <= 2.0)
            and (exit_delta is None or exit_delta >= -0.05)
        ):
            toward_goal = "improving"
            reason = "early_capture_up_with_guardrails"
        elif (
            (early_delta is not None and early_delta <= -5.0)
            or (capture_delta is not None and capture_delta <= -5.0)
            or (fp_delta is not None and fp_delta >= 5.0)
            or (exit_delta is not None and exit_delta <= -0.10)
        ):
            toward_goal = "worsening"
            reason = "one_or_more_objective_guardrails_worsened"
        else:
            toward_goal = "flat"
            reason = "no_material_guardrailed_change"
    return {
        "toward_goal": toward_goal,
        "confidence": confidence,
        "reason": reason,
        "main_blocker": blocker,
        "target": {"early_capture_rate_pct": EARLY_CAPTURE_TARGET_PCT},
    }


def _rl_freshness(latest_rl: dict[str, Any]) -> dict[str, Any]:
    raw = str(latest_rl.get("generated_at_utc") or "")
    age_hours: float | None = None
    try:
        generated = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        age_hours = round((datetime.now(timezone.utc) - generated).total_seconds() / 3600.0, 2)
    except Exception:
        pass
    return {
        "status": "unknown" if age_hours is None else ("fresh" if age_hours <= 48.0 else "stale"),
        "age_hours": age_hours,
        "budget_hours": 48.0,
    }


def build(days: int, end_day: date | None = None) -> dict:
    records: list[dict[str, Any]] = []
    loaded_days: list[str] = []
    coverage_days: list[dict[str, Any]] = []
    for day in _complete_days(days, end_day):
        ds = day.isoformat()
        critic = _load_day_report("top_gainer_critic_", ds, "_final.json")
        goal = _load_day_report("watchlist_top_gainer_goal_", ds, "_22h.json")
        quality = _load_day_report("signal_quality_", ds, "_final.json")
        status, reasons = _coverage_status(critic, goal, quality)
        if critic:
            loaded_days.append(ds)
        row = {"day": ds, "critic": critic, "goal": goal, "quality": quality, "status": status}
        records.append(row)
        coverage_days.append({
            "day": ds,
            "status": status,
            "reasons": reasons,
            "has_critic": bool(critic),
            "has_goal": bool(goal),
            "has_quality": bool(quality),
        })

    objective_eligible = [
        row for row in records
        if _critic_complete(row["critic"]) and _goal_complete(row["goal"])
    ]
    quality_eligible = [row for row in records if _quality_complete(row["quality"])]
    scout = _aggregate_scout(objective_eligible)
    goal = _aggregate_goal(objective_eligible)
    quality = _aggregate_quality(quality_eligible)
    comparison = _comparison(objective_eligible, quality_eligible)
    latest_rl = _load_json(REPORTS / "rl_train_latest.json") or {}
    train = latest_rl.get("train_report", {}) or {}
    test_rank = (train.get("test_group_ranking", {}) or {}).get("top_n", []) or []
    top1 = next((x for x in test_rank if x.get("top_n") == 1), {})
    ml = {
        "scope": "reported_test_group_ranking; holdout timing must be verified separately",
        "model_name": latest_rl.get("model_name"),
        "rows_total": latest_rl.get("rows_total"),
        "critic_rows_total": latest_rl.get("critic_rows_total"),
        "ml_rows_total": latest_rl.get("ml_rows_total"),
        "top1_delta_shadow": latest_rl.get("top1_delta"),
        "test_top1_target_return_delta": top1.get("delta", {}).get("avg_target_return"),
        "test_top1_win_rate_delta": top1.get("delta", {}).get("win_rate"),
        "test_top1_teacher_top_gainer_rate_delta": top1.get("delta", {}).get("teacher_top_gainer_rate"),
        "test_top1_teacher_capture_ratio_delta": top1.get("delta", {}).get("teacher_capture_ratio"),
        "diagnostic_only": True,
    }
    rl = {
        "training_run_index": latest_rl.get("training_run_index"),
        "generated_at_utc": latest_rl.get("generated_at_utc"),
        "freshness": _rl_freshness(latest_rl),
        "collector_last_cycle_stats": latest_rl.get("collector_last_cycle_stats", {}),
        "teacher_rows_total": latest_rl.get("critic_rows_total"),
    }
    verdict = _verdict(scout, quality, comparison)

    return {
        "schema_version": 2,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "days_requested": days,
        "loaded_days": loaded_days,
        "eligible_days": [row["day"] for row in objective_eligible],
        "objective_eligible_days": [row["day"] for row in objective_eligible],
        "quality_eligible_days": [row["day"] for row in quality_eligible],
        "coverage": {
            "policy": (
                "capture uses days with final critic and goal denominator; "
                "quality uses complete signal-quality days; incomplete quality blocks improvement"
            ),
            "days": coverage_days,
            "status_counts": {
                status: sum(1 for row in coverage_days if row["status"] == status)
                for status in ("complete", "partial", "missing")
            },
        },
        "goal": goal,
        "scout": scout,
        "quality": quality,
        "comparison": comparison,
        "ml": ml,
        "rl": rl,
        "verdict": verdict,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=14)
    parser.add_argument("--output")
    args = parser.parse_args()
    payload = build(args.days)
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
