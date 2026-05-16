from __future__ import annotations

import argparse
import json
from datetime import date, datetime, timedelta
from pathlib import Path
from statistics import mean, median


ROOT = Path(__file__).resolve().parents[3]
REPORTS = ROOT / ".runtime" / "reports"


def _load_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _complete_days(days: int, end_day: date | None = None) -> list[date]:
    end = end_day or (datetime.now().date() - timedelta(days=1))
    return [end - timedelta(days=i) for i in range(days - 1, -1, -1)]


def _pct(num: float, den: float) -> float:
    return round((num / den) * 100.0, 2) if den else 0.0


def _avg(values: list[float]) -> float | None:
    return round(mean(values), 4) if values else None


def _med(values: list[float]) -> float | None:
    return round(median(values), 4) if values else None


def build(days: int) -> dict:
    critic_rows: list[dict] = []
    goal_rows: list[dict] = []
    quality_rows: list[dict] = []
    loaded_days: list[str] = []
    for day in _complete_days(days):
        ds = day.isoformat()
        critic = _load_json(REPORTS / f"top_gainer_critic_{ds}_final.json")
        goal = _load_json(REPORTS / f"watchlist_top_gainer_goal_{ds}_22h.json")
        quality = _load_json(REPORTS / f"signal_quality_{ds}_final.json")
        if critic:
            critic_rows.append(critic)
            loaded_days.append(ds)
        if goal:
            goal_rows.append(goal)
        if quality:
            quality_rows.append(quality)

    latest_rl = _load_json(REPORTS / "rl_train_latest.json") or {}
    critic_summaries = [r.get("summary", {}) for r in critic_rows]
    goal_summaries = [r.get("summary", {}) for r in goal_rows]
    quality_summaries = [r.get("summary", {}) for r in quality_rows]

    scout = {
        "days_loaded": len(critic_rows),
        "watchlist_top_bought_avg": _avg([s.get("watchlist_top_bought", 0) for s in critic_summaries]),
        "watchlist_top_count_avg": _avg([s.get("watchlist_top_count", 0) for s in critic_summaries]),
        "watchlist_top_bought_rate_pct": _pct(
            sum(s.get("watchlist_top_bought", 0) for s in critic_summaries),
            sum(s.get("watchlist_top_count", 0) for s in critic_summaries),
        ),
        "early_capture_rate_pct": _pct(
            sum(s.get("watchlist_top_early_captured", 0) for s in critic_summaries),
            sum(s.get("watchlist_top_count", 0) for s in critic_summaries),
        ),
        "false_positive_buys_total": sum(s.get("bot_false_positive_buys", 0) for s in critic_summaries),
    }
    goal = {
        "days_loaded": len(goal_rows),
        "recall_at_cutoff_avg_pct": _avg([s.get("recall_at_cutoff_pct", 0.0) for s in goal_summaries]),
        "median_lead_time_min": _med([s.get("median_lead_time_min", 0.0) for s in goal_summaries]),
        "positive_coverage_avg_pct": _avg([s.get("mandatory_positive_coverage_pct", 0.0) for s in goal_summaries]),
        "false_positive_buys_total": sum(s.get("bot_false_positive_buys", 0) for s in goal_summaries),
    }
    quality = {
        "days_loaded": len(quality_rows),
        "miss_rate_avg_pct": _avg([s.get("miss_rate", 0.0) * 100.0 for s in quality_summaries]),
        "false_positive_rate_avg_pct": _avg([s.get("false_positive_rate", 0.0) * 100.0 for s in quality_summaries]),
        "median_capture_ratio": _med([s.get("capture_ratio_at_entry", {}).get("median") for s in quality_summaries if s.get("capture_ratio_at_entry", {}).get("median") is not None]),
        "median_exit_efficiency": _med([s.get("exit_efficiency", {}).get("median") for s in quality_summaries if s.get("exit_efficiency", {}).get("median") is not None]),
        "median_giveback_pct": _med([s.get("giveback_pct", {}).get("median") for s in quality_summaries if s.get("giveback_pct", {}).get("median") is not None]),
    }

    train = latest_rl.get("train_report", {})
    test_rank = train.get("test_group_ranking", {}).get("top_n", [])
    top1 = next((x for x in test_rank if x.get("top_n") == 1), {})
    ml = {
        "model_name": latest_rl.get("model_name"),
        "rows_total": latest_rl.get("rows_total"),
        "critic_rows_total": latest_rl.get("critic_rows_total"),
        "ml_rows_total": latest_rl.get("ml_rows_total"),
        "top1_delta_shadow": latest_rl.get("top1_delta"),
        "test_top1_target_return_delta": top1.get("delta", {}).get("avg_target_return"),
        "test_top1_win_rate_delta": top1.get("delta", {}).get("win_rate"),
        "test_top1_teacher_top_gainer_rate_delta": top1.get("delta", {}).get("teacher_top_gainer_rate"),
        "test_top1_teacher_capture_ratio_delta": top1.get("delta", {}).get("teacher_capture_ratio"),
    }
    rl = {
        "training_run_index": latest_rl.get("training_run_index"),
        "generated_at_utc": latest_rl.get("generated_at_utc"),
        "collector_last_cycle_stats": latest_rl.get("collector_last_cycle_stats", {}),
        "teacher_rows_total": latest_rl.get("critic_rows_total"),
    }
    verdict = {
        "toward_goal": "insufficient_data",
        "main_blocker": "missing rolling data",
    }
    if scout["watchlist_top_bought_rate_pct"] is not None:
        if scout["watchlist_top_bought_rate_pct"] >= 50 and (goal["recall_at_cutoff_avg_pct"] or 0) >= 50:
            verdict = {"toward_goal": "improving", "main_blocker": "exit quality / precision"}
        else:
            verdict = {"toward_goal": "not_yet", "main_blocker": "top-mover recall remains low"}

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "days_requested": days,
        "loaded_days": loaded_days,
        "goal": goal,
        "scout": scout,
        "quality": quality,
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
