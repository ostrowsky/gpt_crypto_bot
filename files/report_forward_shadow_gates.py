from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from typing import Any
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parent.parent
FILES = ROOT / "files"
REPORTS = ROOT / ".runtime" / "reports"
DEFAULT_RESEARCH_DATASET = FILES / "research_universe_shadow.jsonl"
DEFAULT_MODEL_METADATA = ROOT / ".runtime" / "models" / "research_early_trend_discriminator.json"
DEFAULT_BOT_EVENTS = FILES / "bot_events.jsonl"
DEFAULT_OUTPUT = REPORTS / "forward_shadow_promotion_gates_latest.json"
DEFAULT_TEXT_OUTPUT = REPORTS / "forward_shadow_promotion_gates_latest.txt"
LOCAL_ZONE = ZoneInfo("Europe/Budapest")


@dataclass(frozen=True)
class ForwardGateConfig:
    early_min_mature: int = 30
    early_min_days: int = 5
    early_min_critic_days: int = 5
    early_min_top_movers: int = 5
    early_min_primary_precision_pct: float = 9.0
    early_min_strict_precision_pct: float = 4.5
    early_min_avg_ret_5_pct: float = 0.0
    early_min_avg_ret_10_pct: float = 0.12
    early_max_precision_loss_pp: float = 2.0
    tail_selector: str = "exclude_ema_and_false_cleanup"
    tail_min_mature: int = 30
    tail_min_days: int = 5
    tail_min_avg_delta_pct: float = 0.0
    tail_min_median_delta_pct: float = 0.0
    tail_max_worse_rate_pct: float = 35.0


def build_report(
    *,
    research_dataset: Path = DEFAULT_RESEARCH_DATASET,
    model_metadata: Path = DEFAULT_MODEL_METADATA,
    bot_events: Path = DEFAULT_BOT_EVENTS,
    critics_dir: Path = REPORTS,
    config: ForwardGateConfig | None = None,
) -> dict[str, Any]:
    cfg = config or ForwardGateConfig()
    early = _build_early_gate(research_dataset, model_metadata, critics_dir, cfg)
    tail = _build_tail_gate(bot_events, cfg)
    decisions = {str(early.get("decision")), str(tail.get("decision"))}
    if any(decision.startswith("invalid_") for decision in decisions):
        decision = "measurement_error_fail_closed"
    elif all("ready_for" in decision for decision in decisions):
        decision = "ready_for_separate_full_replays_not_production"
    else:
        decision = "collect_independent_forward_evidence"
    return {
        "generated_at_utc": _fmt_ts(datetime.now(timezone.utc)),
        "mode": "measurement_only",
        "config": asdict(cfg),
        "early_trend": early,
        "observable_tail": tail,
        "decision": decision,
        "production_eligible": False,
        "guardrails": [
            "replay_rows_before_profile_creation_excluded",
            "first_symbol_local_day_candidate_only",
            "canonical_top_movers_from_exchange_top_gainers_in_watchlist",
            "no_buy_sell_telegram_or_portfolio_change",
        ],
    }


def _build_early_gate(dataset: Path, metadata_path: Path, critics_dir: Path, cfg: ForwardGateConfig) -> dict[str, Any]:
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {
            "status": "invalid_metadata",
            "decision": "invalid_early_metadata_fail_closed",
            "error": f"{type(exc).__name__}: {exc}",
            "production_eligible": False,
        }
    profile = str(metadata.get("profile") or "")
    created = _parse_ts(metadata.get("created_at_utc"))
    if not profile or created is None:
        return {
            "status": "invalid_metadata",
            "decision": "invalid_early_metadata_fail_closed",
            "error": "profile_or_created_at_missing",
            "production_eligible": False,
        }

    rows: list[dict[str, Any]] = []
    malformed = 0
    if dataset.exists():
        with dataset.open("r", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except Exception:
                    malformed += 1
                    continue
                if not isinstance(row, dict):
                    malformed += 1
                    continue
                ts = _parse_ts(row.get("ts_utc"))
                if ts is None or ts <= created:
                    continue
                if str(row.get("tf") or "") != "15m" or not bool(row.get("in_trade_watchlist")):
                    continue
                row = dict(row)
                row["_ts"] = ts
                row["_day"] = ts.astimezone(LOCAL_ZONE).date().isoformat()
                rows.append(row)

    rows.sort(key=lambda row: row["_ts"])
    baseline_first: dict[tuple[str, str], dict[str, Any]] = {}
    candidate_first: dict[tuple[str, str], dict[str, Any]] = {}
    annotated_rows = 0
    raw_candidates = 0
    for row in rows:
        symbol = str(row.get("sym") or "").upper()
        day = str(row.get("_day") or "")
        if not symbol or not day:
            continue
        key = (day, symbol)
        if str(row.get("rule_signal") or "none") != "none":
            baseline_first.setdefault(key, row)
        annotation = row.get("early_trend_shadow") or {}
        if str(annotation.get("profile") or "") != profile:
            continue
        annotated_rows += 1
        if bool(annotation.get("candidate")):
            raw_candidates += 1
            candidate_first.setdefault(key, row)

    candidates = list(candidate_first.values())
    mature = [row for row in candidates if _label(row, "ret_5") is not None and _label(row, "ret_10") is not None]
    ret5 = [float(_label(row, "ret_5")) for row in mature]
    ret10 = [float(_label(row, "ret_10")) for row in mature]
    primary = [row for row in mature if float(_label(row, "ret_5")) >= 0.5 and float(_label(row, "ret_10")) >= 1.0]
    strict = [row for row in mature if float(_label(row, "ret_5")) >= 1.0 and float(_label(row, "ret_10")) >= 2.0]
    cohort_days = sorted({str(row.get("_day") or "") for row in candidates})
    north_star = _north_star_forward_check(
        critics_dir=critics_dir,
        cohort_days=cohort_days,
        baseline_first=baseline_first,
        candidate_first=candidate_first,
    )
    metrics = {
        "primary_precision_pct": _pct(len(primary), len(mature)),
        "strict_precision_pct": _pct(len(strict), len(mature)),
        "avg_ret_5_pct": _avg(ret5),
        "median_ret_5_pct": _median(ret5),
        "avg_ret_10_pct": _avg(ret10),
        "median_ret_10_pct": _median(ret10),
    }
    checks = {
        "support": len(mature) >= cfg.early_min_mature and len(cohort_days) >= cfg.early_min_days,
        "primary_precision": _ge(metrics["primary_precision_pct"], cfg.early_min_primary_precision_pct),
        "strict_precision": _ge(metrics["strict_precision_pct"], cfg.early_min_strict_precision_pct),
        "avg_ret_5": _ge(metrics["avg_ret_5_pct"], cfg.early_min_avg_ret_5_pct),
        "avg_ret_10": _ge(metrics["avg_ret_10_pct"], cfg.early_min_avg_ret_10_pct),
        "critic_support": int(north_star.get("critic_days") or 0) >= cfg.early_min_critic_days
        and int(north_star.get("canonical_top_movers") or 0) >= cfg.early_min_top_movers,
        "canonical_recall": _ge(north_star.get("recall_delta_pp"), 0.0),
        "canonical_precision": _ge(north_star.get("precision_delta_pp"), -cfg.early_max_precision_loss_pp),
        "earlier_or_new_capture": int(north_star.get("newly_captured") or 0) + int(north_star.get("earlier_captured") or 0) >= 1,
    }
    if not checks["support"]:
        decision = "collect_forward_labels"
    elif not checks["critic_support"]:
        decision = "collect_final_critic_coverage"
    elif all(checks.values()):
        decision = "ready_for_fee_slippage_portfolio_replay_not_production"
    else:
        decision = "forward_gate_failed_keep_shadow_only"
    return {
        "status": "complete",
        "profile": profile,
        "model_created_at_utc": _fmt_ts(created),
        "coverage": {
            "eligible_rows_after_creation": len(rows),
            "annotated_rows": annotated_rows,
            "raw_candidates": raw_candidates,
            "first_symbol_day_candidates": len(candidates),
            "mature_both": len(mature),
            "pending": len(candidates) - len(mature),
            "local_days": len(cohort_days),
            "first_day": cohort_days[0] if cohort_days else None,
            "last_day": cohort_days[-1] if cohort_days else None,
            "malformed_rows_skipped": malformed,
        },
        "metrics": metrics,
        "north_star": north_star,
        "checks": checks,
        "decision": decision,
        "production_eligible": False,
    }


def _north_star_forward_check(
    *,
    critics_dir: Path,
    cohort_days: list[str],
    baseline_first: dict[tuple[str, str], dict[str, Any]],
    candidate_first: dict[tuple[str, str], dict[str, Any]],
) -> dict[str, Any]:
    top_keys: set[tuple[str, str]] = set()
    covered_days: list[str] = []
    for day in cohort_days:
        path = critics_dir / f"top_gainer_critic_{day}_final.json"
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if str(report.get("phase") or "") != "final" or str(report.get("target_day_local") or "") != day:
            continue
        covered_days.append(day)
        for row in report.get("exchange_top_gainers") or []:
            if isinstance(row, dict) and bool(row.get("in_watchlist")) and row.get("symbol"):
                top_keys.add((day, str(row["symbol"]).upper()))

    covered = set(covered_days)
    base_keys = {key for key in baseline_first if key[0] in covered}
    addon_keys = {key for key in candidate_first if key[0] in covered}
    union_keys = base_keys | addon_keys
    base_hits = base_keys & top_keys
    union_hits = union_keys & top_keys
    newly = (addon_keys - base_keys) & top_keys
    earlier_minutes: list[float] = []
    for key in addon_keys & base_keys & top_keys:
        candidate_ts = candidate_first[key].get("_ts")
        baseline_ts = baseline_first[key].get("_ts")
        if isinstance(candidate_ts, datetime) and isinstance(baseline_ts, datetime) and candidate_ts < baseline_ts:
            earlier_minutes.append((baseline_ts - candidate_ts).total_seconds() / 60.0)
    base_recall = _pct(len(base_hits), len(top_keys))
    union_recall = _pct(len(union_hits), len(top_keys))
    base_precision = _pct(len(base_hits), len(base_keys))
    union_precision = _pct(len(union_hits), len(union_keys))
    return {
        "critic_days": len(covered_days),
        "covered_days": covered_days,
        "canonical_top_movers": len(top_keys),
        "baseline_candidates": len(base_keys),
        "adjunct_union_candidates": len(union_keys),
        "baseline_recall_pct": base_recall,
        "adjunct_recall_pct": union_recall,
        "recall_delta_pp": _delta(union_recall, base_recall),
        "baseline_precision_pct": base_precision,
        "adjunct_precision_pct": union_precision,
        "precision_delta_pp": _delta(union_precision, base_precision),
        "newly_captured": len(newly),
        "earlier_captured": len(earlier_minutes),
        "avg_earlier_minutes": _avg(earlier_minutes),
    }


def _build_tail_gate(events_path: Path, cfg: ForwardGateConfig) -> dict[str, Any]:
    candidates_by_key: dict[tuple[str, str, str, float], list[dict[str, Any]]] = defaultdict(list)
    candidates: list[dict[str, Any]] = []
    malformed = 0
    relevant_labels = 0
    if events_path.exists():
        with events_path.open("r", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                if "observable_tail_shadow_" not in line or cfg.tail_selector not in line:
                    continue
                try:
                    event = json.loads(line)
                except Exception:
                    malformed += 1
                    continue
                if not isinstance(event, dict) or str(event.get("selector") or "") != cfg.tail_selector:
                    continue
                key = _tail_key(event)
                ts = _parse_ts(event.get("ts"))
                if key is None or ts is None:
                    malformed += 1
                    continue
                if event.get("event") == "observable_tail_shadow_candidate":
                    candidate = dict(event)
                    candidate["_ts"] = ts
                    candidate["_labels"] = {}
                    candidates_by_key[key].append(candidate)
                    candidates.append(candidate)
                elif event.get("event") == "observable_tail_shadow_label":
                    try:
                        horizon = int(event.get("horizon"))
                    except Exception:
                        malformed += 1
                        continue
                    for candidate in reversed(candidates_by_key.get(key, [])):
                        if candidate["_ts"] <= ts and horizon not in candidate["_labels"]:
                            candidate["_labels"][horizon] = event
                            relevant_labels += 1
                            break

    mature = [candidate for candidate in candidates if 10 in candidate["_labels"]]
    deltas = [float(candidate["_labels"][10].get("partial_delta_pct") or 0.0) for candidate in mature]
    days = sorted({candidate["_ts"].astimezone(LOCAL_ZONE).date().isoformat() for candidate in mature})
    metrics = {
        "avg_partial_delta_10_pct": _avg(deltas),
        "median_partial_delta_10_pct": _median(deltas),
        "positive_rate_pct": _pct(sum(1 for value in deltas if value > 0.0), len(deltas)),
        "worse_rate_pct": _pct(sum(1 for value in deltas if value < 0.0), len(deltas)),
    }
    checks = {
        "support": len(mature) >= cfg.tail_min_mature and len(days) >= cfg.tail_min_days,
        "avg_delta": _gt(metrics["avg_partial_delta_10_pct"], cfg.tail_min_avg_delta_pct),
        "median_delta": _gt(metrics["median_partial_delta_10_pct"], cfg.tail_min_median_delta_pct),
        "worse_rate": _le(metrics["worse_rate_pct"], cfg.tail_max_worse_rate_pct),
    }
    if not checks["support"]:
        decision = "collect_forward_labels"
    elif all(checks.values()):
        decision = "ready_for_full_portfolio_replay_not_production"
    else:
        decision = "forward_gate_failed_keep_shadow_only"
    return {
        "status": "complete",
        "selector": cfg.tail_selector,
        "coverage": {
            "candidates": len(candidates),
            "matched_labels": relevant_labels,
            "mature_t10": len(mature),
            "pending_t10": len(candidates) - len(mature),
            "local_days": len(days),
            "first_day": days[0] if days else None,
            "last_day": days[-1] if days else None,
            "malformed_relevant_rows_skipped": malformed,
        },
        "metrics": metrics,
        "checks": checks,
        "decision": decision,
        "production_eligible": False,
    }


def _tail_key(event: dict[str, Any]) -> tuple[str, str, str, float] | None:
    try:
        price = round(float(event.get("exit_price")), 8)
    except Exception:
        return None
    symbol = str(event.get("sym") or "").upper()
    timeframe = str(event.get("tf") or "")
    selector = str(event.get("selector") or "")
    if not symbol or not timeframe or not selector or not math.isfinite(price):
        return None
    return symbol, timeframe, selector, price


def _label(row: dict[str, Any], key: str) -> float | None:
    try:
        value = float((row.get("labels") or {}).get(key))
    except Exception:
        return None
    return value if math.isfinite(value) else None


def _parse_ts(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return parsed.astimezone(timezone.utc)
    except Exception:
        return None


def _fmt_ts(value: datetime) -> str:
    return value.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _avg(values: list[float]) -> float | None:
    return round(mean(values), 6) if values else None


def _median(values: list[float]) -> float | None:
    return round(median(values), 6) if values else None


def _pct(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator * 100.0, 4) if denominator else None


def _delta(left: float | None, right: float | None) -> float | None:
    return round(float(left) - float(right), 4) if left is not None and right is not None else None


def _ge(value: Any, threshold: float) -> bool:
    return value is not None and float(value) >= threshold


def _gt(value: Any, threshold: float) -> bool:
    return value is not None and float(value) > threshold


def _le(value: Any, threshold: float) -> bool:
    return value is not None and float(value) <= threshold


def render_text(report: dict[str, Any]) -> str:
    early = report.get("early_trend") or {}
    early_coverage = early.get("coverage") or {}
    early_metrics = early.get("metrics") or {}
    north = early.get("north_star") or {}
    tail = report.get("observable_tail") or {}
    tail_coverage = tail.get("coverage") or {}
    tail_metrics = tail.get("metrics") or {}
    return "\n".join(
        [
            "Forward shadow promotion gates",
            f"decision: {report.get('decision')}",
            "",
            "Early trend:",
            f"  decision={early.get('decision')} mature={early_coverage.get('mature_both', 0)}/{early_coverage.get('first_symbol_day_candidates', 0)} days={early_coverage.get('local_days', 0)}",
            f"  primary={early_metrics.get('primary_precision_pct')}% avg T+5/T+10={early_metrics.get('avg_ret_5_pct')}/{early_metrics.get('avg_ret_10_pct')}%",
            f"  critic days/top={north.get('critic_days', 0)}/{north.get('canonical_top_movers', 0)} recall delta={north.get('recall_delta_pp')}pp earlier/new={north.get('earlier_captured', 0)}/{north.get('newly_captured', 0)}",
            "",
            "Observable tail:",
            f"  decision={tail.get('decision')} mature={tail_coverage.get('mature_t10', 0)}/{tail_coverage.get('candidates', 0)} days={tail_coverage.get('local_days', 0)}",
            f"  avg/median T+10 partial delta={tail_metrics.get('avg_partial_delta_10_pct')}/{tail_metrics.get('median_partial_delta_10_pct')}% worse={tail_metrics.get('worse_rate_pct')}%",
            "",
            "Production behavior unchanged.",
        ]
    ) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Independent forward gates for replay-approved shadow profiles")
    parser.add_argument("--research-dataset", type=Path, default=DEFAULT_RESEARCH_DATASET)
    parser.add_argument("--model-metadata", type=Path, default=DEFAULT_MODEL_METADATA)
    parser.add_argument("--bot-events", type=Path, default=DEFAULT_BOT_EVENTS)
    parser.add_argument("--critics-dir", type=Path, default=REPORTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--text-output", type=Path, default=DEFAULT_TEXT_OUTPUT)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()
    report = build_report(
        research_dataset=args.research_dataset,
        model_metadata=args.model_metadata,
        bot_events=args.bot_events,
        critics_dir=args.critics_dir,
    )
    text = render_text(report)
    if not args.no_save:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        args.text_output.write_text(text, encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2) if args.json else text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
