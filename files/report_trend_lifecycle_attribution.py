from __future__ import annotations

import argparse
import bisect
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable

import research_artifact_provenance as artifact_provenance
import research_event_cohort_store as cohort_store


ROOT = Path(__file__).resolve().parent.parent
FILES = ROOT / "files"
REPORTS = ROOT / ".runtime" / "reports"
COHORT_DB = ROOT / ".runtime" / "research_event_cohorts.sqlite3"
DEFAULT_OUTPUT = REPORTS / "trend_lifecycle_attribution_latest.json"
DEFAULT_TEXT_OUTPUT = REPORTS / "trend_lifecycle_attribution_latest.txt"
ROUND_TRIP_COST_PCT = 0.20
PORTFOLIO_CAPACITY_REASONS = {"portfolio_full", "strategy_cap", "open_cluster_cap"}


def build(
    *,
    files_dir: Path = FILES,
    reports_dir: Path = REPORTS,
    cohort_db: Path = COHORT_DB,
    output: Path = DEFAULT_OUTPUT,
    text_output: Path = DEFAULT_TEXT_OUTPUT,
    lookback_days: int = 0,
    save: bool = True,
) -> dict[str, Any]:
    signal_reports = _load_reports(reports_dir, "signal_quality_*_final.json")
    critic_reports = _load_reports(reports_dir, "top_gainer_critic_*_final.json")
    signal_reports, critic_reports = _apply_lookback(signal_reports, critic_reports, lookback_days)
    report_days = {row["day"] for row in signal_reports if row.get("day")}
    missed = _dedupe_missed(signal_reports)
    late_entries = _dedupe_trade_failures(signal_reports, "late_entries")
    early_exits = _dedupe_trade_failures(signal_reports, "early_exits")
    episode_days = {
        day
        for row in missed
        for day in (_local_day(row.get("start_ts")), _local_day(row.get("peak_ts")))
        if day
    }
    blocked_rows, cohort_sync = cohort_store.load_blocked_intervals(
        files_dir=files_dir,
        allowed_days=report_days | episode_days,
        db_path=cohort_db,
    )
    trade_events, _ = cohort_store.load_trade_events(files_dir=files_dir, db_path=cohort_db, sync=False)
    blocked_index = BlockedIntervalIndex(blocked_rows)
    observation_index = ObservationIndex(files_dir)
    capacity_times, capacity_counts = _capacity_timeline(trade_events)

    missed_rows = [
        _attribute_missed(row, blocked_index, observation_index, capacity_times, capacity_counts)
        for row in missed
    ]
    late_rows = [_attribute_trade_failure(row, "entered_late") for row in late_entries]
    exit_rows = [_attribute_trade_failure(row, "exited_early") for row in early_exits]
    all_cases = missed_rows + late_rows + exit_rows
    ranked = _rank_casebook(all_cases)
    detail_coverage = _detail_coverage(signal_reports)
    latest_day = max(report_days) if report_days else None
    latest_complete = all(
        item["complete"] for item in detail_coverage["by_report"]
        if item["day"] == latest_day
    ) if latest_day else False
    status = "complete" if detail_coverage["all_reports_complete"] else "partial_historical_detail"
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "status": status,
        "latest_day": latest_day,
        "latest_day_complete": latest_complete,
        "scope": {
            "signal_quality_reports": len(signal_reports),
            "critic_reports": len(critic_reports),
            "lookback_days": lookback_days or None,
            "first_signal_day": min(report_days) if report_days else None,
            "last_signal_day": latest_day,
            "unique_missed_episodes_exported": len(missed_rows),
            "unique_late_entries_exported": len(late_rows),
            "unique_early_exits_exported": len(exit_rows),
            "cohort_sync": cohort_sync,
            "observation_rows": observation_index.rows_loaded,
            "observation_window": {
                "first": _iso(observation_index.first_ts_ms) if observation_index.first_ts_ms else None,
                "last": _iso(observation_index.last_ts_ms) if observation_index.last_ts_ms else None,
            },
        },
        "data_quality": detail_coverage,
        "metric_families": {
            "watchlist_top_north_star": _north_star_metrics(critic_reports),
            "broad_trend_lifecycle": _broad_metrics(signal_reports, missed_rows, late_rows, exit_rows),
            "warning": "watchlist-top early capture and broad trend-episode miss rate use different denominators",
        },
        "attribution": {
            "missed_stage_counts": dict(Counter(row["stage"] for row in missed_rows)),
            "missed_stage_opportunity": _stage_opportunity(missed_rows),
            "blocked_reason_summary": _blocked_reason_summary(missed_rows),
            "entered_late": len(late_rows),
            "exited_early": len(exit_rows),
        },
        "casebook": ranked[:100],
        "unranked_no_causal_decision_price": sum(
            1 for row in all_cases if row.get("realizable_opportunity_net_pct") is None
        ),
        "rows": {
            "missed": missed_rows,
            "entered_late": late_rows,
            "exited_early": exit_rows,
        },
        "provenance": artifact_provenance.build_provenance(
            builder="trend_lifecycle_attribution_v1",
            research_config={
                "round_trip_cost_pct": ROUND_TRIP_COST_PCT,
                "blocked_interval_minutes": 15,
                "lookback_days": lookback_days or None,
            },
            input_paths=[
                files_dir / "research_universe_shadow.jsonl",
                files_dir / "v2_shadow_events.jsonl",
                cohort_db,
                *[Path(row["path"]) for row in signal_reports],
                *[Path(row["path"]) for row in critic_reports],
            ],
        ),
    }
    if save:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        text_output.write_text(render_text(payload), encoding="utf-8")
        payload["files"] = {"json": str(output), "txt": str(text_output)}
    return payload


class BlockedIntervalIndex:
    def __init__(self, rows: Iterable[dict[str, Any]]):
        self.rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            item = dict(row)
            item["first_ms"] = _ts_ms(item.get("first_ts"))
            item["last_ms"] = _ts_ms(item.get("last_ts"))
            if item["first_ms"] and item["last_ms"]:
                self.rows[str(item.get("symbol") or "").upper()].append(item)
        for values in self.rows.values():
            values.sort(key=lambda row: row["first_ms"])

    def between(self, symbol: str, start_ms: int, end_ms: int) -> list[dict[str, Any]]:
        return [
            row for row in self.rows.get(symbol.upper(), [])
            if row["first_ms"] <= end_ms and row["last_ms"] >= start_ms
        ]


class ObservationIndex:
    def __init__(self, files_dir: Path):
        self.rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self.times: dict[str, list[int]] = {}
        self.rows_loaded = 0
        self.first_ts_ms = 0
        self.last_ts_ms = 0
        self._load_research(files_dir / "research_universe_shadow.jsonl")
        self._load_v2(files_dir / "v2_shadow_events.jsonl")
        for symbol, values in self.rows.items():
            values.sort(key=lambda row: row["ts_ms"])
            self.times[symbol] = [row["ts_ms"] for row in values]
        all_times = [ts for values in self.times.values() for ts in values]
        if all_times:
            self.first_ts_ms = min(all_times)
            self.last_ts_ms = max(all_times)

    def between(self, symbol: str, start_ms: int, end_ms: int) -> list[dict[str, Any]]:
        symbol = symbol.upper()
        times = self.times.get(symbol, [])
        values = self.rows.get(symbol, [])
        left = bisect.bisect_left(times, start_ms)
        right = bisect.bisect_right(times, end_ms)
        return values[left:right]

    def coverage_available(self, start_ms: int, end_ms: int) -> bool:
        return bool(self.first_ts_ms and self.last_ts_ms and start_ms <= self.last_ts_ms and end_ms >= self.first_ts_ms)

    def _load_research(self, path: Path) -> None:
        for row in _jsonl(path):
            if not bool(row.get("in_trade_watchlist", True)):
                continue
            ts_ms = int(row.get("bar_ts") or _ts_ms(row.get("ts_utc")))
            symbol = str(row.get("sym") or "").upper()
            if not ts_ms or not symbol:
                continue
            rule = str(row.get("rule_signal") or "none").lower()
            self.rows[symbol].append({
                "ts_ms": ts_ms,
                "ts": row.get("ts_utc") or _iso(ts_ms),
                "source": "research_universe_shadow",
                "signal": rule not in {"", "none", "null"},
                "signal_name": rule,
                "price": _num(row.get("price")),
            })
            self.rows_loaded += 1

    def _load_v2(self, path: Path) -> None:
        for row in _jsonl(path):
            if row.get("event") != "v2_shadow_signal":
                continue
            ts_ms = _ts_ms(row.get("ts"))
            symbol = str(row.get("sym") or "").upper()
            if not ts_ms or not symbol:
                continue
            features = row.get("features") if isinstance(row.get("features"), dict) else {}
            self.rows[symbol].append({
                "ts_ms": ts_ms,
                "ts": row.get("ts"),
                "source": "v2_shadow",
                "signal": True,
                "signal_name": row.get("state") or row.get("action"),
                "price": _num(features.get("price")),
            })
            self.rows_loaded += 1


def _attribute_missed(
    row: dict[str, Any],
    blocked_index: BlockedIntervalIndex,
    observations: ObservationIndex,
    capacity_times: list[str],
    capacity_counts: list[int],
) -> dict[str, Any]:
    symbol = str(row.get("sym") or row.get("symbol") or "").upper()
    start_ms, peak_ms = _ts_ms(row.get("start_ts")), _ts_ms(row.get("peak_ts"))
    blockers = blocked_index.between(symbol, start_ms, peak_ms)
    observed = observations.between(symbol, start_ms, peak_ms)
    signals = [item for item in observed if item.get("signal")]
    capacity_blockers = [
        item for item in blockers if str(item.get("reason_code") or "") in PORTFOLIO_CAPACITY_REASONS
    ]
    if capacity_blockers:
        stage = "blocked_by_portfolio_capacity"
        decisions = capacity_blockers
    elif blockers or signals:
        stage = "signaled_but_rejected"
        decisions = blockers + signals
    elif observed:
        stage = "observed_but_not_signaled"
        decisions = observed
    elif observations.coverage_available(start_ms, peak_ms):
        stage = "not_observed"
        decisions = []
    else:
        stage = "observation_coverage_unavailable"
        decisions = []
    decisions.sort(key=lambda item: int(item.get("first_ms") or item.get("ts_ms") or 0))
    decision = decisions[0] if decisions else {}
    decision_ts = decision.get("first_ts") or decision.get("ts")
    decision_price = _num(decision.get("first_price"), _num(decision.get("price")))
    peak_price = _num(row.get("peak_price"))
    opportunity = _net_opportunity(decision_price, peak_price)
    reasons = Counter(str(item.get("reason_code") or "") for item in blockers)
    return {
        "case_type": "missed_trend",
        "stage": stage,
        "day": _local_day(row.get("peak_ts")),
        "symbol": symbol,
        "tf": row.get("tf"),
        "start_ts": row.get("start_ts"),
        "peak_ts": row.get("peak_ts"),
        "move_pct": _round(_num(row.get("move_pct"))),
        "top_mover_rank": row.get("top_mover_rank"),
        "decision_ts": decision_ts,
        "decision_price": decision_price,
        "decision_price_source": (decision.get("first_source") or decision.get("source")) if decision_price is not None else None,
        "realizable_opportunity_net_pct": opportunity,
        "blocked_reasons": dict(reasons),
        "blocked_events": sum(int(item.get("block_count") or 0) for item in blockers),
        "observation_events": len(observed),
        "signal_events": len(signals),
        "open_positions_at_decision": _capacity_at(decision_ts, capacity_times, capacity_counts),
    }


def _attribute_trade_failure(row: dict[str, Any], stage: str) -> dict[str, Any]:
    trend = row.get("trend") if isinstance(row.get("trend"), dict) else {}
    decision_ts = row.get("entry_ts") if stage == "entered_late" else row.get("exit_ts")
    decision_price = _num(row.get("entry_price") if stage == "entered_late" else row.get("exit_price"))
    peak_price = _num(trend.get("peak_price"))
    opportunity = _net_opportunity(decision_price, peak_price)
    if opportunity is None and stage == "exited_early":
        future = _num(row.get("future_favorable_pct"))
        opportunity = _round(future - ROUND_TRIP_COST_PCT) if future is not None else None
    return {
        "case_type": stage,
        "stage": stage,
        "day": _local_day(decision_ts),
        "symbol": str(row.get("sym") or row.get("symbol") or "").upper(),
        "tf": row.get("tf"),
        "start_ts": trend.get("start_ts"),
        "peak_ts": trend.get("peak_ts"),
        "move_pct": _round(_num(trend.get("move_pct"))),
        "top_mover_rank": trend.get("top_mover_rank"),
        "decision_ts": decision_ts,
        "decision_price": decision_price,
        "decision_price_source": "actual_entry" if stage == "entered_late" else "actual_exit",
        "realizable_opportunity_net_pct": opportunity,
        "pnl_pct": _round(_num(row.get("pnl_pct"))),
        "giveback_pct": _round(_num(row.get("giveback_pct"))),
        "exit_reason": row.get("exit_reason"),
    }


def _load_reports(directory: Path, pattern: str) -> list[dict[str, Any]]:
    by_day: dict[str, dict[str, Any]] = {}
    for path in directory.glob(pattern):
        try:
            data = json.loads(path.read_text(encoding="utf-8-sig"))
        except Exception:
            continue
        if not isinstance(data, dict):
            continue
        day = str(data.get("target_day_local") or "")
        if not day:
            continue
        candidate = {"day": day, "path": str(path), "data": data, "mtime": path.stat().st_mtime_ns}
        if day not in by_day or candidate["mtime"] > by_day[day]["mtime"]:
            by_day[day] = candidate
    return sorted(by_day.values(), key=lambda row: row["day"])


def _apply_lookback(
    signal_reports: list[dict[str, Any]],
    critic_reports: list[dict[str, Any]],
    lookback_days: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if lookback_days <= 0:
        return signal_reports, critic_reports
    selected_signal = signal_reports[-lookback_days:]
    if not selected_signal:
        return selected_signal, []
    first_day = selected_signal[0]["day"]
    last_day = selected_signal[-1]["day"]
    return selected_signal, [row for row in critic_reports if first_day <= row["day"] <= last_day]


def _dedupe_missed(reports: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: dict[tuple[str, str, str], dict[str, Any]] = {}
    for report in reports:
        for row in report["data"].get("missed_trends") or []:
            if not isinstance(row, dict):
                continue
            key = (str(row.get("sym") or ""), str(row.get("tf") or ""), str(row.get("start_ts") or ""))
            if all(key):
                rows[key] = row
    return sorted(rows.values(), key=lambda row: (str(row.get("start_ts") or ""), str(row.get("sym") or "")))


def _dedupe_trade_failures(reports: list[dict[str, Any]], bucket: str) -> list[dict[str, Any]]:
    rows: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for report in reports:
        for row in report["data"].get(bucket) or []:
            if not isinstance(row, dict):
                continue
            key = (
                str(row.get("sym") or row.get("symbol") or ""),
                str(row.get("tf") or ""),
                str(row.get("entry_ts") or ""),
                str(row.get("exit_ts") or ""),
            )
            if key[0] and (key[2] or key[3]):
                rows[key] = row
    return sorted(rows.values(), key=lambda row: (str(row.get("entry_ts") or ""), str(row.get("sym") or "")))


def _detail_coverage(reports: list[dict[str, Any]]) -> dict[str, Any]:
    rows = []
    for report in reports:
        data = report["data"]
        summary = data.get("summary") or {}
        explicit = data.get("detail_coverage") or {}
        checks = []
        for bucket, summary_key in (
            ("missed_trends", "missed_trends"),
            ("late_entries", "late_entries"),
            ("early_exits", "early_exits"),
            ("false_positive_buys", "false_positive_buys"),
        ):
            total = int(summary.get(summary_key) or 0)
            exported = len(data.get(bucket) or [])
            meta = explicit.get(bucket) if isinstance(explicit, dict) else None
            complete = bool(meta.get("complete")) if isinstance(meta, dict) else exported >= total
            checks.append({"bucket": bucket, "total": total, "exported": exported, "complete": complete})
        rows.append({"day": report["day"], "complete": all(item["complete"] for item in checks), "buckets": checks})
    return {
        "all_reports_complete": bool(rows) and all(row["complete"] for row in rows),
        "complete_reports": sum(1 for row in rows if row["complete"]),
        "partial_reports": sum(1 for row in rows if not row["complete"]),
        "by_report": rows,
    }


def _north_star_metrics(reports: list[dict[str, Any]]) -> dict[str, Any]:
    recent = reports[-7:]
    denominator = sum(int((row["data"].get("summary") or {}).get("watchlist_top_count") or 0) for row in recent)
    early = sum(int((row["data"].get("summary") or {}).get("watchlist_top_early_captured") or 0) for row in recent)
    bought = sum(int((row["data"].get("summary") or {}).get("watchlist_top_bought") or 0) for row in recent)
    return {
        "window_days": len(recent),
        "watchlist_top": denominator,
        "bought": bought,
        "early_captured": early,
        "early_capture_rate_pct": _round(early / denominator * 100.0) if denominator else None,
    }


def _broad_metrics(
    reports: list[dict[str, Any]],
    missed: list[dict[str, Any]],
    late: list[dict[str, Any]],
    early_exit: list[dict[str, Any]],
) -> dict[str, Any]:
    latest = reports[-1]["data"].get("summary") if reports else {}
    return {
        "latest_day_summary": latest or {},
        "unique_exported_missed_episodes": len(missed),
        "unique_exported_late_entries": len(late),
        "unique_exported_early_exits": len(early_exit),
    }


def _stage_opportunity(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        value = _num(row.get("realizable_opportunity_net_pct"))
        if value is not None:
            grouped[row["stage"]].append(value)
    return {
        stage: {
            "labeled": len(values),
            "avg_net_pct": _round(mean(values)),
            "median_net_pct": _round(median(values)),
        }
        for stage, values in grouped.items()
    }


def _blocked_reason_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    aggregates: dict[str, dict[str, Any]] = {}
    for row in rows:
        opportunity = _num(row.get("realizable_opportunity_net_pct"))
        for reason, count in (row.get("blocked_reasons") or {}).items():
            item = aggregates.setdefault(str(reason), {"cases": 0, "events": 0, "opportunities": []})
            item["cases"] += 1
            item["events"] += int(count or 0)
            if opportunity is not None:
                item["opportunities"].append(opportunity)
    result = []
    for reason, item in aggregates.items():
        values = item.pop("opportunities")
        result.append({
            "reason_code": reason,
            **item,
            "labeled_cases": len(values),
            "avg_remaining_net_pct": _round(mean(values)) if values else None,
            "median_remaining_net_pct": _round(median(values)) if values else None,
        })
    return sorted(result, key=lambda row: (row["cases"], row["events"]), reverse=True)


def _rank_casebook(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    best: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in rows:
        value = _num(row.get("realizable_opportunity_net_pct"))
        if value is None:
            continue
        key = (str(row.get("day") or ""), str(row.get("symbol") or ""), str(row.get("stage") or ""))
        if key not in best or value > float(best[key]["realizable_opportunity_net_pct"]):
            best[key] = row
    return sorted(best.values(), key=lambda row: float(row["realizable_opportunity_net_pct"]), reverse=True)


def _capacity_timeline(events: list[dict[str, Any]]) -> tuple[list[str], list[int]]:
    open_symbols: set[str] = set()
    times: list[str] = []
    counts: list[int] = []
    for row in sorted(events, key=lambda item: str(item.get("ts") or "")):
        symbol = str(row.get("sym") or "")
        if row.get("event") == "entry":
            open_symbols.add(symbol)
        elif row.get("event") == "exit":
            open_symbols.discard(symbol)
        times.append(str(row.get("ts") or ""))
        counts.append(len(open_symbols))
    return times, counts


def _capacity_at(ts: str | None, times: list[str], counts: list[int]) -> int | None:
    if not ts:
        return None
    index = bisect.bisect_right(times, ts) - 1
    return counts[index] if 0 <= index < len(counts) else 0


def _net_opportunity(decision_price: float | None, peak_price: float | None) -> float | None:
    if decision_price is None or peak_price is None or decision_price <= 0:
        return None
    return _round((peak_price / decision_price - 1.0) * 100.0 - ROUND_TRIP_COST_PCT)


def _jsonl(path: Path) -> Iterable[dict[str, Any]]:
    if not path.exists():
        return
    with path.open("rb") as handle:
        for raw in handle:
            try:
                row = json.loads(raw)
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue
            if isinstance(row, dict):
                yield row


def _ts_ms(value: Any) -> int:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return 0
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return int(parsed.timestamp() * 1000)


def _local_day(value: Any) -> str:
    ts_ms = _ts_ms(value)
    if not ts_ms:
        return ""
    return datetime.fromtimestamp(ts_ms / 1000.0, timezone.utc).astimezone(cohort_store.TZ).date().isoformat()


def _iso(ts_ms: int) -> str:
    return datetime.fromtimestamp(ts_ms / 1000.0, timezone.utc).isoformat().replace("+00:00", "Z")


def _num(value: Any, default: float | None = None) -> float | None:
    try:
        return default if value is None else float(value)
    except (TypeError, ValueError):
        return default


def _round(value: float | None) -> float | None:
    return round(float(value), 6) if value is not None else None


def render_text(report: dict[str, Any]) -> str:
    attr = report.get("attribution") or {}
    north = (report.get("metric_families") or {}).get("watchlist_top_north_star") or {}
    lines = [
        "Trend lifecycle attribution",
        f"status: {report.get('status')} latest={report.get('latest_day')} latest_complete={report.get('latest_day_complete')}",
        f"north-star 7d: early={north.get('early_captured')}/{north.get('watchlist_top')} ({north.get('early_capture_rate_pct')}%)",
        "missed stages: " + ", ".join(f"{key}={value}" for key, value in (attr.get("missed_stage_counts") or {}).items()),
        f"entered_late={attr.get('entered_late')} exited_early={attr.get('exited_early')}",
        "",
        "Top realizable opportunities after causal decision:",
    ]
    for row in (report.get("casebook") or [])[:15]:
        lines.append(
            f"  {row.get('day')} {row.get('symbol')} {row.get('stage')} "
            f"remaining_net={row.get('realizable_opportunity_net_pct')}%"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Build broad-trend lifecycle attribution and causal opportunity casebook.")
    parser.add_argument("--files-dir", type=Path, default=FILES)
    parser.add_argument("--reports-dir", type=Path, default=REPORTS)
    parser.add_argument("--cohort-db", type=Path, default=COHORT_DB)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--text-output", type=Path, default=DEFAULT_TEXT_OUTPUT)
    parser.add_argument("--lookback-days", type=int, default=0, help="0 uses every available report")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    payload = build(
        files_dir=args.files_dir,
        reports_dir=args.reports_dir,
        cohort_db=args.cohort_db,
        output=args.output,
        text_output=args.text_output,
        lookback_days=max(0, args.lookback_days),
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2) if args.json else render_text(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
