from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import audit_score_gate_near_miss_band as common
import research_artifact_provenance as artifact_provenance
import research_event_cohort_store as cohort_store


ROOT = Path(__file__).resolve().parent.parent
FILES = ROOT / "files"
REPORTS = ROOT / ".runtime" / "reports"
CANDLE_CACHE = ROOT / ".runtime" / "signal_quality_cache"
COHORT_DB = ROOT / ".runtime" / "research_event_cohorts.sqlite3"
DEFAULT_OUTPUT = REPORTS / "high_volume_breakout_rescue_audit_latest.json"
DEFAULT_TEXT_OUTPUT = REPORTS / "high_volume_breakout_rescue_audit_latest.txt"
FLOOR_RE = re.compile(r"score\s+-?\d+(?:\.\d+)?\s*<\s*(-?\d+(?:\.\d+)?)", re.I)


@dataclass(frozen=True)
class AuditConfig:
    timeframe: str = "15m"
    mode: str = "breakout"
    min_candidate_score: float = 120.0
    min_live_score: float = 28.0
    max_live_score_exclusive: float = 34.0
    min_vol_x: float = 5.0
    required_floor: float = 34.0
    horizon_bars: int = 10
    round_trip_cost_bps: float = 20.0
    train_fraction: float = 0.70
    recent_days: int = 14
    portfolio_capacity: int = 10
    min_holdout_cases: int = 10
    min_top_opportunities: int = 1


def build_audit(
    *,
    files_dir: Path = FILES,
    reports_dir: Path = REPORTS,
    candle_cache: Path = CANDLE_CACHE,
    cohort_db: Path = COHORT_DB,
    cfg: AuditConfig = AuditConfig(),
    output: Path = DEFAULT_OUTPUT,
    text_output: Path = DEFAULT_TEXT_OUTPUT,
    save: bool = True,
) -> dict[str, Any]:
    labels, label_days = common._load_labels_and_days(reports_dir)
    candidates, scan = _scan_candidates(files_dir, label_days, cfg)
    blocked, entries_by_key, cohort_sync = cohort_store.load_replay_inputs(
        files_dir=files_dir,
        allowed_days=label_days,
        db_path=cohort_db,
    )
    trade_events, _ = cohort_store.load_trade_events(
        files_dir=files_dir,
        db_path=cohort_db,
    )
    reasons_by_key: dict[tuple[str, str], set[str]] = {}
    for row in blocked:
        reasons_by_key.setdefault((row["day"], row["symbol"]), set()).add(str(row.get("reason_code") or ""))
    capacity_times, capacity_counts = common._capacity_timeline(trade_events)
    cache = common.CandleCache(candle_cache)

    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        key = (candidate["day"], candidate["symbol"])
        candidate_ts = candidate["ts"]
        entries = entries_by_key.get(key, [])
        earlier_entries = [row for row in entries if str(row.get("ts") or "") <= candidate_ts]
        later_entries = [row for row in entries if str(row.get("ts") or "") > candidate_ts]
        open_count = common._capacity_at(candidate_ts, capacity_times, capacity_counts)
        label = labels.get(key, {})
        rows.append({
            **candidate,
            "is_watchlist_top": bool(label),
            "critic_status": str(label.get("status") or "not_top15"),
            "capture_ratio_at_entry": label.get("capture_ratio_at_entry"),
            "opportunity_from_first_block_pct": label.get("opportunity_from_first_block_pct"),
            "already_bought_before_candidate": bool(earlier_entries),
            "already_bought_before_band": bool(earlier_entries),
            "later_bought_by_control": bool(later_entries),
            "first_later_entry_ts": later_entries[0].get("ts") if later_entries else None,
            "open_positions_at_candidate": open_count,
            "capacity_available": open_count < cfg.portfolio_capacity,
            "other_blockers_seen_same_day": sorted(
                reason for reason in reasons_by_key.get(key, set())
                if reason and reason != "top_gainer_score_gate"
            ),
            **cache.forward_metrics(candidate, cfg),
        })
    rows.sort(key=lambda row: (row["day"], row["ts"], row["symbol"]))
    mature = [row for row in rows if row.get("ret10_net_pct") is not None]
    train, holdout = common._chronological_split(mature, cfg.train_fraction)
    recent_cutoff = common._recent_cutoff(mature, cfg.recent_days)
    recent = [row for row in mature if row["day"] >= recent_cutoff] if recent_cutoff else []
    summary_cfg = common.AuditConfig(
        horizon_bars=cfg.horizon_bars,
        round_trip_cost_bps=cfg.round_trip_cost_bps,
        train_fraction=cfg.train_fraction,
        recent_days=cfg.recent_days,
        portfolio_capacity=cfg.portfolio_capacity,
        min_holdout_cases=cfg.min_holdout_cases,
        min_top_opportunities=cfg.min_top_opportunities,
    )
    segments = {
        "all_mature": common._summary(mature, summary_cfg),
        "train": common._summary(train, summary_cfg),
        "holdout": common._summary(holdout, summary_cfg),
        "recent_stability": common._summary(recent, summary_cfg),
    }
    decision = (
        "advance_high_volume_breakout_to_watch_shadow"
        if all(common._segment_passes(segments[name], summary_cfg) for name in ("all_mature", "holdout", "recent_stability"))
        else "reject_high_volume_breakout_rescue"
    )
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "status": "research_only",
        "config": cfg.__dict__,
        "scope": {
            "label_days": len(label_days),
            "first_label_day": min(label_days) if label_days else None,
            "last_label_day": max(label_days) if label_days else None,
            "first_candidate_day": min((row["day"] for row in rows), default=None),
            "last_candidate_day": max((row["day"] for row in rows), default=None),
            **scan,
            "unique_day_symbols": len(rows),
            "mature_rows": len(mature),
            "candle_cache_files_loaded": cache.loaded_file_count,
            "cohort_sync": cohort_sync,
        },
        "segments": segments,
        "decision": decision,
        "promotion": {
            "watch_shadow": decision == "advance_high_volume_breakout_to_watch_shadow",
            "buy": False,
            "reason": "BUY requires an independent full-policy portfolio replay and forward WATCH evidence",
        },
        "rows": rows,
        "provenance": artifact_provenance.build_provenance(
            builder="high_volume_breakout_rescue_audit_v1",
            research_config=cfg,
            input_paths=[
                files_dir / "bot_events.jsonl",
                files_dir / "agent_events.jsonl",
                cohort_db,
                *([latest_critic] if (latest_critic := artifact_provenance.latest_path(reports_dir, "top_gainer_critic_*_final.json")) else []),
            ],
        ),
    }
    text = render_text(payload)
    if save:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        text_output.write_text(text, encoding="utf-8")
        payload["files"] = {"json": str(output), "txt": str(text_output)}
    return payload


def _scan_candidates(
    files_dir: Path,
    label_days: set[str],
    cfg: AuditConfig,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    selected: dict[tuple[str, str], dict[str, Any]] = {}
    learning_events = 0
    profile_events = 0
    for name in cohort_store.SOURCE_NAMES:
        path = files_dir / name
        if not path.exists():
            continue
        with path.open("rb") as handle:
            for raw in handle:
                if b"blocked_learning_label" not in raw or b"blocked_strong_score_gate" not in raw:
                    continue
                try:
                    row = json.loads(raw)
                except (UnicodeDecodeError, json.JSONDecodeError):
                    continue
                if row.get("event") != "blocked_learning_label" or row.get("label_type") != "blocked_strong_score_gate":
                    continue
                learning_events += 1
                reason = str(row.get("reason") or "")
                floor_match = FLOOR_RE.search(reason)
                floor = float(floor_match.group(1)) if floor_match else None
                candidate_score = common._num(row.get("candidate_score"))
                live_score = common._num(row.get("live_score"))
                vol_x = common._num(row.get("vol_x"))
                if (
                    str(row.get("tf") or "") != cfg.timeframe
                    or str(row.get("mode") or "").lower() != cfg.mode
                    or candidate_score is None
                    or candidate_score < cfg.min_candidate_score
                    or live_score is None
                    or not (cfg.min_live_score <= live_score < cfg.max_live_score_exclusive)
                    or vol_x is None
                    or vol_x < cfg.min_vol_x
                    or floor is None
                    or abs(floor - cfg.required_floor) > 1e-9
                ):
                    continue
                ts = str(row.get("ts") or "")
                day, _ = cohort_store._local_day_hour(ts)
                symbol = str(row.get("sym") or row.get("symbol") or "").upper()
                if day not in label_days or not symbol:
                    continue
                profile_events += 1
                key = (day, symbol)
                candidate = {
                    "day": day,
                    "symbol": symbol,
                    "tf": cfg.timeframe,
                    "mode": cfg.mode,
                    "ts": ts,
                    "ts_ms": common._ts_ms(ts),
                    "price": common._num(row.get("price")),
                    "candidate_score": candidate_score,
                    "live_score": live_score,
                    "vol_x": vol_x,
                    "adx": common._num(row.get("adx")),
                    "rsi": common._num(row.get("rsi")),
                    "daily_range": common._num(row.get("daily_range")),
                    "source": "market_agent" if name.startswith("agent") else "bot",
                    "repeat_count": 1,
                }
                current = selected.get(key)
                if current is None or ts < current["ts"]:
                    if current is not None:
                        candidate["repeat_count"] = int(current.get("repeat_count") or 0) + 1
                    selected[key] = candidate
                else:
                    current["repeat_count"] += 1
    return list(selected.values()), {
        "blocked_learning_events": learning_events,
        "raw_profile_events": profile_events,
    }


def render_text(report: dict[str, Any]) -> str:
    scope = report.get("scope") or {}
    lines = [
        "High-volume breakout score-gate rescue audit",
        f"decision: {report.get('decision')}",
        f"scope: {scope.get('first_candidate_day')}..{scope.get('last_candidate_day')} "
        f"events={scope.get('raw_profile_events')} candidates={scope.get('unique_day_symbols')} mature={scope.get('mature_rows')}",
        "",
    ]
    for name, row in (report.get("segments") or {}).items():
        lines.append(
            f"{name}: n={row.get('n')} eligible={row.get('admission_eligible')} top={row.get('top_candidates')} "
            f"precision={row.get('top_precision_pct')}% ret10 avg/med={row.get('avg_ret10_net_pct')}/{row.get('median_ret10_net_pct')}% "
            f"positive={row.get('ret10_positive_rate_pct')}%"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Frozen high-volume 15m breakout rescue audit.")
    parser.add_argument("--files-dir", type=Path, default=FILES)
    parser.add_argument("--reports-dir", type=Path, default=REPORTS)
    parser.add_argument("--candle-cache", type=Path, default=CANDLE_CACHE)
    parser.add_argument("--cohort-db", type=Path, default=COHORT_DB)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--text-output", type=Path, default=DEFAULT_TEXT_OUTPUT)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    payload = build_audit(
        files_dir=args.files_dir,
        reports_dir=args.reports_dir,
        candle_cache=args.candle_cache,
        cohort_db=args.cohort_db,
        output=args.output,
        text_output=args.text_output,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2) if args.json else render_text(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
