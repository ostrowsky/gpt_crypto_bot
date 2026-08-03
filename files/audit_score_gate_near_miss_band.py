from __future__ import annotations

import argparse
import bisect
import json
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable
from zoneinfo import ZoneInfo

import audit_early_block_rescue_event_replay as event_replay
import research_artifact_provenance as artifact_provenance
import research_event_cohort_store as cohort_store


ROOT = Path(__file__).resolve().parent.parent
FILES = ROOT / "files"
REPORTS = ROOT / ".runtime" / "reports"
CANDLE_CACHE = ROOT / ".runtime" / "signal_quality_cache"
COHORT_DB = ROOT / ".runtime" / "research_event_cohorts.sqlite3"
DEFAULT_OUTPUT = REPORTS / "score_gate_32_33_causal_audit_latest.json"
DEFAULT_TEXT_OUTPUT = REPORTS / "score_gate_32_33_causal_audit_latest.txt"
TZ = ZoneInfo("Europe/Budapest")
SCORE_RE = re.compile(
    r"score\s+(-?\d+(?:\.\d+)?)\s*<\s*(-?\d+(?:\.\d+)?)\s+for\s+(15m|1h)\s+([a-z0-9_]+)",
    re.I,
)
CACHE_RE = re.compile(r"^([A-Z0-9]+)_([^_]+)_(\d+)_(\d+)\.json$")
BAR_MS = {"15m": 900_000, "1h": 3_600_000}


@dataclass(frozen=True)
class AuditConfig:
    min_score: float = 32.0
    max_score_exclusive: float = 34.0
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
    labels, label_days = _load_labels_and_days(reports_dir)
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
    capacity_times, capacity_counts = _capacity_timeline(trade_events)
    cache = CandleCache(candle_cache)

    rows = []
    for candidate in candidates:
        key = (candidate["day"], candidate["symbol"])
        entries = entries_by_key.get(key, [])
        candidate_ts = candidate["ts"]
        earlier_entries = [row for row in entries if str(row.get("ts") or "") <= candidate_ts]
        later_entries = [row for row in entries if str(row.get("ts") or "") > candidate_ts]
        open_count = _capacity_at(candidate_ts, capacity_times, capacity_counts)
        already_bought = bool(earlier_entries)
        capacity_available = open_count < cfg.portfolio_capacity
        label = labels.get(key, {})
        candle_metrics = cache.forward_metrics(candidate, cfg)
        other_reasons = sorted(
            reason for reason in reasons_by_key.get(key, set())
            if reason and reason != "top_gainer_score_gate"
        )
        rows.append({
            **candidate,
            "is_watchlist_top": bool(label),
            "critic_status": str(label.get("status") or "not_top15"),
            "capture_ratio_at_entry": label.get("capture_ratio_at_entry"),
            "opportunity_from_first_block_pct": label.get("opportunity_from_first_block_pct"),
            "already_bought_before_band": already_bought,
            "later_bought_by_control": bool(later_entries),
            "first_later_entry_ts": later_entries[0].get("ts") if later_entries else None,
            "open_positions_at_candidate": open_count,
            "capacity_available": capacity_available,
            "admission_eligible": capacity_available and not already_bought,
            "other_blockers_seen_same_day": other_reasons,
            **candle_metrics,
        })
    rows.sort(key=lambda row: (row["day"], row["ts"], row["symbol"]))
    mature = [row for row in rows if row.get("ret10_net_pct") is not None]
    train, holdout = _chronological_split(mature, cfg.train_fraction)
    recent_cutoff = _recent_cutoff(mature, cfg.recent_days)
    recent = [row for row in mature if row["day"] >= recent_cutoff] if recent_cutoff else []
    segments = {
        "all_mature": _summary(mature, cfg),
        "train": _summary(train, cfg),
        "holdout": _summary(holdout, cfg),
        "recent_stability": _summary(recent, cfg),
    }
    decision = _decision(segments, cfg)
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "status": "research_only",
        "config": cfg.__dict__,
        "scope": {
            "label_days": len(label_days),
            "first_label_day": min(label_days) if label_days else None,
            "last_label_day": max(label_days) if label_days else None,
            "raw_band_events": scan["band_events"],
            "unique_day_symbols": len(rows),
            "mature_rows": len(mature),
            "candle_cache_files_loaded": cache.loaded_file_count,
            "cohort_sync": cohort_sync,
        },
        "attribution": _attribution(rows),
        "segments": segments,
        "decision": decision,
        "promotion": {
            "watch_shadow": decision == "advance_score_32_33_to_watch_shadow",
            "buy": False,
            "reason": "BUY requires independent WATCH forward evidence even when frozen replay passes",
        },
        "rows": rows,
        "provenance": artifact_provenance.build_provenance(
            builder="score_gate_32_33_causal_audit_v1",
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


def render_text(report: dict[str, Any]) -> str:
    lines = [
        "Score-gate 32-33 frozen causal audit",
        f"decision: {report.get('decision')}",
        f"scope: {(report.get('scope') or {}).get('first_label_day')}..{(report.get('scope') or {}).get('last_label_day')} "
        f"candidates={(report.get('scope') or {}).get('unique_day_symbols')} mature={(report.get('scope') or {}).get('mature_rows')}",
        "",
    ]
    for name, row in (report.get("segments") or {}).items():
        lines.append(
            f"{name}: n={row.get('n')} eligible={row.get('admission_eligible')} top={row.get('top_candidates')} "
            f"precision={row.get('top_precision_pct')}% ret10 avg/med={row.get('avg_ret10_net_pct')}/{row.get('median_ret10_net_pct')}% "
            f"positive={row.get('ret10_positive_rate_pct')}% top_opportunities={row.get('earlier_top_opportunities')}"
        )
    attr = report.get("attribution") or {}
    lines.extend([
        "",
        "Attribution:",
        f"  already_bought={attr.get('already_bought_before_band')} later_control_buy={attr.get('later_bought_by_control')} "
        f"capacity_full={attr.get('portfolio_capacity_full')} other_blockers={attr.get('other_blockers_seen')}",
    ])
    return "\n".join(lines) + "\n"


def _load_labels_and_days(reports_dir: Path) -> tuple[dict[tuple[str, str], dict[str, Any]], set[str]]:
    labels = event_replay._load_labels(reports_dir)
    days: set[str] = set()
    for path in reports_dir.glob("top_gainer_critic_*_final.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8-sig"))
        except Exception:
            continue
        day = str(payload.get("target_day_local") or "")
        if day:
            days.add(day)
    return labels, days


def _scan_candidates(
    files_dir: Path,
    label_days: set[str],
    cfg: AuditConfig,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    selected: dict[tuple[str, str], dict[str, Any]] = {}
    band_events = 0
    matched_events = 0
    for name in cohort_store.SOURCE_NAMES:
        path = files_dir / name
        if not path.exists():
            continue
        with path.open("rb") as handle:
            for raw in handle:
                if b"top_gainer_score_gate" not in raw or b"score" not in raw:
                    continue
                try:
                    row = json.loads(raw)
                except (UnicodeDecodeError, json.JSONDecodeError):
                    continue
                if row.get("event") != "blocked":
                    continue
                match = SCORE_RE.search(str(row.get("reason") or ""))
                if not match:
                    continue
                matched_events += 1
                score, floor = float(match.group(1)), float(match.group(2))
                if not (cfg.min_score <= score < cfg.max_score_exclusive):
                    continue
                if abs(floor - cfg.required_floor) > 1e-9:
                    continue
                ts = str(row.get("ts") or "")
                day, _ = cohort_store._local_day_hour(ts)
                symbol = str(row.get("sym") or row.get("symbol") or "").upper()
                if day not in label_days or not symbol:
                    continue
                band_events += 1
                key = (day, symbol)
                candidate = {
                    "day": day,
                    "symbol": symbol,
                    "tf": match.group(3),
                    "mode": match.group(4).lower(),
                    "score": score,
                    "score_floor": floor,
                    "ts": ts,
                    "ts_ms": _ts_ms(ts),
                    "price": _num(row.get("price")),
                    "source": str(row.get("source") or ("market_agent" if name.startswith("agent") else "bot")),
                    "repeat_count": 1,
                }
                current = selected.get(key)
                if current is None:
                    selected[key] = candidate
                elif ts < current["ts"]:
                    candidate["repeat_count"] = int(current.get("repeat_count") or 0) + 1
                    selected[key] = candidate
                else:
                    current["repeat_count"] += 1
    return list(selected.values()), {"matched_events": matched_events, "band_events": band_events}


class CandleCache:
    def __init__(self, directory: Path):
        self.paths: dict[tuple[str, str], list[tuple[int, int, Path]]] = {}
        self.loaded: dict[Path, list[dict[str, Any]]] = {}
        for path in directory.glob("*.json") if directory.exists() else []:
            match = CACHE_RE.match(path.name)
            if not match:
                continue
            symbol, tf, start, end = match.groups()
            self.paths.setdefault((symbol, tf), []).append((int(start), int(end), path))
        for rows in self.paths.values():
            rows.sort(key=lambda item: (item[1] - item[0], -item[2].stat().st_mtime_ns))

    @property
    def loaded_file_count(self) -> int:
        return len(self.loaded)

    def forward_metrics(self, candidate: dict[str, Any], cfg: AuditConfig) -> dict[str, Any]:
        ts_ms = int(candidate.get("ts_ms") or 0)
        tf = str(candidate.get("tf") or "")
        bar_ms = BAR_MS.get(tf)
        price = _num(candidate.get("price"))
        if not ts_ms or not bar_ms or not price or price <= 0:
            return _empty_candle_metrics("invalid_candidate")
        end_needed = ts_ms + cfg.horizon_bars * bar_ms
        for start, end, path in self.paths.get((candidate["symbol"], tf), []):
            if start > ts_ms or end < end_needed:
                continue
            candles = self._load(path)
            if not candles:
                continue
            times = [int(row.get("t") or 0) for row in candles]
            index = bisect.bisect_right(times, ts_ms) - 1
            if index < 0 or index + cfg.horizon_bars >= len(candles):
                continue
            horizon = candles[index + cfg.horizon_bars]
            path_rows = candles[index + 1:index + cfg.horizon_bars + 1]
            cost_pct = cfg.round_trip_cost_bps / 100.0
            close10 = _num(horizon.get("c"))
            close5 = _num(candles[index + 5].get("c")) if index + 5 < len(candles) else None
            if close10 is None:
                continue
            ret5 = None
            if close5 is not None:
                ret5 = (close5 / price - 1.0) * 100.0 - cost_pct
            ret10 = (close10 / price - 1.0) * 100.0 - cost_pct
            mfe10 = (max(_num(row.get("h")) or price for row in path_rows) / price - 1.0) * 100.0
            mae10 = (min(_num(row.get("l")) or price for row in path_rows) / price - 1.0) * 100.0
            return {
                "label_status": "mature",
                "ret5_net_pct": _round(ret5),
                "ret10_net_pct": _round(ret10),
                "mfe10_pct": _round(mfe10),
                "mae10_pct": _round(mae10),
                "candle_cache_path": str(path),
            }
        return _empty_candle_metrics("missing_candles")

    def _load(self, path: Path) -> list[dict[str, Any]]:
        if path not in self.loaded:
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                payload = []
            self.loaded[path] = payload if isinstance(payload, list) else []
        return self.loaded[path]


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


def _capacity_at(ts: str, times: list[str], counts: list[int]) -> int:
    index = bisect.bisect_right(times, ts) - 1
    return counts[index] if 0 <= index < len(counts) else 0


def _chronological_split(rows: list[dict[str, Any]], fraction: float) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    ordered = sorted(rows, key=lambda row: (row["day"], row["ts"], row["symbol"]))
    if len(ordered) <= 1:
        return ordered, []
    cut = max(1, min(len(ordered) - 1, int(len(ordered) * fraction)))
    return ordered[:cut], ordered[cut:]


def _recent_cutoff(rows: list[dict[str, Any]], days: int) -> str:
    if not rows:
        return ""
    last = datetime.fromisoformat(max(row["day"] for row in rows)).date()
    return (last - timedelta(days=max(0, days - 1))).isoformat()


def _summary(rows: list[dict[str, Any]], cfg: AuditConfig) -> dict[str, Any]:
    eligible = [
        row for row in rows
        if row.get("capacity_available") and not row.get("already_bought_before_band")
    ]
    values = [_num(row.get("ret10_net_pct")) for row in eligible]
    values = [value for value in values if value is not None]
    top = [row for row in eligible if row.get("is_watchlist_top")]
    false = [row for row in eligible if not row.get("is_watchlist_top")]
    false_values = [_num(row.get("ret10_net_pct")) for row in false]
    false_values = [value for value in false_values if value is not None]
    opportunities = [
        row for row in top
        if not row.get("already_bought_before_band")
    ]
    return {
        "n": len(rows),
        "capacity_eligible": sum(1 for row in rows if row.get("capacity_available")),
        "admission_eligible": len(eligible),
        "top_candidates": len(top),
        "false_candidates": len(false),
        "top_precision_pct": _round(len(top) / len(eligible) * 100.0) if eligible else 0.0,
        "avg_ret10_net_pct": _avg(values),
        "median_ret10_net_pct": _median(values),
        "ret10_positive_rate_pct": _round(sum(1 for value in values if value > 0) / len(values) * 100.0) if values else 0.0,
        "avg_false_ret10_net_pct": _avg(false_values),
        "median_mae10_pct": _median([_num(row.get("mae10_pct")) for row in eligible]),
        "earlier_top_opportunities": len(opportunities),
        "profitable_top_opportunities": sum(1 for row in opportunities if (_num(row.get("ret10_net_pct")) or 0.0) > 0.0),
    }


def _segment_passes(row: dict[str, Any], cfg: AuditConfig) -> bool:
    return bool(
        int(row.get("admission_eligible") or 0) >= cfg.min_holdout_cases
        and int(row.get("earlier_top_opportunities") or 0) >= cfg.min_top_opportunities
        and (_num(row.get("avg_ret10_net_pct")) or 0.0) > 0.0
        and (_num(row.get("median_ret10_net_pct")) or 0.0) > 0.0
        and (_num(row.get("ret10_positive_rate_pct")) or 0.0) >= 50.0
    )


def _decision(segments: dict[str, dict[str, Any]], cfg: AuditConfig) -> str:
    if all(_segment_passes(segments[name], cfg) for name in ("all_mature", "holdout", "recent_stability")):
        return "advance_score_32_33_to_watch_shadow"
    return "reject_score_32_33_watch_shadow_gate"


def _attribution(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "observed_in_score_band": len(rows),
        "already_bought_before_band": sum(1 for row in rows if row.get("already_bought_before_band")),
        "later_bought_by_control": sum(1 for row in rows if row.get("later_bought_by_control")),
        "portfolio_capacity_full": sum(1 for row in rows if not row.get("capacity_available")),
        "other_blockers_seen": sum(1 for row in rows if row.get("other_blockers_seen_same_day")),
        "watchlist_top_candidates": sum(1 for row in rows if row.get("is_watchlist_top")),
        "not_watchlist_top_candidates": sum(1 for row in rows if not row.get("is_watchlist_top")),
        "missing_candle_labels": sum(1 for row in rows if row.get("label_status") != "mature"),
    }


def _empty_candle_metrics(status: str) -> dict[str, Any]:
    return {
        "label_status": status,
        "ret5_net_pct": None,
        "ret10_net_pct": None,
        "mfe10_pct": None,
        "mae10_pct": None,
        "candle_cache_path": None,
    }


def _ts_ms(value: str) -> int:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return 0
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return int(parsed.timestamp() * 1000)


def _num(value: Any) -> float | None:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def _round(value: float | None) -> float | None:
    return round(float(value), 6) if value is not None else None


def _avg(values: Iterable[float | None]) -> float | None:
    rows = [float(value) for value in values if value is not None]
    return _round(mean(rows)) if rows else None


def _median(values: Iterable[float | None]) -> float | None:
    rows = [float(value) for value in values if value is not None]
    return _round(median(rows)) if rows else None


def main() -> int:
    parser = argparse.ArgumentParser(description="Frozen causal audit of top-gainer score 32-33 near misses.")
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
