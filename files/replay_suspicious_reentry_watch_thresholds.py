from __future__ import annotations

import argparse
import asyncio
import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import aiohttp
import numpy as np

import config
import research_universe_shadow_collector as market


ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = ROOT.parent
EVENTS_FILE = ROOT / "bot_events.jsonl"
WATCHLIST_FILE = ROOT / "watchlist.json"
OUTPUT_FILE = WORKSPACE_ROOT / ".runtime" / "reports" / "suspicious_reentry_watch_threshold_replay_latest.json"
HORIZONS = (2, 5, 10)
ROUND_TRIP_COST_PCT = 2.0 * float(getattr(config, "PAPER_FEE_BPS", 7.5)) / 100.0
EXIT_SCORE_GRID = (0.0, 0.45, 0.50, 0.55, 0.60, 0.65, 0.68, 0.70, 0.75)
MFE_GRID = (0.0, 0.25, 0.50, 0.75, 1.0, 1.5)


def load_watch_decisions(
    events_file: Path = EVENTS_FILE,
    *,
    valid_symbols: set[str] | None = None,
) -> list[dict[str, Any]]:
    if not events_file.exists():
        return []
    if valid_symbols is None:
        valid_symbols = _load_watchlist()
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    with events_file.open("r", encoding="utf-8", errors="replace") as source:
        for raw_line in source:
            try:
                row = json.loads(raw_line)
            except Exception:
                continue
            if not isinstance(row, dict) or row.get("event") != "suspicious_reentry_watch_decision":
                continue
            symbol = str(row.get("sym") or "")
            tf = str(row.get("tf") or "15m")
            ts = str(row.get("ts") or "")
            if not symbol or not ts or (valid_symbols and symbol not in valid_symbols):
                continue
            key = (symbol, tf, ts)
            if key in seen:
                continue
            seen.add(key)
            rows.append(row)
    rows.sort(key=lambda row: str(row.get("ts") or ""))
    return rows


async def run_replay(
    *,
    events_file: Path = EVENTS_FILE,
    output_file: Path = OUTPUT_FILE,
    concurrency: int = 6,
    save: bool = True,
) -> dict[str, Any]:
    decisions = load_watch_decisions(events_file)
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in decisions:
        grouped.setdefault((str(row["sym"]), str(row.get("tf") or "15m")), []).append(row)

    semaphore = asyncio.Semaphore(max(1, int(concurrency)))

    async def fetch_pair(
        session: aiohttp.ClientSession,
        key: tuple[str, str],
        rows: list[dict[str, Any]],
    ) -> tuple[tuple[str, str], Any | None]:
        timestamps = [_ts_ms(row["ts"]) for row in rows]
        bar_ms = market._timeframe_ms(key[1])
        async with semaphore:
            data = await market._fetch_range_klines(
                session,
                key[0],
                key[1],
                min(timestamps),
                max(timestamps) + (max(HORIZONS) + 2) * bar_ms,
            )
        return key, data

    headers = {"User-Agent": "Mozilla/5.0"}
    connector = aiohttp.TCPConnector(limit=max(4, int(concurrency) * 2))
    async with aiohttp.ClientSession(headers=headers, connector=connector) as session:
        fetched = await asyncio.gather(
            *[fetch_pair(session, key, rows) for key, rows in grouped.items()]
        )
    market_data = {key: data for key, data in fetched if data is not None and len(data)}
    labeled = label_decisions(decisions, market_data)
    report = evaluate_thresholds(labeled)
    report.update(
        {
            "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "events_file": str(events_file),
            "decisions_loaded": len(decisions),
            "pairs_requested": len(grouped),
            "pairs_fetched": len(market_data),
            "labels_mature": len(labeled),
            "round_trip_cost_pct": ROUND_TRIP_COST_PCT,
        }
    )
    if save:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def label_decisions(
    decisions: Iterable[dict[str, Any]],
    market_data: dict[tuple[str, str], Any],
) -> list[dict[str, Any]]:
    labeled: list[dict[str, Any]] = []
    for row in decisions:
        key = (str(row.get("sym") or ""), str(row.get("tf") or "15m"))
        data = market_data.get(key)
        if data is None or not len(data):
            continue
        idx = int(np.searchsorted(data["t"], _ts_ms(str(row.get("ts") or "")), side="left"))
        if idx >= len(data) or idx + max(HORIZONS) - 1 >= len(data):
            continue
        entry = _float(data["o"][idx])
        if entry <= 0:
            continue
        item = {
            "ts": str(row.get("ts") or ""),
            "sym": key[0],
            "tf": key[1],
            "decision": str(row.get("decision") or ""),
            "exit_score": _float(row.get("exit_score")),
            "mfe_pct": _float(row.get("mfe_pct")),
            "entry_price": entry,
        }
        for horizon in HORIZONS:
            item[f"ret_{horizon}"] = (_float(data["c"][idx + horizon - 1]) / entry - 1.0) * 100.0
        future_lows = data["l"][idx : idx + max(HORIZONS)].astype(float)
        item["drawdown_10"] = (float(np.min(future_lows)) / entry - 1.0) * 100.0
        labeled.append(item)
    return labeled


def evaluate_thresholds(labeled: list[dict[str, Any]]) -> dict[str, Any]:
    if not labeled:
        return {"status": "no_labels", "profiles": [], "decision": "keep_current"}
    ordered = sorted(labeled, key=lambda row: row["ts"])
    split_idx = max(1, min(len(ordered) - 1, int(len(ordered) * 0.70)))
    train = ordered[:split_idx]
    test = ordered[split_idx:]
    current_score = float(getattr(config, "SUSPICIOUS_REENTRY_SHADOW_EXIT_SCORE_MIN", 0.68))
    current_mfe = float(getattr(config, "SUSPICIOUS_REENTRY_SHADOW_MIN_MFE_PCT", 1.0))
    profiles = []
    for exit_score_min in EXIT_SCORE_GRID:
        for mfe_min in MFE_GRID:
            profiles.append(
                {
                    "exit_score_min": exit_score_min,
                    "mfe_min": mfe_min,
                    "train": _profile_stats(train, exit_score_min, mfe_min),
                    "test": _profile_stats(test, exit_score_min, mfe_min),
                }
            )
    current = next(
        row
        for row in profiles
        if row["exit_score_min"] == current_score and row["mfe_min"] == current_mfe
    )
    eligible = [row for row in profiles if row["train"]["count"] >= 20]
    selected = max(
        eligible or profiles,
        key=lambda row: (
            _or_floor(row["train"]["avg_net_ret5_pct"]),
            _or_floor(row["train"]["positive_after_cost_rate"]),
            row["train"]["count"],
        ),
    )
    test_improvement = _difference(
        selected["test"].get("avg_net_ret5_pct"),
        current["test"].get("avg_net_ret5_pct"),
    )
    passes = bool(
        selected["test"]["count"] >= 10
        and test_improvement is not None
        and test_improvement >= 0.10
        and _or_floor(selected["test"]["positive_after_cost_rate"])
        >= _or_floor(current["test"]["positive_after_cost_rate"])
        and _or_floor(selected["test"]["median_drawdown10_pct"])
        >= _or_floor(current["test"]["median_drawdown10_pct"]) - 0.25
    )
    return {
        "status": "complete",
        "window": {"first": ordered[0]["ts"], "last": ordered[-1]["ts"]},
        "split": {"train_count": len(train), "test_count": len(test), "train_fraction": 0.70},
        "current": current,
        "train_selected": selected,
        "test_avg_net_improvement_pp": test_improvement,
        "promotion_gate_passed": passes,
        "decision": "candidate_for_shadow_recalibration" if passes else "keep_current",
        "profiles": profiles,
    }


def _profile_stats(rows: Iterable[dict[str, Any]], exit_score_min: float, mfe_min: float) -> dict[str, Any]:
    selected = [
        row
        for row in rows
        if _float(row.get("exit_score")) >= exit_score_min and _float(row.get("mfe_pct")) >= mfe_min
    ]
    net = [_float(row.get("ret_5")) - ROUND_TRIP_COST_PCT for row in selected]
    drawdown = [_float(row.get("drawdown_10")) for row in selected]
    return {
        "count": len(selected),
        "avg_net_ret5_pct": _avg(net),
        "median_net_ret5_pct": _median(net),
        "positive_after_cost_rate": _ratio(sum(value > 0.0 for value in net), len(net)),
        "median_drawdown10_pct": _median(drawdown),
    }


def _load_watchlist() -> set[str]:
    try:
        return {str(value) for value in json.loads(WATCHLIST_FILE.read_text(encoding="utf-8")) if value}
    except Exception:
        return set()


def _ts_ms(value: str) -> int:
    return int(datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp() * 1000)


def _float(value: Any) -> float:
    try:
        result = float(value)
        return result if math.isfinite(result) else 0.0
    except Exception:
        return 0.0


def _avg(values: Iterable[float]) -> float | None:
    rows = list(values)
    return round(statistics.mean(rows), 6) if rows else None


def _median(values: Iterable[float]) -> float | None:
    rows = list(values)
    return round(statistics.median(rows), 6) if rows else None


def _ratio(num: int, den: int) -> float | None:
    return round(num / den, 6) if den else None


def _difference(left: Any, right: Any) -> float | None:
    if left is None or right is None:
        return None
    return round(float(left) - float(right), 6)


def _or_floor(value: Any) -> float:
    return float(value) if value is not None else -1e9


def main() -> int:
    parser = argparse.ArgumentParser(description="Replay suspicious re-entry watch thresholds.")
    parser.add_argument("--events", type=Path, default=EVENTS_FILE)
    parser.add_argument("--output", type=Path, default=OUTPUT_FILE)
    parser.add_argument("--concurrency", type=int, default=6)
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args()
    report = asyncio.run(
        run_replay(
            events_file=args.events,
            output_file=args.output,
            concurrency=args.concurrency,
        )
    )
    compact = {
        "status": report.get("status"),
        "decisions_loaded": report.get("decisions_loaded"),
        "labels_mature": report.get("labels_mature"),
        "current": report.get("current"),
        "train_selected": report.get("train_selected"),
        "test_avg_net_improvement_pp": report.get("test_avg_net_improvement_pp"),
        "promotion_gate_passed": report.get("promotion_gate_passed"),
        "decision": report.get("decision"),
    }
    print(json.dumps(report if args.as_json else compact, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
