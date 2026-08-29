from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import math
import random
import statistics
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parent.parent
FILES = ROOT / "files"
REPORTS = ROOT / ".runtime" / "reports"
DEFAULT_EVENTS = FILES / "bot_events.jsonl"
DEFAULT_RESEARCH = FILES / "research_universe_shadow.jsonl"
DEFAULT_OUTPUT = REPORTS / "early_signal_one_bar_persistence_replay.json"

LOCAL_TZ = ZoneInfo("Europe/Budapest")
BAR_MS = 15 * 60 * 1000
MAX_BASE_AGE_MS = 20 * 60 * 1000
JOIN_SECONDS = 120.0
ROUND_TRIP_COST_PCT = 0.20
ELIGIBLE_MODES = {"trend", "strong_trend", "retest"}
VARIANTS = (
    "persistence_structure",
    "persistence_rank",
    "persistence_quality",
)


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, 1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                value = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_number}: {exc}") from exc
            if isinstance(value, dict):
                yield value


def _parse_ts(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed


def _local_day(ts: datetime) -> str:
    return ts.astimezone(LOCAL_TZ).date().isoformat()


def _as_float(value: Any) -> float | None:
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return None
    return converted if math.isfinite(converted) else None


def _file_fingerprint(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "bytes": stat.st_size,
        "mtime_utc": datetime.fromtimestamp(stat.st_mtime, tz=ZoneInfo("UTC")).isoformat(),
        "sha256": digest.hexdigest(),
    }


def _critic_manifest_fingerprint(reports_dir: Path) -> dict[str, Any]:
    paths = sorted(reports_dir.glob("top_gainer_critic_*_final.json"))
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode("utf-8"))
        digest.update(bytes.fromhex(_file_fingerprint(path)["sha256"]))
    return {
        "path": str(reports_dir.resolve()),
        "files": len(paths),
        "manifest_sha256": digest.hexdigest(),
    }


def _load_final_top_labels(reports_dir: Path) -> tuple[set[str], set[tuple[str, str]]]:
    labeled_days: set[str] = set()
    positives: set[tuple[str, str]] = set()
    for path in sorted(reports_dir.glob("top_gainer_critic_*_final.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        day = str(payload.get("target_day_local") or "")
        rows = payload.get("exchange_top_gainers")
        if not day or not isinstance(rows, list):
            continue
        labeled_days.add(day)
        for row in rows:
            if not isinstance(row, dict) or row.get("in_watchlist") is not True:
                continue
            symbol = str(row.get("symbol") or row.get("sym") or "").upper()
            if symbol:
                positives.add((day, symbol))
    return labeled_days, positives


def _load_candidates_and_entries(
    events_path: Path,
    labeled_days: set[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    latest_labels: dict[tuple[str, str], tuple[datetime, dict[str, Any]]] = {}
    deliveries: list[dict[str, Any]] = []
    entries: dict[tuple[str, str], list[datetime]] = defaultdict(list)
    counts: dict[str, int] = defaultdict(int)

    for event in _iter_jsonl(events_path):
        ts = _parse_ts(event.get("ts"))
        if ts is None:
            continue
        symbol = str(event.get("sym") or event.get("symbol") or "").upper()
        timeframe = str(event.get("tf") or "")
        event_name = str(event.get("event") or "")

        if event_name == "entry" and symbol:
            entries[(_local_day(ts), symbol)].append(ts)
            continue

        if event_name == "blocked_learning_label":
            if symbol and timeframe and event.get("reason_code") == "top_gainer_score_gate":
                latest_labels[(symbol, timeframe)] = (ts, event)
            continue

        if event_name != "telegram_delivery" or event.get("delivery_stage") != "ok":
            continue
        if "entry blocked by score gate" not in str(event.get("text_preview") or "").lower():
            continue
        counts["delivered_score_gate"] += 1
        day = _local_day(ts)
        if day not in labeled_days:
            counts["without_final_critic"] += 1
            continue
        prior = latest_labels.get((symbol, timeframe))
        if prior is None:
            counts["join_missing"] += 1
            continue
        label_ts, label = prior
        join_age = (ts - label_ts).total_seconds()
        if not 0.0 <= join_age <= JOIN_SECONDS:
            counts["join_out_of_window"] += 1
            continue
        counts["causally_joined"] += 1
        live_score = _as_float(label.get("live_score"))
        mode = str(label.get("mode") or "")
        if (
            timeframe != "15m"
            or mode not in ELIGIBLE_MODES
            or live_score is None
            or not 32.0 <= live_score < 34.0
        ):
            counts["outside_population"] += 1
            continue
        counts["eligible_deliveries"] += 1
        deliveries.append(
            {
                "local_day": day,
                "symbol": symbol,
                "timeframe": timeframe,
                "mode": mode,
                "live_score": live_score,
                "alert_ts": ts.isoformat(),
                "alert_ts_ms": int(ts.timestamp() * 1000),
                "label_join_age_seconds": round(join_age, 3),
            }
        )

    first_by_day_symbol: dict[tuple[str, str], dict[str, Any]] = {}
    for candidate in sorted(deliveries, key=lambda row: row["alert_ts_ms"]):
        first_by_day_symbol.setdefault(
            (candidate["local_day"], candidate["symbol"]), candidate
        )

    candidates = list(first_by_day_symbol.values())
    for candidate in candidates:
        alert_ts = _parse_ts(candidate["alert_ts"])
        assert alert_ts is not None
        future_entries = [
            ts
            for ts in entries.get((candidate["local_day"], candidate["symbol"]), [])
            if ts > alert_ts
        ]
        candidate["later_buy_after_alert"] = bool(future_entries)
        candidate["first_later_buy_ts"] = (
            min(future_entries).isoformat() if future_entries else None
        )

    counts["deduplicated_candidates"] = len(candidates)
    counts["duplicate_deliveries_removed"] = len(deliveries) - len(candidates)
    return candidates, dict(counts)


def _align_research_pair(
    rows: list[dict[str, Any]],
    alert_ts_ms: int,
    max_age_ms: int = MAX_BASE_AGE_MS,
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    if not rows:
        return None
    ordered = sorted(rows, key=lambda row: int(row["bar_ts"]))
    timestamps = [int(row["bar_ts"]) for row in ordered]
    # bar_ts is the candle-open timestamp. The row is only causal after that
    # candle closes one BAR_MS later.
    latest_observable_open = int(alert_ts_ms) - BAR_MS
    base_index = bisect.bisect_right(timestamps, latest_observable_open) - 1
    if base_index < 0:
        return None
    base = ordered[base_index]
    base_ts = int(base["bar_ts"])
    base_available_ts = base_ts + BAR_MS
    if alert_ts_ms - base_available_ts > max_age_ms:
        return None
    confirm_ts = base_ts + BAR_MS
    confirm_index = bisect.bisect_left(timestamps, confirm_ts, base_index + 1)
    if confirm_index >= len(ordered) or timestamps[confirm_index] != confirm_ts:
        return None
    return base, ordered[confirm_index]


def _near_candidate_window(
    candidate_times: list[int],
    bar_ts: int,
) -> bool:
    index = bisect.bisect_left(candidate_times, bar_ts)
    for candidate_index in (index - 1, index):
        if 0 <= candidate_index < len(candidate_times):
            alert_ts = candidate_times[candidate_index]
            if alert_ts - BAR_MS - MAX_BASE_AGE_MS <= bar_ts <= alert_ts:
                return True
    return False


def _load_research_pairs(
    research_path: Path,
    candidates: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    candidate_times: dict[str, list[int]] = defaultdict(list)
    for candidate in candidates:
        candidate_times[candidate["symbol"]].append(int(candidate["alert_ts_ms"]))
    for values in candidate_times.values():
        values.sort()

    nearby_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    retained_source_rows = 0
    for row in _iter_jsonl(research_path):
        symbol = str(row.get("sym") or "").upper()
        if symbol not in candidate_times:
            continue
        if row.get("tf") != "15m":
            continue
        try:
            bar_ts = int(row["bar_ts"])
        except (KeyError, TypeError, ValueError):
            continue
        if _near_candidate_window(candidate_times[symbol], bar_ts):
            nearby_rows[symbol].append(row)
            retained_source_rows += 1

    aligned: list[dict[str, Any]] = []
    missing_reasons: dict[str, int] = defaultdict(int)
    for candidate in candidates:
        pair = _align_research_pair(
            nearby_rows.get(candidate["symbol"], []),
            int(candidate["alert_ts_ms"]),
        )
        if pair is None:
            missing_reasons["missing_exact_closed_bar_pair"] += 1
            continue
        base, confirm = pair
        if (
            base.get("in_trade_watchlist") is not True
            or confirm.get("in_trade_watchlist") is not True
        ):
            missing_reasons["pair_outside_trade_watchlist"] += 1
            continue
        enriched = dict(candidate)
        enriched["base"] = base
        enriched["confirm"] = confirm
        enriched["confirmation_bar_open_ts_ms"] = int(confirm["bar_ts"])
        enriched["confirmation_available_ts_ms"] = int(confirm["bar_ts"]) + BAR_MS
        buy_ts = _parse_ts(candidate.get("first_later_buy_ts"))
        enriched["later_buy_after_confirmation"] = bool(
            buy_ts is not None
            and int(buy_ts.timestamp() * 1000) > enriched["confirmation_available_ts_ms"]
        )
        aligned.append(enriched)
    return aligned, {
        "retained_research_rows": retained_source_rows,
        "aligned_pairs": len(aligned),
        "missing_pairs": len(candidates) - len(aligned),
        "missing_pair_reasons": dict(missing_reasons),
    }


def _feature(row: dict[str, Any], name: str) -> float | None:
    features = row.get("f")
    if not isinstance(features, dict):
        return None
    return _as_float(features.get(name))


def _variant_matches(
    name: str,
    base: dict[str, Any],
    confirm: dict[str, Any],
) -> bool:
    if name not in VARIANTS:
        raise ValueError(f"unknown frozen variant: {name}")
    base_slope = _feature(base, "slope")
    confirm_slope = _feature(confirm, "slope")
    base_macd = _feature(base, "macd_hist_norm")
    confirm_macd = _feature(confirm, "macd_hist_norm")
    if None in (base_slope, confirm_slope, base_macd, confirm_macd):
        return False
    structure = bool(
        confirm_slope > 0.0
        and confirm_slope - base_slope >= -0.05
        and confirm_macd - base_macd >= -0.02
    )
    if name == "persistence_structure":
        return structure

    base_rank = _as_float(base.get("rank_24h"))
    confirm_rank = _as_float(confirm.get("rank_24h"))
    rank = bool(
        structure
        and base_rank is not None
        and confirm_rank is not None
        and confirm_rank <= base_rank
    )
    if name == "persistence_rank":
        return rank

    volume = _feature(confirm, "vol_x")
    upper_wick = _feature(confirm, "upper_wick_pct")
    body = _feature(confirm, "body_pct")
    rsi = _feature(confirm, "rsi")
    return bool(
        rank
        and volume is not None
        and upper_wick is not None
        and body is not None
        and rsi is not None
        and volume >= 0.8
        and upper_wick <= body
        and rsi <= 75.0
    )


def _split_by_day_with_purge(
    rows: list[dict[str, Any]],
    all_days: list[str] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    days = sorted(set(all_days or [str(row["local_day"]) for row in rows]))
    if len(days) < 5:
        return [], [], [], {
            "status": "insufficient_days",
            "all_days": days,
            "purged_days": [],
        }
    train_cut = max(1, min(len(days) - 2, int(len(days) * 0.60)))
    validation_cut = max(train_cut + 1, min(len(days) - 1, int(len(days) * 0.80)))
    purged = {
        days[train_cut - 1],
        days[train_cut],
        days[validation_cut - 1],
        days[validation_cut],
    }
    train_days = set(days[:train_cut]) - purged
    validation_days = set(days[train_cut:validation_cut]) - purged
    holdout_days = set(days[validation_cut:]) - purged

    train = [row for row in rows if row["local_day"] in train_days]
    validation = [row for row in rows if row["local_day"] in validation_days]
    holdout = [row for row in rows if row["local_day"] in holdout_days]
    return train, validation, holdout, {
        "status": "ok",
        "all_days": days,
        "raw_boundaries": {
            "train_last": days[train_cut - 1],
            "validation_first": days[train_cut],
            "validation_last": days[validation_cut - 1],
            "holdout_first": days[validation_cut],
        },
        "purged_days": sorted(purged),
        "retained_days": {
            "train": sorted(train_days),
            "validation": sorted(validation_days),
            "holdout": sorted(holdout_days),
        },
    }


def _percentile(values: list[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _summary(
    rows: list[dict[str, Any]],
    predicate: Callable[[dict[str, Any]], bool] | None = None,
) -> dict[str, Any]:
    selected = [row for row in rows if predicate is None or predicate(row)]
    count = len(selected)
    final_top = sum(bool(row.get("final_top")) for row in selected)
    buys_after_alert = sum(bool(row.get("later_buy_after_alert")) for row in selected)
    buys_after_confirmation = sum(
        bool(row.get("later_buy_after_confirmation")) for row in selected
    )
    returns = [
        value
        for row in selected
        if (value := _as_float(row.get("ret10_net_pct"))) is not None
    ]
    active_days = len({row["local_day"] for row in selected})
    return {
        "opportunities": count,
        "active_days": active_days,
        "candidates_per_active_day": round(count / active_days, 4) if active_days else None,
        "later_buy_after_alert_count": buys_after_alert,
        "later_buy_after_alert_rate": round(buys_after_alert / count, 6) if count else None,
        "later_buy_after_confirmation_count": buys_after_confirmation,
        "later_buy_after_confirmation_rate": (
            round(buys_after_confirmation / count, 6) if count else None
        ),
        "final_top_count": final_top,
        "final_top_precision": round(final_top / count, 6) if count else None,
        "ret10_net": {
            "mature_count": len(returns),
            "average_pct": round(statistics.fmean(returns), 6) if returns else None,
            "median_pct": round(statistics.median(returns), 6) if returns else None,
            "positive_rate": (
                round(sum(value > 0.0 for value in returns) / len(returns), 6)
                if returns
                else None
            ),
        },
    }


def _metric_differences(
    baseline: dict[str, Any],
    variant: dict[str, Any],
) -> dict[str, float | None]:
    def difference(left: Any, right: Any, scale: float = 1.0) -> float | None:
        if left is None or right is None:
            return None
        return (float(left) - float(right)) * scale

    return {
        "final_top_precision_pp": difference(
            variant["final_top_precision"], baseline["final_top_precision"], 100.0
        ),
        "later_buy_after_alert_rate_pp": difference(
            variant["later_buy_after_alert_rate"],
            baseline["later_buy_after_alert_rate"],
            100.0,
        ),
        "ret10_net_median_pct": difference(
            variant["ret10_net"]["median_pct"],
            baseline["ret10_net"]["median_pct"],
        ),
    }


def _bootstrap_differences(
    rows: list[dict[str, Any]],
    predicate: Callable[[dict[str, Any]], bool],
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    by_day: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_day[row["local_day"]].append(row)
    days = sorted(by_day)
    samples: dict[str, list[float]] = defaultdict(list)
    if not days or iterations <= 0:
        return {"iterations_requested": iterations, "iterations_usable": 0, "intervals": {}}
    rng = random.Random(seed)
    usable = 0
    for _ in range(iterations):
        resampled: list[dict[str, Any]] = []
        for sampled_day in rng.choices(days, k=len(days)):
            resampled.extend(by_day[sampled_day])
        differences = _metric_differences(
            _summary(resampled), _summary(resampled, predicate)
        )
        if all(value is not None for value in differences.values()):
            usable += 1
            for name, value in differences.items():
                assert value is not None
                samples[name].append(float(value))
    return {
        "iterations_requested": iterations,
        "iterations_usable": usable,
        "intervals": {
            name: {
                "p2_5": round(_percentile(values, 0.025), 6),
                "p97_5": round(_percentile(values, 0.975), 6),
            }
            for name, values in samples.items()
            if values
        },
    }


def _evaluate_split(
    rows: list[dict[str, Any]],
    bootstrap_iterations: int,
    seed: int,
) -> dict[str, Any]:
    baseline = _summary(rows)
    variants: dict[str, Any] = {}
    for offset, name in enumerate(VARIANTS):
        predicate = lambda row, variant=name: _variant_matches(
            variant, row["base"], row["confirm"]
        )
        summary = _summary(rows, predicate)
        variants[name] = {
            **summary,
            "difference_vs_baseline": _metric_differences(baseline, summary),
            "day_cluster_bootstrap": _bootstrap_differences(
                rows,
                predicate,
                bootstrap_iterations,
                seed + offset,
            ),
        }
    return {"baseline": baseline, "variants": variants}


def _terminal_decision(holdout: dict[str, Any]) -> dict[str, Any]:
    baseline = holdout["baseline"]
    adequately_powered: list[tuple[str, dict[str, Any]]] = []
    passed: list[tuple[str, dict[str, Any]]] = []
    for name, result in holdout["variants"].items():
        if result["opportunities"] < 30 or result["final_top_count"] < 5:
            continue
        adequately_powered.append((name, result))
        differences = result["difference_vs_baseline"]
        if (
            differences["final_top_precision_pp"] is not None
            and differences["final_top_precision_pp"] >= 2.0
            and differences["later_buy_after_alert_rate_pp"] is not None
            and differences["later_buy_after_alert_rate_pp"] >= -5.0
            and result["ret10_net"]["median_pct"] is not None
            and baseline["ret10_net"]["median_pct"] is not None
            and result["ret10_net"]["median_pct"] >= 0.0
            and result["ret10_net"]["median_pct"]
            >= baseline["ret10_net"]["median_pct"]
        ):
            passed.append((name, result))
    if passed:
        winner = max(
            passed,
            key=lambda item: (
                item[1]["difference_vs_baseline"]["final_top_precision_pp"],
                item[1]["final_top_count"],
            ),
        )
        return {
            "status": "advance_to_independent_shadow",
            "variant": winner[0],
            "reason": "all pre-registered holdout gates passed",
        }
    if baseline["opportunities"] < 30 or baseline["final_top_count"] < 5:
        return {
            "status": "inconclusive_underpowered",
            "variant": None,
            "reason": "holdout baseline has fewer than 30 candidates or five final-top outcomes",
        }
    if not adequately_powered:
        return {
            "status": "inconclusive_underpowered",
            "variant": None,
            "reason": "no frozen variant retains 30 candidates and five final-top outcomes",
        }
    return {
        "status": "rejected",
        "variant": None,
        "reason": "adequately observed frozen variants failed one or more holdout gates",
    }


def run_replay(
    events_path: Path,
    research_path: Path,
    reports_dir: Path,
    bootstrap_iterations: int = 1000,
) -> dict[str, Any]:
    labeled_days, positives = _load_final_top_labels(reports_dir)
    candidates, event_coverage = _load_candidates_and_entries(events_path, labeled_days)
    aligned, research_coverage = _load_research_pairs(research_path, candidates)
    for row in aligned:
        row["final_top"] = (row["local_day"], row["symbol"]) in positives
        labels = row["confirm"].get("labels")
        ret10 = _as_float(labels.get("ret_10")) if isinstance(labels, dict) else None
        row["ret10_net_pct"] = (
            round(ret10 - ROUND_TRIP_COST_PCT, 6) if ret10 is not None else None
        )

    candidate_days = sorted({row["local_day"] for row in candidates})
    train, validation, holdout, split_meta = _split_by_day_with_purge(
        aligned, all_days=candidate_days
    )
    evaluation = {
        "full": _evaluate_split(aligned, bootstrap_iterations, 20260829),
        "train": _evaluate_split(train, bootstrap_iterations, 20260830),
        "validation": _evaluate_split(validation, bootstrap_iterations, 20260831),
        "holdout": _evaluate_split(holdout, bootstrap_iterations, 20260901),
    }
    decision = (
        _terminal_decision(evaluation["holdout"])
        if split_meta.get("status") == "ok"
        else {
            "status": "inconclusive_underpowered",
            "variant": None,
            "reason": "fewer than five causally aligned local days",
        }
    )
    source_days = sorted({row["local_day"] for row in aligned})
    candidate_count = int(event_coverage.get("deduplicated_candidates", 0))
    aligned_count = int(research_coverage.get("aligned_pairs", 0))
    research_coverage["aligned_pair_rate"] = (
        round(aligned_count / candidate_count, 6) if candidate_count else None
    )
    research_coverage["coverage_status"] = (
        "complete" if aligned_count == candidate_count else "partial"
    )
    return {
        "schema_version": 1,
        "study": "early_signal_one_bar_persistence",
        "generated_at_utc": datetime.now(tz=ZoneInfo("UTC")).isoformat(),
        "policy_impact": "none_research_only",
        "decision": decision,
        "population": {
            "timeframe": "15m",
            "modes": sorted(ELIGIBLE_MODES),
            "live_score_interval": "[32,34)",
            "dedupe": "first_successful_delivery_per_local_day_symbol",
            "final_top_source": "exchange_top_gainers where in_watchlist=true",
        },
        "provenance": {
            "events": _file_fingerprint(events_path),
            "research": _file_fingerprint(research_path),
            "final_critics": _critic_manifest_fingerprint(reports_dir),
            "first_aligned_day": source_days[0] if source_days else None,
            "last_aligned_day": source_days[-1] if source_days else None,
            "aligned_days": len(source_days),
            "round_trip_cost_pct": ROUND_TRIP_COST_PCT,
            "feature_time_definition": (
                "bar_ts is candle open; row becomes observable at bar_ts + 15m"
            ),
            "forward_label_definition": (
                "confirmation-row close-to-T+10-close percent return minus 20 bps"
            ),
            "objective_label_definition": (
                "immutable final exchange_top_gainers membership with in_watchlist=true"
            ),
            "policy_epoch_status": (
                "unavailable on legacy early-alert/research rows; mixed historical policy possible"
            ),
        },
        "coverage": {
            "final_critic_days": len(labeled_days),
            "final_top_labels": len(positives),
            "events": event_coverage,
            "research": research_coverage,
        },
        "split": split_meta,
        "evaluation": evaluation,
        "truth_harness_scope": {
            "production_promotion_authorized": False,
            "note": "A replay pass authorizes independent shadow only; current full-harness failures remain blocking.",
        },
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Causal maximum-period replay of frozen one-bar early-signal filters."
    )
    parser.add_argument("--events", type=Path, default=DEFAULT_EVENTS)
    parser.add_argument("--research", type=Path, default=DEFAULT_RESEARCH)
    parser.add_argument("--reports-dir", type=Path, default=REPORTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--json", action="store_true", dest="print_json")
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    result = run_replay(
        args.events,
        args.research,
        args.reports_dir,
        max(0, args.bootstrap),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if args.print_json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(
            json.dumps(
                {
                    "decision": result["decision"],
                    "coverage": result["coverage"],
                    "holdout": result["evaluation"]["holdout"],
                    "output": str(args.output),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
