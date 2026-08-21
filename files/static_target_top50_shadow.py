from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence
from zoneinfo import ZoneInfo

import aiohttp

import config
from validate_external_top50_screen import (
    DAY_MS,
    HOUR_MS,
    Bar,
    _bootstrap_delta,
    _fresh_bar,
    _safe_return,
    merge_binance_rows,
    wilson_interval,
)


ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = ROOT.parent
REPORT_DIR = WORKSPACE_ROOT / ".runtime" / "reports"
WATCHLIST_FILE = ROOT / "watchlist.json"
BINANCE_API = str(getattr(config, "BINANCE_REST", "https://api.binance.com")).rstrip("/")
EXCHANGE_INFO_URL = f"{BINANCE_API}/api/v3/exchangeInfo"
KLINES_URL = f"{BINANCE_API}/api/v3/klines"
SCHEMA_VERSION = 1
CONTRACT_DESCRIPTOR = {
    "name": "static_target_top50_forward_shadow",
    "version": 1,
    "timezone": "Europe/Budapest",
    "observation_time_local": "12:15",
    "observation_grace_minutes": 30,
    "target_time_local": "23:00",
    "label_delay_minutes": 5,
    "source_timeframe": "1h",
    "top_n": 50,
    "selection_size": 10,
    "min_market_symbols": 200,
    "min_watchlist_symbols": 50,
    "min_forward_days": 30,
    "min_observation_coverage": 0.90,
    "min_final_coverage": 0.85,
    "formula": "observation_close / target_minus_24h_close - 1",
    "control": "observation_close / observation_minus_24h_close - 1",
}
CONTRACT_HASH = hashlib.sha256(
    json.dumps(CONTRACT_DESCRIPTOR, sort_keys=True, separators=(",", ":")).encode("utf-8")
).hexdigest()


class ObservationWindowError(RuntimeError):
    pass


class LabelNotMatureError(RuntimeError):
    pass


class EvidenceContractError(RuntimeError):
    pass


def validate_runtime_contract() -> None:
    """Reject config drift until the frozen contract is re-specified/replayed."""
    actual = {
        "timezone": str(getattr(config, "STATIC_TARGET_TOP50_SHADOW_TIMEZONE", "Europe/Budapest")),
        "observation_time_local": (
            f"{int(getattr(config, 'STATIC_TARGET_TOP50_SHADOW_OBSERVATION_HOUR_LOCAL', 12)):02d}:"
            f"{int(getattr(config, 'STATIC_TARGET_TOP50_SHADOW_OBSERVATION_MINUTE_LOCAL', 15)):02d}"
        ),
        "observation_grace_minutes": int(
            getattr(config, "STATIC_TARGET_TOP50_SHADOW_OBSERVATION_GRACE_MINUTES", 30)
        ),
        "target_time_local": (
            f"{int(getattr(config, 'STATIC_TARGET_TOP50_SHADOW_TARGET_HOUR_LOCAL', 23)):02d}:00"
        ),
        "label_delay_minutes": int(
            getattr(config, "STATIC_TARGET_TOP50_SHADOW_LABEL_DELAY_MINUTES", 5)
        ),
        "top_n": int(getattr(config, "STATIC_TARGET_TOP50_SHADOW_TOP_N", 50)),
        "selection_size": int(
            getattr(config, "STATIC_TARGET_TOP50_SHADOW_SELECTION_SIZE", 10)
        ),
        "min_market_symbols": int(
            getattr(config, "STATIC_TARGET_TOP50_SHADOW_MIN_MARKET_SYMBOLS", 200)
        ),
        "min_watchlist_symbols": int(
            getattr(config, "STATIC_TARGET_TOP50_SHADOW_MIN_WATCHLIST_SYMBOLS", 50)
        ),
        "min_forward_days": int(
            getattr(config, "STATIC_TARGET_TOP50_SHADOW_MIN_FORWARD_DAYS", 30)
        ),
        "min_observation_coverage": float(
            getattr(config, "STATIC_TARGET_TOP50_SHADOW_MIN_OBSERVATION_COVERAGE", 0.90)
        ),
        "min_final_coverage": float(
            getattr(config, "STATIC_TARGET_TOP50_SHADOW_MIN_FINAL_COVERAGE", 0.85)
        ),
    }
    mismatches = {
        key: {"expected": CONTRACT_DESCRIPTOR[key], "actual": value}
        for key, value in actual.items()
        if value != CONTRACT_DESCRIPTOR[key]
    }
    if mismatches:
        raise EvidenceContractError(f"frozen runtime contract drift: {mismatches}")


def _iso_utc(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat()


def _clock(local_day: date, value: time, timezone_name: str) -> datetime:
    return datetime.combine(local_day, value, tzinfo=ZoneInfo(timezone_name))


def _paths(reports_dir: Path, local_day: date) -> dict[str, Path]:
    prefix = reports_dir / f"static_target_top50_shadow_{local_day.isoformat()}"
    return {
        "observation": prefix.with_name(prefix.name + "_observation.json"),
        "missed": prefix.with_name(prefix.name + "_missed.json"),
        "final": prefix.with_name(prefix.name + "_final.json"),
    }


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON object required: {path}")
    return payload


def write_json_once(path: Path, payload: Mapping[str, Any]) -> None:
    """Persist immutable evidence without replacing an earlier attempt."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as target:
        json.dump(payload, target, ensure_ascii=False, indent=2)
        target.write("\n")
        target.flush()
        os.fsync(target.fileno())


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _extract_observation_row(
    symbol: str,
    history: Sequence[Bar],
    *,
    observation_ms: int,
    target_ms: int,
) -> dict[str, Any] | None:
    close_times = tuple(bar.close_ts_ms for bar in history)
    observation = _fresh_bar(history, observation_ms, close_times=close_times)
    observation_base = _fresh_bar(history, observation_ms - DAY_MS, close_times=close_times)
    target_base = _fresh_bar(history, target_ms - DAY_MS, close_times=close_times)
    if not all((observation, observation_base, target_base)):
        return None
    obs_bar = observation[1]
    current_return = _safe_return(obs_bar.close, observation_base[1].close)
    static_target_return = _safe_return(obs_bar.close, target_base[1].close)
    if not all(math.isfinite(value) for value in (current_return, static_target_return)):
        return None
    return {
        "symbol": symbol,
        "observation_price": obs_bar.close,
        "observation_bar_close_utc": datetime.fromtimestamp(
            obs_bar.close_ts_ms / 1000, tz=timezone.utc
        ).isoformat(),
        "observation_base_price": observation_base[1].close,
        "observation_base_bar_close_utc": datetime.fromtimestamp(
            observation_base[1].close_ts_ms / 1000, tz=timezone.utc
        ).isoformat(),
        "target_base_price": target_base[1].close,
        "target_base_bar_close_utc": datetime.fromtimestamp(
            target_base[1].close_ts_ms / 1000, tz=timezone.utc
        ).isoformat(),
        "current_return_pct": current_return,
        "static_target_return_pct": static_target_return,
    }


def _selection_row(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "symbol": row["symbol"],
        "current_rank": row["current_rank"],
        "current_return_pct": row["current_return_pct"],
        "static_target_return_pct": row["static_target_return_pct"],
    }


def build_observation_payload(
    histories: Mapping[str, Sequence[Bar]],
    *,
    active_symbols: set[str],
    watchlist: set[str],
    watchlist_sha256: str,
    exchange_info_sha256: str,
    local_day: date,
    observed_at: datetime,
    timezone_name: str = "Europe/Budapest",
    observation_time: time = time(12, 15),
    observation_grace_minutes: int = 30,
    target_time: time = time(23, 0),
    top_n: int = 50,
    selection_size: int = 10,
    min_market_symbols: int = 200,
    min_watchlist_symbols: int = 50,
    fetch_failures: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    if observed_at.tzinfo is None:
        raise ValueError("observed_at must be timezone-aware")
    observation_dt = _clock(local_day, observation_time, timezone_name)
    target_dt = _clock(local_day, target_time, timezone_name)
    grace_end = observation_dt + timedelta(minutes=max(0, observation_grace_minutes))
    observed_local = observed_at.astimezone(ZoneInfo(timezone_name))
    if observed_local < observation_dt or observed_local >= grace_end:
        raise ObservationWindowError(
            f"observation outside frozen window: {observed_local.isoformat()} not in "
            f"[{observation_dt.isoformat()}, {grace_end.isoformat()})"
        )
    observation_ms = int(observation_dt.astimezone(timezone.utc).timestamp() * 1000)
    target_ms = int(target_dt.astimezone(timezone.utc).timestamp() * 1000)
    rows = []
    for symbol in sorted(active_symbols):
        row = _extract_observation_row(
            symbol,
            histories.get(symbol, ()),
            observation_ms=observation_ms,
            target_ms=target_ms,
        )
        if row is not None:
            rows.append(row)
    current_order = sorted(rows, key=lambda item: (-float(item["current_return_pct"]), item["symbol"]))
    current_ranks = {row["symbol"]: index for index, row in enumerate(current_order, start=1)}
    for row in rows:
        row["current_rank"] = current_ranks[row["symbol"]]
    candidates = [row for row in rows if row["symbol"] in watchlist and row["current_rank"] > top_n]
    static_order = sorted(
        candidates, key=lambda item: (-float(item["static_target_return_pct"]), item["symbol"])
    )
    control_order = sorted(
        candidates, key=lambda item: (-float(item["current_return_pct"]), item["symbol"])
    )
    market_valid = len(rows)
    watchlist_valid = sum(row["symbol"] in watchlist for row in rows)
    eligible = (
        market_valid >= min_market_symbols
        and watchlist_valid >= min_watchlist_symbols
        and len(candidates) >= selection_size
    )
    feature_closes = [datetime.fromisoformat(row["observation_bar_close_utc"]) for row in rows]
    feature_cutoff = max(feature_closes) if feature_closes else observation_dt.astimezone(timezone.utc)
    status = "observation_complete" if eligible else "partial"
    return {
        "schema_version": SCHEMA_VERSION,
        "contract_hash": CONTRACT_HASH,
        "generated_at_utc": _iso_utc(observed_at),
        "local_day": local_day.isoformat(),
        "status": status,
        "eligible": eligible,
        "production_effect": "none_shadow_only",
        "contract": {
            **CONTRACT_DESCRIPTOR,
            "timezone": timezone_name,
            "observation_time_local": observation_time.strftime("%H:%M"),
            "observation_grace_minutes": observation_grace_minutes,
            "target_time_local": target_time.strftime("%H:%M"),
            "top_n": top_n,
            "selection_size": selection_size,
        },
        "timing": {
            "observed_at_utc": _iso_utc(observed_at),
            "observation_utc": _iso_utc(observation_dt),
            "feature_cutoff_utc": _iso_utc(feature_cutoff),
            "target_utc": _iso_utc(target_dt),
            "labels_present": False,
        },
        "watchlist": {
            "sha256": watchlist_sha256,
            "count": len(watchlist),
            "symbols": sorted(watchlist),
        },
        "provenance": {
            "exchange_info_sha256": exchange_info_sha256,
            "exchange_info_url": EXCHANGE_INFO_URL,
            "klines_url": KLINES_URL,
            "source_timeframe": "1h",
            "fetch_failure_count": len(fetch_failures),
            "fetch_failures": list(fetch_failures)[:30],
        },
        "coverage": {
            "market_requested": len(active_symbols),
            "market_valid": market_valid,
            "market_coverage": market_valid / len(active_symbols) if active_symbols else None,
            "watchlist_requested": len(watchlist & active_symbols),
            "watchlist_valid": watchlist_valid,
            "minimum_market_symbols": min_market_symbols,
            "minimum_watchlist_symbols": min_watchlist_symbols,
            "minimum_candidate_symbols": selection_size,
        },
        "candidate_population": {
            "count": len(candidates),
            "current_top_n_excluded": top_n,
        },
        "market_reference": sorted(rows, key=lambda item: item["symbol"]),
        "selections": {
            "static_target": [_selection_row(row) for row in static_order[:selection_size]],
            "current_rank": [_selection_row(row) for row in control_order[:selection_size]],
        },
    }


def build_final_payload(
    observation: Mapping[str, Any],
    target_histories: Mapping[str, Sequence[Bar]],
    *,
    labeled_at: datetime,
    min_market_symbols: int = 200,
    label_delay_minutes: int = 5,
    fetch_failures: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    if labeled_at.tzinfo is None:
        raise ValueError("labeled_at must be timezone-aware")
    if int(observation.get("schema_version") or 0) != SCHEMA_VERSION:
        raise EvidenceContractError("unsupported observation schema")
    if str(observation.get("contract_hash") or "") != CONTRACT_HASH:
        raise EvidenceContractError("observation contract hash mismatch")
    timing = observation.get("timing") or {}
    observation_dt = datetime.fromisoformat(str(timing.get("observation_utc")))
    feature_cutoff = datetime.fromisoformat(str(timing.get("feature_cutoff_utc")))
    target_dt = datetime.fromisoformat(str(timing.get("target_utc")))
    mature_at = target_dt + timedelta(minutes=max(0, label_delay_minutes))
    if labeled_at.astimezone(timezone.utc) < mature_at.astimezone(timezone.utc):
        raise LabelNotMatureError(f"labels mature at {mature_at.isoformat()}")
    if feature_cutoff > observation_dt:
        raise EvidenceContractError("feature cutoff is after observation")
    target_ms = int(target_dt.astimezone(timezone.utc).timestamp() * 1000)
    labeled_rows = []
    for reference in observation.get("market_reference") or []:
        symbol = str(reference.get("symbol") or "")
        history = target_histories.get(symbol, ())
        close_times = tuple(bar.close_ts_ms for bar in history)
        target = _fresh_bar(history, target_ms, close_times=close_times)
        target_base = float(reference.get("target_base_price") or 0.0)
        if target is None or target_base <= 0:
            continue
        target_return = _safe_return(target[1].close, target_base)
        if not math.isfinite(target_return):
            continue
        labeled_rows.append({
            "symbol": symbol,
            "target_price": target[1].close,
            "target_bar_close_utc": datetime.fromtimestamp(
                target[1].close_ts_ms / 1000, tz=timezone.utc
            ).isoformat(),
            "target_return_pct": target_return,
            "current_rank": int(reference.get("current_rank") or 0),
        })
    ordered = sorted(labeled_rows, key=lambda item: (-float(item["target_return_pct"]), item["symbol"]))
    target_ranks = {row["symbol"]: index for index, row in enumerate(ordered, start=1)}
    top_n = int((observation.get("contract") or {}).get("top_n") or 50)
    target_top = [row["symbol"] for row in ordered[:top_n]]
    target_top_set = set(target_top)
    watchlist = set((observation.get("watchlist") or {}).get("symbols") or [])
    current_ranks = {
        str(row.get("symbol")): int(row.get("current_rank") or 0)
        for row in observation.get("market_reference") or []
    }
    entrants = sorted(
        symbol for symbol in target_top
        if symbol in watchlist and current_ranks.get(symbol, 0) > top_n
    )
    candidate_symbols = {
        symbol for symbol in watchlist if current_ranks.get(symbol, 0) > top_n
    }
    labeled_candidate_symbols = candidate_symbols & set(target_ranks)
    candidate_count = len(labeled_candidate_symbols)
    metrics: dict[str, Any] = {}
    labeled_selections: dict[str, list[dict[str, Any]]] = {}
    selected_labels_complete = True
    for variant in ("static_target", "current_rank"):
        selections = list((observation.get("selections") or {}).get(variant) or [])
        rows = []
        for selection in selections:
            symbol = str(selection.get("symbol") or "")
            if symbol not in target_ranks:
                selected_labels_complete = False
            rows.append({
                "symbol": symbol,
                "target_rank": target_ranks.get(symbol),
                "is_target_top": symbol in target_top_set,
            })
        hits = sum(bool(row["is_target_top"]) for row in rows)
        metrics[variant] = {
            "top1": {"hits": int(bool(rows and rows[0]["is_target_top"])), "days": int(bool(rows))},
            "topk": {"hits": hits, "selections": len(rows)},
            "entrant_recall": {"hits": hits, "entrants": len(entrants)},
        }
        labeled_selections[variant] = rows
    market_valid = len(labeled_rows)
    eligible = (
        bool(observation.get("eligible"))
        and market_valid >= min_market_symbols
        and selected_labels_complete
    )
    result = {
        "schema_version": SCHEMA_VERSION,
        "contract_hash": CONTRACT_HASH,
        "generated_at_utc": _iso_utc(labeled_at),
        "local_day": str(observation.get("local_day") or ""),
        "status": "complete" if eligible else "partial",
        "eligible": eligible,
        "production_effect": "none_shadow_only",
        "timing": {
            "observation_utc": observation_dt.isoformat(),
            "feature_cutoff_utc": feature_cutoff.isoformat(),
            "target_utc": target_dt.isoformat(),
            "labeled_at_utc": _iso_utc(labeled_at),
            "labels_mature_at_utc": mature_at.isoformat(),
        },
        "coverage": {
            "observation_market_valid": int((observation.get("coverage") or {}).get("market_valid") or 0),
            "target_market_valid": market_valid,
            "minimum_market_symbols": min_market_symbols,
            "fetch_failure_count": len(fetch_failures),
            "fetch_failures": list(fetch_failures)[:30],
            "selected_labels_complete": selected_labels_complete,
            "candidate_labels_requested": len(candidate_symbols),
            "candidate_labels_valid": candidate_count,
        },
        "candidate_population": {
            "hits": len(entrants),
            "candidates": candidate_count,
        },
        "target": {
            "top_n": top_n,
            "top_symbols": target_top,
            "entrant_symbols": entrants,
        },
        "labeled_selections": labeled_selections,
    }
    if eligible:
        result["metrics"] = metrics
    else:
        result["performance_interpretation"] = "unknown_not_miss_or_success"
    return result


def scheduled_action(
    now_local: datetime,
    *,
    reports_dir: Path = REPORT_DIR,
    observation_time: time = time(12, 15),
    observation_grace_minutes: int = 30,
    target_time: time = time(23, 0),
    label_delay_minutes: int = 5,
    start_day: date | None = None,
) -> tuple[str, date] | None:
    if now_local.tzinfo is None:
        raise ValueError("now_local must be timezone-aware")
    if start_day is not None and now_local.date() < start_day:
        return None
    # The current observation is irrecoverable after its grace window and must
    # preempt old, recoverable label catch-up work.
    local_day = now_local.date()
    paths = _paths(reports_dir, local_day)
    starts_at = _clock(local_day, observation_time, str(now_local.tzinfo))
    ends_at = starts_at + timedelta(minutes=max(0, observation_grace_minutes))
    if (
        not paths["observation"].exists()
        and not paths["missed"].exists()
        and starts_at <= now_local < ends_at
    ):
        return "observe", local_day

    # Labels may catch up after restarts because the immutable observation
    # already fixes the prediction.
    candidates = []
    for path in reports_dir.glob("static_target_top50_shadow_*_observation.json"):
        day_text = path.name.removeprefix("static_target_top50_shadow_").removesuffix("_observation.json")
        try:
            candidate_day = date.fromisoformat(day_text)
        except ValueError:
            continue
        final_path = _paths(reports_dir, candidate_day)["final"]
        mature_at = _clock(candidate_day, target_time, str(now_local.tzinfo)) + timedelta(minutes=label_delay_minutes)
        if not final_path.exists() and now_local >= mature_at:
            candidates.append(candidate_day)
    if candidates:
        return "finalize", min(candidates)

    if paths["observation"].exists() or paths["missed"].exists():
        return None
    if now_local >= ends_at:
        return "record_missed", local_day
    return None


def _metric_summary(hits: int, total: int, rate_key: str) -> dict[str, Any]:
    return {
        "hits": hits,
        "denominator": total,
        rate_key: hits / total if total else None,
        "wilson95": wilson_interval(hits, total),
    }


def build_scorecard(
    reports_dir: Path = REPORT_DIR,
    *,
    as_of_day: date | None = None,
    min_eligible_days: int = 30,
    min_observation_coverage: float = 0.90,
    min_final_coverage: float = 0.85,
    bootstrap_samples: int = 5000,
) -> dict[str, Any]:
    as_of_day = as_of_day or datetime.now(ZoneInfo("Europe/Budapest")).date()
    observations = sorted(reports_dir.glob("static_target_top50_shadow_*_observation.json"))
    missed = sorted(reports_dir.glob("static_target_top50_shadow_*_missed.json"))
    contract_violations = []
    observation_by_day: dict[str, dict[str, Any]] = {}
    for path in observations:
        try:
            payload = _read_json(path)
        except Exception as exc:
            contract_violations.append(f"{path.name}: {type(exc).__name__}")
            continue
        local_day = str(payload.get("local_day") or "")
        if (
            int(payload.get("schema_version") or 0) != SCHEMA_VERSION
            or payload.get("contract_hash") != CONTRACT_HASH
            or not local_day
        ):
            contract_violations.append(f"{path.name}: observation_contract")
            continue
        observation_by_day[local_day] = payload
    artifact_days = []
    for path in observations + missed:
        day_text = path.name.removeprefix("static_target_top50_shadow_").rsplit("_", 1)[0]
        try:
            artifact_days.append(date.fromisoformat(day_text))
        except ValueError:
            continue
    first_day = min(artifact_days) if artifact_days else None
    scheduled_slots = (as_of_day - first_day).days + 1 if first_day and as_of_day >= first_day else 0
    observed_slots = len(observations)
    final_files = sorted(reports_dir.glob("static_target_top50_shadow_*_final.json"))
    eligible_reports = []
    for path in final_files:
        try:
            payload = _read_json(path)
        except Exception as exc:
            contract_violations.append(f"{path.name}: {type(exc).__name__}")
            continue
        if (
            int(payload.get("schema_version") or 0) != SCHEMA_VERSION
            or payload.get("contract_hash") != CONTRACT_HASH
        ):
            contract_violations.append(f"{path.name}: final_contract")
            continue
        if payload.get("status") == "complete" and bool(payload.get("eligible")):
            local_day = str(payload.get("local_day") or "")
            observation = observation_by_day.get(local_day)
            if not observation or observation.get("status") != "observation_complete" or not observation.get("eligible"):
                contract_violations.append(f"{path.name}: missing_eligible_observation")
                continue
            try:
                timing = payload.get("timing") or {}
                observation_dt = datetime.fromisoformat(str(timing["observation_utc"]))
                feature_cutoff = datetime.fromisoformat(str(timing["feature_cutoff_utc"]))
                target_dt = datetime.fromisoformat(str(timing["target_utc"]))
                labeled_at = datetime.fromisoformat(str(timing["labeled_at_utc"]))
                mature_at = datetime.fromisoformat(str(timing["labels_mature_at_utc"]))
            except (KeyError, TypeError, ValueError):
                contract_violations.append(f"{path.name}: invalid_timing")
                continue
            if feature_cutoff > observation_dt or mature_at < target_dt or labeled_at < mature_at:
                contract_violations.append(f"{path.name}: timing_violation")
                continue
            eligible_reports.append(payload)
    variant_metrics: dict[str, Any] = {}
    daily_precision: dict[str, list[float]] = {"static_target": [], "current_rank": []}
    for variant in daily_precision:
        top1_hits = top1_days = topk_hits = selections = recall_hits = entrants = 0
        for report in eligible_reports:
            metrics = (report.get("metrics") or {}).get(variant) or {}
            top1 = metrics.get("top1") or {}
            topk = metrics.get("topk") or {}
            recall = metrics.get("entrant_recall") or {}
            top1_hits += int(top1.get("hits") or 0)
            top1_days += int(top1.get("days") or 0)
            day_hits = int(topk.get("hits") or 0)
            day_selections = int(topk.get("selections") or 0)
            topk_hits += day_hits
            selections += day_selections
            recall_hits += int(recall.get("hits") or 0)
            entrants += int(recall.get("entrants") or 0)
            if day_selections:
                daily_precision[variant].append(day_hits / day_selections)
        variant_metrics[variant] = {
            "top1": _metric_summary(top1_hits, top1_days, "rate"),
            "topk": _metric_summary(topk_hits, selections, "precision"),
            "entrant_recall": {
                "hits": recall_hits,
                "entrants": entrants,
                "recall": recall_hits / entrants if entrants else None,
            },
        }
    candidate_hits = sum(int((item.get("candidate_population") or {}).get("hits") or 0) for item in eligible_reports)
    candidate_count = sum(int((item.get("candidate_population") or {}).get("candidates") or 0) for item in eligible_reports)
    base_rate = candidate_hits / candidate_count if candidate_count else None
    for variant, metrics in variant_metrics.items():
        precision = metrics["topk"]["precision"]
        metrics["candidate_base_rate"] = {
            "hits": candidate_hits,
            "candidates": candidate_count,
            "rate": base_rate,
        }
        metrics["precision_lift_over_base"] = (
            precision / base_rate if precision is not None and base_rate not in (None, 0.0) else None
        )
    paired = _bootstrap_delta(
        daily_precision["static_target"],
        daily_precision["current_rank"],
        samples=bootstrap_samples,
        seed=3701,
    )
    observation_coverage = observed_slots / scheduled_slots if scheduled_slots else None
    final_coverage = len(eligible_reports) / scheduled_slots if scheduled_slots else None
    reasons = []
    if contract_violations:
        verdict = "FAIL"
        reasons.append("contract_or_artifact_violation")
    elif len(eligible_reports) < min_eligible_days:
        verdict = "COLLECTING"
        reasons.append("minimum_forward_days_not_reached")
    elif observation_coverage is None or observation_coverage < min_observation_coverage:
        verdict = "INCONCLUSIVE"
        reasons.append("observation_coverage_below_gate")
    elif final_coverage is None or final_coverage < min_final_coverage:
        verdict = "INCONCLUSIVE"
        reasons.append("eligible_final_coverage_below_gate")
    elif paired is None or paired["bootstrap95"][0] <= 0:
        verdict = "INCONCLUSIVE"
        reasons.append("paired_precision_interval_not_positive")
    elif base_rate is None or (variant_metrics["static_target"]["topk"]["precision"] or 0.0) <= base_rate:
        verdict = "INCONCLUSIVE"
        reasons.append("precision_not_above_candidate_base_rate")
    else:
        verdict = "ELIGIBLE_FOR_SEPARATE_PRODUCTION_REVIEW"
        reasons.append("forward_shadow_gate_passed")
    return {
        "schema_version": SCHEMA_VERSION,
        "contract_hash": CONTRACT_HASH,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete" if scheduled_slots else "no_evidence",
        "production_effect": "none_shadow_only",
        "coverage": {
            "first_scheduled_day": first_day.isoformat() if first_day else None,
            "as_of_day": as_of_day.isoformat(),
            "scheduled_slots": scheduled_slots,
            "observed_slots": observed_slots,
            "missed_artifacts": len(missed),
            "final_artifacts": len(final_files),
            "eligible_finals": len(eligible_reports),
            "observation_coverage": observation_coverage,
            "eligible_final_coverage": final_coverage,
        },
        "metrics": variant_metrics,
        "paired_daily_precision_delta_static_vs_current": paired,
        "decision": {
            "verdict": verdict,
            "reasons": reasons,
            "minimum_eligible_days": min_eligible_days,
            "automatic_production_promotion": False,
        },
        "contract_violations": contract_violations,
    }


async def _fetch_json(
    session: aiohttp.ClientSession,
    url: str,
    params: Mapping[str, Any] | None = None,
) -> Any:
    delay = 1.0
    for _ in range(6):
        async with session.get(url, params=params) as response:
            if response.status in (418, 429):
                await asyncio.sleep(float(response.headers.get("Retry-After", delay)))
                delay *= 2.0
                continue
            response.raise_for_status()
            return await response.json()
    raise RuntimeError(f"rate limited after retries: {url}")


async def _fetch_one_history(
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    symbol: str,
    start_ms: int,
    end_ms: int,
) -> tuple[str, tuple[Bar, ...], str | None]:
    try:
        async with semaphore:
            payload = await _fetch_json(session, KLINES_URL, {
                "symbol": symbol,
                "interval": "1h",
                "startTime": start_ms,
                "endTime": end_ms,
                "limit": 64,
            })
        if not isinstance(payload, list):
            raise ValueError("klines response is not a list")
        return symbol, merge_binance_rows((payload,)), None
    except Exception as exc:
        return symbol, (), f"{type(exc).__name__}: {exc}"


async def _fetch_histories(
    session: aiohttp.ClientSession,
    symbols: Sequence[str],
    *,
    start_ms: int,
    end_ms: int,
    concurrency: int,
) -> tuple[dict[str, tuple[Bar, ...]], list[dict[str, str]]]:
    semaphore = asyncio.Semaphore(max(1, concurrency))
    results = await asyncio.gather(*[
        _fetch_one_history(session, semaphore, symbol, start_ms, end_ms)
        for symbol in symbols
    ])
    histories = {}
    failures = []
    for symbol, history, error in results:
        if error:
            failures.append({"symbol": symbol, "error": error})
        else:
            histories[symbol] = history
    return histories, failures


def _active_spot_usdt(exchange_info: Mapping[str, Any]) -> set[str]:
    return {
        str(item.get("symbol"))
        for item in exchange_info.get("symbols", [])
        if item.get("status") == "TRADING"
        and item.get("quoteAsset") == "USDT"
        and bool(item.get("isSpotTradingAllowed"))
    }


async def collect_observation(
    local_day: date,
    *,
    now_local: datetime,
    reports_dir: Path = REPORT_DIR,
) -> dict[str, Any]:
    timezone_name = str(getattr(config, "STATIC_TARGET_TOP50_SHADOW_TIMEZONE", "Europe/Budapest"))
    observation_time = time(
        int(getattr(config, "STATIC_TARGET_TOP50_SHADOW_OBSERVATION_HOUR_LOCAL", 12)),
        int(getattr(config, "STATIC_TARGET_TOP50_SHADOW_OBSERVATION_MINUTE_LOCAL", 15)),
    )
    target_time = time(int(getattr(config, "STATIC_TARGET_TOP50_SHADOW_TARGET_HOUR_LOCAL", 23)), 0)
    observation_dt = _clock(local_day, observation_time, timezone_name)
    target_dt = _clock(local_day, target_time, timezone_name)
    observation_ms = int(observation_dt.astimezone(timezone.utc).timestamp() * 1000)
    target_ms = int(target_dt.astimezone(timezone.utc).timestamp() * 1000)
    timeout = aiohttp.ClientTimeout(total=75)
    concurrency = int(getattr(config, "STATIC_TARGET_TOP50_SHADOW_CONCURRENCY", 8))
    async with aiohttp.ClientSession(timeout=timeout, connector=aiohttp.TCPConnector(limit=max(2, concurrency))) as session:
        exchange_info = await _fetch_json(session, EXCHANGE_INFO_URL)
        if not isinstance(exchange_info, dict):
            raise ValueError("exchangeInfo response is not an object")
        active_symbols = _active_spot_usdt(exchange_info)
        start_ms = min(observation_ms - DAY_MS, target_ms - DAY_MS) - 2 * HOUR_MS
        histories, failures = await _fetch_histories(
            session,
            sorted(active_symbols),
            start_ms=start_ms,
            end_ms=observation_ms,
            concurrency=concurrency,
        )
    watchlist_bytes = WATCHLIST_FILE.read_bytes()
    watchlist = {
        str(item).strip().upper()
        for item in json.loads(watchlist_bytes.decode("utf-8-sig"))
        if str(item).strip()
    }
    exchange_bytes = json.dumps(exchange_info, sort_keys=True, separators=(",", ":")).encode("utf-8")
    payload = build_observation_payload(
        histories,
        active_symbols=active_symbols,
        watchlist=watchlist,
        watchlist_sha256=hashlib.sha256(watchlist_bytes).hexdigest(),
        exchange_info_sha256=hashlib.sha256(exchange_bytes).hexdigest(),
        local_day=local_day,
        observed_at=now_local,
        timezone_name=timezone_name,
        observation_time=observation_time,
        observation_grace_minutes=int(getattr(config, "STATIC_TARGET_TOP50_SHADOW_OBSERVATION_GRACE_MINUTES", 30)),
        target_time=target_time,
        top_n=int(getattr(config, "STATIC_TARGET_TOP50_SHADOW_TOP_N", 50)),
        selection_size=int(getattr(config, "STATIC_TARGET_TOP50_SHADOW_SELECTION_SIZE", 10)),
        min_market_symbols=int(getattr(config, "STATIC_TARGET_TOP50_SHADOW_MIN_MARKET_SYMBOLS", 200)),
        min_watchlist_symbols=int(getattr(config, "STATIC_TARGET_TOP50_SHADOW_MIN_WATCHLIST_SYMBOLS", 50)),
        fetch_failures=failures,
    )
    path = _paths(reports_dir, local_day)["observation"]
    write_json_once(path, payload)
    return {**payload, "files": {"observation": str(path)}}


async def finalize_observation(
    local_day: date,
    *,
    now_local: datetime,
    reports_dir: Path = REPORT_DIR,
) -> dict[str, Any]:
    paths = _paths(reports_dir, local_day)
    observation = _read_json(paths["observation"])
    symbols = [str(row.get("symbol")) for row in observation.get("market_reference") or []]
    target_dt = datetime.fromisoformat(str((observation.get("timing") or {}).get("target_utc")))
    target_ms = int(target_dt.astimezone(timezone.utc).timestamp() * 1000)
    concurrency = int(getattr(config, "STATIC_TARGET_TOP50_SHADOW_CONCURRENCY", 8))
    timeout = aiohttp.ClientTimeout(total=75)
    async with aiohttp.ClientSession(timeout=timeout, connector=aiohttp.TCPConnector(limit=max(2, concurrency))) as session:
        histories, failures = await _fetch_histories(
            session,
            symbols,
            start_ms=target_ms - 2 * HOUR_MS,
            end_ms=target_ms,
            concurrency=concurrency,
        )
    payload = build_final_payload(
        observation,
        histories,
        labeled_at=now_local,
        min_market_symbols=int(getattr(config, "STATIC_TARGET_TOP50_SHADOW_MIN_MARKET_SYMBOLS", 200)),
        label_delay_minutes=int(getattr(config, "STATIC_TARGET_TOP50_SHADOW_LABEL_DELAY_MINUTES", 5)),
        fetch_failures=failures,
    )
    write_json_once(paths["final"], payload)
    return {**payload, "files": {"final": str(paths["final"])}}


def record_missed(local_day: date, *, now_local: datetime, reports_dir: Path = REPORT_DIR) -> dict[str, Any]:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "contract_hash": CONTRACT_HASH,
        "generated_at_utc": _iso_utc(now_local),
        "local_day": local_day.isoformat(),
        "status": "missed_observation_window",
        "eligible": False,
        "reason": "worker_did_not_create_immutable_observation_within_grace_window",
        "performance_interpretation": "unknown_not_miss_or_success",
        "production_effect": "none_shadow_only",
    }
    path = _paths(reports_dir, local_day)["missed"]
    write_json_once(path, payload)
    return {**payload, "files": {"missed": str(path)}}


async def run_scheduled_once(
    *,
    now_local: datetime | None = None,
    reports_dir: Path = REPORT_DIR,
) -> dict[str, Any]:
    validate_runtime_contract()
    timezone_name = str(getattr(config, "STATIC_TARGET_TOP50_SHADOW_TIMEZONE", "Europe/Budapest"))
    tz = ZoneInfo(timezone_name)
    now_local = (now_local or datetime.now(tz)).astimezone(tz)
    action = scheduled_action(
        now_local,
        reports_dir=reports_dir,
        observation_time=time(
            int(getattr(config, "STATIC_TARGET_TOP50_SHADOW_OBSERVATION_HOUR_LOCAL", 12)),
            int(getattr(config, "STATIC_TARGET_TOP50_SHADOW_OBSERVATION_MINUTE_LOCAL", 15)),
        ),
        observation_grace_minutes=int(getattr(config, "STATIC_TARGET_TOP50_SHADOW_OBSERVATION_GRACE_MINUTES", 30)),
        target_time=time(int(getattr(config, "STATIC_TARGET_TOP50_SHADOW_TARGET_HOUR_LOCAL", 23)), 0),
        label_delay_minutes=int(getattr(config, "STATIC_TARGET_TOP50_SHADOW_LABEL_DELAY_MINUTES", 5)),
        start_day=date.fromisoformat(
            str(getattr(config, "STATIC_TARGET_TOP50_SHADOW_START_DAY_LOCAL", "2026-08-22"))
        ),
    )
    if action is None:
        return {"status": "idle", "action": None, "production_effect": "none_shadow_only"}
    action_name, local_day = action
    if action_name == "observe":
        result = await collect_observation(local_day, now_local=now_local, reports_dir=reports_dir)
    elif action_name == "finalize":
        result = await finalize_observation(local_day, now_local=now_local, reports_dir=reports_dir)
    elif action_name == "record_missed":
        result = record_missed(local_day, now_local=now_local, reports_dir=reports_dir)
    else:
        raise RuntimeError(f"unsupported action: {action_name}")
    scorecard = build_scorecard(
        reports_dir,
        as_of_day=now_local.date(),
        min_eligible_days=int(getattr(config, "STATIC_TARGET_TOP50_SHADOW_MIN_FORWARD_DAYS", 30)),
        min_observation_coverage=float(getattr(config, "STATIC_TARGET_TOP50_SHADOW_MIN_OBSERVATION_COVERAGE", 0.90)),
        min_final_coverage=float(getattr(config, "STATIC_TARGET_TOP50_SHADOW_MIN_FINAL_COVERAGE", 0.85)),
    )
    scorecard_path = reports_dir / "static_target_top50_shadow_scorecard_latest.json"
    _write_json_atomic(scorecard_path, scorecard)
    files = dict(result.get("files") or {})
    files["scorecard"] = str(scorecard_path)
    return {"action": action_name, **result, "files": files}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the static-target Top-50 forward shadow once")
    parser.add_argument("--reports-dir", type=Path, default=REPORT_DIR)
    parser.add_argument("--scorecard", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.scorecard:
        result = build_scorecard(args.reports_dir)
        path = args.reports_dir / "static_target_top50_shadow_scorecard_latest.json"
        _write_json_atomic(path, result)
    else:
        result = asyncio.run(run_scheduled_once(reports_dir=args.reports_dir))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
