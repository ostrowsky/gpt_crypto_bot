from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from evidence_capacity import build_power_report, verify_objective_report_contract


FINAL_CRITIC_PATTERN = re.compile(
    r"^top_gainer_critic_(\d{4}-\d{2}-\d{2})_final\.json$"
)
LABEL_VERSION = "exchange_top_filtered_watchlist_v1"
SNAPSHOT_VERSION = "maximum_final_critic_snapshot_v1"
OBJECTIVE_METHOD_VERSION = "day_cluster_bootstrap_baseline_v1"


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _parse_as_of(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (AttributeError, ValueError) as exc:
        raise ValueError("as_of must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("as_of must be timezone-aware")
    return parsed.astimezone(timezone.utc)


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _calendar_days(start: date, end: date) -> list[str]:
    return [
        (start + timedelta(days=offset)).isoformat()
        for offset in range((end - start).days + 1)
    ]


def _cutoffs(day: date, timezone_name: str) -> tuple[str, str]:
    try:
        zone = ZoneInfo(timezone_name)
    except ZoneInfoNotFoundError as exc:
        raise ValueError(f"unknown timezone: {timezone_name}") from exc
    start = datetime.combine(day, time.min, tzinfo=zone).astimezone(timezone.utc)
    end = datetime.combine(day + timedelta(days=1), time.min, tzinfo=zone).astimezone(
        timezone.utc
    )
    return start.isoformat(), end.isoformat()


def _summary_expectations(
    rows: Sequence[Mapping[str, Any]], early_threshold: float
) -> dict[str, int | float | str]:
    filtered = [row for row in rows if row.get("in_watchlist") is True]
    bought = [row for row in filtered if row.get("status") == "bought"]
    early = [
        row
        for row in bought
        if (_number(row.get("capture_ratio")) or 0.0) >= early_threshold
    ]
    denominator = len(filtered)
    return {
        "exchange_top_count": len(rows),
        "exchange_top_in_watchlist": denominator,
        "watchlist_top_count": denominator,
        "watchlist_top_bought": len(bought),
        "watchlist_top_early_captured": len(early),
        "watchlist_top_missed": denominator - len(bought),
        "watchlist_top_capture_rate_pct": round(
            len(bought) / denominator * 100.0, 2
        )
        if denominator
        else 0.0,
        "watchlist_top_early_capture_rate_pct": round(
            len(early) / denominator * 100.0, 2
        )
        if denominator
        else 0.0,
        "watchlist_top_denominator": "exchange_top_filtered_to_watchlist",
    }


def _validate_report(
    path: Path, filename_day: str, payload: Any, source_hash: str
) -> tuple[list[str], dict[str, Any] | None]:
    reasons: list[str] = []
    if not isinstance(payload, Mapping):
        return ["report_not_object"], None
    if payload.get("target_day_local") != filename_day:
        reasons.append("target_day_filename_mismatch")
    if payload.get("phase") != "final":
        reasons.append("phase_not_final")
    settings = payload.get("settings")
    summary = payload.get("summary")
    rows = payload.get("exchange_top_gainers")
    if not isinstance(settings, Mapping):
        reasons.append("settings_missing")
        settings = {}
    if not isinstance(summary, Mapping):
        reasons.append("summary_missing")
        summary = {}
    if not isinstance(rows, list):
        reasons.append("exchange_top_gainers_missing")
        rows = []
    timezone_name = settings.get("timezone")
    if not isinstance(timezone_name, str) or not timezone_name:
        reasons.append("timezone_missing")
        timezone_name = "UTC"
    try:
        day_value = date.fromisoformat(filename_day)
        feature_cutoff, label_cutoff = _cutoffs(day_value, timezone_name)
    except ValueError as exc:
        reasons.append(str(exc))
        day_value = date.fromisoformat(filename_day)
        feature_cutoff, label_cutoff = _cutoffs(day_value, "UTC")
    top_n = _int(settings.get("top_n"))
    if top_n is None or top_n <= 0:
        reasons.append("top_n_invalid")
    elif len(rows) != top_n:
        reasons.append("exchange_top_count_not_top_n")
    early_threshold = _number(settings.get("early_capture_ratio_min"))
    if early_threshold is None or not 0 <= early_threshold <= 1.5:
        reasons.append("early_capture_ratio_min_invalid")
        early_threshold = 0.35
    normalized_rows: list[dict[str, Any]] = []
    symbols: set[str] = set()
    last_change: float | None = None
    for index, raw in enumerate(rows):
        if not isinstance(raw, Mapping):
            reasons.append(f"row_{index}_not_object")
            continue
        symbol = raw.get("symbol")
        if not isinstance(symbol, str) or not symbol:
            reasons.append(f"row_{index}_symbol_invalid")
            continue
        if symbol in symbols:
            reasons.append(f"duplicate_symbol:{symbol}")
        symbols.add(symbol)
        day_open = _number(raw.get("day_open"))
        day_close = _number(raw.get("day_close"))
        day_change = _number(raw.get("day_change_pct"))
        if day_open is None or day_open <= 0 or day_close is None or day_close <= 0:
            reasons.append(f"price_invalid:{symbol}")
        elif day_change is None:
            reasons.append(f"day_change_missing:{symbol}")
        else:
            recomputed = (day_close / day_open - 1.0) * 100.0
            if abs(recomputed - day_change) > 0.02:
                reasons.append(f"day_change_mismatch:{symbol}")
        if day_change is not None and last_change is not None and day_change > last_change + 1e-9:
            reasons.append("exchange_rows_not_descending")
        if day_change is not None:
            last_change = day_change
        if raw.get("in_watchlist") not in {True, False}:
            reasons.append(f"watchlist_flag_invalid:{symbol}")
        normalized_rows.append(dict(raw))
    expected = _summary_expectations(normalized_rows, early_threshold)
    for key, expected_value in expected.items():
        actual = summary.get(key)
        if isinstance(expected_value, float):
            actual_number = _number(actual)
            if actual_number is None or abs(actual_number - expected_value) > 0.011:
                reasons.append(f"summary_mismatch:{key}")
        elif actual != expected_value:
            reasons.append(f"summary_mismatch:{key}")
    if reasons:
        return sorted(set(reasons)), None
    population_payload = {
        "target_day_local": filename_day,
        "timezone": timezone_name,
        "top_n": top_n,
        "rows": [
            {
                "rank": rank,
                "symbol": row["symbol"],
                "day_change_pct": row["day_change_pct"],
                "in_watchlist": row["in_watchlist"],
            }
            for rank, row in enumerate(normalized_rows, start=1)
        ],
    }
    population_hash = _sha256(_canonical_bytes(population_payload))
    filtered = [row for row in normalized_rows if row["in_watchlist"] is True]
    labels: list[dict[str, Any]] = []
    for exchange_rank, row in enumerate(normalized_rows, start=1):
        if row["in_watchlist"] is not True:
            continue
        capture_ratio = _number(row.get("capture_ratio"))
        bought = row.get("status") == "bought"
        early = bought and (capture_ratio or 0.0) >= early_threshold
        identity = _sha256(
            f"{filename_day}|{row['symbol']}|{LABEL_VERSION}|{source_hash}".encode(
                "utf-8"
            )
        )
        labels.append(
            {
                "schema_version": 1,
                "label_id": f"top_mover:{identity}",
                "label_version": LABEL_VERSION,
                "label_definition": "exchange_top_n_filtered_to_frozen_watchlist",
                "symbol": row["symbol"],
                "objective_day": filename_day,
                "event_day_timezone": timezone_name,
                "exchange_rank": exchange_rank,
                "exchange_top_population_size": len(normalized_rows),
                "watchlist_top_denominator": len(filtered),
                "day_change_pct": row["day_change_pct"],
                "bought": bought,
                "early_captured": early,
                "capture_ratio": capture_ratio,
                "early_capture_ratio_min": early_threshold,
                "feature_cutoff": feature_cutoff,
                "label_cutoff": label_cutoff,
                "label_available_at": label_cutoff,
                "coverage_status": "complete",
                "decision_grade": True,
                "source_file": path.name,
                "source_snapshot_hash": source_hash,
                "population_snapshot_hash": population_hash,
            }
        )
    return [], {
        "target_day_local": filename_day,
        "timezone": timezone_name,
        "feature_cutoff": feature_cutoff,
        "label_cutoff": label_cutoff,
        "source_file": path.name,
        "source_snapshot_hash": source_hash,
        "population_snapshot_hash": population_hash,
        "exchange_top_count": len(normalized_rows),
        "watchlist_top_count": expected["watchlist_top_count"],
        "early_captured": expected["watchlist_top_early_captured"],
        "labels": labels,
    }


def _cluster_interval(
    daily: Sequence[tuple[int, int]], snapshot_hash: str, *, samples: int = 4000
) -> tuple[float | None, float | None]:
    if len(daily) < 2 or not any(denominator for _, denominator in daily):
        return None, None
    rng = random.Random(int(snapshot_hash[:16], 16))
    estimates: list[float] = []
    for _ in range(samples):
        selected = [daily[rng.randrange(len(daily))] for _ in range(len(daily))]
        numerator = sum(item[0] for item in selected)
        denominator = sum(item[1] for item in selected)
        if denominator:
            estimates.append(numerator / denominator)
    if not estimates:
        return None, None
    estimates.sort()
    low_index = max(0, math.floor(0.025 * (len(estimates) - 1)))
    high_index = min(len(estimates) - 1, math.ceil(0.975 * (len(estimates) - 1)))
    return estimates[low_index], estimates[high_index]


def build_maximum_critic_audit(
    report_dir: Path | str,
    *,
    as_of: str,
    sesoi: float = 0.05,
) -> dict[str, Any]:
    directory = Path(report_dir)
    observed_at = _parse_as_of(as_of)
    if not isinstance(sesoi, (int, float)) or isinstance(sesoi, bool) or not 0 < float(sesoi) < 1:
        raise ValueError("sesoi must be between zero and one")
    discovered: list[tuple[Path, str]] = []
    if directory.exists():
        for path in directory.iterdir():
            match = FINAL_CRITIC_PATTERN.fullmatch(path.name) if path.is_file() else None
            if match:
                discovered.append((path, match.group(1)))
    discovered.sort(key=lambda item: (item[1], item[0].name))
    file_manifest: list[dict[str, Any]] = []
    invalid: list[dict[str, Any]] = []
    valid_days: list[dict[str, Any]] = []
    discovered_dates = [date.fromisoformat(day) for _, day in discovered]
    for path, filename_day in discovered:
        try:
            raw = path.read_bytes()
            source_hash = _sha256(raw)
            payload = json.loads(raw.decode("utf-8-sig"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            invalid.append(
                {
                    "source_file": path.name,
                    "target_day_local": filename_day,
                    "reasons": [f"unreadable:{type(exc).__name__}"],
                }
            )
            continue
        file_manifest.append(
            {
                "source_file": path.name,
                "target_day_local": filename_day,
                "sha256": source_hash,
                "byte_count": len(raw),
            }
        )
        reasons, day_record = _validate_report(path, filename_day, payload, source_hash)
        if reasons:
            invalid.append(
                {
                    "source_file": path.name,
                    "target_day_local": filename_day,
                    "reasons": reasons,
                }
            )
        elif day_record is not None:
            valid_days.append(day_record)
    manifest_hash = _sha256(_canonical_bytes(file_manifest))
    if discovered_dates:
        calendar = _calendar_days(min(discovered_dates), max(discovered_dates))
        discovered_day_set = {day for _, day in discovered}
        missing_days = [day for day in calendar if day not in discovered_day_set]
    else:
        calendar = []
        missing_days = []
    valid_day_set = {record["target_day_local"] for record in valid_days}
    invalid_days = sorted(
        {record["target_day_local"] for record in invalid if record.get("target_day_local")}
    )
    excluded_days = sorted(set(missing_days) | set(invalid_days))
    labels = [label for day_record in valid_days for label in day_record["labels"]]
    observations = [
        {
            "objective_day": label["objective_day"],
            "outcome": label["early_captured"],
            "coverage_status": "complete",
        }
        for label in labels
    ]
    power = build_power_report(
        observations,
        sesoi=float(sesoi),
        as_of=observed_at.isoformat(),
    )
    numerator = sum(int(label["early_captured"]) for label in labels)
    denominator = len(labels)
    estimate = numerator / denominator if denominator else None
    daily_counts = [
        (int(record["early_captured"]), int(record["watchlist_top_count"]))
        for record in valid_days
    ]
    interval_low, interval_high = _cluster_interval(daily_counts, manifest_hash)
    coverage_denominator = len(calendar)
    coverage_numerator = len(valid_day_set)
    coverage_status = (
        "complete"
        if coverage_denominator > 0 and coverage_numerator == coverage_denominator
        else "partial"
        if coverage_denominator > 0
        else "unknown"
    )
    if discovered_dates:
        timezone_name = valid_days[0]["timezone"] if valid_days else "UTC"
        feature_cutoff, _ = _cutoffs(min(discovered_dates), timezone_name)
        _, label_cutoff = _cutoffs(max(discovered_dates), timezone_name)
    else:
        feature_cutoff = observed_at.isoformat()
        label_cutoff = observed_at.isoformat()
    objective = {
        "metric_id": "watchlist_top_early_capture_v1",
        "action_layer": "BUY",
        "metric_version": "v1",
        "label_version": LABEL_VERSION,
        "method_version": OBJECTIVE_METHOD_VERSION,
        "numerator": numerator,
        "denominator": denominator,
        "coverage_numerator": coverage_numerator,
        "coverage_denominator": coverage_denominator,
        "coverage_status": coverage_status,
        "exclusions": excluded_days,
        "feature_cutoff": feature_cutoff,
        "feature_cutoff_semantics": "aggregate_measurement_window_start_not_prediction_time",
        "label_cutoff": label_cutoff,
        "label_available_at": label_cutoff,
        "estimate": estimate,
        "interval_low": interval_low,
        "interval_high": interval_high,
        "sesoi": float(sesoi),
        "mde": power.get("mde"),
        "effective_sample_size": power.get("effective_sample_size", 0),
        "expected_decision_horizon_days": power.get("expected_decision_horizon_days"),
        "evidence_status": "INSUFFICIENT_EVIDENCE",
        "verdict_rule": "baseline_only_no_directional_verdict",
        "verdict_rule_passed": False,
        "source_manifest_hash": manifest_hash,
    }
    contract_errors = verify_objective_report_contract(objective)
    measurement_grade = (
        coverage_status == "complete"
        and bool(valid_days)
        and not contract_errors
        and denominator > 0
    )
    return {
        "schema_version": 1,
        "snapshot_version": SNAPSHOT_VERSION,
        "generated_as_of": observed_at.isoformat(),
        "status": "COMPLETE" if measurement_grade else "PARTIAL",
        "measurement_grade": measurement_grade,
        "promotion_grade": False,
        "trading_behavior_changed": False,
        "snapshot": {
            "report_directory": str(directory.resolve()),
            "manifest_hash": manifest_hash,
            "discovered_exact_final_files": len(discovered),
            "hashed_readable_files": len(file_manifest),
            "valid_final_days": len(valid_days),
            "invalid_reports": invalid,
            "first_day": min(discovered_dates).isoformat() if discovered_dates else None,
            "last_day": max(discovered_dates).isoformat() if discovered_dates else None,
            "calendar_coverage_numerator": coverage_numerator,
            "calendar_coverage_denominator": coverage_denominator,
            "missing_calendar_days": missing_days,
            "invalid_days": invalid_days,
            "files": file_manifest,
        },
        "labels": labels,
        "objective_report": objective,
        "objective_contract_errors": contract_errors,
        "power_report": power,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit maximum local final-critic history")
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--as-of", required=True)
    parser.add_argument("--sesoi", type=float, default=0.05)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    result = build_maximum_critic_audit(
        args.report_dir, as_of=args.as_of, sesoi=args.sesoi
    )
    rendered = json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
