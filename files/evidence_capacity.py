from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import median
from typing import Any, Iterable, Mapping, Sequence


MOVE_EVENT_VERSION = "move5_v1"
TOP_MOVER_VERSION = "watchlist_top_close_v1"
MISSION_TOP_MOVER_VERSION = "exchange_top_filtered_watchlist_v1"
POWER_METHOD_VERSION = "day_cluster_binary_v1"
THROUGHPUT_METHOD_VERSION = "attempt_throughput_v1"
METRIC_REGISTRY_VERSION = "action_metrics_v1"
REMEDIATION_SCHEMA_VERSION = "harness_remediation_v1"
LEGACY_INVENTORY_VERSION = "legacy_research_v1"

_HEX_64 = re.compile(r"^[0-9a-f]{64}$")
_SPEC_LINK = re.compile(r"`((?:docs|files)/[^`]+)`")
_RESEARCH_STATUS = re.compile(
    r"research|replay|shadow|rejected|reject|complete|validation|backtest",
    re.IGNORECASE,
)


class LabelConflictError(RuntimeError):
    """Raised when an immutable label identity is reused with other content."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256(value: Any) -> str:
    payload = value if isinstance(value, bytes) else _canonical_bytes(value)
    return hashlib.sha256(payload).hexdigest()


def _parse_time(value: Any, field: str) -> datetime:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty ISO-8601 timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{field} is not a valid ISO-8601 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{field} must be timezone-aware")
    return parsed.astimezone(timezone.utc)


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat()


def _positive_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{field} must be finite and positive")
    return result


def _validate_hash(value: Any, field: str) -> str:
    if not isinstance(value, str) or not _HEX_64.fullmatch(value):
        raise ValueError(f"{field} must be a lowercase SHA-256 hex digest")
    return value


def _validated_row(row: Mapping[str, Any]) -> dict[str, Any]:
    symbol = row.get("symbol")
    objective_day = row.get("objective_day")
    timezone_name = row.get("event_day_timezone")
    if not isinstance(symbol, str) or not symbol.strip():
        raise ValueError("symbol must be non-empty")
    if not isinstance(objective_day, str) or not re.fullmatch(
        r"\d{4}-\d{2}-\d{2}", objective_day
    ):
        raise ValueError("objective_day must be YYYY-MM-DD")
    if not isinstance(timezone_name, str) or not timezone_name.strip():
        raise ValueError("event_day_timezone must be non-empty")
    reference_time = _parse_time(row.get("reference_time"), "reference_time")
    cutoff = _parse_time(row.get("label_cutoff"), "label_cutoff")
    if cutoff <= reference_time:
        raise ValueError("label_cutoff must be after reference_time")
    reference_price = _positive_number(row.get("reference_price"), "reference_price")
    universe_hash = _validate_hash(
        row.get("universe_snapshot_hash"), "universe_snapshot_hash"
    )
    source_hash = _validate_hash(row.get("source_snapshot_hash"), "source_snapshot_hash")
    coverage_status = str(row.get("coverage_status", "unknown")).lower()
    if coverage_status not in {"complete", "partial", "unknown"}:
        raise ValueError("coverage_status must be complete, partial, or unknown")
    raw_bars = row.get("bars")
    if not isinstance(raw_bars, list):
        raise ValueError("bars must be a list")
    bars: list[dict[str, Any]] = []
    previous: datetime | None = None
    for index, bar in enumerate(raw_bars):
        if not isinstance(bar, Mapping):
            raise ValueError(f"bars[{index}] must be an object")
        close_time = _parse_time(bar.get("close_time"), f"bars[{index}].close_time")
        if previous is not None and close_time <= previous:
            raise ValueError("bars must be strictly ordered by close_time")
        previous = close_time
        high = _positive_number(bar.get("high"), f"bars[{index}].high")
        close = _positive_number(bar.get("close"), f"bars[{index}].close")
        if high < close:
            raise ValueError(f"bars[{index}].high cannot be below close")
        bars.append({"close_time": close_time, "high": high, "close": close})
    return {
        "symbol": symbol.strip().upper(),
        "objective_day": objective_day,
        "event_day_timezone": timezone_name,
        "reference_time": reference_time,
        "reference_price": reference_price,
        "label_cutoff": cutoff,
        "coverage_status": coverage_status,
        "universe_snapshot_hash": universe_hash,
        "source_snapshot_hash": source_hash,
        "bars": bars,
        "raw_bar_count": len(raw_bars),
    }


def _label_identity(
    symbol: str, objective_day: str, version: str, universe_snapshot_hash: str
) -> str:
    return _sha256(
        f"{symbol}|{objective_day}|{version}|{universe_snapshot_hash}".encode("utf-8")
    )


def build_move_event_labels(
    rows: Sequence[Mapping[str, Any]],
    *,
    as_of: str,
    event_version: str = MOVE_EVENT_VERSION,
    event_threshold: float = 0.05,
    midpoint_threshold: float = 0.025,
) -> list[dict[str, Any]]:
    """Build hindsight event labels without admitting partial or forming data."""
    observed_at = _parse_time(as_of, "as_of")
    if event_version != MOVE_EVENT_VERSION:
        raise ValueError(f"unsupported event_version: {event_version}")
    if not 0 < midpoint_threshold < event_threshold:
        raise ValueError("thresholds must satisfy 0 < midpoint < event")
    labels: list[dict[str, Any]] = []
    for raw in rows:
        row = _validated_row(raw)
        event_id = _label_identity(
            row["symbol"],
            row["objective_day"],
            event_version,
            row["universe_snapshot_hash"],
        )
        included = [
            bar
            for bar in row["bars"]
            if row["reference_time"] < bar["close_time"] <= row["label_cutoff"]
        ]
        common = {
            "schema_version": 1,
            "event_id": event_id,
            "event_version": event_version,
            "label_definition": (
                f"closed_bar_high_from_fixed_reference>={event_threshold:.6f};"
                f"midpoint={midpoint_threshold:.6f};cutoff_bound"
            ),
            "symbol": row["symbol"],
            "objective_day": row["objective_day"],
            "event_day_timezone": row["event_day_timezone"],
            "reference_time": _iso(row["reference_time"]),
            "reference_price": row["reference_price"],
            "label_cutoff": _iso(row["label_cutoff"]),
            "label_available_at": _iso(row["label_cutoff"]),
            "event_threshold": event_threshold,
            "midpoint_threshold": midpoint_threshold,
            "coverage_status": row["coverage_status"],
            "raw_bar_count": row["raw_bar_count"],
            "included_bar_count": len(included),
            "universe_snapshot_hash": row["universe_snapshot_hash"],
            "source_snapshot_hash": row["source_snapshot_hash"],
        }
        if observed_at < row["label_cutoff"]:
            labels.append(
                {
                    **common,
                    "status": "NOT_MATURE",
                    "decision_grade": False,
                    "event_occurred": None,
                    "first_midpoint_crossing_time": None,
                    "first_event_crossing_time": None,
                }
            )
            continue
        if row["coverage_status"] != "complete" or not included:
            labels.append(
                {
                    **common,
                    "status": "PARTIAL",
                    "decision_grade": False,
                    "event_occurred": None,
                    "first_midpoint_crossing_time": None,
                    "first_event_crossing_time": None,
                }
            )
            continue
        midpoint_price = row["reference_price"] * (1.0 + midpoint_threshold)
        event_price = row["reference_price"] * (1.0 + event_threshold)
        midpoint_cross = next(
            (bar["close_time"] for bar in included if bar["high"] >= midpoint_price), None
        )
        event_cross = next(
            (bar["close_time"] for bar in included if bar["high"] >= event_price), None
        )
        labels.append(
            {
                **common,
                "label_id": f"move_event:{event_id}",
                "status": "CONFIRMED" if event_cross else "NOT_EVENT",
                "decision_grade": True,
                "event_occurred": event_cross is not None,
                "first_midpoint_crossing_time": _iso(midpoint_cross)
                if midpoint_cross
                else None,
                "first_event_crossing_time": _iso(event_cross) if event_cross else None,
            }
        )
    return labels


def build_top_mover_labels(
    rows: Sequence[Mapping[str, Any]],
    *,
    as_of: str,
    top_k: int = 20,
    label_version: str = TOP_MOVER_VERSION,
) -> dict[str, Any]:
    """Rank one complete frozen universe-day using a fixed close-return label."""
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
        raise ValueError("top_k must be a positive integer")
    if label_version != TOP_MOVER_VERSION:
        raise ValueError(f"unsupported label_version: {label_version}")
    observed_at = _parse_time(as_of, "as_of")
    validated = [_validated_row(row) for row in rows]
    if not validated:
        return {
            "schema_version": 1,
            "label_version": label_version,
            "status": "NO_EVIDENCE",
            "decision_grade": False,
            "population_size": 0,
            "configured_top_k": top_k,
            "effective_top_k": 0,
            "labels": [],
        }
    group_keys = {
        (
            row["objective_day"],
            row["event_day_timezone"],
            row["universe_snapshot_hash"],
            row["source_snapshot_hash"],
            row["label_cutoff"],
        )
        for row in validated
    }
    if len(group_keys) != 1:
        raise ValueError("top-mover input must contain exactly one frozen universe-day")
    first = validated[0]
    common = {
        "schema_version": 1,
        "label_version": label_version,
        "objective_day": first["objective_day"],
        "event_day_timezone": first["event_day_timezone"],
        "label_definition": "rank_fixed_reference_to_last_closed_cutoff_close",
        "label_cutoff": _iso(first["label_cutoff"]),
        "label_available_at": _iso(first["label_cutoff"]),
        "universe_snapshot_hash": first["universe_snapshot_hash"],
        "source_snapshot_hash": first["source_snapshot_hash"],
        "population_size": len(validated),
        "configured_top_k": top_k,
        "effective_top_k": min(top_k, len(validated)),
    }
    if observed_at < first["label_cutoff"]:
        return {
            **common,
            "status": "NOT_MATURE",
            "decision_grade": False,
            "labels": [],
        }
    ranked: list[tuple[dict[str, Any], dict[str, Any], float]] = []
    partial_symbols: list[str] = []
    for row in validated:
        included = [
            bar
            for bar in row["bars"]
            if row["reference_time"] < bar["close_time"] <= row["label_cutoff"]
        ]
        if row["coverage_status"] != "complete" or not included:
            partial_symbols.append(row["symbol"])
            continue
        final_return = included[-1]["close"] / row["reference_price"] - 1.0
        ranked.append((row, included[-1], final_return))
    if partial_symbols:
        return {
            **common,
            "status": "PARTIAL",
            "decision_grade": False,
            "partial_symbols": sorted(partial_symbols),
            "labels": [],
        }
    ranked.sort(key=lambda item: (-item[2], item[0]["symbol"]))
    labels: list[dict[str, Any]] = []
    for rank, (row, final_bar, final_return) in enumerate(ranked, start=1):
        identity = _label_identity(
            row["symbol"],
            row["objective_day"],
            label_version,
            row["universe_snapshot_hash"],
        )
        labels.append(
            {
                "schema_version": 1,
                "label_id": f"top_mover:{identity}",
                "label_version": label_version,
                "label_definition": common["label_definition"],
                "symbol": row["symbol"],
                "objective_day": row["objective_day"],
                "event_day_timezone": row["event_day_timezone"],
                "rank": rank,
                "is_top_mover": rank <= common["effective_top_k"],
                "final_return": final_return,
                "population_size": common["population_size"],
                "configured_top_k": top_k,
                "effective_top_k": common["effective_top_k"],
                "reference_time": _iso(row["reference_time"]),
                "label_cutoff": common["label_cutoff"],
                "label_available_at": common["label_available_at"],
                "final_bar_close_time": _iso(final_bar["close_time"]),
                "coverage_status": "complete",
                "decision_grade": True,
                "universe_snapshot_hash": row["universe_snapshot_hash"],
                "source_snapshot_hash": row["source_snapshot_hash"],
            }
        )
    return {**common, "status": "COMPLETE", "decision_grade": True, "labels": labels}


class ImmutableLabelLedger:
    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)

    def read_all(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        records: list[dict[str, Any]] = []
        for line_no, line in enumerate(
            self.path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid label ledger JSON at line {line_no}") from exc
            if not isinstance(record, dict) or not record.get("label_id"):
                raise ValueError(f"invalid label ledger record at line {line_no}")
            records.append(record)
        return records

    def append(self, labels: Iterable[Mapping[str, Any]]) -> int:
        incoming = [dict(label) for label in labels]
        existing = self.read_all()
        by_id = {str(record["label_id"]): _canonical_bytes(record) for record in existing}
        pending: list[dict[str, Any]] = []
        pending_by_id: dict[str, bytes] = {}
        for label in incoming:
            label_id = label.get("label_id")
            if not isinstance(label_id, str) or not label_id:
                raise ValueError("every immutable label requires label_id")
            encoded = _canonical_bytes(label)
            prior = by_id.get(label_id, pending_by_id.get(label_id))
            if prior is not None:
                if prior != encoded:
                    raise LabelConflictError(f"immutable label conflict: {label_id}")
                continue
            pending.append(label)
            pending_by_id[label_id] = encoded
        if not pending:
            return 0
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8", newline="\n") as handle:
            for record in pending:
                handle.write(_canonical_bytes(record).decode("utf-8"))
                handle.write("\n")
        return len(pending)


def _normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def build_power_report(
    observations: Sequence[Mapping[str, Any]],
    *,
    sesoi: float,
    as_of: str,
    inconclusive_budget: float = 0.80,
    alpha_z: float = 1.96,
) -> dict[str, Any]:
    """Build a conservative binary-outcome planning report by objective day."""
    observed_at = _parse_time(as_of, "as_of")
    sesoi_value = _positive_number(sesoi, "sesoi")
    if not 0 < inconclusive_budget < 1:
        raise ValueError("inconclusive_budget must be between zero and one")
    complete: list[Mapping[str, Any]] = []
    days: set[str] = set()
    positives = 0
    for index, observation in enumerate(observations):
        day = observation.get("objective_day")
        if not isinstance(day, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", day):
            raise ValueError(f"observations[{index}].objective_day must be YYYY-MM-DD")
        outcome = observation.get("outcome")
        if outcome not in {True, False, None}:
            raise ValueError(f"observations[{index}].outcome must be boolean or null")
        if observation.get("coverage_status") == "complete" and outcome is not None:
            complete.append(observation)
            days.add(day)
            positives += int(outcome)
    denominator = len(complete)
    raw_count = len(observations)
    day_count = len(days)
    coverage = denominator / raw_count if raw_count else None
    if denominator == 0 or day_count == 0:
        return {
            "schema_version": 1,
            "method_version": POWER_METHOD_VERSION,
            "status": "NO_EVIDENCE",
            "raw_event_count": raw_count,
            "complete_objective_days": 0,
            "effective_sample_size": 0,
            "base_rate_numerator": 0,
            "base_rate_denominator": 0,
            "base_rate": None,
            "variance": None,
            "coverage_numerator": denominator,
            "coverage_denominator": raw_count,
            "coverage": coverage,
            "downtime_event_count": raw_count - denominator,
            "sesoi": sesoi_value,
            "mde": None,
            "expected_confidence_interval_width": None,
            "estimated_inconclusive_probability": None,
            "inconclusive_budget": inconclusive_budget,
            "expected_decision_horizon_days": None,
            "earliest_expected_maturity_date": None,
        }
    base_rate = positives / denominator
    variance = base_rate * (1.0 - base_rate)
    # Worst-case Bernoulli variance is intentionally retained for admission
    # planning. The observed base rate is descriptive and may be unstable.
    planning_variance = 0.25
    effective_n = day_count
    standard_error_two_arm = math.sqrt(2.0 * planning_variance / effective_n)
    mde = alpha_z * standard_error_two_arm
    interval_width = 2.0 * alpha_z * math.sqrt(planning_variance / effective_n)
    inconclusive_probability = _normal_cdf(
        (mde - sesoi_value) / standard_error_two_arm
    ) - _normal_cdf((-mde - sesoi_value) / standard_error_two_arm)
    required_days = max(
        1,
        math.ceil(2.0 * planning_variance * (alpha_z / sesoi_value) ** 2),
    )
    additional_days = max(0, required_days - effective_n)
    maturity = observed_at.date() + timedelta(days=additional_days)
    status = (
        "UNDERPOWERED"
        if inconclusive_probability > inconclusive_budget or mde > sesoi_value
        else "POWER_FEASIBLE"
    )
    return {
        "schema_version": 1,
        "method_version": POWER_METHOD_VERSION,
        "status": status,
        "raw_event_count": raw_count,
        "complete_objective_days": day_count,
        "effective_sample_size": effective_n,
        "base_rate_numerator": positives,
        "base_rate_denominator": denominator,
        "base_rate": base_rate,
        "variance": variance,
        "coverage_numerator": denominator,
        "coverage_denominator": raw_count,
        "coverage": coverage,
        "downtime_event_count": raw_count - denominator,
        "sesoi": sesoi_value,
        "mde": mde,
        "expected_confidence_interval_width": interval_width,
        "estimated_inconclusive_probability": inconclusive_probability,
        "inconclusive_budget": inconclusive_budget,
        "required_complete_days": required_days,
        "expected_decision_horizon_days": additional_days,
        "earliest_expected_maturity_date": maturity.isoformat(),
    }


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _percentile(values: Sequence[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, math.ceil(quantile * len(ordered)) - 1)
    return ordered[index]


def build_evidence_throughput_report(
    events: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    attempts: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for index, event in enumerate(events):
        attempt_id = event.get("attempt_id")
        if not isinstance(attempt_id, str) or not attempt_id:
            raise ValueError(f"events[{index}] requires attempt_id")
        if event.get("occurred_at") is not None:
            _parse_time(event.get("occurred_at"), f"events[{index}].occurred_at")
        attempts[attempt_id].append(event)
    if not attempts:
        return {
            "schema_version": 1,
            "method_version": THROUGHPUT_METHOD_VERSION,
            "status": "NO_EVIDENCE",
            "attempt_count": 0,
            "terminal_attempt_count": 0,
            "terminal_rate_numerator": 0,
            "terminal_rate_denominator": 0,
            "terminal_rate": None,
            "terminal_reason_counts": {},
            "median_hours_to_terminal": None,
            "p90_hours_to_terminal": None,
            "missing_duration_count": 0,
            "power_feasible_numerator": 0,
            "power_feasible_denominator": 0,
            "power_feasible_rate": None,
            "label_loss_numerator": 0,
            "label_loss_denominator": 0,
            "label_loss_rate": None,
            "evidence_reuse_numerator": 0,
            "evidence_reuse_denominator": 0,
            "evidence_reuse_rate": None,
        }
    terminal_reasons: Counter[str] = Counter()
    durations: list[float] = []
    missing_duration = 0
    terminal_count = 0
    power_denominator = power_numerator = 0
    expected_labels = observed_labels = 0
    reuse_denominator = reuse_numerator = 0
    for attempt_events in attempts.values():
        # The ledger is append-only. Phase -1 deliberately had no timestamps;
        # retain its order and report duration as unknown instead of inventing it.
        ordered = list(attempt_events)
        starts = [
            event
            for event in ordered
            if event.get("event_type") == "attempt_started"
            or event.get("stage") == "OBSERVED"
        ]
        terminals = [
            event
            for event in ordered
            if event.get("event_type") == "attempt_terminal"
            or event.get("status") == "TERMINAL"
        ]
        source = starts[0] if starts else ordered[0]
        if isinstance(source.get("power_feasible"), bool):
            power_denominator += 1
            power_numerator += int(source["power_feasible"])
        if isinstance(source.get("expected_labels"), int) and isinstance(
            source.get("observed_labels"), int
        ):
            expected = max(0, source["expected_labels"])
            observed = max(0, min(source["observed_labels"], expected))
            expected_labels += expected
            observed_labels += observed
        if isinstance(source.get("evidence_reused"), bool):
            reuse_denominator += 1
            reuse_numerator += int(source["evidence_reused"])
        if terminals:
            terminal_count += 1
            terminal = terminals[-1]
            terminal_reasons[str(terminal.get("outcome_reason") or "unknown")] += 1
            if starts and starts[0].get("occurred_at") and terminal.get("occurred_at"):
                start_at = _parse_time(starts[0]["occurred_at"], "occurred_at")
                terminal_at = _parse_time(terminal["occurred_at"], "occurred_at")
                if terminal_at >= start_at:
                    durations.append((terminal_at - start_at).total_seconds() / 3600.0)
                else:
                    missing_duration += 1
            else:
                missing_duration += 1
    label_loss = expected_labels - observed_labels
    attempt_count = len(attempts)
    return {
        "schema_version": 1,
        "method_version": THROUGHPUT_METHOD_VERSION,
        "status": "COMPLETE" if terminal_count == attempt_count else "PARTIAL",
        "attempt_count": attempt_count,
        "terminal_attempt_count": terminal_count,
        "terminal_rate_numerator": terminal_count,
        "terminal_rate_denominator": attempt_count,
        "terminal_rate": _ratio(terminal_count, attempt_count),
        "terminal_reason_counts": dict(sorted(terminal_reasons.items())),
        "median_hours_to_terminal": median(durations) if durations else None,
        "p90_hours_to_terminal": _percentile(durations, 0.90),
        "missing_duration_count": missing_duration,
        "power_feasible_numerator": power_numerator,
        "power_feasible_denominator": power_denominator,
        "power_feasible_rate": _ratio(power_numerator, power_denominator),
        "label_loss_numerator": label_loss,
        "label_loss_denominator": expected_labels,
        "label_loss_rate": _ratio(label_loss, expected_labels),
        "evidence_reuse_numerator": reuse_numerator,
        "evidence_reuse_denominator": reuse_denominator,
        "evidence_reuse_rate": _ratio(reuse_numerator, reuse_denominator),
    }


def action_layer_metric_registry() -> list[dict[str, Any]]:
    entries = [
        ("coverage_move5_v1", "WATCH", "early Move5 alerts", "confirmed Move5 events", MOVE_EVENT_VERSION, "steering_only", ["RESEARCH_PRIORITIZATION"]),
        ("precision_alert5_v1", "WATCH", "eligible alerts later confirming Move5", "unique eligible symbol-day alerts", MOVE_EVENT_VERSION, "watch_guardrail", ["WATCH_SHADOW"]),
        ("watchlist_top_capture_v1", "BUY", "canonical top movers bought", "canonical top movers", MISSION_TOP_MOVER_VERSION, "mission_objective", ["BUY_REPLAY"]),
        ("watchlist_top_early_capture_v1", "BUY", "top movers bought before registered early cutoff", "canonical top movers", MISSION_TOP_MOVER_VERSION, "mission_objective", ["BUY_REPLAY"]),
        ("trade_precision_v1", "BUY", "BUY decisions meeting registered outcome", "unique BUY decisions", "buy_outcome_v1", "buy_guardrail", ["BUY_REPLAY"]),
        ("exit_efficiency_v1", "SELL", "realized return relative to post-entry MFE", "mature closed positions with MFE", "exit_path_v1", "sell_objective", ["SELL_REPLAY"]),
        ("giveback_v1", "SELL", "MFE minus realized return", "mature closed positions with MFE", "exit_path_v1", "sell_guardrail", ["SELL_REPLAY"]),
        ("replacement_uplift_v1", "PORTFOLIO", "paired candidate-minus-protected portfolio outcome", "eligible replacement decisions", "portfolio_pair_v1", "portfolio_objective", ["PORTFOLIO_REPLAY"]),
        ("net_alpha_after_costs_v1", "PORTFOLIO", "unified portfolio return minus benchmark", "complete unified capital curves", "portfolio_curve_v1", "promotion_guardrail", ["PORTFOLIO_REPLAY", "PORTFOLIO_PROMOTION"]),
        ("model_proxy_score_v1", "OBSERVATION", "correct proxy predictions", "provenance-verified holdout rows", "model_holdout_v1", "diagnostic_only", ["MODEL_DIAGNOSTIC"]),
    ]
    return [
        {
            "registry_version": METRIC_REGISTRY_VERSION,
            "metric_id": metric_id,
            "action_layer": layer,
            "numerator": numerator,
            "denominator": denominator,
            "label_version": label_version,
            "availability_rule": "label_mature_and_coverage_complete",
            "decision_use": decision_use,
            "allowed_decisions": allowed,
            "cross_layer_substitution_allowed": False,
        }
        for metric_id, layer, numerator, denominator, label_version, decision_use, allowed in entries
    ]


_FINDING_POLICY = {
    "TH03": {
        "category": "model pipeline",
        "blocked": ["model_achievement_claim", "model_promotion"],
        "allowed": ["provenance_repair", "read_only_diagnosis", "rule_replay"],
        "repair": "Record immutable feature/label timing, label definition, evaluation scope, and provenance-verified rows.",
        "verify": "pyembed\\python.exe files\\truth_harness.py full",
    },
    "TH10": {
        "category": "learning worker",
        "blocked": ["current_rl_achievement_claim", "rl_promotion"],
        "allowed": ["worker_repair", "measurement_repair", "unrelated_watch_research"],
        "repair": "Regenerate RL evidence from current inputs with freshness and provenance fields, or retain stale status.",
        "verify": "pyembed\\python.exe files\\truth_harness.py full",
    },
}


def _finding_policy(check_id: str) -> dict[str, Any]:
    for prefix, policy in _FINDING_POLICY.items():
        if check_id.startswith(prefix):
            return policy
    return {
        "category": "truth harness",
        "blocked": ["claims_in_affected_scope", "promotion_in_affected_scope"],
        "allowed": ["measurement_repair", "read_only_diagnosis"],
        "repair": "Repair the named invariant and retain the finding until the full Harness verifies it.",
        "verify": "pyembed\\python.exe files\\truth_harness.py full",
    }


def build_harness_remediation_ledger(
    harness_payload: Mapping[str, Any], *, review_at: str
) -> dict[str, Any]:
    review_time = _parse_time(review_at, "review_at")
    generated_at = _parse_time(harness_payload.get("generated_at"), "generated_at")
    findings = harness_payload.get("findings")
    if not isinstance(findings, list):
        raise ValueError("Harness payload findings must be a list")
    records: list[dict[str, Any]] = []
    for index, finding in enumerate(findings):
        if not isinstance(finding, Mapping):
            raise ValueError(f"findings[{index}] must be an object")
        check_id = str(finding.get("check_id") or "UNKNOWN")
        policy = _finding_policy(check_id)
        evidence = str(finding.get("evidence") or "")
        message = str(finding.get("message") or "unnamed Harness finding")
        invariant = str(finding.get("invariant") or "UNKNOWN")
        record_identity = {
            "check_id": check_id,
            "invariant": invariant,
            "message": message,
            "evidence": evidence,
            "generated_at": _iso(generated_at),
        }
        records.append(
            {
                "finding_id": _sha256(record_identity),
                "check_id": check_id,
                "invariant": invariant,
                "severity": str(finding.get("severity") or "unknown"),
                "message": message,
                "affected_scope": policy["blocked"],
                "blocked_actions": policy["blocked"],
                "allowed_work": policy["allowed"],
                "owner": "repository maintainer",
                "triage_category": policy["category"],
                "repair_task": str(finding.get("remediation") or policy["repair"]),
                "verification_command": policy["verify"],
                "state": "OPEN",
                "acknowledgement_target": "next_weekly_triage",
                "review_at": _iso(review_time),
                "escalation_state": "NOT_ESCALATED",
                "source_evidence_hash": _sha256(evidence.encode("utf-8")),
                "waived": False,
            }
        )
    return {
        "schema_version": REMEDIATION_SCHEMA_VERSION,
        "source_harness_status": str(harness_payload.get("status") or "unknown"),
        "source_generated_at": _iso(generated_at),
        "finding_count": len(records),
        "findings": records,
    }


def _research_sources(root: Path, feature_index_path: Path) -> list[Path]:
    candidates: set[Path] = set()
    reports = root / "docs" / "reports"
    if reports.exists():
        candidates.update(path for path in reports.rglob("*.md") if path.is_file())
    if feature_index_path.exists():
        text = feature_index_path.read_text(encoding="utf-8-sig")
        for line in text.splitlines():
            if not _RESEARCH_STATUS.search(line):
                continue
            for relative in _SPEC_LINK.findall(line):
                path = root.joinpath(*relative.split("/"))
                if path.suffix.lower() == ".md":
                    candidates.add(path)
    return sorted(candidates, key=lambda path: path.as_posix().lower())


def _load_negative_registry(
    root: Path, registry_path: Path
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    if not registry_path.exists():
        return {}, []
    try:
        payload = json.loads(registry_path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        return {}, [f"registry_unreadable:{type(exc).__name__}"]
    if not isinstance(payload, dict) or not isinstance(payload.get("entries"), list):
        return {}, ["registry_schema_invalid"]
    by_source: dict[str, dict[str, Any]] = {}
    seen_ids: set[str] = set()
    errors: list[str] = []
    for index, raw in enumerate(payload["entries"]):
        prefix = f"entry_{index}"
        if not isinstance(raw, dict):
            errors.append(f"{prefix}:not_object")
            continue
        negative_id = raw.get("negative_id")
        source = raw.get("source")
        digest = raw.get("source_sha256")
        period = raw.get("period")
        required_text = ("population", "metric", "evidence_summary")
        if not isinstance(negative_id, str) or not negative_id:
            errors.append(f"{prefix}:negative_id_invalid")
            continue
        if negative_id in seen_ids:
            errors.append(f"{prefix}:duplicate_negative_id:{negative_id}")
            continue
        seen_ids.add(negative_id)
        if not isinstance(source, str) or not source or source.startswith(("/", "\\")):
            errors.append(f"{prefix}:source_invalid")
            continue
        if not isinstance(digest, str) or not _HEX_64.fullmatch(digest):
            errors.append(f"{prefix}:source_sha256_invalid")
            continue
        if raw.get("verdict") != "rejected":
            errors.append(f"{prefix}:verdict_not_rejected")
            continue
        if not all(isinstance(raw.get(field), str) and raw.get(field) for field in required_text):
            errors.append(f"{prefix}:required_text_invalid")
            continue
        if not isinstance(period, dict):
            errors.append(f"{prefix}:period_invalid")
            continue
        try:
            period_start = datetime.fromisoformat(str(period.get("start"))).date()
            period_end = datetime.fromisoformat(str(period.get("end"))).date()
        except ValueError:
            errors.append(f"{prefix}:period_invalid")
            continue
        if period_end < period_start:
            errors.append(f"{prefix}:period_reversed")
            continue
        normalized_source = Path(source).as_posix()
        if normalized_source in by_source:
            errors.append(f"{prefix}:duplicate_source:{normalized_source}")
            continue
        by_source[normalized_source] = dict(raw)
    return by_source, errors


def migrate_legacy_research_inventory(
    root: Path | str,
    feature_index_path: Path | str,
    confirmed_registry_path: Path | str | None = None,
) -> dict[str, Any]:
    root_path = Path(root).resolve()
    index_path = Path(feature_index_path)
    if not index_path.is_absolute():
        index_path = root_path / index_path
    registry_path = (
        Path(confirmed_registry_path)
        if confirmed_registry_path is not None
        else root_path / "docs" / "specs" / "legacy-negative-results-registry.json"
    )
    if not registry_path.is_absolute():
        registry_path = root_path / registry_path
    registry, registry_errors = _load_negative_registry(root_path, registry_path)
    sources = _research_sources(root_path, index_path)
    items: list[dict[str, Any]] = []
    canonical_by_hash: dict[str, str] = {}
    counts: Counter[str] = Counter()
    registry_accepted = 0
    registry_hash_mismatch = 0
    for path in sources:
        try:
            payload = path.read_bytes()
            payload.decode("utf-8-sig")
        except (OSError, UnicodeError) as exc:
            state = "MIGRATION_ERROR"
            item = {
                "source": path.relative_to(root_path).as_posix()
                if path.is_relative_to(root_path)
                else str(path),
                "state": state,
                "content_hash": None,
                "repair_task": f"Make source readable without inventing evidence: {type(exc).__name__}",
            }
        else:
            content_hash = hashlib.sha256(payload).hexdigest()
            relative = (
                path.relative_to(root_path).as_posix()
                if path.is_relative_to(root_path)
                else str(path)
            )
            duplicate_of = canonical_by_hash.get(content_hash)
            if duplicate_of is not None:
                state = "DUPLICATE"
                item = {
                    "source": relative,
                    "state": state,
                    "content_hash": content_hash,
                    "duplicate_of": duplicate_of,
                }
            else:
                canonical_by_hash[content_hash] = relative
                reviewed = registry.get(relative)
                reviewed_match = bool(
                    reviewed and reviewed.get("source_sha256") == content_hash
                )
                if reviewed and not reviewed_match:
                    registry_hash_mismatch += 1
                state = "CONFIRMED_NEGATIVE" if reviewed_match else "LEGACY_UNVERIFIED"
                item = {
                    "source": relative,
                    "state": state,
                    "content_hash": content_hash,
                    "decision_grade": state == "CONFIRMED_NEGATIVE",
                    "similarity_use": "hard_negative"
                    if state == "CONFIRMED_NEGATIVE"
                    else "warning_only",
                }
                if reviewed_match:
                    registry_accepted += 1
                    item["negative_id"] = reviewed["negative_id"]
                    item["period"] = reviewed["period"]
                    item["population"] = reviewed["population"]
                    item["metric"] = reviewed["metric"]
                    item["verdict"] = reviewed["verdict"]
                    item["evidence_summary"] = reviewed["evidence_summary"]
                elif reviewed:
                    item["registry_status"] = "hash_mismatch"
        counts[state] += 1
        items.append(item)
    discovered_relative = {
        path.relative_to(root_path).as_posix()
        for path in sources
        if path.is_relative_to(root_path)
    }
    registry_orphans = sorted(set(registry) - discovered_relative)
    return {
        "schema_version": LEGACY_INVENTORY_VERSION,
        "declared_roots": ["docs/reports", "docs/FEATURE_SPEC_INDEX.md research entries"],
        "discovered_count": len(sources),
        "migrated_count": len(items),
        "state_counts": dict(sorted(counts.items())),
        "complete": len(sources) == len(items),
        "registry": {
            "path": str(registry_path),
            "entry_count": len(registry),
            "accepted_count": registry_accepted,
            "hash_mismatch_count": registry_hash_mismatch,
            "orphan_count": len(registry_orphans),
            "orphan_sources": registry_orphans,
            "errors": registry_errors,
        },
        "items": items,
    }


_OBJECTIVE_REQUIRED = {
    "metric_id",
    "action_layer",
    "metric_version",
    "label_version",
    "method_version",
    "numerator",
    "denominator",
    "coverage_numerator",
    "coverage_denominator",
    "coverage_status",
    "exclusions",
    "feature_cutoff",
    "label_cutoff",
    "label_available_at",
    "estimate",
    "interval_low",
    "interval_high",
    "sesoi",
    "mde",
    "effective_sample_size",
    "expected_decision_horizon_days",
    "evidence_status",
    "verdict_rule",
    "verdict_rule_passed",
}


def verify_objective_report_contract(report: Mapping[str, Any]) -> list[str]:
    errors = [f"missing required field: {field}" for field in sorted(_OBJECTIVE_REQUIRED - report.keys())]
    numeric_nonnegative = (
        "numerator",
        "denominator",
        "coverage_numerator",
        "coverage_denominator",
        "effective_sample_size",
        "expected_decision_horizon_days",
    )
    for field in numeric_nonnegative:
        if field not in report:
            continue
        value = report[field]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or value < 0:
            errors.append(f"{field} must be finite and non-negative")
    for numerator, denominator in (("numerator", "denominator"), ("coverage_numerator", "coverage_denominator")):
        if numerator in report and denominator in report and isinstance(report[numerator], (int, float)) and isinstance(report[denominator], (int, float)) and report[numerator] > report[denominator]:
            errors.append(f"{numerator} cannot exceed {denominator}")
    if (
        report.get("coverage_status") == "complete"
        and report.get("coverage_numerator") != report.get("coverage_denominator")
    ):
        errors.append("complete coverage requires equal coverage numerator and denominator")
    if report.get("denominator") == 0 and report.get("estimate") is not None:
        errors.append("estimate must be null when denominator is zero")
    for field in ("estimate", "interval_low", "interval_high", "sesoi", "mde"):
        value = report.get(field)
        if value is not None and (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            errors.append(f"{field} must be finite numeric or null")
    if all(field in report for field in ("interval_low", "estimate", "interval_high")):
        low, estimate, high = report["interval_low"], report["estimate"], report["interval_high"]
        if all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in (low, estimate, high)) and not low <= estimate <= high:
            errors.append("interval must contain estimate")
    for field in ("feature_cutoff", "label_cutoff", "label_available_at"):
        if field in report:
            try:
                _parse_time(report[field], field)
            except ValueError as exc:
                errors.append(str(exc))
    if all(field in report for field in ("label_cutoff", "label_available_at")):
        try:
            if _parse_time(report["label_available_at"], "label_available_at") < _parse_time(report["label_cutoff"], "label_cutoff"):
                errors.append("label_available_at cannot precede label_cutoff")
        except ValueError:
            pass
    if all(field in report for field in ("feature_cutoff", "label_cutoff")):
        try:
            if _parse_time(report["feature_cutoff"], "feature_cutoff") > _parse_time(
                report["label_cutoff"], "label_cutoff"
            ):
                errors.append("feature_cutoff cannot follow label_cutoff")
        except ValueError:
            pass
    if report.get("verdict_rule_passed") not in {True, False}:
        errors.append("verdict_rule_passed must be boolean")
    if not isinstance(report.get("verdict_rule"), str) or not report.get("verdict_rule"):
        errors.append("verdict_rule must be non-empty")
    if report.get("evidence_status") == "IMPROVING" and (
        errors
        or report.get("coverage_status") != "complete"
        or not report.get("denominator")
        or report.get("effective_sample_size", 0) <= 0
        or report.get("verdict_rule_passed") is not True
    ):
        errors.append(
            "IMPROVING is not allowed without a complete contract and passed verdict rule"
        )
    return errors


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _load_jsonl(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def repository_audit(
    root: Path,
    harness_json: Path,
    *,
    review_at: str,
    attempt_ledger: Path | None = None,
) -> dict[str, Any]:
    harness_payload = _load_json(harness_json)
    remediation = build_harness_remediation_ledger(harness_payload, review_at=review_at)
    inventory = migrate_legacy_research_inventory(
        root, root / "docs" / "FEATURE_SPEC_INDEX.md"
    )
    throughput = build_evidence_throughput_report(_load_jsonl(attempt_ledger))
    blockers = [
        "canonical_snapshot_not_supplied",
        "objective_power_report_not_computed",
    ]
    if remediation["source_harness_status"].lower() != "pass":
        blockers.append("truth_harness_not_pass")
    return {
        "schema_version": "phase0_repository_audit_v1",
        "phase": "PHASE_0",
        "status": "IN_PROGRESS" if blockers else "COMPLETE",
        "trading_behavior_changed": False,
        "metric_registry": action_layer_metric_registry(),
        "harness_remediation": remediation,
        "legacy_inventory": inventory,
        "evidence_throughput": throughput,
        "exit_blockers": blockers,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 0 evidence-capacity audit")
    subparsers = parser.add_subparsers(dest="command", required=True)
    audit = subparsers.add_parser("repository-audit")
    audit.add_argument("--root", type=Path, required=True)
    audit.add_argument("--harness-json", type=Path, required=True)
    audit.add_argument("--review-at", required=True)
    audit.add_argument("--attempt-ledger", type=Path)
    audit.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.command == "repository-audit":
        result = repository_audit(
            args.root.resolve(),
            args.harness_json.resolve(),
            review_at=args.review_at,
            attempt_ledger=args.attempt_ledger.resolve() if args.attempt_ledger else None,
        )
        rendered = json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered, encoding="utf-8")
        else:
            print(rendered, end="")
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
