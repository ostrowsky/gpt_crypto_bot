"""Causal maximum-period validation of static-target plus 3h rank velocity."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import subprocess
from collections import defaultdict
from dataclasses import asdict, dataclass, replace
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Iterable, Mapping, Sequence
from zoneinfo import ZoneInfo

from validate_external_top50_screen import (
    DAY_MS,
    DEFAULT_LEGACY_CACHE,
    DEFAULT_TAIL_CACHE,
    DEFAULT_WATCHLIST,
    HOUR_MS,
    Bar,
    CacheFile,
    _date_range,
    _default_period,
    _fresh_bar,
    _load_json_rows,
    _safe_return,
    discover_cache_files,
    merge_binance_rows,
    wilson_interval,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT = ROOT / ".runtime" / "reports" / "static_target_rank_velocity_validation_latest.json"
DEFAULT_SNAPSHOT = ROOT / ".runtime" / "research" / "static_target_rank_velocity_snapshot.json"
DEFAULT_CONTRACT = ROOT / ".runtime" / "research" / "static_target_rank_velocity_contract.json"
SNAPSHOT_KIND = "static_target_rank_velocity_normalized_v1"
POLICIES = ("static_target", "static_target_rank_velocity_v1")
RANK_VELOCITY_HOURS = 3
RANK_VELOCITY_WEIGHT = 0.25
POWER_FACTOR_95_80 = 2.8015852181129683
DEFAULT_BOOTSTRAP_SEED = 220826


@dataclass(frozen=True, slots=True)
class FeatureRow:
    symbol: str
    current_return: float
    static_target_return: float
    prior_current_return: float
    target_return: float


@dataclass(frozen=True, slots=True)
class Candidate:
    symbol: str
    current_return: float
    static_target_return: float
    prior_current_return: float
    current_market_rank: int
    prior_market_rank: int
    rank_velocity_3h: int
    target_rank: int
    is_target_top: bool


@dataclass(frozen=True, slots=True)
class DaySnapshot:
    local_day: str
    market_symbol_count: int
    watchlist_symbol_count: int
    candidates: tuple[Candidate, ...]
    target_entrant_symbols: frozenset[str]


@dataclass(frozen=True, slots=True)
class RankedCandidate:
    symbol: str
    static_target_return: float
    rank_velocity_3h: int
    target_rank: int
    is_target_top: bool
    static_target_percentile: float
    rank_velocity_percentile: float
    composite_score: float


class ValidationError(ValueError):
    pass


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256_payload(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def _extract_feature_row(
    symbol: str,
    history: Sequence[Bar],
    *,
    observation_ms: int,
    target_ms: int,
    close_times: Sequence[int] | None = None,
) -> FeatureRow | None:
    prior_ms = observation_ms - RANK_VELOCITY_HOURS * HOUR_MS
    values = {
        "observation": _fresh_bar(history, observation_ms, close_times=close_times),
        "observation_base": _fresh_bar(
            history, observation_ms - DAY_MS, close_times=close_times
        ),
        "prior": _fresh_bar(history, prior_ms, close_times=close_times),
        "prior_base": _fresh_bar(history, prior_ms - DAY_MS, close_times=close_times),
        "target_base": _fresh_bar(history, target_ms - DAY_MS, close_times=close_times),
        "target": _fresh_bar(history, target_ms, close_times=close_times),
    }
    if not all(values.values()):
        return None
    observation = values["observation"][1]
    observation_base = values["observation_base"][1]
    prior = values["prior"][1]
    prior_base = values["prior_base"][1]
    target_base = values["target_base"][1]
    target = values["target"][1]
    result = FeatureRow(
        symbol=symbol,
        current_return=_safe_return(observation.close, observation_base.close),
        static_target_return=_safe_return(observation.close, target_base.close),
        prior_current_return=_safe_return(prior.close, prior_base.close),
        target_return=_safe_return(target.close, target_base.close),
    )
    if not all(math.isfinite(value) for value in (
        result.current_return,
        result.static_target_return,
        result.prior_current_return,
        result.target_return,
    )):
        return None
    return result


def _snapshot_from_rows(
    rows: Sequence[FeatureRow],
    *,
    watchlist: set[str],
    local_day: date,
    top_n: int,
    min_market_symbols: int,
    min_watchlist_symbols: int,
    min_candidates: int,
) -> DaySnapshot | None:
    valid_watchlist = {row.symbol for row in rows if row.symbol in watchlist}
    if len(rows) < min_market_symbols or len(valid_watchlist) < min_watchlist_symbols:
        return None
    current_order = sorted(rows, key=lambda row: (-row.current_return, row.symbol))
    prior_order = sorted(rows, key=lambda row: (-row.prior_current_return, row.symbol))
    target_order = sorted(rows, key=lambda row: (-row.target_return, row.symbol))
    current_ranks = {row.symbol: index + 1 for index, row in enumerate(current_order)}
    prior_ranks = {row.symbol: index + 1 for index, row in enumerate(prior_order)}
    target_ranks = {row.symbol: index + 1 for index, row in enumerate(target_order)}
    target_top = {row.symbol for row in target_order[:top_n]}
    candidates = []
    for row in rows:
        current_rank = current_ranks[row.symbol]
        if row.symbol not in watchlist or current_rank <= top_n:
            continue
        prior_rank = prior_ranks[row.symbol]
        candidates.append(Candidate(
            symbol=row.symbol,
            current_return=row.current_return,
            static_target_return=row.static_target_return,
            prior_current_return=row.prior_current_return,
            current_market_rank=current_rank,
            prior_market_rank=prior_rank,
            rank_velocity_3h=prior_rank - current_rank,
            target_rank=target_ranks[row.symbol],
            is_target_top=row.symbol in target_top,
        ))
    if len(candidates) < min_candidates:
        return None
    entrant_symbols = frozenset(
        symbol for symbol in target_top
        if symbol in watchlist and current_ranks.get(symbol, 0) > top_n
    )
    return DaySnapshot(
        local_day=local_day.isoformat(),
        market_symbol_count=len(rows),
        watchlist_symbol_count=len(valid_watchlist),
        candidates=tuple(sorted(candidates, key=lambda row: row.symbol)),
        target_entrant_symbols=entrant_symbols,
    )


def build_day_snapshot(
    histories: Mapping[str, Sequence[Bar]],
    *,
    watchlist: set[str],
    local_day: date,
    timezone_name: str = "Europe/Budapest",
    observation_time: time = time(12, 15),
    target_time: time = time(23, 0),
    top_n: int = 50,
    min_market_symbols: int = 200,
    min_watchlist_symbols: int = 50,
    min_candidates: int = 10,
) -> DaySnapshot | None:
    tz = ZoneInfo(timezone_name)
    observation_ms = int(datetime.combine(
        local_day, observation_time, tzinfo=tz
    ).astimezone(timezone.utc).timestamp() * 1000)
    target_ms = int(datetime.combine(
        local_day, target_time, tzinfo=tz
    ).astimezone(timezone.utc).timestamp() * 1000)
    rows = []
    for symbol, history in histories.items():
        close_times = tuple(bar.close_ts_ms for bar in history)
        row = _extract_feature_row(
            symbol,
            history,
            observation_ms=observation_ms,
            target_ms=target_ms,
            close_times=close_times,
        )
        if row is not None:
            rows.append(row)
    return _snapshot_from_rows(
        rows,
        watchlist=watchlist,
        local_day=local_day,
        top_n=top_n,
        min_market_symbols=min_market_symbols,
        min_watchlist_symbols=min_watchlist_symbols,
        min_candidates=min_candidates,
    )


def replace_snapshot_day(snapshot: DaySnapshot, local_day: str) -> DaySnapshot:
    return replace(snapshot, local_day=local_day)


def _percentile(value: float, population: Sequence[float]) -> float:
    if len(population) <= 1:
        return 1.0
    return sum(other < value for other in population) / (len(population) - 1)


def rank_candidates(
    candidates: Sequence[Candidate],
    policy: str,
) -> list[RankedCandidate]:
    if policy not in POLICIES:
        raise ValidationError(f"unsupported policy: {policy}")
    static_values = tuple(row.static_target_return for row in candidates)
    velocity_values = tuple(float(row.rank_velocity_3h) for row in candidates)
    scored = [
        RankedCandidate(
            symbol=row.symbol,
            static_target_return=row.static_target_return,
            rank_velocity_3h=row.rank_velocity_3h,
            target_rank=row.target_rank,
            is_target_top=row.is_target_top,
            static_target_percentile=_percentile(row.static_target_return, static_values),
            rank_velocity_percentile=_percentile(float(row.rank_velocity_3h), velocity_values),
            composite_score=(
                _percentile(row.static_target_return, static_values)
                + RANK_VELOCITY_WEIGHT
                * _percentile(float(row.rank_velocity_3h), velocity_values)
            ),
        )
        for row in candidates
    ]
    if policy == "static_target":
        return sorted(scored, key=lambda row: (-row.static_target_return, row.symbol))
    return sorted(
        scored,
        key=lambda row: (-row.composite_score, -row.static_target_return, row.symbol),
    )


def _candidate_from_payload(row: Mapping[str, Any]) -> Candidate:
    current_rank = int(row["current_market_rank"])
    prior_rank = int(row["prior_market_rank"])
    velocity = int(row["rank_velocity_3h"])
    if velocity != prior_rank - current_rank:
        raise ValidationError("rank velocity does not match registered ranks")
    candidate = Candidate(
        symbol=str(row["symbol"]),
        current_return=float(row["current_return"]),
        static_target_return=float(row["static_target_return"]),
        prior_current_return=float(row["prior_current_return"]),
        current_market_rank=current_rank,
        prior_market_rank=prior_rank,
        rank_velocity_3h=velocity,
        target_rank=int(row["target_rank"]),
        is_target_top=bool(row["is_target_top"]),
    )
    numeric = (
        candidate.current_return,
        candidate.static_target_return,
        candidate.prior_current_return,
        float(candidate.current_market_rank),
        float(candidate.prior_market_rank),
        float(candidate.rank_velocity_3h),
        float(candidate.target_rank),
    )
    if not candidate.symbol or not all(math.isfinite(value) for value in numeric):
        raise ValidationError("invalid candidate feature")
    return candidate


def _bootstrap(values: Sequence[float], *, samples: int, seed: int) -> list[float] | None:
    if not values:
        return None
    rng = random.Random(seed)
    estimates = [
        mean(values[rng.randrange(len(values))] for _ in values)
        for _ in range(max(1, samples))
    ]
    estimates.sort()
    return [
        round(estimates[int(0.025 * (len(estimates) - 1))], 6),
        round(estimates[int(0.975 * (len(estimates) - 1))], 6),
    ]


def _safe_rate(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 6) if denominator else None


def _compute_slice(
    days: Sequence[Mapping[str, Any]],
    *,
    selection_size: int,
    bootstrap_samples: int,
    bootstrap_seed: int,
    minimum_practical_effect: float,
) -> dict[str, Any]:
    totals = {
        policy: {"top1": 0, "hits": 0, "selections": 0}
        for policy in POLICIES
    }
    daily_trace = []
    differences = []
    candidate_count = 0
    entrant_total = 0
    seen_days: set[str] = set()
    for day in days:
        local_day = str(day.get("local_day") or "")
        if not local_day or local_day in seen_days:
            raise ValidationError("missing or duplicate local day")
        seen_days.add(local_day)
        raw_candidates = day.get("candidates")
        if not isinstance(raw_candidates, list) or len(raw_candidates) < selection_size:
            raise ValidationError(f"insufficient candidates on {local_day}")
        candidates = tuple(_candidate_from_payload(row) for row in raw_candidates)
        symbols = [row.symbol for row in candidates]
        if len(symbols) != len(set(symbols)):
            raise ValidationError(f"duplicate symbol on {local_day}")
        target_symbols = sorted(str(value) for value in day.get("target_entrant_symbols") or [])
        labeled = sorted(row.symbol for row in candidates if row.is_target_top)
        if target_symbols != labeled:
            raise ValidationError(f"target entrant mismatch on {local_day}")
        candidate_count += len(candidates)
        entrant_total += len(target_symbols)
        precisions = {}
        trace: dict[str, Any] = {"local_day": local_day}
        for policy in POLICIES:
            selected = rank_candidates(candidates, policy)[:selection_size]
            hits = sum(int(row.is_target_top) for row in selected)
            totals[policy]["top1"] += int(bool(selected) and selected[0].is_target_top)
            totals[policy]["hits"] += hits
            totals[policy]["selections"] += len(selected)
            precision = hits / len(selected) if selected else 0.0
            precisions[policy] = precision
            trace[policy] = {
                "selected_symbols": [row.symbol for row in selected],
                "hits": hits,
                "selections": len(selected),
                "precision": round(precision, 6),
            }
        differences.append(
            precisions["static_target_rank_velocity_v1"] - precisions["static_target"]
        )
        daily_trace.append(trace)
    eligible_days = len(daily_trace)
    base_rate = entrant_total / candidate_count if candidate_count else None
    policies = {}
    for policy in POLICIES:
        total = totals[policy]
        precision = total["hits"] / total["selections"] if total["selections"] else None
        recall = total["hits"] / entrant_total if entrant_total else None
        policies[policy] = {
            "top1_hits": total["top1"],
            "top1_days": eligible_days,
            "top1_rate": _safe_rate(total["top1"], eligible_days),
            "top1_wilson95": wilson_interval(total["top1"], eligible_days),
            "topk_hits": total["hits"],
            "topk_selections": total["selections"],
            "topk_precision": round(precision, 6) if precision is not None else None,
            "topk_wilson95": wilson_interval(total["hits"], total["selections"]),
            "entrant_hits": total["hits"],
            "entrant_total": entrant_total,
            "entrant_recall": round(recall, 6) if recall is not None else None,
            "precision_lift_over_base": (
                round(precision / base_rate, 6)
                if precision is not None and base_rate not in (None, 0.0)
                else None
            ),
        }
    delta = mean(differences) if differences else 0.0
    sigma = stdev(differences) if len(differences) > 1 else 0.0
    mde = POWER_FACTOR_95_80 * sigma / math.sqrt(len(differences)) if differences else None
    return {
        "eligible_days": eligible_days,
        "candidate_count": candidate_count,
        "target_entrant_count": entrant_total,
        "candidate_base_rate": _safe_rate(entrant_total, candidate_count),
        "policies": policies,
        "paired_daily_precision_delta": round(delta, 6),
        "paired_bootstrap95": _bootstrap(
            differences, samples=bootstrap_samples, seed=bootstrap_seed
        ),
        "power": {
            "method_version": "paired_day_mean_mde_95_80_v1",
            "effective_sample_size": eligible_days,
            "paired_day_sd": round(sigma, 6),
            "mde": round(mde, 6) if mde is not None else None,
            "sesoi": minimum_practical_effect,
        },
        "daily_trace": daily_trace,
    }


def compute_metrics(snapshot: Mapping[str, Any], contract: Mapping[str, Any]) -> dict[str, Any]:
    days = snapshot.get("days")
    if not isinstance(days, list):
        raise ValidationError("snapshot days missing")
    protocol = contract.get("validation_protocol") or {}
    selection_size = int(protocol.get("selection_size") or 0)
    holdout_days = int(protocol.get("holdout_days") or 0)
    bootstrap_samples = int(protocol.get("bootstrap_samples") or 0)
    seed = int(protocol.get("bootstrap_seed") or 0)
    sesoi = float(contract.get("minimum_practical_effect") or 0.0)
    if selection_size <= 0 or holdout_days <= 0 or bootstrap_samples <= 0 or sesoi <= 0:
        raise ValidationError("invalid validation protocol")
    ordered_days = sorted(days, key=lambda row: str(row.get("local_day") or ""))
    holdout_count = min(holdout_days, len(ordered_days))
    development = ordered_days[:-holdout_count] if holdout_count else ordered_days
    holdout = ordered_days[-holdout_count:] if holdout_count else []
    kwargs = {
        "selection_size": selection_size,
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_seed": seed,
        "minimum_practical_effect": sesoi,
    }
    return {
        "metric_version": contract.get("objective_metric_version"),
        "full_period": _compute_slice(ordered_days, **kwargs),
        "development_period": _compute_slice(development, **kwargs) if development else None,
        "holdout_period": _compute_slice(holdout, **kwargs),
        "holdout_requested_days": holdout_days,
    }


def _decision(
    metrics: Mapping[str, Any],
    contract: Mapping[str, Any],
    *,
    decision_grade_input: bool,
) -> dict[str, Any]:
    full = metrics["full_period"]
    holdout = metrics["holdout_period"]
    candidate_name = "static_target_rank_velocity_v1"
    baseline_name = "static_target"
    candidate = full["policies"][candidate_name]
    baseline = full["policies"][baseline_name]
    holdout_candidate = holdout["policies"][candidate_name]
    holdout_baseline = holdout["policies"][baseline_name]
    sesoi = float(contract["minimum_practical_effect"])
    min_days = int(contract["validation_protocol"]["minimum_eligible_days"])
    min_holdout = int(contract["validation_protocol"]["minimum_holdout_days"])
    interval = full.get("paired_bootstrap95")
    mde = full.get("power", {}).get("mde")
    reasons = []
    if not decision_grade_input:
        reasons.append("snapshot_not_decision_grade")
    if full["eligible_days"] < min_days:
        reasons.append("minimum_eligible_days_not_reached")
    if holdout["eligible_days"] < min_holdout:
        reasons.append("minimum_holdout_days_not_reached")
    if full["paired_daily_precision_delta"] < sesoi:
        reasons.append("maximum_period_delta_below_sesoi")
    if candidate["entrant_recall"] is None or baseline["entrant_recall"] is None:
        reasons.append("maximum_period_recall_unknown")
    elif candidate["entrant_recall"] < baseline["entrant_recall"]:
        reasons.append("maximum_period_recall_worse")
    if holdout_candidate["entrant_recall"] is None or holdout_baseline["entrant_recall"] is None:
        reasons.append("holdout_recall_unknown")
    elif holdout_candidate["entrant_recall"] < holdout_baseline["entrant_recall"]:
        reasons.append("holdout_recall_worse")
    if holdout["paired_daily_precision_delta"] < 0.0:
        reasons.append("holdout_delta_negative")
    if candidate["precision_lift_over_base"] is None or candidate["precision_lift_over_base"] <= 1.0:
        reasons.append("precision_not_above_base_rate")
    if interval is None or interval[0] <= 0.0:
        reasons.append("maximum_period_interval_not_positive")
    if mde is None or mde > sesoi:
        reasons.append("maximum_period_underpowered")

    rejected = any(reason in reasons for reason in (
        "maximum_period_delta_below_sesoi",
        "maximum_period_recall_worse",
        "holdout_recall_worse",
        "holdout_delta_negative",
        "precision_not_above_base_rate",
    ))
    if not reasons:
        verdict = "SUPPORTED_FOR_FORWARD_SHADOW_ONLY"
    elif rejected:
        verdict = "REJECTED"
    else:
        verdict = "INCONCLUSIVE"
    return {
        "verdict": verdict,
        "reasons": reasons,
        "automatic_production_promotion": False,
    }


def normalized_snapshot_payload(
    snapshots: Sequence[DaySnapshot],
    *,
    provenance: Mapping[str, Any],
    watchlist_sha256: str,
    source_contract: Mapping[str, Any],
) -> dict[str, Any]:
    days = []
    for snapshot in sorted(snapshots, key=lambda row: row.local_day):
        days.append({
            "local_day": snapshot.local_day,
            "market_symbol_count": snapshot.market_symbol_count,
            "watchlist_symbol_count": snapshot.watchlist_symbol_count,
            "target_entrant_symbols": sorted(snapshot.target_entrant_symbols),
            "candidates": [asdict(candidate) for candidate in snapshot.candidates],
        })
    requested = int(provenance.get("requested_days") or 0)
    coverage = len(days) / requested if requested else 0.0
    return {
        "schema_version": 1,
        "snapshot_kind": SNAPSHOT_KIND,
        "decision_grade_input": bool(
            len(days) >= 30
            and coverage >= 0.95
            and int(provenance.get("malformed_file_count") or 0) == 0
            and provenance.get("used_content_hash")
        ),
        "source_contract": dict(source_contract),
        "provenance": {
            **dict(provenance),
            "calendar_coverage": round(coverage, 6),
            "watchlist_sha256": watchlist_sha256,
            "builder_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "git_commit": _git_commit(),
        },
        "days": days,
    }


def build_contract(
    snapshot: Mapping[str, Any],
    *,
    selection_size: int = 10,
    holdout_days: int = 90,
    bootstrap_samples: int = 5000,
    minimum_practical_effect: float = 0.02,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "hypothesis_id": "static-target-rank-velocity-v1",
        "action_layer": "WATCH",
        "objective_metric_version": "top50_top10_precision_delta_v1",
        "guardrail_versions": [
            "top50_entrant_recall_noninferiority_v1",
            "candidate_base_rate_lift_v1",
            "chronological_holdout_noninferiority_v1",
        ],
        "snapshot_sha256": _sha256_payload(snapshot),
        "feature_contract": {
            "rank_velocity_hours": RANK_VELOCITY_HOURS,
            "rank_velocity_weight": RANK_VELOCITY_WEIGHT,
            "formula": "static_target_percentile + 0.25*rank_velocity_3h_percentile",
            "target_features_forbidden": True,
        },
        "minimum_practical_effect": minimum_practical_effect,
        "validation_protocol": {
            "selection_size": selection_size,
            "holdout_days": holdout_days,
            "minimum_eligible_days": 30,
            "minimum_holdout_days": 90,
            "bootstrap_samples": bootstrap_samples,
            "bootstrap_seed": DEFAULT_BOOTSTRAP_SEED,
            "paired_unit": "local_day",
        },
        "production_effect": "none_research_only",
    }


def validate_payload(
    snapshot: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    snapshot_hash = _sha256_payload(snapshot)
    if snapshot_hash != contract.get("snapshot_sha256"):
        raise ValidationError("snapshot hash does not match frozen contract")
    feature_contract = contract.get("feature_contract") or {}
    if (
        int(feature_contract.get("rank_velocity_hours") or 0) != RANK_VELOCITY_HOURS
        or float(feature_contract.get("rank_velocity_weight") or 0.0) != RANK_VELOCITY_WEIGHT
    ):
        raise ValidationError("feature contract does not match frozen implementation")
    metrics = compute_metrics(snapshot, contract)
    return {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "snapshot_sha256": snapshot_hash,
        "contract_sha256": _sha256_payload(contract),
        "metrics": metrics,
        "decision": _decision(
            metrics,
            contract,
            decision_grade_input=bool(snapshot.get("decision_grade_input")),
        ),
        "production_effect": "none_research_only",
        "limitations": [
            "Historical exchangeInfo snapshots are unavailable; universe membership is inferred from bars.",
            "One-hour bars cannot reproduce intrabar order flow or depth trajectories.",
            "This WATCH ranking result is not portfolio profitability or permission to trade.",
        ],
    }


def build_snapshots_from_cache(
    selected_files: Mapping[str, Sequence[CacheFile]],
    *,
    watchlist: set[str],
    start_day: date,
    end_day: date,
    timezone_name: str,
    top_n: int,
    min_market_symbols: int,
    min_watchlist_symbols: int,
    min_candidates: int,
) -> tuple[list[DaySnapshot], dict[str, Any]]:
    tz = ZoneInfo(timezone_name)
    days = tuple(_date_range(start_day, end_day))
    clocks = {}
    for local_day in days:
        observation = datetime.combine(local_day, time(12, 15), tzinfo=tz).astimezone(timezone.utc)
        target = datetime.combine(local_day, time(23, 0), tzinfo=tz).astimezone(timezone.utc)
        clocks[local_day] = (int(observation.timestamp() * 1000), int(target.timestamp() * 1000))
    by_day: dict[date, list[FeatureRow]] = defaultdict(list)
    used_files: list[Path] = []
    malformed_files: list[str] = []
    symbols_with_rows = 0
    for symbol in sorted(selected_files):
        batches = []
        for cache_file in selected_files[symbol]:
            try:
                batches.append(_load_json_rows(cache_file.path))
                used_files.append(cache_file.path)
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                malformed_files.append(f"{cache_file.path}: {type(exc).__name__}: {exc}")
        history = merge_binance_rows(batches)
        if not history:
            continue
        symbols_with_rows += 1
        close_times = tuple(bar.close_ts_ms for bar in history)
        for local_day, (observation_ms, target_ms) in clocks.items():
            row = _extract_feature_row(
                symbol,
                history,
                observation_ms=observation_ms,
                target_ms=target_ms,
                close_times=close_times,
            )
            if row is not None:
                by_day[local_day].append(row)
    snapshots = []
    rejected_days = []
    for local_day in days:
        snapshot = _snapshot_from_rows(
            by_day.get(local_day, ()),
            watchlist=watchlist,
            local_day=local_day,
            top_n=top_n,
            min_market_symbols=min_market_symbols,
            min_watchlist_symbols=min_watchlist_symbols,
            min_candidates=min_candidates,
        )
        if snapshot is None:
            rejected_days.append(local_day.isoformat())
        else:
            snapshots.append(snapshot)
    digest = hashlib.sha256()
    for path in sorted(set(used_files), key=str):
        digest.update(str(path.resolve()).encode("utf-8"))
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
    return snapshots, {
        "cache_symbol_count": len(selected_files),
        "symbols_with_rows": symbols_with_rows,
        "used_file_count": len(set(used_files)),
        "used_content_hash": digest.hexdigest(),
        "malformed_file_count": len(malformed_files),
        "malformed_files": malformed_files[:20],
        "requested_days": len(days),
        "eligible_days": len(snapshots),
        "rejected_days": rejected_days,
    }


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _parse_day(value: str) -> date:
    return date.fromisoformat(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-cache", type=Path, default=DEFAULT_LEGACY_CACHE)
    parser.add_argument("--tail-cache", type=Path, default=DEFAULT_TAIL_CACHE)
    parser.add_argument("--watchlist", type=Path, default=DEFAULT_WATCHLIST)
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--snapshot-output", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--contract-output", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--timezone", default="Europe/Budapest")
    parser.add_argument("--start-day", type=_parse_day)
    parser.add_argument("--end-day", type=_parse_day)
    parser.add_argument("--top-n", type=int, default=50)
    parser.add_argument("--selection-size", type=int, default=10)
    parser.add_argument("--min-market-symbols", type=int, default=200)
    parser.add_argument("--min-watchlist-symbols", type=int, default=50)
    parser.add_argument("--min-candidates", type=int, default=10)
    parser.add_argument("--holdout-days", type=int, default=90)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    selected = discover_cache_files((args.legacy_cache, args.tail_cache))
    default_start, default_end = _default_period(selected, args.timezone)
    start_day = args.start_day or default_start
    end_day = args.end_day or default_end
    watchlist_raw = args.watchlist.read_bytes()
    watchlist = set(json.loads(watchlist_raw.decode("utf-8-sig")))
    snapshots, provenance = build_snapshots_from_cache(
        selected,
        watchlist=watchlist,
        start_day=start_day,
        end_day=end_day,
        timezone_name=args.timezone,
        top_n=args.top_n,
        min_market_symbols=args.min_market_symbols,
        min_watchlist_symbols=args.min_watchlist_symbols,
        min_candidates=args.min_candidates,
    )
    source_contract = {
        "source": "local Binance Spot 1h cache",
        "timezone": args.timezone,
        "observation_time_local": "12:15",
        "target_time_local": "23:00",
        "start_day": start_day.isoformat(),
        "end_day": end_day.isoformat(),
        "top_n": args.top_n,
        "selection_size": args.selection_size,
        "min_market_symbols": args.min_market_symbols,
        "min_watchlist_symbols": args.min_watchlist_symbols,
        "min_candidates": args.min_candidates,
    }
    snapshot = normalized_snapshot_payload(
        snapshots,
        provenance=provenance,
        watchlist_sha256=hashlib.sha256(watchlist_raw).hexdigest(),
        source_contract=source_contract,
    )
    contract = build_contract(
        snapshot,
        selection_size=args.selection_size,
        holdout_days=args.holdout_days,
        bootstrap_samples=args.bootstrap_samples,
    )
    result = validate_payload(snapshot, contract)
    from verify_static_target_rank_velocity import verify_payload

    verification = verify_payload(snapshot, contract, result)
    if not verification["valid"]:
        result["status"] = "invalid_result"
        result["decision"] = {
            "verdict": "INVALID_RESULT",
            "reasons": verification["errors"],
            "automatic_production_promotion": False,
        }
    result["verification"] = verification
    _write_json(args.snapshot_output, snapshot)
    _write_json(args.contract_output, contract)
    _write_json(args.output, result)
    full = result["metrics"]["full_period"]
    holdout = result["metrics"]["holdout_period"]
    baseline = full["policies"]["static_target"]
    candidate = full["policies"]["static_target_rank_velocity_v1"]
    print(
        f"status={result['status']} verdict={result['decision']['verdict']} "
        f"period={start_day}..{end_day} eligible={full['eligible_days']}"
    )
    print(
        f"full top10={baseline['topk_hits']}/{baseline['topk_selections']}"
        f"->{candidate['topk_hits']}/{candidate['topk_selections']} "
        f"delta={full['paired_daily_precision_delta']:.6f} "
        f"ci={full['paired_bootstrap95']} mde={full['power']['mde']}"
    )
    print(
        f"holdout days={holdout['eligible_days']} "
        f"delta={holdout['paired_daily_precision_delta']:.6f} "
        f"ci={holdout['paired_bootstrap95']}"
    )
    print(f"verification={verification['status']} report={args.output}")
    return 0 if verification["valid"] and snapshots else 2


if __name__ == "__main__":
    raise SystemExit(main())
