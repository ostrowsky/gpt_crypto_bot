"""Independent verifier for static-target rank-velocity validation evidence."""

from __future__ import annotations

import hashlib
import json
import math
import random
from statistics import mean, stdev
from typing import Any, Mapping, Sequence


POLICIES = ("static_target", "static_target_rank_velocity_v1")
SNAPSHOT_KIND = "static_target_rank_velocity_normalized_v1"
RANK_VELOCITY_HOURS = 3
RANK_VELOCITY_WEIGHT = 0.25
POWER_FACTOR_95_80 = 2.8015852181129683


class VerificationError(ValueError):
    pass


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _safe_rate(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 6) if denominator else None


def _wilson(hits: int, total: int, z: float = 1.959963984540054) -> list[float] | None:
    if total <= 0:
        return None
    p = hits / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denominator
    margin = z * math.sqrt(
        (p * (1.0 - p) + z * z / (4.0 * total)) / total
    ) / denominator
    return [round(max(0.0, center - margin), 6), round(min(1.0, center + margin), 6)]


def _percentile(value: float, population: Sequence[float]) -> float:
    if len(population) <= 1:
        return 1.0
    return sum(other < value for other in population) / (len(population) - 1)


def _normalize_candidates(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    normalized = []
    symbols: set[str] = set()
    for row in rows:
        symbol = str(row.get("symbol") or "")
        if not symbol or symbol in symbols:
            raise VerificationError("missing or duplicate candidate symbol")
        symbols.add(symbol)
        current_rank = int(row["current_market_rank"])
        prior_rank = int(row["prior_market_rank"])
        velocity = int(row["rank_velocity_3h"])
        if velocity != prior_rank - current_rank:
            raise VerificationError("rank velocity mismatch")
        values = {
            "symbol": symbol,
            "current_return": float(row["current_return"]),
            "static_target_return": float(row["static_target_return"]),
            "prior_current_return": float(row["prior_current_return"]),
            "current_market_rank": current_rank,
            "prior_market_rank": prior_rank,
            "rank_velocity_3h": velocity,
            "target_rank": int(row["target_rank"]),
            "is_target_top": bool(row["is_target_top"]),
        }
        numeric = [value for key, value in values.items() if key not in ("symbol", "is_target_top")]
        if not all(math.isfinite(float(value)) for value in numeric):
            raise VerificationError("non-finite candidate value")
        normalized.append(values)
    return normalized


def _rank(rows: Sequence[Mapping[str, Any]], policy: str) -> list[dict[str, Any]]:
    static_values = tuple(float(row["static_target_return"]) for row in rows)
    velocity_values = tuple(float(row["rank_velocity_3h"]) for row in rows)
    scored = []
    for row in rows:
        static_percentile = _percentile(float(row["static_target_return"]), static_values)
        velocity_percentile = _percentile(float(row["rank_velocity_3h"]), velocity_values)
        scored.append({
            **dict(row),
            "static_target_percentile": static_percentile,
            "rank_velocity_percentile": velocity_percentile,
            "composite_score": static_percentile + RANK_VELOCITY_WEIGHT * velocity_percentile,
        })
    if policy == "static_target":
        return sorted(scored, key=lambda row: (-row["static_target_return"], row["symbol"]))
    if policy == "static_target_rank_velocity_v1":
        return sorted(
            scored,
            key=lambda row: (
                -row["composite_score"],
                -row["static_target_return"],
                row["symbol"],
            ),
        )
    raise VerificationError(f"unsupported policy: {policy}")


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


def _compute_slice(
    days: Sequence[Mapping[str, Any]],
    *,
    selection_size: int,
    bootstrap_samples: int,
    bootstrap_seed: int,
    minimum_practical_effect: float,
) -> dict[str, Any]:
    totals = {policy: {"top1": 0, "hits": 0, "selections": 0} for policy in POLICIES}
    differences = []
    daily_trace = []
    candidate_count = 0
    entrant_total = 0
    seen_days: set[str] = set()
    for day in days:
        local_day = str(day.get("local_day") or "")
        if not local_day or local_day in seen_days:
            raise VerificationError("missing or duplicate local day")
        seen_days.add(local_day)
        raw = day.get("candidates")
        if not isinstance(raw, list) or len(raw) < selection_size:
            raise VerificationError("insufficient candidates")
        candidates = _normalize_candidates(raw)
        target_symbols = sorted(str(value) for value in day.get("target_entrant_symbols") or [])
        labeled = sorted(row["symbol"] for row in candidates if row["is_target_top"])
        if target_symbols != labeled:
            raise VerificationError("target entrant mismatch")
        candidate_count += len(candidates)
        entrant_total += len(target_symbols)
        trace: dict[str, Any] = {"local_day": local_day}
        precisions = {}
        for policy in POLICIES:
            selected = _rank(candidates, policy)[:selection_size]
            hits = sum(int(row["is_target_top"]) for row in selected)
            totals[policy]["top1"] += int(bool(selected) and selected[0]["is_target_top"])
            totals[policy]["hits"] += hits
            totals[policy]["selections"] += len(selected)
            precision = hits / len(selected) if selected else 0.0
            precisions[policy] = precision
            trace[policy] = {
                "selected_symbols": [row["symbol"] for row in selected],
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
            "top1_wilson95": _wilson(total["top1"], eligible_days),
            "topk_hits": total["hits"],
            "topk_selections": total["selections"],
            "topk_precision": round(precision, 6) if precision is not None else None,
            "topk_wilson95": _wilson(total["hits"], total["selections"]),
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


def _compute_metrics(snapshot: Mapping[str, Any], contract: Mapping[str, Any]) -> dict[str, Any]:
    days = snapshot.get("days")
    if not isinstance(days, list):
        raise VerificationError("snapshot days missing")
    protocol = contract.get("validation_protocol") or {}
    selection_size = int(protocol.get("selection_size") or 0)
    holdout_days = int(protocol.get("holdout_days") or 0)
    bootstrap_samples = int(protocol.get("bootstrap_samples") or 0)
    seed = int(protocol.get("bootstrap_seed") or 0)
    sesoi = float(contract.get("minimum_practical_effect") or 0.0)
    if selection_size <= 0 or holdout_days <= 0 or bootstrap_samples <= 0 or sesoi <= 0:
        raise VerificationError("invalid protocol")
    ordered = sorted(days, key=lambda row: str(row.get("local_day") or ""))
    holdout_count = min(holdout_days, len(ordered))
    development = ordered[:-holdout_count] if holdout_count else ordered
    holdout = ordered[-holdout_count:] if holdout_count else []
    kwargs = {
        "selection_size": selection_size,
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_seed": seed,
        "minimum_practical_effect": sesoi,
    }
    return {
        "metric_version": contract.get("objective_metric_version"),
        "full_period": _compute_slice(ordered, **kwargs),
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
    candidate = full["policies"]["static_target_rank_velocity_v1"]
    baseline = full["policies"]["static_target"]
    holdout_candidate = holdout["policies"]["static_target_rank_velocity_v1"]
    holdout_baseline = holdout["policies"]["static_target"]
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


def verify_payload(
    snapshot: Mapping[str, Any],
    contract: Mapping[str, Any],
    result: Mapping[str, Any],
) -> dict[str, Any]:
    errors = []
    snapshot_hash = _sha256(snapshot)
    contract_hash = _sha256(contract)
    if snapshot.get("snapshot_kind") != SNAPSHOT_KIND:
        errors.append("snapshot_kind")
    if contract.get("snapshot_sha256") != snapshot_hash:
        errors.append("registered_snapshot_sha256")
    if result.get("snapshot_sha256") != snapshot_hash:
        errors.append("result_snapshot_sha256")
    if result.get("contract_sha256") != contract_hash:
        errors.append("result_contract_sha256")
    feature = contract.get("feature_contract") or {}
    if int(feature.get("rank_velocity_hours") or 0) != RANK_VELOCITY_HOURS:
        errors.append("rank_velocity_hours")
    if float(feature.get("rank_velocity_weight") or 0.0) != RANK_VELOCITY_WEIGHT:
        errors.append("rank_velocity_weight")
    try:
        metrics = _compute_metrics(snapshot, contract)
        decision = _decision(
            metrics,
            contract,
            decision_grade_input=bool(snapshot.get("decision_grade_input")),
        )
    except (KeyError, TypeError, ValueError, VerificationError) as exc:
        return {
            "schema_version": 1,
            "status": "INVALID_RESULT",
            "valid": False,
            "errors": [*errors, f"recompute:{type(exc).__name__}:{exc}"],
            "snapshot_sha256": snapshot_hash,
            "contract_sha256": contract_hash,
        }
    if _canonical_bytes(result.get("metrics")) != _canonical_bytes(metrics):
        errors.append("metrics")
    if _canonical_bytes(result.get("decision")) != _canonical_bytes(decision):
        errors.append("decision")
    if result.get("production_effect") != "none_research_only":
        errors.append("production_effect")
    return {
        "schema_version": 1,
        "status": "VERIFIED_RESULT" if not errors else "INVALID_RESULT",
        "valid": not errors,
        "errors": errors,
        "snapshot_sha256": snapshot_hash,
        "contract_sha256": contract_hash,
        "verified_metrics": metrics,
        "verified_decision": decision,
    }
