"""Deterministic validator for the Phase 1 static-target WATCH experiment."""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import math
import random
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Mapping, Sequence


VALIDATOR_ID = "static_target_top50_validator_v1"
SCHEMA_VERSION = 1
POWER_FACTOR_95_80 = 2.8015852181129683


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


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _load(path: Path) -> tuple[dict[str, Any], bytes]:
    raw = path.read_bytes()
    payload = json.loads(raw.decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValidationError(f"{path.name} must contain an object")
    return payload, raw


def _bootstrap_delta(values: Sequence[float], *, samples: int, seed: int) -> list[float]:
    if not values:
        return []
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


def compute_metrics(snapshot: Mapping[str, Any], contract: Mapping[str, Any]) -> dict[str, Any]:
    protocol = contract.get("validation_protocol") or {}
    selection_size = int(protocol.get("selection_size") or 0)
    bootstrap_samples = int(protocol.get("bootstrap_samples") or 0)
    seed = int(protocol.get("bootstrap_seed") or 0)
    if selection_size <= 0 or bootstrap_samples <= 0:
        raise ValidationError("invalid registered selection/bootstrap contract")
    days = snapshot.get("days")
    if not isinstance(days, list):
        raise ValidationError("snapshot days are missing")

    aggregate = {
        "current_rank": {"top1_hits": 0, "topk_hits": 0, "selections": 0},
        "static_target": {"top1_hits": 0, "topk_hits": 0, "selections": 0},
    }
    daily_trace: list[dict[str, Any]] = []
    candidate_count = 0
    entrant_total = 0
    differences: list[float] = []
    seen_days: set[str] = set()
    for day in days:
        if not isinstance(day, dict):
            raise ValidationError("day row must be an object")
        local_day = str(day.get("local_day") or "")
        if not local_day or local_day in seen_days:
            raise ValidationError("missing or duplicate local day")
        seen_days.add(local_day)
        candidates = day.get("candidates")
        if not isinstance(candidates, list) or not candidates:
            continue
        symbols: set[str] = set()
        normalized = []
        for row in candidates:
            symbol = str(row.get("symbol") or "")
            if not symbol or symbol in symbols:
                raise ValidationError(f"invalid candidate identity on {local_day}")
            symbols.add(symbol)
            current_return = float(row["current_return"])
            static_return = float(row["static_target_return"])
            if not math.isfinite(current_return) or not math.isfinite(static_return):
                raise ValidationError(f"non-finite feature on {local_day}/{symbol}")
            normalized.append({
                "symbol": symbol,
                "current_return": current_return,
                "static_target_return": static_return,
                "is_target_top": bool(row.get("is_target_top")),
            })
        target_symbols = sorted(str(value) for value in day.get("target_entrant_symbols") or [])
        observed_targets = sorted(row["symbol"] for row in normalized if row["is_target_top"])
        if target_symbols != observed_targets:
            raise ValidationError(f"target denominator mismatch on {local_day}")
        candidate_count += len(normalized)
        entrant_total += len(target_symbols)
        day_metrics: dict[str, Any] = {"local_day": local_day}
        precisions = {}
        for variant, key_name in (
            ("current_rank", "current_return"),
            ("static_target", "static_target_return"),
        ):
            ordered = sorted(normalized, key=lambda row: (-row[key_name], row["symbol"]))
            selected = ordered[:selection_size]
            hits = sum(int(row["is_target_top"]) for row in selected)
            aggregate[variant]["top1_hits"] += int(bool(selected) and selected[0]["is_target_top"])
            aggregate[variant]["topk_hits"] += hits
            aggregate[variant]["selections"] += len(selected)
            precision = hits / len(selected) if selected else 0.0
            precisions[variant] = precision
            day_metrics[variant] = {
                "selected_symbols": [row["symbol"] for row in selected],
                "hits": hits,
                "selections": len(selected),
                "precision": round(precision, 6),
            }
        differences.append(precisions["static_target"] - precisions["current_rank"])
        daily_trace.append(day_metrics)

    eligible_days = len(daily_trace)
    base_hits = entrant_total
    base_rate = base_hits / candidate_count if candidate_count else None
    policies = {}
    for variant in ("current_rank", "static_target"):
        row = aggregate[variant]
        precision = row["topk_hits"] / row["selections"] if row["selections"] else None
        recall = row["topk_hits"] / entrant_total if entrant_total else None
        policies[variant] = {
            "top1_hits": row["top1_hits"],
            "top1_days": eligible_days,
            "top1_rate": _safe_rate(row["top1_hits"], eligible_days),
            "topk_hits": row["topk_hits"],
            "topk_selections": row["selections"],
            "topk_precision": round(precision, 6) if precision is not None else None,
            "entrant_hits": row["topk_hits"],
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
        "metric_version": contract.get("objective_metric_version"),
        "eligible_days": eligible_days,
        "candidate_count": candidate_count,
        "target_entrant_count": entrant_total,
        "candidate_base_rate": _safe_rate(base_hits, candidate_count),
        "policies": policies,
        "paired_daily_precision_delta": round(delta, 6),
        "paired_bootstrap95": _bootstrap_delta(
            differences, samples=bootstrap_samples, seed=seed
        ),
        "power": {
            "method_version": "paired_day_mean_mde_95_80_v1",
            "effective_sample_size": eligible_days,
            "paired_day_sd": round(sigma, 6),
            "mde": round(mde, 6) if mde is not None else None,
            "sesoi": float(contract.get("minimum_practical_effect") or 0.0),
        },
        "daily_trace": daily_trace,
    }


def validate(
    snapshot_path: Path,
    manifest_path: Path,
    contract_path: Path,
    key: bytes,
    *,
    corrupt_result: bool = False,
) -> dict[str, Any]:
    snapshot, snapshot_raw = _load(snapshot_path)
    manifest, manifest_raw = _load(manifest_path)
    contract, contract_raw = _load(contract_path)
    expected_snapshot_hash = str(contract.get("snapshot_sha256") or "")
    if _sha256(snapshot_raw) != expected_snapshot_hash:
        raise ValidationError("snapshot hash does not match contract")
    if manifest.get("snapshot_sha256") != expected_snapshot_hash:
        raise ValidationError("manifest snapshot hash does not match contract")
    if _sha256(manifest_raw) != contract.get("snapshot_manifest_sha256"):
        raise ValidationError("manifest hash does not match contract")
    metrics = compute_metrics(snapshot, contract)
    if corrupt_result:
        metrics["paired_daily_precision_delta"] = round(
            float(metrics["paired_daily_precision_delta"]) + 0.01, 6
        )
    unsigned = {
        "schema_version": SCHEMA_VERSION,
        "validator_id": VALIDATOR_ID,
        "attempt_id": contract.get("attempt_id"),
        "hypothesis_id": contract.get("hypothesis_id"),
        "snapshot_sha256": _sha256(snapshot_raw),
        "snapshot_manifest_sha256": _sha256(manifest_raw),
        "contract_sha256": _sha256(contract_raw),
        "decision_grade_input": bool(snapshot.get("decision_grade_input")),
        "metrics": metrics,
    }
    signature = hmac.new(key, _canonical_bytes(unsigned), hashlib.sha256).hexdigest()
    return {**unsigned, "attestation": {"algorithm": "HMAC-SHA256", "signature": signature}}


def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--key", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--corrupt-result", action="store_true")
    args = parser.parse_args()
    key = bytes.fromhex(args.key.read_text(encoding="ascii"))
    payload = validate(
        args.snapshot,
        args.manifest,
        args.contract,
        key,
        corrupt_result=args.corrupt_result,
    )
    raw = _canonical_bytes(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as handle:
        handle.write(raw)
        handle.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
