"""Independent verifier for the Phase 1 static-target result bundle."""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import random
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Mapping, Sequence


POWER_FACTOR_95_80 = 2.8015852181129683


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
        raise ValueError(f"{path.name} must contain an object")
    return payload, raw


def _rate(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 6) if denominator else None


def _bootstrap(values: Sequence[float], samples: int, seed: int) -> list[float]:
    rng = random.Random(seed)
    estimates = [
        mean(values[rng.randrange(len(values))] for _ in values)
        for _ in range(max(1, samples))
    ]
    estimates.sort()
    return [
        round(estimates[int(0.025 * (len(estimates) - 1))], 6),
        round(estimates[int(0.975 * (len(estimates) - 1))], 6),
    ] if values else []


def _recompute(snapshot: Mapping[str, Any], contract: Mapping[str, Any]) -> dict[str, Any]:
    protocol = contract["validation_protocol"]
    k = int(protocol["selection_size"])
    samples = int(protocol["bootstrap_samples"])
    seed = int(protocol["bootstrap_seed"])
    totals = {
        "current_rank": {"top1": 0, "hits": 0, "n": 0},
        "static_target": {"top1": 0, "hits": 0, "n": 0},
    }
    daily = []
    diffs = []
    candidate_count = 0
    entrant_total = 0
    seen_days = set()
    for day in snapshot["days"]:
        day_name = str(day["local_day"])
        if day_name in seen_days:
            raise ValueError("duplicate local day")
        seen_days.add(day_name)
        candidates = list(day["candidates"])
        if not candidates:
            continue
        symbols = [str(row["symbol"]) for row in candidates]
        if len(symbols) != len(set(symbols)):
            raise ValueError("duplicate candidate symbol")
        target_symbols = sorted(str(value) for value in day["target_entrant_symbols"])
        labeled = sorted(str(row["symbol"]) for row in candidates if bool(row["is_target_top"]))
        if target_symbols != labeled:
            raise ValueError("target entrant mismatch")
        candidate_count += len(candidates)
        entrant_total += len(target_symbols)
        trace = {"local_day": day_name}
        precision = {}
        for variant, field in (
            ("current_rank", "current_return"),
            ("static_target", "static_target_return"),
        ):
            for row in candidates:
                if not math.isfinite(float(row[field])):
                    raise ValueError("non-finite ranking feature")
            chosen = sorted(candidates, key=lambda row: (-float(row[field]), str(row["symbol"])))[:k]
            hits = sum(int(bool(row["is_target_top"])) for row in chosen)
            totals[variant]["top1"] += int(bool(chosen) and bool(chosen[0]["is_target_top"]))
            totals[variant]["hits"] += hits
            totals[variant]["n"] += len(chosen)
            precision[variant] = hits / len(chosen) if chosen else 0.0
            trace[variant] = {
                "selected_symbols": [str(row["symbol"]) for row in chosen],
                "hits": hits,
                "selections": len(chosen),
                "precision": round(precision[variant], 6),
            }
        diffs.append(precision["static_target"] - precision["current_rank"])
        daily.append(trace)
    base_rate = entrant_total / candidate_count if candidate_count else None
    policies = {}
    for variant in ("current_rank", "static_target"):
        row = totals[variant]
        p = row["hits"] / row["n"] if row["n"] else None
        recall = row["hits"] / entrant_total if entrant_total else None
        policies[variant] = {
            "top1_hits": row["top1"],
            "top1_days": len(daily),
            "top1_rate": _rate(row["top1"], len(daily)),
            "topk_hits": row["hits"],
            "topk_selections": row["n"],
            "topk_precision": round(p, 6) if p is not None else None,
            "entrant_hits": row["hits"],
            "entrant_total": entrant_total,
            "entrant_recall": round(recall, 6) if recall is not None else None,
            "precision_lift_over_base": (
                round(p / base_rate, 6) if p is not None and base_rate not in (None, 0.0) else None
            ),
        }
    sigma = stdev(diffs) if len(diffs) > 1 else 0.0
    mde = POWER_FACTOR_95_80 * sigma / math.sqrt(len(diffs)) if diffs else None
    return {
        "metric_version": contract.get("objective_metric_version"),
        "eligible_days": len(daily),
        "candidate_count": candidate_count,
        "target_entrant_count": entrant_total,
        "candidate_base_rate": _rate(entrant_total, candidate_count),
        "policies": policies,
        "paired_daily_precision_delta": round(mean(diffs), 6) if diffs else 0.0,
        "paired_bootstrap95": _bootstrap(diffs, samples, seed),
        "power": {
            "method_version": "paired_day_mean_mde_95_80_v1",
            "effective_sample_size": len(daily),
            "paired_day_sd": round(sigma, 6),
            "mde": round(mde, 6) if mde is not None else None,
            "sesoi": float(contract.get("minimum_practical_effect") or 0.0),
        },
        "daily_trace": daily,
    }


def verify(
    snapshot_path: Path,
    manifest_path: Path,
    contract_path: Path,
    result_path: Path,
    key: bytes,
) -> dict[str, Any]:
    errors = []
    snapshot, snapshot_raw = _load(snapshot_path)
    manifest, manifest_raw = _load(manifest_path)
    contract, contract_raw = _load(contract_path)
    result, _ = _load(result_path)
    attestation = result.get("attestation") if isinstance(result.get("attestation"), dict) else {}
    unsigned = dict(result)
    unsigned.pop("attestation", None)
    expected_signature = hmac.new(key, _canonical_bytes(unsigned), hashlib.sha256).hexdigest()
    if attestation.get("algorithm") != "HMAC-SHA256" or not hmac.compare_digest(
        str(attestation.get("signature") or ""), expected_signature
    ):
        errors.append("attestation")
    if result.get("snapshot_sha256") != _sha256(snapshot_raw):
        errors.append("snapshot_sha256")
    if result.get("snapshot_manifest_sha256") != _sha256(manifest_raw):
        errors.append("snapshot_manifest_sha256")
    if result.get("contract_sha256") != _sha256(contract_raw):
        errors.append("contract_sha256")
    if contract.get("snapshot_sha256") != _sha256(snapshot_raw):
        errors.append("registered_snapshot_sha256")
    if contract.get("snapshot_manifest_sha256") != _sha256(manifest_raw):
        errors.append("registered_snapshot_manifest_sha256")
    if manifest.get("snapshot_sha256") != _sha256(snapshot_raw):
        errors.append("manifest_snapshot_sha256")
    if manifest.get("snapshot_kind") != snapshot.get("snapshot_kind"):
        errors.append("manifest_snapshot_kind")
    if manifest.get("metric_version") != contract.get("objective_metric_version"):
        errors.append("manifest_metric_version")
    provenance = snapshot.get("provenance") if isinstance(snapshot.get("provenance"), dict) else {}
    required_manifest = {
        "policy_epoch": manifest.get("policy_epoch") or provenance.get("policy_epoch"),
        "source_cache_hash": manifest.get("source_cache_hash") or provenance.get("used_content_hash"),
    }
    if bool(snapshot.get("decision_grade_input")):
        required_manifest.update({
            "phase0_completion_sha256": manifest.get("phase0_completion_sha256"),
            "builder_sha256": manifest.get("builder_sha256"),
            "git_commit": manifest.get("git_commit"),
        })
    for name, value in required_manifest.items():
        expected_length = 40 if name == "git_commit" else 64 if name.endswith("sha256") or name.endswith("hash") else 1
        if not isinstance(value, str) or len(value) < expected_length:
            errors.append(f"manifest_{name}")
    if result.get("attempt_id") != contract.get("attempt_id"):
        errors.append("attempt_id")
    recomputed = _recompute(snapshot, contract)
    if _canonical_bytes(result.get("metrics")) != _canonical_bytes(recomputed):
        errors.append("metrics")
    valid = not errors
    return {
        "schema_version": 1,
        "status": "VERIFIED_RESULT" if valid else "INVALID_RESULT",
        "valid": valid,
        "errors": errors,
        "attempt_id": contract.get("attempt_id"),
        "snapshot_sha256": _sha256(snapshot_raw),
        "contract_sha256": _sha256(contract_raw),
        "validator_result_sha256": _sha256(result_path.read_bytes()),
        "decision_grade": bool(valid and snapshot.get("decision_grade_input")),
        "verified_metrics": recomputed,
    }
