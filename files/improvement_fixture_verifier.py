"""Independent raw-snapshot verifier for the Phase -1 smoke validator.

Do not import the validator here.  A second implementation is the control: the
validator trace and summary are untrusted comparison evidence.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
from pathlib import Path
from typing import Any


def _canonical_bytes(payload: dict[str, Any]) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _load(path: Path) -> tuple[dict[str, Any], bytes]:
    raw = Path(path).read_bytes()
    payload = json.loads(raw.decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected object in {path}")
    return payload, raw


def _policy_threshold(contract: dict[str, Any], key: str) -> float:
    policy = contract[key]
    if policy.get("kind") != "threshold":
        raise ValueError(f"unsupported policy: {key}")
    threshold = float(policy["threshold"])
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError(f"invalid threshold: {key}")
    return threshold


def _close(left: Any, right: float) -> bool:
    return isinstance(left, (int, float)) and not isinstance(left, bool) and math.isclose(
        float(left), right, rel_tol=0.0, abs_tol=1e-12
    )


def verify_result_bundle(
    snapshot_path: Path,
    contract_path: Path,
    result_bundle: dict[str, Any],
    attestation_key: bytes,
) -> dict[str, Any]:
    """Rebuild policy decisions from raw inputs and compare every derivative."""

    snapshot, snapshot_raw = _load(Path(snapshot_path))
    contract, contract_raw = _load(Path(contract_path))
    rows = snapshot.get("rows")
    if not isinstance(rows, list) or not rows or len(rows) > 64:
        raise ValueError("raw snapshot row count must be in 1..64")

    errors: list[str] = []
    attestation = result_bundle.get("attestation")
    signed_payload = dict(result_bundle)
    signed_payload.pop("attestation", None)
    expected_signature = hmac.new(
        attestation_key, _canonical_bytes(signed_payload), hashlib.sha256
    ).hexdigest()
    attestation_valid = bool(
        isinstance(attestation, dict)
        and attestation.get("algorithm") == "hmac-sha256"
        and attestation.get("key_id") == contract.get("attestation_key_id")
        and isinstance(attestation.get("signature"), str)
        and hmac.compare_digest(attestation["signature"], expected_signature)
    )
    if not attestation_valid:
        errors.append("attestation")

    identity_checks = {
        "schema_version": 1,
        "attempt_id": contract.get("attempt_id"),
        "hypothesis_id": contract.get("hypothesis_id"),
        "validator_id": contract.get("validator_id"),
        "fixture_id": snapshot.get("fixture_id"),
        "snapshot_sha256": _sha256(snapshot_raw),
        "contract_sha256": _sha256(contract_raw),
        "metric": "accuracy",
        "decision_grade": False,
        "trading_conclusion_allowed": False,
    }
    for field, expected in identity_checks.items():
        if result_bundle.get(field) != expected:
            errors.append(field)

    baseline_threshold = _policy_threshold(contract, "baseline_policy")
    candidate_threshold = _policy_threshold(contract, "candidate_policy")
    reconstructed_trace: list[dict[str, Any]] = []
    baseline_correct = 0
    candidate_correct = 0
    seen: set[str] = set()
    for raw_row in rows:
        if not isinstance(raw_row, dict) or set(raw_row) != {"row_id", "score", "label"}:
            raise ValueError("invalid raw row")
        row_id = raw_row["row_id"]
        score = float(raw_row["score"])
        label = raw_row["label"]
        if not isinstance(row_id, str) or not row_id or row_id in seen:
            raise ValueError("raw row ids must be unique")
        if not math.isfinite(score) or not 0.0 <= score <= 1.0 or label not in (0, 1):
            raise ValueError("invalid raw row value")
        seen.add(row_id)
        baseline_prediction = 1 if score >= baseline_threshold else 0
        candidate_prediction = 1 if score >= candidate_threshold else 0
        baseline_correct += 1 if baseline_prediction == label else 0
        candidate_correct += 1 if candidate_prediction == label else 0
        reconstructed_trace.append(
            {
                "row_id": row_id,
                "baseline_prediction": baseline_prediction,
                "candidate_prediction": candidate_prediction,
            }
        )

    denominator = len(rows)
    baseline_metric = baseline_correct / denominator
    candidate_metric = candidate_correct / denominator
    delta = candidate_metric - baseline_metric
    if result_bundle.get("denominator") != denominator:
        errors.append("denominator")
    if not _close(result_bundle.get("baseline_metric"), baseline_metric):
        errors.append("baseline_metric")
    if not _close(result_bundle.get("candidate_metric"), candidate_metric):
        errors.append("candidate_metric")
    if not _close(result_bundle.get("delta"), delta):
        errors.append("delta")
    if result_bundle.get("validator_trace") != reconstructed_trace:
        errors.append("validator_trace")

    return {
        "valid": not errors,
        "errors": sorted(set(errors)),
        "attestation_valid": attestation_valid,
        "denominator": denominator,
        "baseline_metric": baseline_metric,
        "candidate_metric": candidate_metric,
        "delta": delta,
        "decision_grade": False,
        "trading_conclusion_allowed": False,
    }
