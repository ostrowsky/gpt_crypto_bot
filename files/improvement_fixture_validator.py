"""Synthetic Phase -1 validator for the improvement control plane.

This module deliberately has no dependency on the trading stack.  It proves a
result-bundle protocol over a tiny checked-in fixture; it is not a market
validator and its output cannot support a trading conclusion.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
from pathlib import Path
from typing import Any


MAX_FIXTURE_ROWS = 64
RESULT_SCHEMA_VERSION = 1


class FixtureValidationError(ValueError):
    """The frozen fixture or registered smoke contract is invalid."""


def _canonical_bytes(payload: dict[str, Any]) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _load_object(path: Path) -> tuple[dict[str, Any], bytes]:
    raw = path.read_bytes()
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FixtureValidationError(f"invalid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise FixtureValidationError(f"expected JSON object: {path}")
    return payload, raw


def _validated_rows(snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    if snapshot.get("schema_version") != 1:
        raise FixtureValidationError("unsupported fixture schema_version")
    if snapshot.get("fixture_id") != "control-plane-smoke-v1":
        raise FixtureValidationError("unexpected fixture_id")
    rows = snapshot.get("rows")
    if not isinstance(rows, list) or not rows or len(rows) > MAX_FIXTURE_ROWS:
        raise FixtureValidationError("fixture row count must be in 1..64")

    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"row_id", "score", "label"}:
            raise FixtureValidationError("invalid fixture row shape")
        row_id = row["row_id"]
        score = row["score"]
        label = row["label"]
        if not isinstance(row_id, str) or not row_id or row_id in seen:
            raise FixtureValidationError("row_id must be non-empty and unique")
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            raise FixtureValidationError("score must be numeric")
        if not math.isfinite(float(score)) or not 0.0 <= float(score) <= 1.0:
            raise FixtureValidationError("score must be finite and in [0, 1]")
        if isinstance(label, bool) or label not in (0, 1):
            raise FixtureValidationError("label must be 0 or 1")
        seen.add(row_id)
    return rows


def _threshold(contract: dict[str, Any], name: str) -> float:
    policy = contract.get(name)
    if not isinstance(policy, dict) or set(policy) != {"kind", "threshold"}:
        raise FixtureValidationError(f"invalid {name}")
    if policy.get("kind") != "threshold":
        raise FixtureValidationError(f"unsupported {name} kind")
    value = policy.get("threshold")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FixtureValidationError(f"invalid {name} threshold")
    value = float(value)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise FixtureValidationError(f"invalid {name} threshold")
    return value


class FixtureDeltaValidatorAdapter:
    """Compute a fixed accuracy delta and emit an attested result bundle."""

    validator_id = "fixture-delta-validator-v1"

    def validate(
        self,
        snapshot_path: Path,
        contract_path: Path,
        attestation_key: bytes,
        *,
        corrupt_summary: bool = False,
    ) -> dict[str, Any]:
        snapshot, snapshot_raw = _load_object(Path(snapshot_path))
        contract, contract_raw = _load_object(Path(contract_path))
        rows = _validated_rows(snapshot)

        if contract.get("contract_version") != 1:
            raise FixtureValidationError("unsupported contract_version")
        if contract.get("validator_id") != self.validator_id:
            raise FixtureValidationError("validator_id mismatch")
        if contract.get("fixture_id") != snapshot.get("fixture_id"):
            raise FixtureValidationError("fixture_id mismatch")
        if contract.get("metric") != "accuracy":
            raise FixtureValidationError("unsupported metric")
        if contract.get("snapshot_sha256") != _sha256_bytes(snapshot_raw):
            raise FixtureValidationError("snapshot hash mismatch")
        if contract.get("decision_grade") is not False:
            raise FixtureValidationError("smoke contract cannot be decision grade")
        if contract.get("trading_conclusion_allowed") is not False:
            raise FixtureValidationError("smoke contract cannot allow trading conclusions")
        if not isinstance(attestation_key, bytes) or len(attestation_key) < 32:
            raise FixtureValidationError("attestation key must contain at least 32 bytes")

        baseline_threshold = _threshold(contract, "baseline_policy")
        candidate_threshold = _threshold(contract, "candidate_policy")
        trace: list[dict[str, Any]] = []
        baseline_correct = 0
        candidate_correct = 0
        for row in rows:
            baseline_prediction = int(float(row["score"]) >= baseline_threshold)
            candidate_prediction = int(float(row["score"]) >= candidate_threshold)
            baseline_correct += int(baseline_prediction == row["label"])
            candidate_correct += int(candidate_prediction == row["label"])
            trace.append(
                {
                    "row_id": row["row_id"],
                    "baseline_prediction": baseline_prediction,
                    "candidate_prediction": candidate_prediction,
                }
            )

        denominator = len(rows)
        baseline_metric = baseline_correct / denominator
        candidate_metric = candidate_correct / denominator
        if corrupt_summary:
            # The bundle remains correctly attested.  Only an independent raw
            # recompute can detect this confident, internally consistent lie.
            candidate_metric -= 0.125

        payload: dict[str, Any] = {
            "schema_version": RESULT_SCHEMA_VERSION,
            "attempt_id": contract.get("attempt_id"),
            "hypothesis_id": contract.get("hypothesis_id"),
            "validator_id": self.validator_id,
            "fixture_id": snapshot.get("fixture_id"),
            "snapshot_sha256": _sha256_bytes(snapshot_raw),
            "contract_sha256": _sha256_bytes(contract_raw),
            "metric": "accuracy",
            "denominator": denominator,
            "baseline_metric": baseline_metric,
            "candidate_metric": candidate_metric,
            "delta": candidate_metric - baseline_metric,
            "validator_trace": trace,
            "decision_grade": False,
            "trading_conclusion_allowed": False,
        }
        signature = hmac.new(
            attestation_key, _canonical_bytes(payload), hashlib.sha256
        ).hexdigest()
        return {
            **payload,
            "attestation": {
                "algorithm": "hmac-sha256",
                "key_id": contract.get("attestation_key_id"),
                "signature": signature,
            },
        }
