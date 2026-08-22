"""Durable deterministic Phase 1 experiment loop."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import secrets
import subprocess
import sys
import uuid
from collections import Counter
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Mapping

from phase1_static_target_snapshot import build_real_snapshot
from phase1_static_target_verifier import verify as verify_static_target_result


ROOT = Path(__file__).resolve().parents[1]
FILES = Path(__file__).resolve().parent
VALIDATOR_ID = "static_target_top50_validator_v1"
CAPABILITY_ID = "watch_ranking.static_target_v1"
METRIC_ID = "static_target_top50_daily_precision_delta_v1"
REASON_REGISTRY_VERSION = 1
STAGES = frozenset(
    {"OBSERVED", "PREPARED", "REGISTERED", "VALIDATING", "FORWARD", "DECIDED", "CLOSED"}
)
STATUSES = frozenset({"ACTIVE", "WAITING", "TERMINAL"})
OUTCOME_REASONS = frozenset({
    "snapshot_invalid",
    "waiting_for_data",
    "power_expansion",
    "metric_redesign",
    "needs_validator",
    "contract_rejected",
    "supported",
    "refuted",
    "underpowered",
    "accepted_unknown",
    "invalid_result",
    "budget_exhausted",
    "forward_rejected",
    "operator_rejected",
    "rolled_back",
})
POWER_FACTOR_95_80 = 2.8015852181129683
CAPABILITY_REGISTRY = {
    CAPABILITY_ID: {
        "action_layer": "WATCH",
        "objective_metric_version": METRIC_ID,
        "validator_id": VALIDATOR_ID,
        "runtime_effect": "none_research_only",
    }
}


class AttemptIntegrityError(RuntimeError):
    pass


class LeaseBusyError(RuntimeError):
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


def _file_hash(path: Path) -> str:
    return _sha256(path.read_bytes())


def _iso(value: datetime | None = None) -> str:
    return (value or datetime.now(timezone.utc)).astimezone(timezone.utc).isoformat().replace(
        "+00:00", "Z"
    )


def _parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def _write_immutable(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != raw:
            raise AttemptIntegrityError(f"immutable artifact conflict: {path.name}")
        return
    try:
        with path.open("xb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError:
        if path.read_bytes() != raw:
            raise AttemptIntegrityError(f"immutable artifact conflict: {path.name}")


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_bytes(_canonical_bytes(payload))
    os.replace(temporary, path)


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise AttemptIntegrityError(f"{path.name} must contain an object")
    return payload


def validator_registry() -> dict[str, dict[str, Any]]:
    path = FILES / "phase1_static_target_validator.py"
    return {
        VALIDATOR_ID: {
            "path": str(path),
            "sha256": _file_hash(path),
            "network_access": False,
            "production_write": False,
        }
    }


def build_static_target_contract(
    *,
    attempt_id: str,
    snapshot_sha256: str,
    snapshot_id: str,
    snapshot_manifest_sha256: str | None = None,
    retry_of: str | None = None,
    phase0_completion_sha256: str | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "hypothesis_id": "static-target-top50-ranking",
        "version": 1,
        "parent_hypothesis_id": None,
        "attempt_id": attempt_id,
        "retry_of": retry_of,
        "action_layer": "WATCH",
        "objective_metric_version": METRIC_ID,
        "guardrail_versions": [
            "static_target_top50_entrant_recall_noninferiority_v1",
            "static_target_top50_base_rate_lift_v1",
            "static_target_top50_coverage_v1",
        ],
        "incident_or_cohort_refs": [
            "docs/specs/external-top50-screen-validation.md",
            f"phase0:{phase0_completion_sha256 or 'fixture_only'}",
        ],
        "causal_mechanism": (
            "Use the target clock's fixed 24h denominator at the observation clock to avoid "
            "rank distortion caused by a moving observation-time denominator."
        ),
        "target_capability": CAPABILITY_ID,
        "proposed_change": {
            "baseline": "rank candidate current_return descending",
            "candidate": "rank candidate static_target_return descending",
            "selection_size": 10,
        },
        "affected_population": (
            "watchlist spot-USDT candidates outside observation-time Binance Top-50"
        ),
        "expected_effect_and_horizon": (
            "At least +2pp paired daily Top-10 precision by same-day 23:00 local label."
        ),
        "competing_explanation": (
            "The historical effect is a denominator artifact or a small set of correlated market days."
        ),
        "falsifier": (
            "Paired day-cluster bootstrap lower bound is not above zero, recall falls, or lift is <=1."
        ),
        "minimum_practical_effect": 0.02,
        "registered_validator_id": VALIDATOR_ID,
        "validation_protocol": {
            "protocol_version": "static_target_top50_maximum_period_v1",
            "timezone": "Europe/Budapest",
            "observation_time_local": "12:15",
            "target_time_local": "23:00",
            "top_n": 50,
            "selection_size": 10,
            "minimum_eligible_days": 30,
            "minimum_calendar_coverage": 0.95,
            "bootstrap_samples": 5000,
            "bootstrap_seed": 2104,
            "power_method": "paired_day_mean_mde_95_80_v1",
        },
        "rollback_flag": "STATIC_TARGET_TOP50_SHADOW_ENABLED",
        "evidence_snapshot_id": snapshot_id,
        "snapshot_sha256": snapshot_sha256,
        "snapshot_manifest_sha256": snapshot_manifest_sha256 or snapshot_sha256,
        "production_effect": "none_research_only",
    }


def precheck_contract(
    contract: Mapping[str, Any],
    *,
    capability_registry: Mapping[str, Mapping[str, Any]] = CAPABILITY_REGISTRY,
    validator_registry: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    required = {
        "hypothesis_id", "version", "parent_hypothesis_id", "attempt_id",
        "action_layer", "objective_metric_version", "guardrail_versions",
        "incident_or_cohort_refs", "causal_mechanism", "target_capability",
        "proposed_change", "affected_population", "expected_effect_and_horizon",
        "competing_explanation", "falsifier", "minimum_practical_effect",
        "registered_validator_id", "validation_protocol", "rollback_flag",
        "evidence_snapshot_id", "snapshot_sha256",
        "snapshot_manifest_sha256",
    }
    missing = sorted(name for name in required if name not in contract)
    capability = capability_registry.get(str(contract.get("target_capability") or ""))
    if missing or capability is None:
        return {
            "status": "TERMINAL",
            "outcome_reason": "contract_rejected",
            "errors": [*(f"missing:{name}" for name in missing), "unknown_capability"]
            if capability is None else [*(f"missing:{name}" for name in missing)],
        }
    errors = []
    if capability.get("action_layer") != contract.get("action_layer"):
        errors.append("action_layer_binding")
    if capability.get("objective_metric_version") != contract.get("objective_metric_version"):
        errors.append("objective_metric_binding")
    if capability.get("validator_id") != contract.get("registered_validator_id"):
        errors.append("validator_binding")
    if len(str(contract.get("snapshot_sha256") or "")) != 64:
        errors.append("snapshot_sha256")
    if errors:
        return {"status": "TERMINAL", "outcome_reason": "contract_rejected", "errors": errors}
    registry = validator_registry if validator_registry is not None else globals()["validator_registry"]()
    if str(contract.get("registered_validator_id")) not in registry:
        return {"status": "WAITING", "outcome_reason": "needs_validator", "errors": []}
    return {"status": "ACTIVE", "outcome_reason": None, "errors": []}


def acquire_lease(
    attempt_dir: Path,
    *,
    owner: str,
    now: datetime | None = None,
    lease_seconds: int = 300,
) -> dict[str, Any]:
    now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    path = attempt_dir / "lease.json"
    recovered = False
    previous_owner = None
    if path.exists():
        existing = _load_json(path)
        expires = _parse_time(str(existing.get("expires_at")))
        previous_owner = existing.get("owner")
        if expires > now and previous_owner != owner:
            raise LeaseBusyError(f"attempt lease held by {previous_owner} until {_iso(expires)}")
        recovered = expires <= now and previous_owner != owner
    payload = {
        "schema_version": 1,
        "owner": owner,
        "acquired_at": _iso(now),
        "expires_at": _iso(now + timedelta(seconds=max(1, lease_seconds))),
        "recovered_expired_lease": recovered,
        "previous_owner": previous_owner if recovered else None,
    }
    _write_json_atomic(path, payload)
    return payload


def _release_lease(attempt_dir: Path, owner: str) -> None:
    path = attempt_dir / "lease.json"
    if not path.exists():
        return
    try:
        payload = _load_json(path)
    except (OSError, ValueError, json.JSONDecodeError, AttemptIntegrityError):
        return
    if payload.get("owner") == owner:
        path.unlink(missing_ok=True)


def _read_ledger(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise AttemptIntegrityError(f"invalid ledger row {number}")
        rows.append(payload)
    return rows


def _append_event(
    ledger_path: Path,
    *,
    attempt_id: str,
    event_type: str,
    stage: str,
    status: str,
    outcome_reason: str | None = None,
    input_hash: str | None = None,
    output_hash: str | None = None,
    retry_of: str | None = None,
    decision_grade: bool = False,
    recovered_expired_lease: bool = False,
) -> dict[str, Any]:
    if stage not in STAGES or status not in STATUSES:
        raise AttemptIntegrityError("unknown stage/status")
    if outcome_reason is not None and outcome_reason not in OUTCOME_REASONS:
        raise AttemptIntegrityError("unknown outcome reason")
    event_key = f"{attempt_id}:{event_type}:{stage}:{outcome_reason or 'none'}"
    stable = {
        "schema_version": 1,
        "reason_registry_version": REASON_REGISTRY_VERSION,
        "event_key": event_key,
        "attempt_id": attempt_id,
        "event_type": event_type,
        "stage": stage,
        "status": status,
        "outcome_reason": outcome_reason,
        "input_hash": input_hash,
        "output_hash": output_hash,
        "retry_of": retry_of,
        "decision_grade": bool(decision_grade),
        "production_effect": "none_research_only",
        "recovered_expired_lease": bool(recovered_expired_lease),
    }
    rows = _read_ledger(ledger_path)
    prior = rows[-1] if rows else None
    for row in rows:
        if row.get("event_key") == event_key:
            existing_stable = {key: row.get(key) for key in stable}
            if existing_stable != stable:
                raise AttemptIntegrityError(f"ledger event conflict: {event_key}")
            return row
    event = {
        **stable,
        "occurred_at": _iso(),
        "prior_stage": prior.get("stage") if prior else None,
        "prior_status": prior.get("status") if prior else None,
        "prior_event_hash": _sha256(_canonical_bytes(prior)) if prior else None,
    }
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    with ledger_path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    return event


def _terminal(ledger_path: Path, attempt_id: str) -> dict[str, Any] | None:
    for row in reversed(_read_ledger(ledger_path)):
        if row.get("attempt_id") == attempt_id and row.get("status") == "TERMINAL":
            return row
    return None


def _power_precheck(snapshot: Mapping[str, Any], contract: Mapping[str, Any]) -> dict[str, Any]:
    protocol = contract["validation_protocol"]
    k = int(protocol["selection_size"])
    differences = []
    candidate_count = 0
    entrant_count = 0
    for day in snapshot.get("days") or []:
        candidates = list(day.get("candidates") or [])
        if not candidates:
            continue
        current = sorted(candidates, key=lambda row: (-float(row["current_return"]), str(row["symbol"])))[:k]
        static = sorted(candidates, key=lambda row: (-float(row["static_target_return"]), str(row["symbol"])))[:k]
        current_precision = sum(int(bool(row["is_target_top"])) for row in current) / len(current)
        static_precision = sum(int(bool(row["is_target_top"])) for row in static) / len(static)
        differences.append(static_precision - current_precision)
        candidate_count += len(candidates)
        entrant_count += len(day.get("target_entrant_symbols") or [])
    n = len(differences)
    sigma = stdev(differences) if n > 1 else 0.0
    mde = POWER_FACTOR_95_80 * sigma / math.sqrt(n) if n else None
    requested = int((snapshot.get("source_contract") or {}).get("requested_day_count") or 0)
    coverage = n / requested if requested else 0.0
    sesoi = float(contract["minimum_practical_effect"])
    feasible = bool(
        n >= int(protocol["minimum_eligible_days"])
        and candidate_count > 0
        and entrant_count > 0
        and mde is not None
        and mde <= sesoi
        and coverage >= float(protocol["minimum_calendar_coverage"])
    )
    return {
        "method_version": "paired_day_mean_mde_95_80_v1",
        "status": "FEASIBLE" if feasible else "UNDERPOWERED",
        "feasible": feasible,
        "effective_sample_size": n,
        "candidate_count": candidate_count,
        "target_entrant_count": entrant_count,
        "calendar_coverage": round(coverage, 6),
        "paired_delta_estimate": round(mean(differences), 6) if differences else None,
        "paired_day_sd": round(sigma, 6),
        "mde": round(mde, 6) if mde is not None else None,
        "sesoi": sesoi,
    }


def _govern(verification: Mapping[str, Any], contract: Mapping[str, Any]) -> str:
    if verification.get("status") != "VERIFIED_RESULT" or not verification.get("valid"):
        return "invalid_result"
    metrics = verification["verified_metrics"]
    candidate = metrics["policies"]["static_target"]
    baseline = metrics["policies"]["current_rank"]
    interval = metrics.get("paired_bootstrap95") or []
    supported = bool(
        int(metrics.get("eligible_days") or 0)
        >= int(contract["validation_protocol"]["minimum_eligible_days"])
        and float(metrics.get("paired_daily_precision_delta") or 0.0)
        >= float(contract["minimum_practical_effect"])
        and len(interval) == 2
        and float(interval[0]) > 0.0
        and float(candidate.get("entrant_recall") or 0.0)
        >= float(baseline.get("entrant_recall") or 0.0)
        and float(candidate.get("precision_lift_over_base") or 0.0) > 1.0
    )
    return "supported" if supported else "refuted"


def _write_dead_letter(attempt_dir: Path, *, attempt_id: str, reason: str, evidence: Any) -> None:
    payload = {
        "schema_version": 1,
        "attempt_id": attempt_id,
        "outcome_reason": reason,
        "evidence": evidence,
        "production_effect": "none_research_only",
    }
    _write_immutable(attempt_dir / "dead_letter.json", _canonical_bytes(payload))


def run_attempt(
    state_dir: Path,
    attempt_id: str,
    *,
    snapshot_payload: Mapping[str, Any],
    retry_of: str | None = None,
    retry_reason: str | None = None,
    corrupt_result: bool = False,
    validator_timeout_seconds: int = 120,
    lease_seconds: int = 300,
) -> dict[str, Any]:
    if retry_of and not retry_reason:
        raise AttemptIntegrityError("retry_reason is required when retry_of is set")
    state_dir = Path(state_dir).resolve()
    attempt_dir = state_dir / "attempts" / attempt_id
    ledger_path = state_dir / "attempt_ledger.jsonl"
    snapshot_path = attempt_dir / "snapshot.json"
    manifest_path = attempt_dir / "snapshot_manifest.json"
    contract_path = attempt_dir / "hypothesis_contract.json"
    key_path = attempt_dir / "validator_attestation.key"
    result_path = attempt_dir / "validator_result.json"
    verification_path = attempt_dir / "verification.json"
    decision_path = attempt_dir / "decision.json"
    snapshot_raw = _canonical_bytes(dict(snapshot_payload))
    snapshot_hash = _sha256(snapshot_raw)
    _write_immutable(snapshot_path, snapshot_raw)
    provenance = snapshot_payload.get("provenance") if isinstance(snapshot_payload.get("provenance"), Mapping) else {}
    manifest = {
        "schema_version": 1,
        "snapshot_id": f"sha256:{snapshot_hash}",
        "snapshot_sha256": snapshot_hash,
        "snapshot_kind": snapshot_payload.get("snapshot_kind"),
        "policy_epoch": provenance.get("policy_epoch"),
        "metric_version": METRIC_ID,
        "source_cache_hash": provenance.get("used_content_hash"),
        "phase0_completion_sha256": provenance.get("phase0_completion_sha256"),
        "builder_sha256": provenance.get("snapshot_builder_sha256"),
        "git_commit": provenance.get("git_commit"),
        "decision_grade_input": bool(snapshot_payload.get("decision_grade_input")),
    }
    _write_immutable(manifest_path, _canonical_bytes(manifest))
    contract = build_static_target_contract(
        attempt_id=attempt_id,
        snapshot_sha256=snapshot_hash,
        snapshot_id=manifest["snapshot_id"],
        snapshot_manifest_sha256=_sha256(_canonical_bytes(manifest)),
        retry_of=retry_of,
        phase0_completion_sha256=provenance.get("phase0_completion_sha256"),
    )
    contract["retry_reason"] = retry_reason
    contract["validation_protocol"]["validator_sha256"] = _file_hash(
        FILES / "phase1_static_target_validator.py"
    )
    contract["validation_protocol"]["verifier_sha256"] = _file_hash(
        FILES / "phase1_static_target_verifier.py"
    )
    contract_raw = _canonical_bytes(contract)
    _write_immutable(contract_path, contract_raw)
    existing_terminal = _terminal(ledger_path, attempt_id)
    if existing_terminal is not None:
        return {
            "terminal": existing_terminal,
            "verification": _load_json(verification_path) if verification_path.exists() else None,
            "decision": _load_json(decision_path) if decision_path.exists() else None,
        }

    owner = f"{os.getpid()}-{uuid.uuid4().hex}"
    lease = acquire_lease(
        attempt_dir, owner=owner, lease_seconds=lease_seconds
    )
    try:
        _append_event(
            ledger_path,
            attempt_id=attempt_id,
            event_type="attempt_started",
            stage="OBSERVED",
            status="ACTIVE",
            input_hash=snapshot_hash,
            retry_of=retry_of,
            recovered_expired_lease=bool(lease.get("recovered_expired_lease")),
        )
        _append_event(
            ledger_path,
            attempt_id=attempt_id,
            event_type="snapshot_created",
            stage="PREPARED",
            status="ACTIVE",
            input_hash=snapshot_hash,
            output_hash=_file_hash(manifest_path),
            retry_of=retry_of,
        )
        precheck = precheck_contract(contract)
        _append_event(
            ledger_path,
            attempt_id=attempt_id,
            event_type="contract_precheck",
            stage="REGISTERED",
            status=str(precheck["status"]),
            outcome_reason=precheck.get("outcome_reason"),
            input_hash=_sha256(contract_raw),
            output_hash=_sha256(_canonical_bytes(precheck)),
            retry_of=retry_of,
        )
        if precheck["status"] != "ACTIVE":
            if precheck["status"] == "WAITING":
                return {"terminal": None, "waiting": precheck}
            _write_dead_letter(
                attempt_dir,
                attempt_id=attempt_id,
                reason=str(precheck["outcome_reason"]),
                evidence=precheck,
            )
            terminal = _append_event(
                ledger_path,
                attempt_id=attempt_id,
                event_type="attempt_terminal",
                stage="CLOSED",
                status="TERMINAL",
                outcome_reason=str(precheck["outcome_reason"]),
                input_hash=_sha256(contract_raw),
                retry_of=retry_of,
            )
            return {"terminal": terminal, "precheck": precheck}

        power = _power_precheck(snapshot_payload, contract)
        power_path = attempt_dir / "power_precheck.json"
        _write_immutable(power_path, _canonical_bytes(power))
        _append_event(
            ledger_path,
            attempt_id=attempt_id,
            event_type="power_gate",
            stage="REGISTERED",
            status="ACTIVE" if power["feasible"] else "TERMINAL",
            outcome_reason=None if power["feasible"] else "underpowered",
            input_hash=snapshot_hash,
            output_hash=_file_hash(power_path),
            retry_of=retry_of,
        )
        if not power["feasible"]:
            terminal = _append_event(
                ledger_path,
                attempt_id=attempt_id,
                event_type="attempt_terminal",
                stage="CLOSED",
                status="TERMINAL",
                outcome_reason="underpowered",
                input_hash=snapshot_hash,
                output_hash=_file_hash(power_path),
                retry_of=retry_of,
                decision_grade=bool(snapshot_payload.get("decision_grade_input")),
            )
            return {"terminal": terminal, "power": power}

        if key_path.exists():
            key = bytes.fromhex(key_path.read_text(encoding="ascii"))
        else:
            key = secrets.token_bytes(32)
            _write_immutable(key_path, key.hex().encode("ascii"))
        _append_event(
            ledger_path,
            attempt_id=attempt_id,
            event_type="validator_started",
            stage="VALIDATING",
            status="ACTIVE",
            input_hash=_sha256(snapshot_raw + contract_raw),
            retry_of=retry_of,
        )
        if not result_path.exists():
            command = [
                sys.executable,
                str(FILES / "phase1_static_target_validator.py"),
                "--snapshot", str(snapshot_path),
                "--manifest", str(manifest_path),
                "--contract", str(contract_path),
                "--key", str(key_path),
                "--output", str(result_path),
            ]
            if corrupt_result:
                command.append("--corrupt-result")
            try:
                subprocess.run(
                    command,
                    cwd=FILES,
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=max(1, validator_timeout_seconds),
                )
            except subprocess.TimeoutExpired as exc:
                _write_dead_letter(
                    attempt_dir,
                    attempt_id=attempt_id,
                    reason="budget_exhausted",
                    evidence={"timeout_seconds": validator_timeout_seconds, "command": command},
                )
                terminal = _append_event(
                    ledger_path,
                    attempt_id=attempt_id,
                    event_type="attempt_terminal",
                    stage="CLOSED",
                    status="TERMINAL",
                    outcome_reason="budget_exhausted",
                    input_hash=_sha256(snapshot_raw + contract_raw),
                    retry_of=retry_of,
                )
                return {"terminal": terminal, "error": str(exc)}
            except subprocess.CalledProcessError as exc:
                _write_dead_letter(
                    attempt_dir,
                    attempt_id=attempt_id,
                    reason="invalid_result",
                    evidence={"returncode": exc.returncode, "stderr": exc.stderr[-4000:]},
                )
                terminal = _append_event(
                    ledger_path,
                    attempt_id=attempt_id,
                    event_type="attempt_terminal",
                    stage="CLOSED",
                    status="TERMINAL",
                    outcome_reason="invalid_result",
                    input_hash=_sha256(snapshot_raw + contract_raw),
                    retry_of=retry_of,
                )
                return {"terminal": terminal, "error": exc.stderr}
        _append_event(
            ledger_path,
            attempt_id=attempt_id,
            event_type="validator_completed",
            stage="VALIDATING",
            status="ACTIVE",
            input_hash=_sha256(snapshot_raw + contract_raw),
            output_hash=_file_hash(result_path),
            retry_of=retry_of,
        )
        verification = verify_static_target_result(
            snapshot_path, manifest_path, contract_path, result_path, key
        )
        _write_immutable(verification_path, _canonical_bytes(verification))
        _append_event(
            ledger_path,
            attempt_id=attempt_id,
            event_type="result_verification",
            stage="VALIDATING",
            status="ACTIVE" if verification["valid"] else "TERMINAL",
            outcome_reason=None if verification["valid"] else "invalid_result",
            input_hash=_file_hash(result_path),
            output_hash=_file_hash(verification_path),
            retry_of=retry_of,
            decision_grade=bool(verification.get("decision_grade")),
        )
        outcome = _govern(verification, contract)
        if outcome == "invalid_result":
            _write_dead_letter(
                attempt_dir,
                attempt_id=attempt_id,
                reason=outcome,
                evidence={"verification_errors": verification.get("errors")},
            )
        decision = {
            "schema_version": 1,
            "attempt_id": attempt_id,
            "input_status": verification["status"],
            "outcome_reason": outcome,
            "automatic_production_promotion": False,
            "authorized_next_action": (
                "continued_or_separately_reviewed_silent_shadow"
                if outcome == "supported" else "no_candidate_advancement"
            ),
            "production_effect": "none_research_only",
            "decision_grade": bool(verification.get("decision_grade")),
        }
        _write_immutable(decision_path, _canonical_bytes(decision))
        _append_event(
            ledger_path,
            attempt_id=attempt_id,
            event_type="governor_decision",
            stage="DECIDED",
            status="ACTIVE",
            outcome_reason=outcome,
            input_hash=_file_hash(verification_path),
            output_hash=_file_hash(decision_path),
            retry_of=retry_of,
            decision_grade=bool(verification.get("decision_grade")),
        )
        terminal = _append_event(
            ledger_path,
            attempt_id=attempt_id,
            event_type="attempt_terminal",
            stage="CLOSED",
            status="TERMINAL",
            outcome_reason=outcome,
            input_hash=_file_hash(verification_path),
            output_hash=_file_hash(decision_path),
            retry_of=retry_of,
            decision_grade=bool(verification.get("decision_grade")),
        )
        return {
            "terminal": terminal,
            "power": power,
            "verification": verification,
            "decision": decision,
        }
    finally:
        _release_lease(attempt_dir, owner)


def build_status(state_dir: Path, *, now: datetime | None = None) -> dict[str, Any]:
    now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    rows = _read_ledger(Path(state_dir) / "attempt_ledger.jsonl")
    attempts = sorted({str(row.get("attempt_id")) for row in rows if row.get("attempt_id")})
    terminals = {}
    waiting = set()
    first_times = {}
    last_transition = None
    retries = 0
    for row in rows:
        attempt_id = str(row.get("attempt_id") or "")
        if row.get("event_type") == "attempt_started" and row.get("retry_of"):
            retries += 1
        if row.get("occurred_at"):
            occurred = _parse_time(str(row["occurred_at"]))
            first_times.setdefault(attempt_id, occurred)
            if last_transition is None or occurred > last_transition:
                last_transition = occurred
        if row.get("status") == "TERMINAL":
            terminals[attempt_id] = str(row.get("outcome_reason"))
            waiting.discard(attempt_id)
        elif row.get("status") == "WAITING" and attempt_id not in terminals:
            waiting.add(attempt_id)
    active = [attempt for attempt in attempts if attempt not in terminals and attempt not in waiting]
    open_attempts = active + sorted(waiting)
    queue_ages = [
        (now - first_times[attempt]).total_seconds() / 3600.0
        for attempt in open_attempts
        if attempt in first_times
    ]
    dead_letters = len(list((Path(state_dir) / "attempts").glob("*/dead_letter.json")))
    return {
        "schema_version": 1,
        "program_mode": "NORMAL",
        "attempt_count": len(attempts),
        "terminal_count": len(terminals),
        "waiting_count": len(waiting),
        "active_count": len(active),
        "terminal_rate": len(terminals) / len(attempts) if attempts else None,
        "terminal_reason_counts": dict(sorted(Counter(terminals.values()).items())),
        "retry_count": retries,
        "dead_letter_count": dead_letters,
        "oldest_queue_age_hours": round(max(queue_ages), 6) if queue_ages else None,
        "last_transition_at": _iso(last_transition) if last_transition else None,
        "production_effect": "none_research_only",
    }


def run_real_static_target_attempt(
    state_dir: Path,
    attempt_id: str,
    *,
    start_day: date,
    end_day: date,
    phase0_completion_path: Path,
    validator_timeout_seconds: int,
    retry_of: str | None = None,
    retry_reason: str | None = None,
) -> dict[str, Any]:
    state_dir = Path(state_dir).resolve()
    snapshot_path = state_dir / "attempts" / attempt_id / "snapshot.json"
    if snapshot_path.exists():
        snapshot = _load_json(snapshot_path)
    elif retry_of and (state_dir / "attempts" / retry_of / "snapshot.json").exists():
        snapshot = _load_json(state_dir / "attempts" / retry_of / "snapshot.json")
    else:
        snapshot = build_real_snapshot(
            start_day=start_day,
            end_day=end_day,
            phase0_completion_path=phase0_completion_path,
        )
    return run_attempt(
        state_dir,
        attempt_id,
        snapshot_payload=snapshot,
        retry_of=retry_of,
        retry_reason=retry_reason,
        validator_timeout_seconds=validator_timeout_seconds,
    )


def _parse_day(value: str) -> date:
    return date.fromisoformat(value)


def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run-static-target")
    run.add_argument("--state-dir", type=Path, required=True)
    run.add_argument("--attempt-id", required=True)
    run.add_argument("--start-day", type=_parse_day, required=True)
    run.add_argument("--end-day", type=_parse_day, required=True)
    run.add_argument("--phase0-completion", type=Path, required=True)
    run.add_argument("--validator-timeout-seconds", type=int, default=120)
    run.add_argument("--retry-of")
    run.add_argument("--retry-reason")
    status = sub.add_parser("status")
    status.add_argument("--state-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "run-static-target":
        payload = run_real_static_target_attempt(
            args.state_dir,
            args.attempt_id,
            start_day=args.start_day,
            end_day=args.end_day,
            phase0_completion_path=args.phase0_completion,
            validator_timeout_seconds=args.validator_timeout_seconds,
            retry_of=args.retry_of,
            retry_reason=args.retry_reason,
        )
    else:
        payload = build_status(args.state_dir)
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2))
    terminal = payload.get("terminal") if isinstance(payload, dict) else None
    return 0 if args.command == "status" or terminal else 2


if __name__ == "__main__":
    raise SystemExit(_main())
