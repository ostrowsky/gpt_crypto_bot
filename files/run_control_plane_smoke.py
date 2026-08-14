"""Run the Phase -1 continuous-improvement control-plane walking skeleton."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import secrets
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from improvement_fixture_validator import FixtureDeltaValidatorAdapter
from improvement_fixture_verifier import verify_result_bundle


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DEFAULT_FIXTURE = HERE / "testdata" / "control_plane_smoke_fixture.json"
OUTCOME_REASON_REGISTRY_VERSION = 1
PHASE_MINUS_ONE_OUTCOME_REASONS = frozenset(
    {"observed", "validation_started", "protocol_verified", "invalid_result"}
)
STAGES = frozenset(
    {"OBSERVED", "PREPARED", "REGISTERED", "VALIDATING", "FORWARD", "DECIDED", "CLOSED"}
)
STATUSES = frozenset({"ACTIVE", "WAITING", "TERMINAL"})
_ATTEMPT_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$")


class AttemptIntegrityError(RuntimeError):
    """An existing attempt conflicts with immutable input or transition state."""


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


def _read_ledger(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    entries: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError as exc:
            raise AttemptIntegrityError(f"invalid ledger line {line_number}") from exc
        if not isinstance(entry, dict):
            raise AttemptIntegrityError(f"invalid ledger entry {line_number}")
        entries.append(entry)
    return entries


def _append_transition(
    ledger_path: Path,
    *,
    attempt_id: str,
    stage: str,
    status: str,
    outcome_reason: str,
) -> dict[str, Any]:
    if stage not in STAGES or status not in STATUSES:
        raise AttemptIntegrityError("unknown stage or status")
    if outcome_reason not in PHASE_MINUS_ONE_OUTCOME_REASONS:
        raise AttemptIntegrityError("unknown Phase -1 outcome reason")
    transition_key = f"{attempt_id}:{stage}:{outcome_reason}"
    candidate = {
        "ledger_schema_version": 1,
        "reason_registry_version": OUTCOME_REASON_REGISTRY_VERSION,
        "transition_key": transition_key,
        "attempt_id": attempt_id,
        "stage": stage,
        "status": status,
        "outcome_reason": outcome_reason,
        "evidence_kind": "synthetic_protocol_fixture",
        "decision_grade": False,
        "trading_conclusion_allowed": False,
    }
    existing = _read_ledger(ledger_path)
    for entry in existing:
        if entry.get("transition_key") == transition_key:
            if entry != candidate:
                raise AttemptIntegrityError(f"transition conflict: {transition_key}")
            return entry
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    with ledger_path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(candidate, sort_keys=True, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    return candidate


def _terminal_for_attempt(ledger_path: Path, attempt_id: str) -> dict[str, Any] | None:
    for entry in reversed(_read_ledger(ledger_path)):
        if entry.get("attempt_id") == attempt_id and entry.get("status") == "TERMINAL":
            return entry
    return None


def _contract(attempt_id: str, snapshot_raw: bytes, key: bytes) -> dict[str, Any]:
    return {
        "contract_version": 1,
        "attempt_id": attempt_id,
        "hypothesis_id": "phase-minus-one-fixture-threshold",
        "validator_id": FixtureDeltaValidatorAdapter.validator_id,
        "fixture_id": "control-plane-smoke-v1",
        "metric": "accuracy",
        "baseline_policy": {"kind": "threshold", "threshold": 0.75},
        "candidate_policy": {"kind": "threshold", "threshold": 0.55},
        "snapshot_sha256": _sha256(snapshot_raw),
        "attestation_key_id": _sha256(key)[:16],
        "decision_grade": False,
        "trading_conclusion_allowed": False,
    }


def _validated_state_dir(state_dir: Path) -> Path:
    resolved = Path(state_dir).resolve()
    try:
        resolved.relative_to(ROOT.resolve())
    except ValueError:
        return resolved
    raise AttemptIntegrityError("smoke state directory must be outside the repository")


def run_attempt(
    state_dir: Path,
    attempt_id: str,
    *,
    corrupt_result: bool = False,
    fixture_path: Path = DEFAULT_FIXTURE,
) -> dict[str, Any]:
    """Run or idempotently resume one isolated synthetic attempt."""

    if not _ATTEMPT_ID.fullmatch(attempt_id):
        raise AttemptIntegrityError("unsafe attempt_id")
    state_dir = _validated_state_dir(state_dir)
    attempt_dir = state_dir / "attempts" / attempt_id
    ledger_path = state_dir / "attempt_ledger.jsonl"
    snapshot_path = attempt_dir / "snapshot.json"
    contract_path = attempt_dir / "contract.json"
    key_path = attempt_dir / "validator_attestation.key"
    result_path = attempt_dir / "validator_result.json"
    verification_path = attempt_dir / "verification.json"

    source_raw = Path(fixture_path).read_bytes()
    _write_immutable(snapshot_path, source_raw)
    if key_path.exists():
        key = bytes.fromhex(key_path.read_text(encoding="ascii"))
    else:
        key = secrets.token_bytes(32)
        _write_immutable(key_path, key.hex().encode("ascii"))
    contract = _contract(attempt_id, source_raw, key)
    _write_immutable(contract_path, _canonical_bytes(contract))

    _append_transition(
        ledger_path,
        attempt_id=attempt_id,
        stage="OBSERVED",
        status="ACTIVE",
        outcome_reason="observed",
    )
    terminal = _terminal_for_attempt(ledger_path, attempt_id)
    if terminal is not None:
        if not verification_path.exists():
            raise AttemptIntegrityError("terminal attempt has no verification artifact")
        return {
            "terminal": terminal,
            "verification": json.loads(verification_path.read_text(encoding="utf-8")),
        }

    _append_transition(
        ledger_path,
        attempt_id=attempt_id,
        stage="VALIDATING",
        status="ACTIVE",
        outcome_reason="validation_started",
    )
    bundle = FixtureDeltaValidatorAdapter().validate(
        snapshot_path,
        contract_path,
        key,
        corrupt_summary=corrupt_result,
    )
    _write_immutable(result_path, _canonical_bytes(bundle))
    verification = verify_result_bundle(snapshot_path, contract_path, bundle, key)
    _write_immutable(verification_path, _canonical_bytes(verification))
    outcome_reason = "protocol_verified" if verification["valid"] else "invalid_result"
    terminal = _append_transition(
        ledger_path,
        attempt_id=attempt_id,
        stage="CLOSED",
        status="TERMINAL",
        outcome_reason=outcome_reason,
    )
    return {"terminal": terminal, "verification": verification}


def run_smoke_suite(state_dir: Path) -> dict[str, Any]:
    started = time.monotonic()
    valid = run_attempt(Path(state_dir), "phase-minus-one-valid")
    corrupt = run_attempt(
        Path(state_dir), "phase-minus-one-corrupt", corrupt_result=True
    )
    elapsed = time.monotonic() - started
    passed = (
        valid["terminal"]["outcome_reason"] == "protocol_verified"
        and corrupt["terminal"]["outcome_reason"] == "invalid_result"
        and elapsed < 10.0
    )
    return {
        "status": "pass" if passed else "fail",
        "elapsed_seconds": elapsed,
        "valid": valid,
        "corrupt": corrupt,
        "evidence_kind": "synthetic_protocol_fixture",
        "decision_grade": False,
        "trading_conclusion_allowed": False,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-dir", type=Path)
    parser.add_argument("--mode", choices=("suite", "valid", "corrupt"), default="suite")
    parser.add_argument("--attempt-id", default="phase-minus-one-manual")
    return parser.parse_args()


def _main() -> int:
    args = _parse_args()

    def execute(state_dir: Path) -> dict[str, Any]:
        if args.mode == "suite":
            return run_smoke_suite(state_dir)
        started = time.monotonic()
        result = run_attempt(
            state_dir,
            args.attempt_id,
            corrupt_result=args.mode == "corrupt",
        )
        expected = "invalid_result" if args.mode == "corrupt" else "protocol_verified"
        elapsed = time.monotonic() - started
        return {
            "status": "pass"
            if result["terminal"]["outcome_reason"] == expected and elapsed < 10.0
            else "fail",
            "elapsed_seconds": elapsed,
            "result": result,
            "evidence_kind": "synthetic_protocol_fixture",
            "decision_grade": False,
            "trading_conclusion_allowed": False,
        }

    if args.state_dir is not None:
        payload = execute(args.state_dir)
    else:
        with tempfile.TemporaryDirectory(prefix="gpt-control-plane-smoke-") as td:
            payload = execute(Path(td))
    json.dump(payload, sys.stdout, ensure_ascii=False, sort_keys=True)
    sys.stdout.write("\n")
    return 0 if payload["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(_main())
