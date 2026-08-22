from __future__ import annotations

import json
import os
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

import phase1_experiment_loop as loop
import phase1_static_target_verifier as verifier


UTC = timezone.utc


def _snapshot(days: int = 40) -> dict:
    rows = []
    start = datetime(2026, 1, 1, tzinfo=UTC)
    for day_index in range(days):
        candidates = []
        for index in range(20):
            is_target = index in (18, 19)
            candidates.append({
                "symbol": f"S{index:02d}USDT",
                "current_return": float(20 - index),
                "static_target_return": float(index),
                "target_rank": index + 1 if is_target else 100 + index,
                "is_target_top": is_target,
            })
        rows.append({
            "local_day": (start + timedelta(days=day_index)).date().isoformat(),
            "market_symbol_count": 250,
            "watchlist_symbol_count": 80,
            "target_entrant_symbols": ["S18USDT", "S19USDT"],
            "candidates": candidates,
        })
    return {
        "schema_version": 1,
        "snapshot_kind": "static_target_top50_normalized_v1",
        "decision_grade_input": False,
        "source_contract": {
            "timezone": "Europe/Budapest",
            "observation_time_local": "12:15",
            "target_time_local": "23:00",
            "top_n": 50,
            "selection_size": 10,
            "requested_day_count": days,
        },
        "provenance": {
            "used_content_hash": "a" * 64,
            "watchlist_sha256": "b" * 64,
            "policy_epoch": "pe-test",
            "eligible_days": days,
            "rejected_days": [],
        },
        "days": rows,
    }


class Phase1ContractTest(unittest.TestCase):
    def test_invented_capability_is_rejected(self) -> None:
        contract = loop.build_static_target_contract(
            attempt_id="a1", snapshot_sha256="a" * 64, snapshot_id="snap-a1"
        )
        contract["target_capability"] = "invented.capability"
        result = loop.precheck_contract(contract)
        self.assertEqual(result["outcome_reason"], "contract_rejected")

    def test_missing_validator_waits_without_fallback(self) -> None:
        contract = loop.build_static_target_contract(
            attempt_id="a1", snapshot_sha256="a" * 64, snapshot_id="snap-a1"
        )
        result = loop.precheck_contract(contract, validator_registry={})
        self.assertEqual(result["status"], "WAITING")
        self.assertEqual(result["outcome_reason"], "needs_validator")

    def test_underpowered_attempt_stops_before_validator(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = loop.run_attempt(Path(tmp), "underpowered", snapshot_payload=_snapshot(5))
            attempt = Path(tmp) / "attempts" / "underpowered"
            self.assertEqual(result["terminal"]["outcome_reason"], "underpowered")
            self.assertFalse((attempt / "validator_result.json").exists())


class Phase1DurabilityTest(unittest.TestCase):
    def test_relative_state_directory_is_resolved_before_validator_subprocess(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            previous = Path.cwd()
            try:
                os.chdir(tmp)
                result = loop.run_attempt(
                    Path("relative-state"), "relative", snapshot_payload=_snapshot()
                )
            finally:
                os.chdir(previous)
        self.assertEqual(result["terminal"]["outcome_reason"], "supported")

    def test_valid_attempt_is_verified_supported_and_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state = Path(tmp)
            first = loop.run_attempt(state, "real-fixture", snapshot_payload=_snapshot())
            ledger_before = (state / "attempt_ledger.jsonl").read_bytes()
            second = loop.run_attempt(state, "real-fixture", snapshot_payload=_snapshot())
            ledger_after = (state / "attempt_ledger.jsonl").read_bytes()

            self.assertEqual(first["terminal"]["outcome_reason"], "supported")
            self.assertEqual(first["verification"]["status"], "VERIFIED_RESULT")
            self.assertFalse(first["terminal"]["decision_grade"])
            self.assertEqual(second["terminal"], first["terminal"])
            self.assertEqual(ledger_after, ledger_before)

    def test_corrupt_validator_result_is_dead_lettered(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state = Path(tmp)
            result = loop.run_attempt(
                state,
                "corrupt-result",
                snapshot_payload=_snapshot(),
                corrupt_result=True,
            )
            attempt = state / "attempts" / "corrupt-result"
            self.assertEqual(result["terminal"]["outcome_reason"], "invalid_result")
            self.assertEqual(result["verification"]["status"], "INVALID_RESULT")
            self.assertTrue((attempt / "dead_letter.json").exists())

    def test_conflicting_snapshot_is_rejected_on_resume(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state = Path(tmp)
            loop.run_attempt(state, "immutable", snapshot_payload=_snapshot())
            changed = _snapshot()
            changed["days"][0]["candidates"][0]["current_return"] = 999.0
            with self.assertRaises(loop.AttemptIntegrityError):
                loop.run_attempt(state, "immutable", snapshot_payload=changed)

    def test_active_lease_blocks_and_expired_lease_reconciles(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            attempt_dir = Path(tmp) / "attempts" / "leased"
            attempt_dir.mkdir(parents=True)
            now = datetime(2026, 8, 22, 12, tzinfo=UTC)
            loop.acquire_lease(attempt_dir, owner="one", now=now, lease_seconds=60)
            with self.assertRaises(loop.LeaseBusyError):
                loop.acquire_lease(attempt_dir, owner="two", now=now, lease_seconds=60)
            recovered = loop.acquire_lease(
                attempt_dir,
                owner="two",
                now=now + timedelta(seconds=61),
                lease_seconds=60,
            )
            self.assertTrue(recovered["recovered_expired_lease"])

    def test_status_has_unknown_empty_ratios_and_terminal_telemetry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state = Path(tmp)
            empty = loop.build_status(state)
            self.assertEqual(empty["attempt_count"], 0)
            self.assertIsNone(empty["terminal_rate"])
            loop.run_attempt(state, "complete", snapshot_payload=_snapshot())
            status = loop.build_status(state)
            self.assertEqual(status["terminal_count"], 1)
            self.assertEqual(status["terminal_reason_counts"], {"supported": 1})
            self.assertIsNotNone(status["last_transition_at"])

    def test_retry_has_new_attempt_and_visible_retry_link(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state = Path(tmp)
            first = loop.run_attempt(
                state, "failed", snapshot_payload=_snapshot(), corrupt_result=True
            )
            second = loop.run_attempt(
                state,
                "retry",
                snapshot_payload=_snapshot(),
                retry_of="failed",
                retry_reason="validator_path_repaired",
            )
            rows = [
                json.loads(line)
                for line in (state / "attempt_ledger.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(first["terminal"]["outcome_reason"], "invalid_result")
            self.assertEqual(second["terminal"]["outcome_reason"], "supported")
            self.assertTrue(any(row.get("attempt_id") == "retry" and row.get("retry_of") == "failed" for row in rows))

    def test_verifier_does_not_import_validator_aggregation(self) -> None:
        source = (Path(__file__).with_name("phase1_static_target_verifier.py")).read_text(
            encoding="utf-8"
        )
        self.assertNotIn("phase1_static_target_validator", source)
        self.assertNotIn("validate_external_top50_screen", source)

    def test_verifier_rejects_snapshot_manifest_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state = Path(tmp)
            loop.run_attempt(state, "manifest", snapshot_payload=_snapshot())
            attempt = state / "attempts" / "manifest"
            manifest_path = attempt / "snapshot_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["snapshot_sha256"] = "f" * 64
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            key = bytes.fromhex((attempt / "validator_attestation.key").read_text(encoding="ascii"))
            result = verifier.verify(
                attempt / "snapshot.json",
                manifest_path,
                attempt / "hypothesis_contract.json",
                attempt / "validator_result.json",
                key,
            )
        self.assertEqual(result["status"], "INVALID_RESULT")
        self.assertIn("manifest_snapshot_sha256", result["errors"])


if __name__ == "__main__":
    unittest.main()
