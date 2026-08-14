from __future__ import annotations

import ast
import json
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parent.parent
FILES = ROOT / "files"
FIXTURE = FILES / "testdata" / "control_plane_smoke_fixture.json"
VALIDATOR_SOURCE = FILES / "improvement_fixture_validator.py"
VERIFIER_SOURCE = FILES / "improvement_fixture_verifier.py"
RUNNER_SOURCE = FILES / "run_control_plane_smoke.py"

if str(FILES) not in sys.path:
    sys.path.insert(0, str(FILES))

from improvement_fixture_validator import FixtureDeltaValidatorAdapter  # noqa: E402
from improvement_fixture_verifier import verify_result_bundle  # noqa: E402
import run_control_plane_smoke as control_plane  # noqa: E402
from run_control_plane_smoke import (  # noqa: E402
    OUTCOME_REASON_REGISTRY_VERSION,
    PHASE_MINUS_ONE_OUTCOME_REASONS,
    AttemptIntegrityError,
    run_attempt,
)


class ControlPlaneWalkingSkeletonTest(unittest.TestCase):
    def test_checked_in_fixture_is_small_raw_and_has_expected_delta(self) -> None:
        payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
        self.assertEqual(payload["schema_version"], 1)
        self.assertEqual(payload["fixture_id"], "control-plane-smoke-v1")
        rows = payload["rows"]
        self.assertGreater(len(rows), 0)
        self.assertLessEqual(len(rows), 64)
        self.assertEqual(len({row["row_id"] for row in rows}), len(rows))
        self.assertTrue(all(set(row) == {"row_id", "score", "label"} for row in rows))
        self.assertTrue(all(0.0 <= float(row["score"]) <= 1.0 for row in rows))
        self.assertTrue(all(row["label"] in (0, 1) for row in rows))

        baseline = [int(float(row["score"]) >= 0.75) for row in rows]
        candidate = [int(float(row["score"]) >= 0.55) for row in rows]
        baseline_accuracy = sum(
            prediction == row["label"] for prediction, row in zip(baseline, rows)
        ) / len(rows)
        candidate_accuracy = sum(
            prediction == row["label"] for prediction, row in zip(candidate, rows)
        ) / len(rows)
        self.assertAlmostEqual(baseline_accuracy, 0.75)
        self.assertAlmostEqual(candidate_accuracy, 1.0)
        self.assertAlmostEqual(candidate_accuracy - baseline_accuracy, 0.25)

    def test_valid_attempt_reaches_non_trading_verified_terminal(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            result = run_attempt(Path(td), "attempt-valid")

        terminal = result["terminal"]
        self.assertEqual(terminal["stage"], "CLOSED")
        self.assertEqual(terminal["status"], "TERMINAL")
        self.assertEqual(terminal["outcome_reason"], "protocol_verified")
        self.assertFalse(terminal["decision_grade"])
        self.assertFalse(terminal["trading_conclusion_allowed"])
        self.assertTrue(result["verification"]["valid"])
        self.assertAlmostEqual(result["verification"]["delta"], 0.25)

    def test_signed_but_corrupted_validator_bundle_is_independently_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            result = run_attempt(Path(td), "attempt-corrupt", corrupt_result=True)

        terminal = result["terminal"]
        self.assertEqual(terminal["stage"], "CLOSED")
        self.assertEqual(terminal["status"], "TERMINAL")
        self.assertEqual(terminal["outcome_reason"], "invalid_result")
        self.assertFalse(result["verification"]["valid"])
        self.assertIn("candidate_metric", result["verification"]["errors"])
        self.assertTrue(
            result["verification"]["attestation_valid"],
            "the independent recompute, not signature failure, must catch corruption",
        )

    def test_same_attempt_is_restart_safe_and_does_not_duplicate_transitions(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            state_dir = Path(td)
            first = run_attempt(state_dir, "attempt-resume")
            ledger = state_dir / "attempt_ledger.jsonl"
            first_lines = ledger.read_text(encoding="utf-8").splitlines()

            second = run_attempt(state_dir, "attempt-resume")
            second_lines = ledger.read_text(encoding="utf-8").splitlines()

        self.assertEqual(first["terminal"], second["terminal"])
        self.assertEqual(first_lines, second_lines)
        self.assertEqual(len(first_lines), 3)

    def test_crash_after_validator_result_resumes_at_verification_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            state_dir = Path(td)
            with patch.object(
                control_plane,
                "verify_result_bundle",
                side_effect=RuntimeError("simulated verifier outage"),
            ):
                with self.assertRaisesRegex(RuntimeError, "simulated verifier outage"):
                    control_plane.run_attempt(state_dir, "attempt-boundary")

            ledger = state_dir / "attempt_ledger.jsonl"
            interrupted_lines = ledger.read_text(encoding="utf-8").splitlines()
            result_path = (
                state_dir
                / "attempts"
                / "attempt-boundary"
                / "validator_result.json"
            )
            self.assertEqual(len(interrupted_lines), 2)
            self.assertTrue(result_path.exists())

            resumed = control_plane.run_attempt(state_dir, "attempt-boundary")
            final_lines = ledger.read_text(encoding="utf-8").splitlines()

        self.assertEqual(resumed["terminal"]["outcome_reason"], "protocol_verified")
        self.assertEqual(len(final_lines), 3)

    def test_existing_attempt_rejects_a_changed_fixture(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            state_dir = Path(td)
            run_attempt(state_dir, "attempt-conflict")
            altered = state_dir / "altered_fixture.json"
            payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
            payload["rows"][0]["score"] = 0.01
            altered.write_text(json.dumps(payload), encoding="utf-8")

            with self.assertRaises(AttemptIntegrityError):
                run_attempt(state_dir, "attempt-conflict", fixture_path=altered)

    def test_repository_and_production_runtime_state_paths_are_refused(self) -> None:
        forbidden = ROOT / ".runtime" / "control-plane-smoke-must-not-exist"
        with self.assertRaisesRegex(AttemptIntegrityError, "outside the repository"):
            run_attempt(forbidden, "attempt-forbidden")
        self.assertFalse(forbidden.exists())

    def test_phase_minus_one_reason_registry_is_closed_and_versioned(self) -> None:
        self.assertEqual(OUTCOME_REASON_REGISTRY_VERSION, 1)
        self.assertEqual(
            PHASE_MINUS_ONE_OUTCOME_REASONS,
            frozenset(
                {
                    "observed",
                    "validation_started",
                    "protocol_verified",
                    "invalid_result",
                }
            ),
        )

    def test_validator_and_verifier_are_separate_and_import_no_trading_stack(self) -> None:
        banned = {
            "replay_backtest",
            "monitor",
            "strategy",
            "config",
            "portfolio_alpha",
            "requests",
            "socket",
            "urllib",
        }

        for source in (VALIDATOR_SOURCE, VERIFIER_SOURCE):
            tree = ast.parse(source.read_text(encoding="utf-8"))
            imports: set[str] = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imports.update(alias.name.split(".")[0] for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    imports.add(node.module.split(".")[0])
            self.assertTrue(banned.isdisjoint(imports), (source.name, imports & banned))

        verifier_text = VERIFIER_SOURCE.read_text(encoding="utf-8")
        self.assertNotIn("improvement_fixture_validator", verifier_text)
        self.assertIsNot(FixtureDeltaValidatorAdapter, verify_result_bundle)

    def test_cli_suite_finishes_under_budget_and_creates_no_release_state(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            started = time.monotonic()
            completed = subprocess.run(
                [
                    sys.executable,
                    str(RUNNER_SOURCE),
                    "--state-dir",
                    td,
                    "--mode",
                    "suite",
                ],
                cwd=ROOT,
                check=False,
                capture_output=True,
                text=True,
                timeout=10,
            )
            elapsed = time.monotonic() - started
            state_paths = [path.relative_to(td).as_posix() for path in Path(td).rglob("*")]

        self.assertEqual(completed.returncode, 0, completed.stderr or completed.stdout)
        payload = json.loads(completed.stdout)
        self.assertEqual(payload["status"], "pass")
        self.assertLess(payload["elapsed_seconds"], 10.0)
        self.assertLess(elapsed, 10.0)
        self.assertEqual(payload["valid"]["terminal"]["outcome_reason"], "protocol_verified")
        self.assertEqual(payload["corrupt"]["terminal"]["outcome_reason"], "invalid_result")
        self.assertFalse(any("release" in path.lower() for path in state_paths))
        self.assertFalse(any("promotion" in path.lower() for path in state_paths))


if __name__ == "__main__":
    unittest.main()
