from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path

import truth_harness


def _git(root: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=root, check=True, capture_output=True, text=True)


class TruthHarnessTest(unittest.TestCase):
    def test_payload_fails_on_blocking_finding(self) -> None:
        audit = truth_harness.Audit("test")
        audit.add("X", "TH-01", "error", "broken")

        self.assertEqual(audit.payload()["status"], "fail")
        self.assertEqual(audit.payload()["blocking_count"], 1)

    def test_runtime_artifact_is_rejected_from_staged_scope(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _git(root, "init")
            runtime = root / ".runtime" / "reports"
            runtime.mkdir(parents=True)
            (runtime / "x.json").write_text("{}", encoding="utf-8")
            _git(root, "add", ".runtime/reports/x.json")

            audit = truth_harness.Audit("change")
            truth_harness.audit_change(audit, root, staged=True)

        self.assertTrue(any(f.check_id == "TH12_STAGED_SCOPE" and "Runtime" in f.message for f in audit.findings))

    def test_material_change_requires_spec_test_and_index(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _git(root, "init")
            files = root / "files"
            files.mkdir()
            (files / "monitor.py").write_text("x = 1\n", encoding="utf-8")
            _git(root, "add", "files/monitor.py")

            audit = truth_harness.Audit("change")
            truth_harness.audit_change(audit, root, staged=True)

        messages = [finding.message for finding in audit.findings]
        self.assertIn("Material change has no staged feature spec", messages)
        self.assertIn("Material change has no staged focused tests", messages)
        self.assertIn("Material change does not update the feature spec index", messages)

    def test_runtime_name_detection_covers_models_and_positions(self) -> None:
        self.assertTrue(truth_harness._is_runtime("files/positions.json"))
        self.assertTrue(truth_harness._is_runtime("files/ml_candidate_ranker_report.json"))
        self.assertFalse(truth_harness._is_runtime("files/monitor.py"))

    def test_model_provenance_rejects_overlap_and_empty_verified_cohort(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            reports = root / ".runtime" / "reports"
            reports.mkdir(parents=True)
            (reports / "rl_train_latest.json").write_text(
                json.dumps({
                    "evaluation_provenance": {
                        "feature_time": "closed bars",
                        "label_time": "future bars",
                        "label_definition": {"ret_5": "T+5"},
                        "evaluation_scope": "out_of_sample_time_holdout",
                        "cross_split_group_overlap_count": 1,
                        "cross_split_group_overlap_examples": ["2026-08-13T10:00:00Z"],
                        "data_provenance": {"verified_rows": 0},
                    }
                }),
                encoding="utf-8",
            )
            audit = truth_harness.Audit("full")
            truth_harness.audit_model_provenance(audit, root)

        messages = [finding.message for finding in audit.findings]
        self.assertIn("Chronological holdout shares decision groups across splits", messages)
        self.assertIn("Model evidence contains no provenance-verified rows", messages)


if __name__ == "__main__":
    unittest.main()
