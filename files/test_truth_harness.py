from __future__ import annotations

import json
import hashlib
import subprocess
import tempfile
import unittest
from pathlib import Path

import truth_harness
from unittest.mock import patch


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

    def test_portfolio_alpha_rejects_legacy_shallow_claim(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            reports = root / ".runtime" / "reports"
            reports.mkdir(parents=True)
            (reports / "portfolio_alpha_latest.json").write_text(
                json.dumps({"net_alpha_after_costs": 1.0, "benchmark": "BTCUSDT"}),
                encoding="utf-8",
            )
            audit = truth_harness.Audit("full")
            truth_harness.audit_portfolio_alpha(audit, root)

        self.assertTrue(any(f.check_id == "TH11_PORTFOLIO_ALPHA" for f in audit.findings))

    def test_portfolio_alpha_accepts_complete_current_contract(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            reports = root / ".runtime" / "reports"
            reports.mkdir(parents=True)
            payload = {
                "metric_contract": "canonical_unified_ten_slot_alpha_v1",
                "decision_grade": True,
                "evidence_grade": "decision_grade",
                "portfolio_contract": {"capacity": 10, "same_symbol_concurrency": 1},
                "cost_contract": {"fee_bps_per_side": 7.5, "slippage_bps_per_side": 5.0},
                "portfolio": {"net_return_after_costs_pct": 2.0},
                "benchmark": {
                    "name": "BTCUSDT_buy_and_hold_same_closed_bar_window",
                    "status": "complete",
                    "net_return_after_costs_pct": 0.5,
                },
                "net_alpha_after_costs": 1.5,
                "window": {"requested_days": 30, "established_max_replay_days": 30, "period_coverage": 0.99},
                "coverage": {"valuation_coverage": 1.0, "contract_violations": []},
                "provenance": {
                    "policy_epoch": "pe-current",
                    "policy_hash": "hash",
                    "universe_hash": "universe",
                    "source_hashes": {},
                    "trade_stream_hash": "trades",
                    "price_stream_hash": "prices",
                },
            }
            for source_name in ("portfolio_alpha.py", "replay_backtest.py"):
                source = root / "files" / source_name
                source.parent.mkdir(parents=True, exist_ok=True)
                source.write_text(source_name, encoding="utf-8")
                payload["provenance"]["source_hashes"][source_name] = hashlib.sha256(source.read_bytes()).hexdigest()
            (reports / "canonical_portfolio_alpha_latest.json").write_text(json.dumps(payload), encoding="utf-8")
            audit = truth_harness.Audit("full")
            with patch("policy_provenance.current_policy_manifest", return_value={"policy_epoch": "pe-current"}):
                truth_harness.audit_portfolio_alpha(audit, root)

        self.assertFalse(any(f.check_id == "TH11_PORTFOLIO_ALPHA" for f in audit.findings))

    def test_portfolio_alpha_fails_on_current_policy_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            reports = root / ".runtime" / "reports"
            reports.mkdir(parents=True)
            payload = {
                "metric_contract": "canonical_unified_ten_slot_alpha_v1",
                "decision_grade": True,
                "evidence_grade": "decision_grade",
                "portfolio_contract": {"capacity": 10, "same_symbol_concurrency": 1},
                "cost_contract": {"fee_bps_per_side": 7.5, "slippage_bps_per_side": 5.0},
                "portfolio": {"net_return_after_costs_pct": 2.0},
                "benchmark": {
                    "name": "BTCUSDT_buy_and_hold_same_closed_bar_window",
                    "status": "complete",
                    "net_return_after_costs_pct": 0.5,
                },
                "net_alpha_after_costs": 1.5,
                "window": {"requested_days": 30, "established_max_replay_days": 30, "period_coverage": 1.0},
                "coverage": {"valuation_coverage": 1.0, "contract_violations": []},
                "provenance": {
                    "policy_epoch": "pe-old", "policy_hash": "hash", "universe_hash": "universe",
                    "source_hashes": {}, "trade_stream_hash": "trades", "price_stream_hash": "prices",
                },
            }
            for source_name in ("portfolio_alpha.py", "replay_backtest.py"):
                source = root / "files" / source_name
                source.parent.mkdir(parents=True, exist_ok=True)
                source.write_text(source_name, encoding="utf-8")
                payload["provenance"]["source_hashes"][source_name] = hashlib.sha256(source.read_bytes()).hexdigest()
            (reports / "canonical_portfolio_alpha_latest.json").write_text(json.dumps(payload), encoding="utf-8")
            audit = truth_harness.Audit("full")
            with patch("policy_provenance.current_policy_manifest", return_value={"policy_epoch": "pe-current"}):
                truth_harness.audit_portfolio_alpha(audit, root)

        evidence = " ".join(f.evidence for f in audit.findings)
        self.assertIn("current_policy_epoch", evidence)


if __name__ == "__main__":
    unittest.main()
