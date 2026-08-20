from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from evidence_capacity import (
    ImmutableLabelLedger,
    LabelConflictError,
    action_layer_metric_registry,
    build_evidence_throughput_report,
    build_harness_remediation_ledger,
    build_move_event_labels,
    build_power_report,
    build_top_mover_labels,
    migrate_legacy_research_inventory,
    verify_objective_report_contract,
)


UTC = timezone.utc


def _row(
    symbol: str,
    final_close: float,
    *,
    high: float | None = None,
    coverage_status: str = "complete",
) -> dict:
    return {
        "symbol": symbol,
        "objective_day": "2026-08-01",
        "event_day_timezone": "UTC",
        "universe_snapshot_hash": "a" * 64,
        "source_snapshot_hash": "b" * 64,
        "reference_time": "2026-08-01T00:00:00+00:00",
        "reference_price": 100.0,
        "label_cutoff": "2026-08-02T00:00:00+00:00",
        "coverage_status": coverage_status,
        "bars": [
            {
                "close_time": "2026-08-01T06:00:00+00:00",
                "high": 102.6,
                "close": 102.0,
            },
            {
                "close_time": "2026-08-01T12:00:00+00:00",
                "high": high if high is not None else final_close,
                "close": final_close,
            },
        ],
    }


class CanonicalLabelTest(unittest.TestCase):
    def test_move_event_is_stable_and_uses_closed_pre_cutoff_bars(self) -> None:
        row = _row("AAAUSDT", 105.5, high=106.0)
        row["bars"].append(
            {
                "close_time": "2026-08-02T01:00:00+00:00",
                "high": 150.0,
                "close": 150.0,
            }
        )
        first = build_move_event_labels(
            [row], as_of="2026-08-02T00:01:00+00:00"
        )[0]
        second = build_move_event_labels(
            [row], as_of="2026-08-03T00:01:00+00:00"
        )[0]

        self.assertEqual(first, second)
        self.assertEqual(first["status"], "CONFIRMED")
        self.assertTrue(first["decision_grade"])
        self.assertEqual(first["included_bar_count"], 2)
        self.assertEqual(
            first["first_midpoint_crossing_time"], "2026-08-01T06:00:00+00:00"
        )
        self.assertEqual(
            first["first_event_crossing_time"], "2026-08-01T12:00:00+00:00"
        )

    def test_partial_or_immature_input_never_enters_event_denominator(self) -> None:
        partial = build_move_event_labels(
            [_row("AAAUSDT", 106.0, coverage_status="partial")],
            as_of="2026-08-02T00:01:00+00:00",
        )[0]
        immature = build_move_event_labels(
            [_row("BBBUSDT", 106.0)], as_of="2026-08-01T23:59:00+00:00"
        )[0]

        self.assertEqual(partial["status"], "PARTIAL")
        self.assertFalse(partial["decision_grade"])
        self.assertEqual(immature["status"], "NOT_MATURE")
        self.assertFalse(immature["decision_grade"])

    def test_timezone_naive_input_is_rejected(self) -> None:
        row = _row("AAAUSDT", 106.0)
        row["label_cutoff"] = "2026-08-02T00:00:00"
        with self.assertRaises(ValueError):
            build_move_event_labels([row], as_of="2026-08-02T00:01:00+00:00")

    def test_top_mover_labels_bind_population_and_reject_partial_day(self) -> None:
        result = build_top_mover_labels(
            [_row("AAAUSDT", 110.0), _row("BBBUSDT", 103.0)],
            as_of="2026-08-02T00:01:00+00:00",
            top_k=1,
        )
        self.assertEqual(result["status"], "COMPLETE")
        self.assertEqual(result["population_size"], 2)
        self.assertEqual(result["labels"][0]["symbol"], "AAAUSDT")
        self.assertTrue(result["labels"][0]["is_top_mover"])
        self.assertFalse(result["labels"][1]["is_top_mover"])

        partial = build_top_mover_labels(
            [
                _row("AAAUSDT", 110.0),
                _row("BBBUSDT", 103.0, coverage_status="partial"),
            ],
            as_of="2026-08-02T00:01:00+00:00",
            top_k=1,
        )
        self.assertEqual(partial["status"], "PARTIAL")
        self.assertFalse(partial["decision_grade"])
        self.assertEqual(partial["labels"], [])

    def test_immutable_ledger_is_idempotent_and_rejects_conflict(self) -> None:
        label = build_move_event_labels(
            [_row("AAAUSDT", 106.0)], as_of="2026-08-02T00:01:00+00:00"
        )[0]
        with tempfile.TemporaryDirectory() as tmp:
            ledger = ImmutableLabelLedger(Path(tmp) / "labels.jsonl")
            self.assertEqual(ledger.append([label]), 1)
            self.assertEqual(ledger.append([label]), 0)
            before = ledger.path.read_bytes()
            conflict = dict(label)
            conflict["status"] = "NOT_EVENT"
            with self.assertRaises(LabelConflictError):
                ledger.append([conflict])
            self.assertEqual(ledger.path.read_bytes(), before)


class CapacityAndGovernanceTest(unittest.TestCase):
    def test_power_report_uses_days_not_symbol_rows(self) -> None:
        observations = [
            {"objective_day": day, "outcome": outcome, "coverage_status": "complete"}
            for day, outcome in (
                ("2026-08-01", True),
                ("2026-08-01", False),
                ("2026-08-02", False),
                ("2026-08-02", True),
                ("2026-08-03", False),
            )
        ]
        report = build_power_report(
            observations,
            sesoi=0.10,
            as_of="2026-08-04T00:00:00+00:00",
        )
        self.assertEqual(report["raw_event_count"], 5)
        self.assertEqual(report["complete_objective_days"], 3)
        self.assertEqual(report["effective_sample_size"], 3)
        self.assertEqual(report["base_rate_numerator"], 2)
        self.assertEqual(report["base_rate_denominator"], 5)
        self.assertEqual(report["status"], "UNDERPOWERED")
        self.assertGreater(report["mde"], report["sesoi"])

    def test_empty_throughput_has_unknown_ratios(self) -> None:
        report = build_evidence_throughput_report([])
        self.assertEqual(report["status"], "NO_EVIDENCE")
        self.assertIsNone(report["terminal_rate"])
        self.assertEqual(report["terminal_rate_denominator"], 0)
        self.assertIsNone(report["power_feasible_rate"])

    def test_throughput_preserves_denominators_and_terminal_reasons(self) -> None:
        events = [
            {
                "attempt_id": "a1",
                "event_type": "attempt_started",
                "occurred_at": "2026-08-01T00:00:00+00:00",
                "power_feasible": True,
                "expected_labels": 10,
                "observed_labels": 8,
                "evidence_reused": True,
            },
            {
                "attempt_id": "a1",
                "event_type": "attempt_terminal",
                "occurred_at": "2026-08-01T01:00:00+00:00",
                "outcome_reason": "supported",
            },
            {
                "attempt_id": "a2",
                "event_type": "attempt_started",
                "occurred_at": "2026-08-02T00:00:00+00:00",
                "power_feasible": False,
                "expected_labels": 5,
                "observed_labels": 5,
                "evidence_reused": False,
            },
        ]
        report = build_evidence_throughput_report(events)
        self.assertEqual(report["terminal_rate_numerator"], 1)
        self.assertEqual(report["terminal_rate_denominator"], 2)
        self.assertEqual(report["terminal_reason_counts"], {"supported": 1})
        self.assertEqual(report["label_loss_numerator"], 2)
        self.assertEqual(report["label_loss_denominator"], 15)
        self.assertEqual(report["evidence_reuse_numerator"], 1)
        self.assertEqual(report["evidence_reuse_denominator"], 2)

    def test_phase_minus_one_ledger_is_counted_without_invented_duration(self) -> None:
        events = [
            {
                "attempt_id": "smoke-1",
                "stage": "OBSERVED",
                "status": "ACTIVE",
                "outcome_reason": "observed",
            },
            {
                "attempt_id": "smoke-1",
                "stage": "CLOSED",
                "status": "TERMINAL",
                "outcome_reason": "protocol_verified",
            },
        ]
        report = build_evidence_throughput_report(events)
        self.assertEqual(report["terminal_rate_numerator"], 1)
        self.assertEqual(report["terminal_rate_denominator"], 1)
        self.assertEqual(report["missing_duration_count"], 1)
        self.assertIsNone(report["median_hours_to_terminal"])

    def test_metric_registry_has_exact_action_boundaries(self) -> None:
        registry = action_layer_metric_registry()
        self.assertGreaterEqual(len(registry), 8)
        metric_ids = {entry["metric_id"] for entry in registry}
        self.assertEqual(len(metric_ids), len(registry))
        for entry in registry:
            self.assertIn(
                entry["action_layer"], {"OBSERVATION", "WATCH", "BUY", "SELL", "PORTFOLIO"}
            )
            self.assertTrue(entry["numerator"])
            self.assertTrue(entry["denominator"])
            self.assertTrue(entry["label_version"])
        move5 = next(entry for entry in registry if entry["metric_id"] == "coverage_move5_v1")
        self.assertEqual(move5["action_layer"], "WATCH")
        self.assertEqual(move5["decision_use"], "steering_only")
        self.assertNotIn("BUY", move5["allowed_decisions"])
        mission = next(
            entry
            for entry in registry
            if entry["metric_id"] == "watchlist_top_early_capture_v1"
        )
        self.assertEqual(
            mission["label_version"], "exchange_top_filtered_watchlist_v1"
        )

    def test_harness_findings_receive_owned_non_waiving_remediation(self) -> None:
        payload = {
            "generated_at": "2026-08-14T12:00:00+00:00",
            "status": "fail",
            "findings": [
                {
                    "check_id": "TH03_MODEL_PROVENANCE",
                    "invariant": "TH-03/TH-04",
                    "severity": "error",
                    "message": "missing provenance",
                    "evidence": "missing=['feature_time']",
                    "remediation": "",
                }
            ],
        }
        ledger = build_harness_remediation_ledger(
            payload, review_at="2026-08-21T09:00:00+00:00"
        )
        self.assertEqual(ledger["source_harness_status"], "fail")
        self.assertEqual(len(ledger["findings"]), 1)
        finding = ledger["findings"][0]
        self.assertEqual(finding["owner"], "repository maintainer")
        self.assertEqual(finding["state"], "OPEN")
        self.assertTrue(finding["blocked_actions"])
        self.assertTrue(finding["allowed_work"])
        self.assertTrue(finding["repair_task"])
        self.assertTrue(finding["verification_command"])
        self.assertFalse(finding["waived"])

    def test_legacy_inventory_covers_every_discovered_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            reports = root / "docs" / "reports"
            specs = root / "docs" / "specs"
            reports.mkdir(parents=True)
            specs.mkdir(parents=True)
            (reports / "a.md").write_text("Result: rejected. no denominator", encoding="utf-8")
            (reports / "b.md").write_text("Result: rejected. no denominator", encoding="utf-8")
            (specs / "idea.md").write_text("Status: research-only", encoding="utf-8")
            index = root / "docs" / "FEATURE_SPEC_INDEX.md"
            index.write_text(
                "| Idea | rejected | `docs/specs/idea.md` | metric | next |\n",
                encoding="utf-8",
            )
            inventory = migrate_legacy_research_inventory(root, index)
            self.assertEqual(inventory["discovered_count"], 3)
            self.assertEqual(inventory["migrated_count"], 3)
            self.assertEqual(sum(inventory["state_counts"].values()), 3)
            self.assertIn("DUPLICATE", inventory["state_counts"])
            self.assertIn("LEGACY_UNVERIFIED", inventory["state_counts"])

    def test_reviewed_negative_requires_matching_source_hash(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            reports = root / "docs" / "reports"
            specs = root / "docs" / "specs"
            reports.mkdir(parents=True)
            specs.mkdir(parents=True)
            source = reports / "negative.md"
            source.write_text("Result: rejected with complete evidence", encoding="utf-8")
            source_hash = __import__("hashlib").sha256(source.read_bytes()).hexdigest()
            index = root / "docs" / "FEATURE_SPEC_INDEX.md"
            index.write_text("", encoding="utf-8")
            registry = specs / "legacy-negative-results-registry.json"
            registry.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "entries": [
                            {
                                "negative_id": "negative-v1",
                                "source": "docs/reports/negative.md",
                                "source_sha256": source_hash,
                                "period": {"start": "2026-01-01", "end": "2026-01-31"},
                                "population": "all eligible cases",
                                "metric": "registered_net_delta",
                                "verdict": "rejected",
                                "evidence_summary": "candidate failed the registered gate",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            inventory = migrate_legacy_research_inventory(root, index, registry)
            self.assertEqual(inventory["state_counts"]["CONFIRMED_NEGATIVE"], 1)
            self.assertEqual(inventory["registry"]["accepted_count"], 1)

            source.write_text("changed after review", encoding="utf-8")
            stale = migrate_legacy_research_inventory(root, index, registry)
            self.assertEqual(stale["state_counts"]["LEGACY_UNVERIFIED"], 1)
            self.assertEqual(stale["registry"]["hash_mismatch_count"], 1)

    def test_objective_report_contract_fails_closed(self) -> None:
        valid = {
            "metric_id": "coverage_move5_v1",
            "action_layer": "WATCH",
            "metric_version": "v1",
            "label_version": "move5_v1",
            "method_version": "day_cluster_binary_v1",
            "numerator": 5,
            "denominator": 10,
            "coverage_numerator": 20,
            "coverage_denominator": 20,
            "coverage_status": "complete",
            "exclusions": [],
            "feature_cutoff": "2026-08-01T00:00:00+00:00",
            "label_cutoff": "2026-08-02T00:00:00+00:00",
            "label_available_at": "2026-08-02T00:00:00+00:00",
            "estimate": 0.5,
            "interval_low": 0.2,
            "interval_high": 0.8,
            "sesoi": 0.1,
            "mde": 0.2,
            "effective_sample_size": 10,
            "expected_decision_horizon_days": 21,
            "evidence_status": "INSUFFICIENT_EVIDENCE",
            "verdict_rule": "registered_interval_v1",
            "verdict_rule_passed": False,
        }
        self.assertEqual(verify_objective_report_contract(valid), [])
        invalid = dict(valid)
        invalid.pop("denominator")
        invalid["evidence_status"] = "IMPROVING"
        errors = verify_objective_report_contract(invalid)
        self.assertTrue(any("denominator" in error for error in errors))
        self.assertTrue(any("IMPROVING" in error for error in errors))

        unsupported_improvement = dict(valid)
        unsupported_improvement["evidence_status"] = "IMPROVING"
        unsupported_improvement["verdict_rule_passed"] = False
        errors = verify_objective_report_contract(unsupported_improvement)
        self.assertTrue(any("verdict rule" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
