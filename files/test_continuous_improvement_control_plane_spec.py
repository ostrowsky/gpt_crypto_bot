from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
CONTROL_PLANE = ROOT / "docs" / "specs" / "continuous-improvement-control-plane.md"
ROADMAP = ROOT / "docs" / "specs" / "learning-loop-architecture-roadmap.md"
POLICY_EPOCH = ROOT / "docs" / "specs" / "policy-epoch-label-provenance.md"
PHASE_ZERO = ROOT / "docs" / "specs" / "phase0-evidence-capacity.md"
SPEC_INDEX = ROOT / "docs" / "FEATURE_SPEC_INDEX.md"


class ContinuousImprovementControlPlaneSpecTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.spec = CONTROL_PLANE.read_text(encoding="utf-8")
        cls.roadmap = ROADMAP.read_text(encoding="utf-8")
        cls.policy_epoch = POLICY_EPOCH.read_text(encoding="utf-8")
        cls.phase_zero = PHASE_ZERO.read_text(encoding="utf-8")
        cls.index = SPEC_INDEX.read_text(encoding="utf-8")

    def test_walking_skeleton_precedes_measurement_platform(self) -> None:
        skeleton = self.spec.index("### Phase -1: walking skeleton")
        phase_zero = self.spec.index("### Phase 0: evidence capacity")
        self.assertLess(skeleton, phase_zero)
        self.assertIn("FixtureDeltaValidatorAdapter", self.spec)
        self.assertIn("files/improvement_fixture_validator.py", self.spec)
        self.assertIn("control_plane_smoke_fixture.json", self.spec)
        self.assertIn("under ten seconds", self.spec)
        self.assertIn("FixtureDeltaValidatorAdapter", self.index)
        self.assertIn("Walking skeleton first", self.roadmap)

    def test_result_is_verified_before_governor(self) -> None:
        self.assertIn('RESULT --> VERIFY["Independent Result Verification"]', self.spec)
        self.assertIn('VERIFY --> GOVERNOR["Deterministic Decision Policy"]', self.spec)
        self.assertNotIn("RESULT --> GOVERNOR", self.spec)
        self.assertIn("INVALID_RESULT", self.spec)
        self.assertIn("orchestrator-frozen immutable raw", self.spec)
        self.assertIn("registered candidate policy", self.spec)
        self.assertIn("decision trace is comparison evidence", self.spec)
        self.assertIn(
            "must not import the validator's aggregation implementation", self.spec
        )

    def test_single_owner_and_honest_debt_contract(self) -> None:
        self.assertIn("All rows have the same accountable owner", self.spec)
        self.assertIn("repository maintainer", self.spec)
        self.assertIn("ACCEPTED_DEBT", self.spec)
        self.assertIn("Acknowledge SLO", self.spec)
        self.assertIn("one triage queue", self.roadmap)

    def test_power_cost_and_cadence_are_first_class(self) -> None:
        self.assertIn("### 7.3 Power-expansion track", self.spec)
        self.assertIn("CycleBudget", self.spec)
        self.assertIn("BUDGET_EXHAUSTED", self.spec)
        self.assertIn("### 16.4 Program cadence and capacity", self.spec)
        self.assertIn("at most 12 decision-grade forward hypothesis versions", self.spec)
        self.assertIn("scarce ceiling, not a delivery target", self.spec)
        self.assertIn("cost per terminal", self.roadmap)

    def test_policy_epoch_has_semantic_invalidation_rules(self) -> None:
        for text in (self.spec, self.policy_epoch):
            self.assertIn("semantic", text.lower())
            self.assertIn("decision-trace", text)
            self.assertIn("historical_only", text)
        self.assertIn("Market regime is an observation", self.policy_epoch)

    def test_existing_negative_memory_must_be_migrated(self) -> None:
        self.assertIn("### 10.4 Bootstrap migration of research memory", self.spec)
        self.assertIn("LEGACY_UNVERIFIED", self.spec)
        self.assertIn("47 legacy", self.spec)
        self.assertIn(
            "every known legacy research artifact has a migration state", self.spec
        )

    def test_regime_reopen_and_operator_deviation_are_auditable(self) -> None:
        self.assertIn("reopen_basis=regime_shift", self.spec)
        self.assertIn("rule_verdict", self.spec)
        self.assertIn("operator_decision", self.spec)
        self.assertIn("decision_deviation", self.spec)
        self.assertIn("masked, randomized baseline/candidate digest", self.spec)

    def test_outcome_and_frozen_world_acceptance_are_explicit(self) -> None:
        self.assertIn("### 9.4 Frozen-world meta-evaluation", self.spec)
        self.assertIn("operator_baseline=NOT_AVAILABLE", self.spec)
        self.assertIn("AgentNecessityGate", self.spec)
        self.assertIn("LOOP_RECOVERY", self.spec)
        self.assertIn("last known-good vertical slice", self.spec)

    def test_state_model_separates_stage_status_and_reason(self) -> None:
        self.assertIn("stage", self.spec)
        self.assertIn("status", self.spec)
        self.assertIn("outcome_reason", self.spec)
        self.assertIn("EVIDENCE_CAPACITY_RECOVERY", self.spec)
        self.assertNotIn("SNAPSHOT_READY | SNAPSHOT_INVALID", self.spec)

    def test_program_level_power_failure_changes_the_roadmap(self) -> None:
        self.assertIn("fewer than 25% of the trailing eight", self.spec)
        self.assertIn("last four primary prechecks", self.spec)
        self.assertIn("pauses LLM-selected admissions", self.roadmap)
        self.assertIn("power-feasible share", self.roadmap)

    def test_phase_zero_is_measurement_only_and_fail_closed(self) -> None:
        self.assertIn("production trading policy unchanged", self.phase_zero)
        self.assertIn("ImmutableLabelLedger", self.phase_zero)
        self.assertIn("day_cluster_binary_v1", self.phase_zero)
        self.assertIn("null` ratios", self.phase_zero)
        self.assertIn("LEGACY_UNVERIFIED", self.phase_zero)
        self.assertIn("Full Phase 0 remains open", self.phase_zero)
        self.assertIn("phase0-evidence-capacity.md", self.index)
        self.assertIn("Phase 0 evidence capacity in progress", self.roadmap)
        self.assertIn("2026-08-14 Phase 0 evidence checkpoint", self.roadmap)
        self.assertIn("MDE=14.07pp", self.roadmap)
        self.assertIn("0` provenance-verified rows", self.roadmap)
        self.assertIn("Five reviewed", self.roadmap)
        self.assertIn("2026-08-20 maximum-period replay recovery checkpoint", self.roadmap)
        self.assertIn("net alpha `-52.30pp`", self.roadmap)
        self.assertIn("baseline measurements, not improvement deltas", self.roadmap)


if __name__ == "__main__":
    unittest.main()
