from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone

import research_artifact_provenance as provenance


class ResearchArtifactProvenanceTests(unittest.TestCase):
    def test_current_artifact_passes_policy_and_age_gate(self) -> None:
        now = datetime(2026, 8, 3, 12, 0, tzinfo=timezone.utc)
        artifact = {
            "generated_at_utc": now.isoformat().replace("+00:00", "Z"),
            "provenance": provenance.build_provenance(
                builder="builder_v1",
                research_config={"days": 14},
                generated_at=now,
            ),
        }

        result = provenance.artifact_freshness(
            artifact,
            expected_builder="builder_v1",
            expected_research_config={"days": 14},
            max_age_hours=36,
            now=now,
        )

        self.assertEqual(result["status"], "fresh")
        self.assertEqual(result["reasons"], [])

    def test_age_and_research_config_mismatch_are_explicit(self) -> None:
        now = datetime(2026, 8, 3, 12, 0, tzinfo=timezone.utc)
        generated = now - timedelta(hours=48)
        artifact = {
            "provenance": provenance.build_provenance(
                builder="builder_v1",
                research_config={"days": 7},
                generated_at=generated,
            )
        }

        result = provenance.artifact_freshness(
            artifact,
            expected_builder="builder_v1",
            expected_research_config={"days": 14},
            max_age_hours=36,
            now=now,
        )

        self.assertEqual(result["status"], "stale")
        self.assertIn("age_budget_exceeded", result["reasons"])
        self.assertIn("research_config_hash_mismatch", result["reasons"])

    def test_policy_snapshot_excludes_sensitive_names(self) -> None:
        snapshot = provenance.current_policy_snapshot()

        self.assertTrue(snapshot)
        self.assertFalse(any("TOKEN" in key or "SECRET" in key or "KEY" in key for key in snapshot))


if __name__ == "__main__":
    unittest.main()
