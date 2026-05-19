from __future__ import annotations

import unittest

from audit_v2_selector_failure_decomposition import _latest_choice, _stale_bucket, _recommendation


class SelectorFailureDecompositionTest(unittest.TestCase):
    def test_stale_bucket(self) -> None:
        self.assertEqual(_stale_bucket(None), "no_choice")
        self.assertEqual(_stale_bucket(1), "age_0_1")
        self.assertEqual(_stale_bucket(4), "age_2_4")
        self.assertEqual(_stale_bucket(8), "age_5_8")
        self.assertEqual(_stale_bucket(9), "age_9_plus")

    def test_recommendation_prioritizes_stale_loss(self) -> None:
        self.assertIn("TTL", _recommendation("candidate_suppressed_open", -100.0))
        self.assertIn("split entry and exit", _recommendation("candidate_suppressed_open", 0.0))
        self.assertIn("downside guard", _recommendation("candidate_enabled_open", 0.0))


if __name__ == "__main__":
    unittest.main()
