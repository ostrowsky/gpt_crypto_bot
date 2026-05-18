from __future__ import annotations

import unittest

from audit_v2_market_environment_separability import _rank_feature_deltas


class MarketEnvironmentSeparabilityTest(unittest.TestCase):
    def test_rank_feature_deltas_orders_by_abs_delta(self) -> None:
        ranked = _rank_feature_deltas(
            {
                "candidate_favorable": {"a": 2.0, "b": 1.0},
                "candidate_unfavorable": {"a": 1.0, "b": 1.5},
            }
        )
        self.assertEqual([item["feature"] for item in ranked], ["a", "b"])


if __name__ == "__main__":
    unittest.main()
