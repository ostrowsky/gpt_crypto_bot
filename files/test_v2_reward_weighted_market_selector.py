from __future__ import annotations

import unittest

from audit_v2_reward_weighted_market_selector import _selector_verdict, _verdict_rank


class RewardWeightedMarketSelectorTest(unittest.TestCase):
    def test_verdict_rejects_trivial_or_losing_selectors(self) -> None:
        self.assertEqual(_selector_verdict({"candidate_share": 0.01, "switched_delta_vs_base": 10.0}, 5.0), "reject_trivial_candidate_count")
        self.assertEqual(_selector_verdict({"candidate_share": 0.10, "switched_delta_vs_base": -1.0}, -5.0), "reject_loses_to_base")
        self.assertEqual(_selector_verdict({"candidate_share": 0.10, "switched_delta_vs_base": 2.0}, 3.0), "reject_loses_to_candidate")
        self.assertEqual(_selector_verdict({"candidate_share": 0.10, "switched_delta_vs_base": 4.0}, 3.0), "promising_next_full_replay_gate")

    def test_verdict_rank_orders_promising_above_losing(self) -> None:
        self.assertEqual(_verdict_rank({"candidate_share": 0.01, "switched_delta_vs_base": 10.0}, 5.0), 0)
        self.assertEqual(_verdict_rank({"candidate_share": 0.10, "switched_delta_vs_base": -1.0}, -5.0), 1)
        self.assertEqual(_verdict_rank({"candidate_share": 0.10, "switched_delta_vs_base": 2.0}, 3.0), 2)
        self.assertEqual(_verdict_rank({"candidate_share": 0.10, "switched_delta_vs_base": 4.0}, 3.0), 3)


if __name__ == "__main__":
    unittest.main()
