from __future__ import annotations

import unittest

from audit_v2_reward_weighted_market_selector_offline_replay import _latest_choice, _decision


class RewardWeightedMarketSelectorOfflineReplayTest(unittest.TestCase):
    def test_latest_choice_uses_most_recent_anchor(self) -> None:
        choices = [
            {"anchor_ts_ms": 1000, "policy": "base"},
            {"anchor_ts_ms": 2000, "policy": "candidate"},
        ]
        self.assertIsNone(_latest_choice(choices, 999))
        self.assertEqual(_latest_choice(choices, 1000)["policy"], "base")
        self.assertEqual(_latest_choice(choices, 2500)["policy"], "candidate")

    def test_decision_requires_reward_and_trade_count_gates(self) -> None:
        summaries = {
            "fixed_base": {"total_reward": 0.0, "trade_count": 10},
            "fixed_candidate": {"total_reward": 10.0, "trade_count": 10},
            "reward_weighted_market_switch": {"total_reward": 11.0, "trade_count": 12},
        }
        self.assertTrue(_decision(summaries)["promotion_gate_passed"])
        summaries["reward_weighted_market_switch"]["trade_count"] = 20
        self.assertFalse(_decision(summaries)["promotion_gate_passed"])


if __name__ == "__main__":
    unittest.main()
