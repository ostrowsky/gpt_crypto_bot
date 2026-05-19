from __future__ import annotations

import unittest

from audit_v2_selected_feature_market_switch_replay import _verdict


class SelectedFeatureMarketSwitchReplayTest(unittest.TestCase):
    def test_verdict_requires_classifier_and_reward_gates(self) -> None:
        self.assertEqual(_verdict({"accuracy_edge": 0.01}, 10.0, 0.0), "reject_classifier_gate")
        self.assertEqual(_verdict({"accuracy_edge": 0.04}, -1.0, -5.0), "reject_loses_to_base")
        self.assertEqual(_verdict({"accuracy_edge": 0.04}, 2.0, 3.0), "reject_loses_to_candidate")
        self.assertEqual(_verdict({"accuracy_edge": 0.04}, 4.0, 3.0), "promising_next_full_replay_gate")


if __name__ == "__main__":
    unittest.main()
