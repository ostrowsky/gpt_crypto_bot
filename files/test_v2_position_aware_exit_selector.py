from __future__ import annotations

import unittest

from audit_v2_position_aware_exit_selector import _decision


class PositionAwareExitSelectorTest(unittest.TestCase):
    def test_decision_requires_reward_trade_and_giveback_gates(self) -> None:
        out = {
            "fixed_candidate": {
                "summary": {"trade_count": 100},
                "exit": {"avg_giveback_penalty": -2.0},
            },
            "profile": {
                "summary": {"trade_count": 105},
                "exit": {"avg_giveback_penalty": -2.05},
            },
        }
        ranked = [{"policy": "profile", "delta_vs_fixed_candidate": 1.0, "total_reward": 2.0}]
        self.assertTrue(_decision(out, ranked)["promotion_gate_passed"])
        out["profile"]["summary"]["trade_count"] = 120
        self.assertFalse(_decision(out, ranked)["promotion_gate_passed"])


if __name__ == "__main__":
    unittest.main()
