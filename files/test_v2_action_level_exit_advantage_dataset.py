from __future__ import annotations

import unittest

from build_v2_action_level_exit_advantage_dataset import _pnl_pct, _finalize_trade_rows


class ActionLevelExitAdvantageDatasetTest(unittest.TestCase):
    def test_pnl_pct(self) -> None:
        self.assertAlmostEqual(_pnl_pct(110.0, 100.0), 10.0)
        self.assertEqual(_pnl_pct(110.0, 0.0), 0.0)

    def test_finalize_trade_rows_labels_sell_advantage(self) -> None:
        rows = [
            {"candidate_action": "hold", "sell_now_reward": 1.0, "candidate_step_reward": 0.25},
            {"candidate_action": "sell", "sell_now_reward": -2.0, "candidate_step_reward": -2.0},
        ]
        out = _finalize_trade_rows(rows)
        self.assertEqual(out[0]["continuation_reward"], -1.75)
        self.assertEqual(out[0]["sell_advantage"], 2.75)
        self.assertTrue(out[0]["sell_advantage_positive"])


if __name__ == "__main__":
    unittest.main()
