from __future__ import annotations

import unittest

from run_post_block_delayed_confirmation_replay import _decision, _summary


class PostBlockDelayedConfirmationReplayTest(unittest.TestCase):
    def test_summary_counts_top15_precision(self) -> None:
        payload = _summary([
            {"label_top15": True, "ret_120m_pct": 1.0, "ret_240m_pct": 2.0, "eod_ret_pct": 1.5, "mfe_to_eod_pct": 3.0, "mae_to_eod_pct": -1.0},
            {"label_top15": False, "ret_120m_pct": -1.0, "ret_240m_pct": -2.0, "eod_ret_pct": -1.5, "mfe_to_eod_pct": 1.0, "mae_to_eod_pct": -3.0},
        ])
        self.assertEqual(payload["count"], 2)
        self.assertEqual(payload["top15_precision"], 0.5)
        self.assertEqual(payload["positive_120m_rate"], 0.5)

    def test_decision_requires_forward_edge(self) -> None:
        payload = {
            "selected": {"count": 10, "mean_ret_240m_pct": 1.0, "mean_mfe_to_eod_pct": 3.0, "mean_mae_to_eod_pct": -1.0},
            "baseline_all_post_block_holdout": {"mean_ret_240m_pct": 0.0},
        }
        self.assertEqual(_decision(payload), "advance_to_fee_slippage_exit_replay")
        payload["selected"]["mean_ret_240m_pct"] = -1.0
        self.assertEqual(_decision(payload), "research_only_rejected_no_forward_edge_after_confirmation")


if __name__ == "__main__":
    unittest.main()
