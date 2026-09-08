from __future__ import annotations

import unittest

import replay_learned_exit_tail_policy as replay
import report_exit_quality as exit_quality


class LearnedExitTailPolicyReplayTest(unittest.TestCase):
    def test_quantile_is_interpolated_and_bounded(self) -> None:
        self.assertEqual(replay._quantile([0.0, 1.0, 2.0, 3.0, 4.0], 0.80), 3.2)
        self.assertEqual(replay._quantile([1.0, 2.0], -1.0), 1.0)
        self.assertEqual(replay._quantile([1.0, 2.0], 2.0), 2.0)
        self.assertIsNone(replay._quantile([], 0.8))

    def test_policy_metrics_apply_zero_delta_to_unselected_rows(self) -> None:
        rows = [
            {
                "pnl_pct": 1.0,
                "selected_by_train_threshold": True,
                "tail_pnl_pct": 1.5,
                "tail_delta_pct": 0.5,
            },
            {
                "pnl_pct": -0.5,
                "selected_by_train_threshold": False,
                "tail_pnl_pct": -0.5,
                "tail_delta_pct": 0.0,
            },
        ]

        metrics = replay._policy_metrics(rows, "tail")

        self.assertEqual(metrics["test_n"], 2)
        self.assertEqual(metrics["action_missing_n"], 0)
        self.assertEqual(metrics["selected_n"], 1)
        self.assertEqual(metrics["selected_avg_delta_pct"], 0.5)
        self.assertEqual(metrics["overall_avg_delta_pct"], 0.25)

    def test_gate_requires_downside_and_sample_safety(self) -> None:
        passing = {
            "test_n": 100,
            "selected_n": 25,
            "selected_avg_delta_pct": 0.2,
            "selected_median_delta_pct": 0.05,
            "selected_worse_rate_pct": 30.0,
            "selected_p10_delta_pct": -0.5,
            "overall_avg_delta_pct": 0.05,
            "policy_median_pnl_pct": 0.4,
            "baseline_median_pnl_pct": 0.3,
            "policy_win_rate_pct": 60.0,
            "baseline_win_rate_pct": 58.0,
        }
        self.assertTrue(replay._passes_gate(passing))
        self.assertFalse(replay._passes_gate({**passing, "selected_p10_delta_pct": -0.8}))
        self.assertFalse(replay._passes_gate({**passing, "selected_n": 19}))
        self.assertFalse(replay._passes_gate({**passing, "action_missing_n": 1}))

    def test_case_key_normalizes_missing_timeframe(self) -> None:
        left = {"day": "2026-09-01", "sym": "ETCUSDT", "pnl_pct": "1.25"}
        right = {"day": "2026-09-01", "sym": "ETCUSDT", "tf": "15m", "pnl_pct": 1.25}
        self.assertEqual(replay._case_key(left), replay._case_key(right))

    def test_exit_audit_case_preserves_causal_prices_for_replay(self) -> None:
        case = exit_quality._compact_case(
            "2026-09-01",
            {"sym": "ETCUSDT", "entry_price": "20.5", "exit_price": 21.0},
            exit_quality.ExitAuditConfig(),
        )
        self.assertEqual(case["entry_price"], 20.5)
        self.assertEqual(case["exit_price"], 21.0)


if __name__ == "__main__":
    unittest.main()
