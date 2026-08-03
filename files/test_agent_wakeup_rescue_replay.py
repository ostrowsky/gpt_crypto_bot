from __future__ import annotations

import unittest
from unittest.mock import patch


class TestAgentWakeupRescueReplay(unittest.TestCase):
    def test_cli_accepts_paired_agent_allowed_control(self) -> None:
        import replay_backtest as rb

        with patch("sys.argv", ["replay_backtest.py", "--compare-variant", "agent_allowed"]):
            args = rb.parse_args()

        self.assertEqual(args.compare_variant, "agent_allowed")

    def _candidate(self):
        import replay_backtest as rb

        return rb.ReplayCandidate(
            sym="TESTUSDT",
            tf="15m",
            mode="breakout",
            ts_ms=10_000_000,
            i=0,
            price=1.0,
            trail_k=1.0,
            max_hold_bars=1,
            score=1.0,
            ranker_final_score=0.0,
            ranker_top_gainer_prob=0.0,
            top_gainer_score=34.0,
            four_h_context_score=0.0,
            four_h_context_label="",
            rsi=60.0,
            daily_range=8.0,
            vol_x=2.0,
            adx=22.0,
            intraday_change_pct=1.0,
        )

    def test_requires_structural_before_wakeup(self) -> None:
        import replay_backtest as rb

        candidate = self._candidate()
        self.assertTrue(
            rb._agent_wakeup_rescue_replay_candidate_ok(
                candidate,
                34.0,
                {("TESTUSDT", "15m"): [8_000_000]},
                {("TESTUSDT", "15m"): [9_000_000]},
            )
        )
        self.assertFalse(
            rb._agent_wakeup_rescue_replay_candidate_ok(
                candidate,
                34.0,
                {("TESTUSDT", "15m"): [9_500_000]},
                {("TESTUSDT", "15m"): [9_000_000]},
            )
        )

    def test_rejects_stale_wakeup(self) -> None:
        import replay_backtest as rb

        candidate = self._candidate()
        self.assertFalse(
            rb._agent_wakeup_rescue_replay_candidate_ok(
                candidate,
                34.0,
                {("TESTUSDT", "15m"): [-13_000_000]},
                {("TESTUSDT", "15m"): [-12_000_000]},
            )
        )

    def test_rejects_stale_structural_alert(self) -> None:
        import replay_backtest as rb

        candidate = self._candidate()
        self.assertFalse(
            rb._agent_wakeup_rescue_replay_candidate_ok(
                candidate,
                34.0,
                {("TESTUSDT", "15m"): [-90_000_000]},
                {("TESTUSDT", "15m"): [9_000_000]},
            )
        )


if __name__ == "__main__":
    unittest.main()
