from __future__ import annotations

import unittest


class TestAgentModeRescueReplay(unittest.TestCase):
    def test_allowed_modes_pass_without_rescue_checks(self) -> None:
        import replay_backtest as rb

        candidate = rb.ReplayCandidate(
            sym="TESTUSDT",
            tf="15m",
            mode="trend",
            ts_ms=0,
            i=0,
            price=1.0,
            trail_k=1.0,
            max_hold_bars=1,
            score=1.0,
            ranker_final_score=0.0,
            ranker_top_gainer_prob=0.0,
            top_gainer_score=1.0,
            four_h_context_score=0.0,
            four_h_context_label="",
            rsi=50.0,
            daily_range=99.0,
            vol_x=0.0,
            adx=0.0,
            intraday_change_pct=0.0,
        )
        self.assertTrue(rb._agent_mode_rescue_replay_candidate_ok(candidate, 34.0))

    def test_disabled_breakout_requires_narrow_profile(self) -> None:
        import replay_backtest as rb

        candidate = rb.ReplayCandidate(
            sym="TESTUSDT",
            tf="15m",
            mode="breakout",
            ts_ms=0,
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
        self.assertTrue(rb._agent_mode_rescue_replay_candidate_ok(candidate, 34.0))
        candidate.vol_x = 1.99
        self.assertFalse(rb._agent_mode_rescue_replay_candidate_ok(candidate, 34.0))


if __name__ == "__main__":
    unittest.main()
