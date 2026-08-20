from __future__ import annotations

import unittest
from unittest.mock import patch

import replay_backtest as rb


class ObjectiveLeaderAdmissionReplayTest(unittest.TestCase):
    def _candidate(self, **overrides) -> rb.ReplayCandidate:
        values = {
            "sym": "TESTUSDT",
            "tf": "15m",
            "mode": "strong_trend",
            "ts_ms": 1_000_000,
            "i": 0,
            "price": 1.0,
            "trail_k": 2.5,
            "max_hold_bars": 48,
            "score": 180.0,
            "ranker_final_score": 0.0,
            "ranker_top_gainer_prob": 0.0,
            "top_gainer_score": 90.0,
            "four_h_context_score": 0.0,
            "four_h_context_label": "",
            "rsi": 75.0,
            "daily_range": 10.0,
            "vol_x": 1.5,
            "adx": 25.0,
            "intraday_change_pct": 2.0,
        }
        values.update(overrides)
        return rb.ReplayCandidate(**values)

    def test_strong_continuation_allows_only_bounded_high_quality_overheat(self) -> None:
        self.assertIsNone(
            rb._chase_guard_reason_for_replay_variant(
                variant="chase_guard_strong_continuation",
                tf="15m",
                mode="strong_trend",
                rsi=80.0,
                daily_range=10.0,
                candidate_score=180.0,
                adx=25.0,
                vol_x=1.5,
            )
        )
        self.assertIn(
            "RSI",
            str(
                rb._chase_guard_reason_for_replay_variant(
                    variant="chase_guard_strong_continuation",
                    tf="15m",
                    mode="strong_trend",
                    rsi=80.0,
                    daily_range=10.0,
                    candidate_score=149.9,
                    adx=25.0,
                    vol_x=1.5,
                )
            ),
        )
        self.assertIn(
            "daily_range",
            str(
                rb._chase_guard_reason_for_replay_variant(
                    variant="chase_guard_strong_continuation",
                    tf="15m",
                    mode="impulse_speed",
                    rsi=80.0,
                    daily_range=25.1,
                    candidate_score=200.0,
                    adx=30.0,
                    vol_x=2.0,
                )
            ),
        )

    def test_objective_leader_profile_is_causal_and_bounded(self) -> None:
        self.assertTrue(rb._objective_leader_replay_candidate_ok(self._candidate()))
        self.assertFalse(
            rb._objective_leader_replay_candidate_ok(
                self._candidate(top_gainer_score=79.9)
            )
        )
        self.assertFalse(
            rb._objective_leader_replay_candidate_ok(
                self._candidate(daily_range=12.1)
            )
        )
        self.assertFalse(
            rb._objective_leader_replay_candidate_ok(self._candidate(rsi=76.1))
        )

    def test_cli_accepts_research_only_variants(self) -> None:
        for variant in (
            "chase_guard_strong_continuation",
            "objective_slot_reserve",
            "objective_leader_combined",
        ):
            with self.subTest(variant=variant), patch(
                "sys.argv", ["replay_backtest.py", "--variant", variant]
            ):
                self.assertEqual(rb.parse_args().variant, variant)


if __name__ == "__main__":
    unittest.main()
