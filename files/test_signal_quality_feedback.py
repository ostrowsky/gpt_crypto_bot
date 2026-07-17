from __future__ import annotations

import unittest

import config
import signal_quality_feedback as feedback


class SignalQualityFeedbackCooldownReplayTest(unittest.TestCase):
    def test_replay_validated_base_cooldown_is_two_bars(self) -> None:
        self.assertEqual(config.COOLDOWN_BARS, 2)
        self.assertEqual(config.AGENT_MAIN_EXIT_COOLDOWN_BARS, 8)

    def test_feedback_embeds_max_period_and_stability_evidence(self) -> None:
        payload = feedback.build_feedback({"target_day_local": "2026-07-17", "summary": {}})
        validation = payload["validation"]["cooldown_bars"]

        self.assertEqual(validation["window"], "30d ending 2026-07-17")
        self.assertEqual(validation["stability"]["window"], "14d ending 2026-07-17")
        self.assertTrue(feedback._cooldown_validation_improves(validation))
        self.assertTrue(payload["policy"]["apply_cooldown_relaxation"])
        self.assertEqual(payload["policy"]["cooldown_bars"], 2)

    def test_failed_stability_window_blocks_auto_apply(self) -> None:
        comparison = {
            "baseline": {
                "pnl_total": -10.0,
                "pnl_avg": -1.0,
                "win_rate": 0.40,
                "trade_precision": 0.30,
                "top20_recall": 1.0,
                "cooldown_harm_pct": 100.0,
            },
            "variant": {
                "pnl_total": -5.0,
                "pnl_avg": -0.5,
                "win_rate": 0.45,
                "trade_precision": 0.35,
                "top20_recall": 1.0,
                "cooldown_harm_pct": 50.0,
            },
        }
        validation = {"status": "replay_confirmed", **comparison, "stability": comparison.copy()}
        validation["stability"] = {
            "baseline": comparison["baseline"],
            "variant": {**comparison["variant"], "pnl_total": -20.0},
        }

        self.assertFalse(feedback._cooldown_validation_improves(validation))


if __name__ == "__main__":
    unittest.main()
