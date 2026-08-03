from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone

import numpy as np

import replay_suspicious_reentry_watch_thresholds as replay


class SuspiciousReentryWatchThresholdReplayTests(unittest.TestCase):
    def test_label_decisions_uses_next_candle_open_without_lookahead(self) -> None:
        start = datetime(2026, 7, 1, tzinfo=timezone.utc)
        timestamps = np.array(
            [int((start + timedelta(minutes=15 * i)).timestamp() * 1000) for i in range(12)],
            dtype=np.int64,
        )
        data = np.zeros(12, dtype=[("t", "i8"), ("o", "f8"), ("h", "f8"), ("l", "f8"), ("c", "f8"), ("v", "f8")])
        data["t"] = timestamps
        data["o"] = np.arange(100.0, 112.0)
        data["c"] = np.arange(101.0, 113.0)
        data["h"] = data["c"] + 1.0
        data["l"] = data["o"] - 1.0
        row = {
            "ts": (start + timedelta(minutes=1)).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "sym": "AAAUSDT",
            "tf": "15m",
            "decision": "registered",
            "exit_score": 0.8,
            "mfe_pct": 2.0,
        }

        labeled = replay.label_decisions([row], {("AAAUSDT", "15m"): data})

        self.assertEqual(len(labeled), 1)
        self.assertEqual(labeled[0]["entry_price"], 101.0)
        self.assertAlmostEqual(labeled[0]["ret_5"], (106.0 / 101.0 - 1.0) * 100.0)

    def test_threshold_selection_requires_out_of_sample_improvement(self) -> None:
        start = datetime(2026, 7, 1, tzinfo=timezone.utc)
        rows = []
        for i in range(100):
            high_reward = i % 2 == 0
            rows.append(
                {
                    "ts": (start + timedelta(minutes=15 * i)).strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "exit_score": 0.60 if high_reward else 0.75,
                    "mfe_pct": 1.2,
                    "ret_5": 1.0 if high_reward else 0.2,
                    "drawdown_10": -0.2,
                }
            )

        result = replay.evaluate_thresholds(rows)

        self.assertLess(result["train_selected"]["exit_score_min"], 0.68)
        self.assertTrue(result["promotion_gate_passed"])
        self.assertEqual(result["decision"], "candidate_for_shadow_recalibration")


if __name__ == "__main__":
    unittest.main()
