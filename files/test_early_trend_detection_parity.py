import unittest
from datetime import datetime, timezone
from unittest.mock import patch

import numpy as np

import strategy
import replay_backtest


def _false(*args, **kwargs):
    return False, "no"


class EarlyTrendDetectionParityTests(unittest.TestCase):
    def test_today_signal_helper_counts_impulse_speed(self):
        feat = {}
        c = np.array([1.0, 1.01, 1.02], dtype=float)

        with patch.object(strategy, "check_entry_conditions", _false), \
             patch.object(strategy, "check_breakout_conditions", _false), \
             patch.object(strategy, "check_retest_conditions", _false), \
             patch.object(strategy, "check_impulse_conditions", _false), \
             patch.object(strategy, "check_alignment_conditions", _false), \
             patch.object(strategy, "check_trend_surge_conditions", return_value=(True, "surge")):
            ok, mode = strategy._live_entry_signal_mode(feat, 1, c, tf="15m")

        self.assertTrue(ok)
        self.assertEqual(mode, "impulse_speed")

    def test_alignment_is_not_counted_when_disabled(self):
        feat = {}
        c = np.array([1.0, 1.01, 1.02], dtype=float)

        with patch.object(strategy, "check_entry_conditions", _false), \
             patch.object(strategy, "check_breakout_conditions", _false), \
             patch.object(strategy, "check_retest_conditions", _false), \
             patch.object(strategy, "check_impulse_conditions", _false), \
             patch.object(strategy, "check_trend_surge_conditions", _false), \
             patch.object(strategy, "check_alignment_conditions", return_value=(True, "alignment")), \
             patch.object(strategy.config, "ALIGNMENT_BUY_ENABLED", False, create=True):
            ok, mode = strategy._live_entry_signal_mode(feat, 1, c, tf="15m")

        self.assertFalse(ok)
        self.assertEqual(mode, "")

    def test_alignment_is_counted_when_enabled(self):
        feat = {}
        c = np.array([1.0, 1.01, 1.02], dtype=float)

        with patch.object(strategy, "check_entry_conditions", _false), \
             patch.object(strategy, "check_breakout_conditions", _false), \
             patch.object(strategy, "check_retest_conditions", _false), \
             patch.object(strategy, "check_impulse_conditions", _false), \
             patch.object(strategy, "check_trend_surge_conditions", _false), \
             patch.object(strategy, "check_alignment_conditions", return_value=(True, "alignment")), \
             patch.object(strategy.config, "ALIGNMENT_BUY_ENABLED", True, create=True):
            ok, mode = strategy._live_entry_signal_mode(feat, 1, c, tf="15m")

        self.assertTrue(ok)
        self.assertEqual(mode, "alignment")

    def test_replay_intraday_uses_local_day_open(self):
        dtype = [("t", "i8"), ("o", "f8"), ("c", "f8")]
        rows = np.array(
            [
                (int(datetime(2026, 5, 27, 21, 45, tzinfo=timezone.utc).timestamp() * 1000), 100.0, 101.0),
                (int(datetime(2026, 5, 27, 22, 0, tzinfo=timezone.utc).timestamp() * 1000), 200.0, 202.0),
                (int(datetime(2026, 5, 27, 23, 0, tzinfo=timezone.utc).timestamp() * 1000), 205.0, 210.0),
            ],
            dtype=dtype,
        )

        change = replay_backtest._intraday_change_pct_from_data(rows, 2)

        self.assertAlmostEqual(change, 5.0, places=6)


if __name__ == "__main__":
    unittest.main()
