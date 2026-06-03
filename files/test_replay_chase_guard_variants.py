from __future__ import annotations

import unittest

import config
from replay_backtest import _chase_guard_reason_for_replay_variant


class ReplayChaseGuardVariantTest(unittest.TestCase):
    def test_rsi_off_keeps_daily_range_guard(self) -> None:
        reason = _chase_guard_reason_for_replay_variant(
            variant="chase_guard_rsi_off",
            tf="15m",
            mode="trend",
            rsi=50.0,
            daily_range=30.0,
        )
        self.assertIn("daily_range", str(reason))

    def test_rsi_off_allows_rsi_only_overheat(self) -> None:
        reason = _chase_guard_reason_for_replay_variant(
            variant="chase_guard_rsi_off",
            tf="15m",
            mode="trend",
            rsi=90.0,
            daily_range=10.0,
        )
        self.assertIsNone(reason)

    def test_rsi_82_blocks_above_82_but_allows_old_76_threshold(self) -> None:
        self.assertIsNone(
            _chase_guard_reason_for_replay_variant(
                variant="chase_guard_rsi_82",
                tf="15m",
                mode="trend",
                rsi=80.0,
                daily_range=10.0,
            )
        )
        self.assertIn(
            "RSI",
            str(
                _chase_guard_reason_for_replay_variant(
                    variant="chase_guard_rsi_82",
                    tf="15m",
                    mode="trend",
                    rsi=83.0,
                    daily_range=10.0,
                )
            ),
        )

    def test_variant_does_not_mutate_config(self) -> None:
        original = config.TOP_GAINER_CHASE_GUARD_MAX_RSI
        _chase_guard_reason_for_replay_variant(
            variant="chase_guard_rsi_off",
            tf="15m",
            mode="trend",
            rsi=90.0,
            daily_range=10.0,
        )
        self.assertEqual(config.TOP_GAINER_CHASE_GUARD_MAX_RSI, original)


if __name__ == "__main__":
    unittest.main()
