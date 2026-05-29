from __future__ import annotations

import unittest

import numpy as np

import config
from replay_backtest import ReplayTrade, _exit_discriminator_shadow_score, _update_trade_progress


class ProtectedTrailingExitCandleReplayTest(unittest.TestCase):
    def test_protected_exit_holds_first_weak_exit_then_trails_out(self) -> None:
        old_values = {
            name: getattr(config, name, None)
            for name in (
                "PROTECTED_EXIT_MIN_MFE_PCT",
                "PROTECTED_EXIT_MIN_CURRENT_PNL_PCT",
                "PROTECTED_EXIT_TRAIL_ATR_K",
                "PROTECTED_EXIT_PROFIT_FLOOR_PCT",
                "PROTECTED_EXIT_MAX_HOLD_BARS",
            )
        }
        try:
            config.PROTECTED_EXIT_MIN_MFE_PCT = 0.5
            config.PROTECTED_EXIT_MIN_CURRENT_PNL_PCT = -1.5
            config.PROTECTED_EXIT_TRAIL_ATR_K = 0.9
            config.PROTECTED_EXIT_PROFIT_FLOOR_PCT = 0.05
            config.PROTECTED_EXIT_MAX_HOLD_BARS = 4

            data = np.zeros(4, dtype=[("t", "i8"), ("c", "f8"), ("h", "f8"), ("l", "f8")])
            data["t"] = np.array([1, 2, 3, 4], dtype=np.int64)
            data["c"] = np.array([100.0, 102.0, 101.0, 100.0], dtype=float)
            data["h"] = np.array([100.0, 102.0, 101.0, 100.0], dtype=float)
            data["l"] = np.array([100.0, 102.0, 101.0, 100.0], dtype=float)
            feat = {
                "atr": np.array([1.0, 1.0, 1.0, 1.0], dtype=float),
                "rsi": np.array([55.0, 60.0, 58.0, 50.0], dtype=float),
                "ema_fast": np.array([99.0, 100.0, 101.5, 101.0], dtype=float),
                "ema_slow": np.array([98.0, 99.0, 100.0, 100.5], dtype=float),
                "ema200": np.array([97.0, 98.0, 99.0, 99.5], dtype=float),
                "adx": np.array([25.0, 26.0, 27.0, 27.0], dtype=float),
                "slope": np.array([0.1, 0.2, 0.1, -0.1], dtype=float),
            }
            trade = ReplayTrade(
                sym="AAAUSDT",
                tf="15m",
                mode="strong_trend",
                entry_ts=1,
                entry_price=100.0,
                entry_i=0,
                trail_k=2.0,
                max_hold_bars=20,
                trail_stop=0.0,
            )

            self.assertIsNone(
                _update_trade_progress(
                    trade,
                    data,
                    feat,
                    2,
                    ts_ms=3,
                    protected_trailing_exit=True,
                )
            )
            self.assertTrue(trade.protected_exit_active)
            self.assertGreaterEqual(trade.trail_stop, 100.05)

            reason = _update_trade_progress(
                trade,
                data,
                feat,
                3,
                ts_ms=4,
                protected_trailing_exit=True,
            )
            self.assertIsNotNone(reason)
            self.assertIn("protected trailing ATR trail stop", str(reason))
        finally:
            for name, value in old_values.items():
                if value is None and hasattr(config, name):
                    delattr(config, name)
                elif value is not None:
                    setattr(config, name, value)

    def test_protected_weak_only_does_not_hold_ema_exit(self) -> None:
        data = np.zeros(3, dtype=[("t", "i8"), ("c", "f8"), ("h", "f8"), ("l", "f8")])
        data["t"] = np.array([1, 2, 3], dtype=np.int64)
        data["c"] = np.array([100.0, 102.0, 101.0], dtype=float)
        data["h"] = np.array([100.0, 102.0, 101.0], dtype=float)
        data["l"] = np.array([100.0, 102.0, 101.0], dtype=float)
        feat = {
            "atr": np.array([1.0, 1.0, 1.0], dtype=float),
            "rsi": np.array([55.0, 60.0, 58.0], dtype=float),
            "ema_fast": np.array([99.0, 100.0, 101.5], dtype=float),
            "ema_slow": np.array([98.0, 99.0, 100.0], dtype=float),
            "ema200": np.array([97.0, 98.0, 99.0], dtype=float),
            "adx": np.array([25.0, 26.0, 27.0], dtype=float),
            "slope": np.array([0.1, 0.2, 0.1], dtype=float),
        }
        trade = ReplayTrade(
            sym="AAAUSDT",
            tf="15m",
            mode="strong_trend",
            entry_ts=1,
            entry_price=100.0,
            entry_i=0,
            trail_k=2.0,
            max_hold_bars=20,
            trail_stop=0.0,
        )

        reason = _update_trade_progress(
            trade,
            data,
            feat,
            2,
            ts_ms=3,
            protected_weak_only=True,
        )
        self.assertIsNotNone(reason)
        self.assertFalse(trade.protected_exit_active)
        self.assertNotIn("protected", str(reason).lower())

    def test_exit_discriminator_shadow_score_marks_high_risk_exit(self) -> None:
        trade = ReplayTrade(
            sym="AAAUSDT",
            tf="15m",
            mode="trend",
            entry_ts=1,
            entry_price=100.0,
            entry_i=0,
            trail_k=2.0,
            max_hold_bars=20,
            trail_stop=0.0,
            capture_ratio_at_entry=0.75,
            max_favorable_pct=3.5,
            bars_held=3,
        )

        score = _exit_discriminator_shadow_score(
            trade=trade,
            reason="WEAK: RSI divergence",
            current_pnl=1.0,
        )

        self.assertGreaterEqual(score, 0.68)

    def test_exit_discriminator_shadow_policy_holds_high_risk_weak_exit(self) -> None:
        old_values = {
            name: getattr(config, name, None)
            for name in (
                "EXIT_DISCRIMINATOR_HOLD_SCORE_MIN",
                "EXIT_DISCRIMINATOR_MIN_MFE_PCT",
                "EXIT_DISCRIMINATOR_MIN_CURRENT_PNL_PCT",
                "EXIT_DISCRIMINATOR_MAX_HOLD_BARS",
                "EXIT_DISCRIMINATOR_TRAIL_ATR_K",
                "EXIT_DISCRIMINATOR_PROFIT_FLOOR_PCT",
            )
        }
        try:
            config.EXIT_DISCRIMINATOR_HOLD_SCORE_MIN = 0.68
            config.EXIT_DISCRIMINATOR_MIN_MFE_PCT = 1.0
            config.EXIT_DISCRIMINATOR_MIN_CURRENT_PNL_PCT = -0.25
            config.EXIT_DISCRIMINATOR_MAX_HOLD_BARS = 2
            config.EXIT_DISCRIMINATOR_TRAIL_ATR_K = 0.75
            config.EXIT_DISCRIMINATOR_PROFIT_FLOOR_PCT = 0.10

            data = np.zeros(4, dtype=[("t", "i8"), ("c", "f8"), ("h", "f8"), ("l", "f8")])
            data["t"] = np.array([1, 2, 3, 4], dtype=np.int64)
            data["c"] = np.array([100.0, 103.5, 101.0, 100.0], dtype=float)
            data["h"] = np.array([100.0, 103.5, 101.0, 100.0], dtype=float)
            data["l"] = np.array([100.0, 103.5, 101.0, 100.0], dtype=float)
            feat = {
                "atr": np.array([1.0, 1.0, 1.0, 1.0], dtype=float),
                "rsi": np.array([55.0, 70.0, 58.0, 50.0], dtype=float),
                "ema_fast": np.array([99.0, 100.0, 100.0, 101.0], dtype=float),
                "ema_slow": np.array([98.0, 99.0, 99.5, 100.5], dtype=float),
                "ema200": np.array([97.0, 98.0, 99.0, 99.5], dtype=float),
                "adx": np.array([25.0, 26.0, 27.0, 27.0], dtype=float),
                "slope": np.array([0.1, 0.2, 0.1, -0.1], dtype=float),
                "rsi_divergence": np.array([0.0, 0.0, 1.0, 0.0], dtype=float),
            }
            trade = ReplayTrade(
                sym="AAAUSDT",
                tf="15m",
                mode="trend",
                entry_ts=1,
                entry_price=100.0,
                entry_i=0,
                trail_k=3.0,
                max_hold_bars=20,
                trail_stop=0.0,
                capture_ratio_at_entry=0.75,
            )

            self.assertIsNone(
                _update_trade_progress(
                    trade,
                    data,
                    feat,
                    2,
                    ts_ms=3,
                    exit_discriminator_shadow_policy=True,
                )
            )
            self.assertTrue(trade.discriminator_exit_active)
            self.assertGreaterEqual(trade.discriminator_exit_score, 0.68)
        finally:
            for name, value in old_values.items():
                if value is None and hasattr(config, name):
                    delattr(config, name)
                elif value is not None:
                    setattr(config, name, value)


if __name__ == "__main__":
    unittest.main()
