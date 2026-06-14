import unittest

import config
import monitor
from monitor import (
    MonitorState,
    OpenPosition,
    _register_suspicious_reentry_watch,
    _suspicious_reentry_candidate_ok,
    _suspicious_reentry_exit_score,
)


class SuspiciousReentryShadowTests(unittest.TestCase):
    def setUp(self) -> None:
        self._old = {
            "SUSPICIOUS_REENTRY_SHADOW_ENABLED": getattr(config, "SUSPICIOUS_REENTRY_SHADOW_ENABLED", None),
            "SUSPICIOUS_REENTRY_SHADOW_EXIT_SCORE_MIN": getattr(config, "SUSPICIOUS_REENTRY_SHADOW_EXIT_SCORE_MIN", None),
            "SUSPICIOUS_REENTRY_SHADOW_MIN_MFE_PCT": getattr(config, "SUSPICIOUS_REENTRY_SHADOW_MIN_MFE_PCT", None),
            "SUSPICIOUS_REENTRY_SHADOW_WINDOW_BARS": getattr(config, "SUSPICIOUS_REENTRY_SHADOW_WINDOW_BARS", None),
            "SUSPICIOUS_REENTRY_SHADOW_MIN_CANDIDATE_SCORE": getattr(config, "SUSPICIOUS_REENTRY_SHADOW_MIN_CANDIDATE_SCORE", None),
            "SUSPICIOUS_REENTRY_SHADOW_MIN_ADX": getattr(config, "SUSPICIOUS_REENTRY_SHADOW_MIN_ADX", None),
        }
        config.SUSPICIOUS_REENTRY_SHADOW_ENABLED = True
        config.SUSPICIOUS_REENTRY_SHADOW_EXIT_SCORE_MIN = 0.68
        config.SUSPICIOUS_REENTRY_SHADOW_MIN_MFE_PCT = 1.0
        config.SUSPICIOUS_REENTRY_SHADOW_WINDOW_BARS = 8
        config.SUSPICIOUS_REENTRY_SHADOW_MIN_CANDIDATE_SCORE = 38.0
        config.SUSPICIOUS_REENTRY_SHADOW_MIN_ADX = 18.0

    def tearDown(self) -> None:
        for name, value in self._old.items():
            if value is None:
                try:
                    delattr(config, name)
                except AttributeError:
                    pass
            else:
                setattr(config, name, value)

    def test_exit_score_marks_high_mfe_weak_exit_as_suspicious(self) -> None:
        score = _suspicious_reentry_exit_score(
            mode="trend",
            reason="WEAK: RSI divergence",
            current_pnl=1.0,
            max_favorable_pct=3.5,
            bars_held=3,
        )
        self.assertGreaterEqual(score, 0.68)

    def test_register_watch_after_suspicious_exit(self) -> None:
        state = MonitorState()
        pos = OpenPosition(
            symbol="AAAUSDT",
            tf="15m",
            entry_price=100.0,
            entry_bar=10,
            entry_ts=1_000_000,
            entry_ema20=99.0,
            entry_slope=0.5,
            entry_adx=30.0,
            entry_rsi=61.0,
            entry_vol_x=2.0,
            signal_mode="trend",
            bars_elapsed=3,
            max_price_since_entry=104.0,
        )
        _register_suspicious_reentry_watch(
            state,
            pos,
            exit_reason="WEAK: RSI divergence",
            exit_price=101.0,
            exit_ts=2_000_000,
            pnl_pct=1.0,
        )
        watch = state.suspicious_reentry_watch.get("AAAUSDT")
        self.assertIsNotNone(watch)
        self.assertGreaterEqual(watch["exit_score"], 0.68)
        self.assertAlmostEqual(watch["mfe_pct"], 4.0)
        self.assertGreater(watch["until_ts"], 2_000_000)

    def test_watch_decision_is_logged_for_registration_and_rejection(self) -> None:
        events = []
        original = monitor.botlog.log_suspicious_reentry_watch_decision
        monitor.botlog.log_suspicious_reentry_watch_decision = lambda **kwargs: events.append(kwargs)
        try:
            state = MonitorState()
            pos = OpenPosition(
                symbol="AAAUSDT",
                tf="15m",
                entry_price=100.0,
                entry_bar=10,
                entry_ts=1_000_000,
                entry_ema20=99.0,
                entry_slope=0.5,
                entry_adx=30.0,
                entry_rsi=61.0,
                entry_vol_x=2.0,
                signal_mode="trend",
                bars_elapsed=3,
                max_price_since_entry=104.0,
            )
            _register_suspicious_reentry_watch(
                state,
                pos,
                exit_reason="WEAK: RSI divergence",
                exit_price=101.0,
                exit_ts=2_000_000,
                pnl_pct=1.0,
            )
            low_score_pos = OpenPosition(
                symbol="BBBUSDT",
                tf="15m",
                entry_price=100.0,
                entry_bar=10,
                entry_ts=1_000_000,
                entry_ema20=99.0,
                entry_slope=0.5,
                entry_adx=30.0,
                entry_rsi=61.0,
                entry_vol_x=2.0,
                signal_mode="trend",
                bars_elapsed=30,
                max_price_since_entry=100.5,
            )
            _register_suspicious_reentry_watch(
                state,
                low_score_pos,
                exit_reason="time exit",
                exit_price=99.8,
                exit_ts=2_000_000,
                pnl_pct=-0.2,
            )
        finally:
            monitor.botlog.log_suspicious_reentry_watch_decision = original

        self.assertEqual(events[0]["decision"], "registered")
        self.assertEqual(events[1]["decision"], "rejected_exit_score")
        self.assertIn("score_floor", events[1])

    def test_candidate_confirmation_requires_score_and_adx(self) -> None:
        self.assertTrue(_suspicious_reentry_candidate_ok(mode="trend", candidate_score=42.0, adx=22.0))
        self.assertFalse(_suspicious_reentry_candidate_ok(mode="trend", candidate_score=30.0, adx=22.0))
        self.assertFalse(_suspicious_reentry_candidate_ok(mode="trend", candidate_score=42.0, adx=12.0))
        self.assertFalse(_suspicious_reentry_candidate_ok(mode="noise", candidate_score=42.0, adx=22.0))


if __name__ == "__main__":
    unittest.main()
