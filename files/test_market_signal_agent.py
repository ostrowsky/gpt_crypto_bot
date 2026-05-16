from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


class TestMarketSignalAgentPersistence(unittest.TestCase):
    def test_roundtrip_positions(self) -> None:
        import market_signal_agent as msa

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            with patch.object(msa, "POSITIONS_FILE", tmp_path / "agent_positions.json"):
                pos = msa.AgentPosition(
                    symbol="TESTUSDT",
                    tf="15m",
                    entry_price=1.23,
                    entry_bar=10,
                    entry_ts=1234567890,
                    entry_ema20=1.2,
                    entry_slope=0.45,
                    entry_adx=25.0,
                    entry_rsi=61.0,
                    entry_vol_x=1.7,
                    forecast_return_pct=0.9,
                    today_change_pct=4.2,
                    predictions={3: None},
                    bars_elapsed=2,
                    signal_mode="breakout",
                    trail_k=1.5,
                    max_hold_bars=8,
                    trail_stop=1.1,
                    last_bar_ts=1234569999,
                )
                msa._save_positions({"TESTUSDT|15m": pos})
                restored = msa._load_positions()
                self.assertIn("TESTUSDT|15m", restored)
                self.assertEqual(restored["TESTUSDT|15m"].symbol, "TESTUSDT")
                self.assertEqual(restored["TESTUSDT|15m"].signal_mode, "breakout")
                self.assertAlmostEqual(restored["TESTUSDT|15m"].trail_stop, 1.1)


class TestCompareAgentVsBot(unittest.TestCase):
    def test_match_events_by_symbol_tf_and_time(self) -> None:
        import compare_agent_vs_bot as cab

        base = datetime(2026, 3, 29, 10, 0, tzinfo=timezone.utc)
        agent_events = [
            {"event": "entry", "sym": "TESTUSDT", "tf": "15m", "mode": "breakout", "_ts": base},
            {"event": "exit", "sym": "TESTUSDT", "tf": "15m", "mode": "breakout", "_ts": base + timedelta(minutes=45)},
        ]
        bot_events = [
            {"event": "entry", "sym": "TESTUSDT", "tf": "15m", "mode": "breakout", "_ts": base + timedelta(minutes=10)},
            {"event": "exit", "sym": "TESTUSDT", "tf": "15m", "mode": "breakout", "_ts": base + timedelta(minutes=50)},
            {"event": "entry", "sym": "MISSUSDT", "tf": "1h", "mode": "trend", "_ts": base},
        ]
        payload = cab._match_events(agent_events, bot_events, tolerance_minutes=90)
        self.assertEqual(len(payload["matched"]), 2)
        self.assertEqual(len(payload["agent_only"]), 0)
        self.assertEqual(len(payload["bot_only"]), 1)


class TestMarketSignalAgentEntryGates(unittest.TestCase):
    def test_trend_uses_stricter_adx_floor(self) -> None:
        import market_signal_agent as msa

        report = SimpleNamespace(today_confirmed=True, today_signals=3, best_accuracy=80.0)
        with patch.object(msa.config, "AGENT_TREND_MIN_ADX", 35.0):
            reason = msa._candidate_block_reason(
                symbol="TESTUSDT",
                tf="15m",
                mode="trend",
                today_change_pct=2.5,
                forecast_proxy_pct=1.0,
                leader_score=20.0,
                daily_range=5.0,
                adx=34.9,
                rsi=62.0,
                vol_x=2.0,
                report=report,
            )
        self.assertEqual(reason, "ADX 34.9 < 35.0")

    def test_4h_leader_requires_strength_confirmation(self) -> None:
        import market_signal_agent as msa

        report = SimpleNamespace(today_confirmed=True, today_signals=3, best_accuracy=80.0)
        with (
            patch.object(msa.config, "AGENT_4H_LEADER_STRENGTH_GATE_ENABLED", True),
            patch.object(msa.config, "AGENT_4H_LEADER_STRENGTH_MIN_VOL_X", 3.0),
            patch.object(msa.config, "AGENT_4H_LEADER_STRENGTH_MIN_TODAY_CHANGE_PCT", 10.0),
        ):
            blocked = msa._candidate_block_reason(
                symbol="TESTUSDT",
                tf="15m",
                mode="4h_leader_watch",
                today_change_pct=9.5,
                forecast_proxy_pct=5.0,
                leader_score=60.0,
                daily_range=12.0,
                adx=36.0,
                rsi=68.0,
                vol_x=2.9,
                report=report,
            )
            allowed = msa._candidate_block_reason(
                symbol="TESTUSDT",
                tf="15m",
                mode="4h_leader_watch",
                today_change_pct=10.5,
                forecast_proxy_pct=5.0,
                leader_score=60.0,
                daily_range=12.0,
                adx=36.0,
                rsi=68.0,
                vol_x=3.1,
                report=report,
            )
        self.assertIn("4h leader strength gate", blocked)
        self.assertEqual(allowed, "")


if __name__ == "__main__":
    unittest.main()
