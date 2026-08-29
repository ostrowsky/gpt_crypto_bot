from __future__ import annotations

import asyncio
import unittest
from datetime import datetime, timezone
from unittest.mock import patch

import config
import monitor


def _utc_ms(value: str) -> int:
    return int(datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp() * 1000)


class EarlySignalReentryLabelingTests(unittest.TestCase):
    def test_entry_context_marks_later_same_day_entry_as_reentry(self) -> None:
        state = monitor.MonitorState()

        first = monitor._register_entry_alert_context(
            state,
            sym="ICPUSDT",
            entry_ts_ms=_utc_ms("2026-08-29T04:00:00Z"),
        )
        second = monitor._register_entry_alert_context(
            state,
            sym="ICPUSDT",
            entry_ts_ms=_utc_ms("2026-08-29T09:45:00Z"),
        )

        self.assertFalse(first["is_reentry"])
        self.assertEqual(first["ordinal"], 1)
        self.assertTrue(second["is_reentry"])
        self.assertEqual(second["ordinal"], 2)
        self.assertEqual(second["first_entry_local_time"], "06:00")

    def test_entry_context_resets_on_new_local_day(self) -> None:
        state = monitor.MonitorState()
        monitor._register_entry_alert_context(
            state,
            sym="ICPUSDT",
            entry_ts_ms=_utc_ms("2026-08-29T21:45:00Z"),
        )

        next_day = monitor._register_entry_alert_context(
            state,
            sym="ICPUSDT",
            entry_ts_ms=_utc_ms("2026-08-29T22:15:00Z"),
        )

        self.assertFalse(next_day["is_reentry"])
        self.assertEqual(next_day["ordinal"], 1)
        self.assertEqual(next_day["first_entry_local_time"], "00:15")

    def test_score_gate_alert_is_explicit_early_non_trading_signal(self) -> None:
        state = monitor.MonitorState()
        sent: list[str] = []

        async def fake_send(text: str) -> None:
            sent.append(text)

        with patch.object(config, "TOP_GAINER_SCORE_GATE_STRONG_ALERTS_ENABLED", True), \
             patch.object(config, "TOP_GAINER_SCORE_GATE_STRONG_ALERT_MIN_CANDIDATE_SCORE", 100.0), \
             patch.object(config, "TOP_GAINER_SCORE_GATE_STRONG_ALERT_MIN_LIVE_SCORE", 28.0), \
             patch.object(config, "TOP_GAINER_SCORE_GATE_STRONG_ALERT_MAX_DEFICIT", 6.0), \
             patch.object(monitor, "_top_gainer_live_score", return_value=31.4), \
             patch.object(monitor, "_top_gainer_score_gate_min_for_mode", return_value=34.0):
            asyncio.run(
                monitor._maybe_send_top_gainer_score_gate_strong_alert(
                    send=fake_send,
                    state=state,
                    sym="ICPUSDT",
                    tf="15m",
                    mode="breakout",
                    price=2.392,
                    candidate_score=135.7,
                    intraday_change_pct=1.4,
                    daily_range=3.1,
                    vol_x=4.23,
                    adx=23.9,
                    rsi=65.1,
                    ranker_info=None,
                    reason="score 31.40 < 34.00",
                    bar_ts=_utc_ms("2026-08-28T23:30:00Z"),
                )
            )

        self.assertEqual(len(sent), 1)
        self.assertIn("РАННИЙ СИГНАЛ РОСТА", sent[0])
        self.assertIn("бот увидел", sent[0].lower())
        self.assertIn("позиция не открыта", sent[0])


if __name__ == "__main__":
    unittest.main()
