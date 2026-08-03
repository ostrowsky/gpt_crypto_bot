from __future__ import annotations

import asyncio
import unittest
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import patch

import config
import monitor


class TopGainerWatchAlertTests(unittest.TestCase):
    def test_broad_watch_is_disabled_after_failed_causal_gate(self) -> None:
        self.assertFalse(config.TOP_GAINER_WATCH_ALERTS_ENABLED)

    def test_local_day_dedup_does_not_raise_and_resets_next_day(self) -> None:
        sent: list[str] = []

        async def send(text: str) -> None:
            sent.append(text)

        state = SimpleNamespace()
        day_one = int(datetime(2026, 8, 2, 10, tzinfo=timezone.utc).timestamp() * 1000)
        day_two = int(datetime(2026, 8, 3, 10, tzinfo=timezone.utc).timestamp() * 1000)

        async def run() -> None:
            kwargs = {
                "send": send,
                "state": state,
                "sym": "AAAUSDT",
                "tf": "1h",
                "mode": "impulse_speed",
                "price": 1.0,
                "intraday_change_pct": 1.0,
                "daily_range": 3.0,
                "vol_x": 2.0,
                "adx": 30.0,
                "rsi": 60.0,
                "ranker_info": None,
                "reason": "score below floor",
            }
            await monitor._maybe_send_top_gainer_watch_alert(bar_ts=day_one, **kwargs)
            await monitor._maybe_send_top_gainer_watch_alert(bar_ts=day_one, **kwargs)
            await monitor._maybe_send_top_gainer_watch_alert(bar_ts=day_two, **kwargs)

        with (
            patch.object(config, "TOP_GAINER_WATCH_ALERTS_ENABLED", True),
            patch.object(config, "TOP_GAINER_WATCH_ALERT_MIN_SCORE", 30.0),
            patch.object(config, "TOP_GAINER_SCORE_GATE_MIN_SCORE", 34.0),
            patch.object(monitor, "_top_gainer_live_score", return_value=32.0),
        ):
            asyncio.run(run())

        self.assertEqual(len(sent), 2)
        self.assertTrue(all("WATCH ONLY" in text for text in sent))


if __name__ == "__main__":
    unittest.main()
