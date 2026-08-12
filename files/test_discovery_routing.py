from __future__ import annotations

import asyncio
import unittest
from unittest.mock import AsyncMock, patch

import numpy as np

import monitor
from strategy import CoinReport


def _report(symbol: str, *, mode: str, score: float) -> CoinReport:
    return CoinReport(
        symbol=symbol,
        tf="15m",
        today_signals=1,
        today_accuracy={},
        today_confirmed=True,
        best_horizon=5,
        best_accuracy=score,
        in_play=True,
        signal_now=True,
        current_price=1.0,
        current_slope=0.1,
        current_rsi=60.0,
        current_adx=25.0,
        current_vol_x=2.0,
        current_macd=0.1,
        signal_mode=mode,
    )


class DiscoveryRoutingTests(unittest.TestCase):
    def test_unchanged_and_open_candidates_do_not_abort_later_additions(self) -> None:
        keep = _report("KEEPUSDT", mode="strong_trend", score=90.0)
        held = _report("HELDUSDT", mode="trend", score=80.0)
        new = _report("NEWUSDT", mode="retest", score=70.0)
        state = monitor.MonitorState(hot_coins=[keep], positions={"HELDUSDT": object()})
        data = np.zeros(2, dtype=[("t", "i8")])
        data["t"] = [1, 2]
        reports = {row.symbol: row for row in (keep, held, new)}

        async def analyze(symbol: str, tf: str, payload: np.ndarray) -> CoinReport:
            return reports[symbol]

        with patch("monitor.config.load_watchlist", return_value=["KEEPUSDT", "HELDUSDT", "NEWUSDT"]), \
             patch("monitor.config.TIMEFRAMES", ("15m",)), \
             patch("monitor.fetch_klines", new=AsyncMock(return_value=data)), \
             patch("monitor._analyze_coin_live", new=AsyncMock(side_effect=analyze)):
            added = asyncio.run(monitor._discover_new_hot_coins(AsyncMock(), state, AsyncMock()))

        self.assertEqual(added, 1)
        self.assertEqual([row.symbol for row in state.hot_coins], ["KEEPUSDT", "NEWUSDT"])
        self.assertIn("NEWUSDT", state.recent_discoveries)


if __name__ == "__main__":
    unittest.main()
