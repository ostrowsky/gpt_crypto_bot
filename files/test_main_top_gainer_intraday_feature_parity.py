from __future__ import annotations

import unittest
from datetime import datetime, timezone

import numpy as np

import monitor
from strategy import CoinReport, HorizonAccuracy


def _ms(iso: str) -> int:
    return int(datetime.fromisoformat(iso.replace("Z", "+00:00")).timestamp() * 1000)


class MainTopGainerIntradayFeatureParityTest(unittest.TestCase):
    def test_intraday_change_supports_structured_array_and_local_day_open(self) -> None:
        candles = np.array(
            [
                (_ms("2026-05-27T21:00:00Z"), 0.097, 0.098),
                (_ms("2026-05-27T22:00:00Z"), 0.098, 0.099),  # Europe/Budapest local day open
                (_ms("2026-05-28T10:00:00Z"), 0.101, 0.1023),
                (_ms("2026-05-28T12:00:00Z"), 0.103, 0.1040),
            ],
            dtype=[("t", "i8"), ("o", "f8"), ("c", "f8")],
        )

        change = monitor._intraday_change_pct_from_data(candles, 3)

        self.assertAlmostEqual(change, (0.1040 / 0.098 - 1.0) * 100.0, places=6)

    def test_bat_like_top_gainer_score_passes_when_intraday_component_present(self) -> None:
        old_score = monitor._top_gainer_live_score(
            mode="impulse_speed",
            intraday_change_pct=0.0,
            daily_range=8.4463,
            vol_x=3.634,
            adx=19.32,
            rsi=69.17,
            ranker_info=None,
        )
        fixed_score = monitor._top_gainer_live_score(
            mode="impulse_speed",
            intraday_change_pct=6.0143,
            daily_range=8.4463,
            vol_x=3.634,
            adx=19.32,
            rsi=69.17,
            ranker_info=None,
        )

        self.assertLess(old_score, 34.0)
        self.assertGreaterEqual(fixed_score, 34.0)

    def test_report_move_features_backfill_zero_fields(self) -> None:
        candles = np.array(
            [
                (_ms("2026-05-27T22:00:00Z"), 0.098, 0.099),
                (_ms("2026-05-28T10:00:00Z"), 0.101, 0.1023),
            ],
            dtype=[("t", "i8"), ("o", "f8"), ("c", "f8")],
        )
        report = CoinReport(
            symbol="BATUSDT",
            tf="1h",
            today_signals=0,
            today_accuracy={3: HorizonAccuracy(3, 0, 0)},
            today_confirmed=False,
            best_horizon=0,
            best_accuracy=0.0,
            in_play=False,
        )

        today, forecast = monitor._ensure_report_move_features(
            report,
            data=candles,
            i=1,
            slope=0.6191,
            adx=18.13,
            vol_x=4.24,
            rsi=63.9,
        )

        self.assertGreater(today, 0.0)
        self.assertGreater(forecast, 0.0)
        self.assertAlmostEqual(report.today_change_pct, today)
        self.assertAlmostEqual(report.forecast_return_pct, forecast)


if __name__ == "__main__":
    unittest.main()
