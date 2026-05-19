from __future__ import annotations

import unittest

from audit_v2_market_breadth_observation_store import (
    _aggregate_market_breadth,
    _ret,
    _top_decile_minus_median,
)


class MarketBreadthObservationStoreTest(unittest.TestCase):
    def test_aggregate_market_breadth_computes_core_shares(self) -> None:
        rows = [
            {
                "symbol": "BTCUSDT",
                "ret_1bar_pct": 1.0,
                "ret_4bar_pct": 2.0,
                "ret_8bar_pct": 3.0,
                "ret_day_pct": 4.0,
                "above_ema20": 1.0,
                "above_ema50": 1.0,
                "price_vs_ema20_pct": 0.5,
                "volume_gt_mean20": 1.0,
            },
            {
                "symbol": "ETHUSDT",
                "ret_1bar_pct": -1.0,
                "ret_4bar_pct": -2.0,
                "ret_8bar_pct": -3.0,
                "ret_day_pct": -4.0,
                "above_ema20": 0.0,
                "above_ema50": 0.0,
                "price_vs_ema20_pct": -0.5,
                "volume_gt_mean20": 0.0,
            },
        ]
        features = _aggregate_market_breadth(rows, tracked_symbols=4)
        self.assertEqual(features["available_symbols"], 2.0)
        self.assertEqual(features["available_share"], 0.5)
        self.assertEqual(features["ret4_positive_share"], 0.5)
        self.assertEqual(features["above_ema20_share"], 0.5)
        self.assertEqual(features["btc_ret_day_pct"], 4.0)
        self.assertEqual(features["eth_ret_day_pct"], -4.0)

    def test_ret_handles_zero_previous_price(self) -> None:
        self.assertEqual(_ret(10.0, 0.0), 0.0)
        self.assertAlmostEqual(_ret(11.0, 10.0), 10.0)

    def test_top_decile_minus_median_is_positive_for_leaders(self) -> None:
        self.assertGreater(_top_decile_minus_median([-1, 0, 1, 10, 20]), 0.0)


if __name__ == "__main__":
    unittest.main()
