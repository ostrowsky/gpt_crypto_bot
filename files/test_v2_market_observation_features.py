from __future__ import annotations

import unittest

from audit_v2_market_observation_features import _breadth_features


class MarketObservationFeaturesTest(unittest.TestCase):
    def test_breadth_features_compute_positive_shares(self) -> None:
        rows = [
            {
                "v1_projected_structural": {
                    "price_vs_ema20_pct": 1.0,
                    "rsi": 55,
                    "projected_forecast_proxy_pct": 1.2,
                    "projected_leader_score_trend": 3.0,
                }
            },
            {
                "v1_projected_structural": {
                    "price_vs_ema20_pct": -1.0,
                    "rsi": 45,
                    "projected_forecast_proxy_pct": 0.8,
                    "projected_leader_score_trend": 5.0,
                }
            },
        ]
        features = _breadth_features(rows)
        self.assertEqual(features["breadth_ema20_positive_share"], 0.5)
        self.assertEqual(features["breadth_rsi50_share"], 0.5)
        self.assertEqual(features["breadth_forecast_gt1_share"], 0.5)


if __name__ == "__main__":
    unittest.main()
