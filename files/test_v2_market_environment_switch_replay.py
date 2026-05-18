from __future__ import annotations

import unittest

from audit_v2_market_environment_switch_replay import _nearest_centroid_label


class MarketEnvironmentSwitchReplayTest(unittest.TestCase):
    def test_nearest_centroid_label_uses_feature_distance(self) -> None:
        train = [
            {
                "label": "candidate_favorable",
                "features": {
                    "adx": 1.0,
                    "projected_leader_score_trend": 1.0,
                    "daily_range_pct": 1.0,
                    "projected_forecast_proxy_pct": 1.0,
                    "price_vs_ema20_pct": 1.0,
                },
            },
            {
                "label": "candidate_unfavorable",
                "features": {
                    "adx": 10.0,
                    "projected_leader_score_trend": 10.0,
                    "daily_range_pct": 10.0,
                    "projected_forecast_proxy_pct": 10.0,
                    "price_vs_ema20_pct": 10.0,
                },
            },
        ]
        self.assertEqual(
            _nearest_centroid_label(
                train,
                {
                    "adx": 1.2,
                    "projected_leader_score_trend": 1.2,
                    "daily_range_pct": 1.2,
                    "projected_forecast_proxy_pct": 1.2,
                    "price_vs_ema20_pct": 1.2,
                },
            ),
            "candidate_favorable",
        )


if __name__ == "__main__":
    unittest.main()
