from __future__ import annotations

import unittest

from audit_v2_market_environment_horizon_belief import _nearest_centroid_prediction


class MarketEnvironmentHorizonBeliefTest(unittest.TestCase):
    def test_nearest_centroid_prediction_returns_confidence(self) -> None:
        features = {
            "adx": 1.0,
            "projected_leader_score_trend": 1.0,
            "daily_range_pct": 1.0,
            "projected_forecast_proxy_pct": 1.0,
            "price_vs_ema20_pct": 1.0,
            "noise_share": 1.0,
            "emerging_share": 0.0,
            "mature_share": 0.0,
            "belief_late_mass": 0.0,
        }
        history = [
            {"label": "candidate_favorable", "features": features},
            {"label": "candidate_unfavorable", "features": {name: 10.0 for name in features}},
        ]
        pred, confidence = _nearest_centroid_prediction(history, features)
        self.assertEqual(pred, "candidate_favorable")
        self.assertGreater(confidence, 0.5)


if __name__ == "__main__":
    unittest.main()
