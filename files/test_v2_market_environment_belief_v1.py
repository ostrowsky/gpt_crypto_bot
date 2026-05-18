from __future__ import annotations

import unittest

from audit_v2_market_environment_belief_v1 import _observation_probs


class MarketEnvironmentBeliefV1Test(unittest.TestCase):
    def test_observation_probs_are_normalized(self) -> None:
        features = {
            "adx": 1.0,
            "projected_leader_score_trend": 1.0,
            "daily_range_pct": 1.0,
            "projected_forecast_proxy_pct": 1.0,
            "price_vs_ema20_pct": 1.0,
        }
        history = [
            {"label": "candidate_favorable", "features": features},
            {
                "label": "candidate_unfavorable",
                "features": {name: 10.0 for name in features},
            },
        ]
        probs = _observation_probs(history, features)
        self.assertAlmostEqual(sum(probs.values()), 1.0)
        self.assertGreater(probs["candidate_favorable"], probs["candidate_unfavorable"])


if __name__ == "__main__":
    unittest.main()
