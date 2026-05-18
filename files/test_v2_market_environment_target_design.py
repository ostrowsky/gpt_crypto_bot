from __future__ import annotations

import unittest

from audit_v2_market_environment_target_design import _state_mix


class MarketEnvironmentTargetDesignTest(unittest.TestCase):
    def test_state_mix_is_normalized(self) -> None:
        mix = _state_mix(["noise", "noise", "mature_trend"])
        self.assertEqual(mix, {"mature_trend": 0.333333, "noise": 0.666667})


if __name__ == "__main__":
    unittest.main()
