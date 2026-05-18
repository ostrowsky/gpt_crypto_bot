from __future__ import annotations

import unittest


class TestV2BeliefCalibration(unittest.TestCase):
    def test_key_projection(self) -> None:
        import audit_v2_belief_calibration as audit

        item = {
            "self_bias": 0.7,
            "temperature": 1.0,
            "metrics": {"macro_f1": 0.3},
            "delta_vs_isolated": {"macro_f1": 0.1},
        }
        self.assertEqual(audit._key(item)["self_bias"], 0.7)
