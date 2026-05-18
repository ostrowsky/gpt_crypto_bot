from __future__ import annotations

import unittest

from audit_v2_temporal_exit_robustness import _profile_name


class TemporalExitRobustnessTest(unittest.TestCase):
    def test_profile_name_is_stable_and_readable(self) -> None:
        self.assertEqual(
            _profile_name(-0.15, 0.10),
            "mature_decay_0_15_late_rise_0_10",
        )


if __name__ == "__main__":
    unittest.main()
