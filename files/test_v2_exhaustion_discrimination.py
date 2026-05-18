from __future__ import annotations

import unittest

from audit_v2_exhaustion_discrimination import _standardized_mean_difference


class ExhaustionDiscriminationTest(unittest.TestCase):
    def test_standardized_mean_difference_is_positive_when_a_is_higher(self) -> None:
        self.assertGreater(_standardized_mean_difference([3.0, 4.0, 5.0], [1.0, 2.0, 3.0]), 0.0)


if __name__ == "__main__":
    unittest.main()
