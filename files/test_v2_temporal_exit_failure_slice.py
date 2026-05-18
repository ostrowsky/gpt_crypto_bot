from __future__ import annotations

import unittest

from audit_v2_temporal_exit_failure_slice import _component_delta


class TemporalExitFailureSliceTest(unittest.TestCase):
    def test_component_delta_is_candidate_minus_base(self) -> None:
        self.assertEqual(
            _component_delta({"a": 1.0, "b": 2.0}, {"a": 3.0, "c": 4.0}),
            {"a": 2.0, "b": -2.0, "c": 4.0},
        )


if __name__ == "__main__":
    unittest.main()
