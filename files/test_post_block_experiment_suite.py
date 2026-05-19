from __future__ import annotations

import unittest

from run_post_block_experiment_suite import _base, _passes_gate


class PostBlockExperimentSuiteTest(unittest.TestCase):
    def test_base_rate(self) -> None:
        rows = [{"x": True}, {"x": False}]
        self.assertEqual(_base(rows, lambda r: r["x"])["rate"], 0.5)

    def test_gate_requires_lift_and_support(self) -> None:
        self.assertTrue(_passes_gate(20, 5, 0.25, 0.05))
        self.assertFalse(_passes_gate(5, 5, 1.0, 0.05))
        self.assertFalse(_passes_gate(20, 2, 0.1, 0.05))


if __name__ == "__main__":
    unittest.main()
