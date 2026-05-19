from __future__ import annotations

import unittest

from audit_v2_exit_advantage_model_families import _bin, _decision, _mean_table, _quantile_edges


class ExitAdvantageModelFamiliesTest(unittest.TestCase):
    def test_quantile_edges_and_bins(self) -> None:
        edges = _quantile_edges([0, 1, 2, 3, 4, 5], bins=3)
        self.assertEqual(len(edges), 2)
        self.assertEqual(_bin(-1, edges), 0)
        self.assertEqual(_bin(99, edges), 2)

    def test_mean_table_respects_min_support(self) -> None:
        table = _mean_table({(0,): [1.0, 3.0], (1,): [9.0]}, min_support=2)
        self.assertEqual(table, {(0,): 2.0})

    def test_decision_requires_beating_always_sell(self) -> None:
        best = {"best_threshold": {"captured_advantage_sum": 9.0, "sell_rate": 0.4}}
        self.assertEqual(
            _decision(best, {"always_sell": 10.0}),
            "research_only_rejected_underperforms_always_sell_proxy",
        )
        self.assertEqual(
            _decision(best, {"always_sell": 1.0}),
            "research_only_advance_to_full_offline_replay",
        )


if __name__ == "__main__":
    unittest.main()
