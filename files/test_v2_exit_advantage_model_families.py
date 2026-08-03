from __future__ import annotations

import unittest

from audit_v2_exit_advantage_model_families import (
    _bin,
    _candidate,
    _chronological_day_split,
    _decision,
    _feature_value,
    _mean_table,
    _quantile_edges,
)
import numpy as np


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

    def test_position_context_features_are_read_from_causal_top_level_fields(self) -> None:
        row = {
            "bars_held": 7,
            "unrealized_pnl_pct": 1.25,
            "mfe_pct": 2.5,
            "giveback_pct": 1.25,
            "candidate_action": "sell",
            "features": {},
        }

        self.assertEqual(_feature_value(row, "bars_held"), 7.0)
        self.assertEqual(_feature_value(row, "giveback_pct"), 1.25)
        self.assertEqual(_feature_value(row, "candidate_action_sell"), 1.0)

    def test_day_split_is_disjoint_and_purges_boundary_days(self) -> None:
        rows = [
            {"local_day": f"2026-05-{day:02d}", "ts_ms": day, "sell_advantage": 0.0, "features": {}}
            for day in range(1, 21)
        ]

        train, validation, holdout, split = _chronological_day_split(
            rows,
            train_fraction=0.60,
            validation_fraction=0.20,
            purge_days=1,
        )

        self.assertTrue(train and validation and holdout)
        self.assertTrue(set(split["train_days"]).isdisjoint(split["validation_days"]))
        self.assertTrue(set(split["validation_days"]).isdisjoint(split["holdout_days"]))
        self.assertEqual(len(split["purged_days"]), 4)

    def test_threshold_is_selected_on_validation_not_holdout(self) -> None:
        validation_rows = [
            {"sell_advantage": 2.0, "features": {}},
            {"sell_advantage": -3.0, "features": {}},
        ]
        holdout_rows = [
            {"sell_advantage": -3.0, "features": {}},
            {"sell_advantage": 2.0, "features": {}},
        ]
        candidate = _candidate(
            "synthetic",
            ["x"],
            [],
            {},
            0.0,
            validation_rows,
            np.array([2.0, -2.0]),
            holdout_rows,
            np.array([2.0, -2.0]),
        )

        self.assertEqual(candidate["selected_threshold_on_validation"]["threshold"], -1.0)
        self.assertEqual(candidate["holdout_result"]["threshold"], -1.0)


if __name__ == "__main__":
    unittest.main()
