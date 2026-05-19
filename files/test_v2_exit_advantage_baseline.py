from __future__ import annotations

import unittest

import numpy as np

from train_v2_exit_advantage_baseline import _chronological_split, _evaluate_threshold, _fit_ridge, _predict


class ExitAdvantageBaselineTest(unittest.TestCase):
    def test_chronological_split_orders_by_existing_order(self) -> None:
        rows = [{"ts_ms": i, "sell_advantage": 0.0, "features": {}} for i in range(10)]
        train, holdout = _chronological_split(rows, train_fraction=0.70)
        self.assertEqual(len(train), 7)
        self.assertEqual(len(holdout), 3)
        self.assertEqual(train[-1]["ts_ms"], 6)
        self.assertEqual(holdout[0]["ts_ms"], 7)

    def test_ridge_learns_simple_linear_signal(self) -> None:
        x = np.array([[-1.0], [0.0], [1.0], [2.0]])
        y = np.array([-2.0, 0.0, 2.0, 4.0])
        model = _fit_ridge(x, y, alpha=0.0)
        pred = _predict(x, model)
        self.assertLess(float(np.mean(np.abs(pred - y))), 1e-8)

    def test_threshold_metrics_count_bad_sells(self) -> None:
        rows = [
            {"sell_advantage": 2.0},
            {"sell_advantage": -2.0},
            {"sell_advantage": 0.5},
        ]
        pred = np.array([2.0, 2.0, -1.0])
        metrics = _evaluate_threshold(rows, pred, threshold=0.0)
        self.assertEqual(metrics["sell_count"], 2)
        self.assertEqual(metrics["bad_sell_count"], 1)
        self.assertEqual(metrics["strong_precision"], 0.5)


if __name__ == "__main__":
    unittest.main()
