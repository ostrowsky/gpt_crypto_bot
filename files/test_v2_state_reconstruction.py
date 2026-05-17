from __future__ import annotations

import unittest


class TestV2StateReconstruction(unittest.TestCase):
    def test_split_is_chronological(self) -> None:
        from v2.state import SymbolState
        from v2.state_reconstruction import ReconstructionRow, chronological_split

        rows = [
            ReconstructionRow("AAA", f"2026-05-0{day}", day, (1.0,) * 9, SymbolState.NOISE, 1.0)
            for day in range(1, 6)
        ]
        train, test = chronological_split(rows, 0.6)
        self.assertEqual([row.local_day for row in train], ["2026-05-01", "2026-05-02", "2026-05-03"])
        self.assertEqual([row.local_day for row in test], ["2026-05-04", "2026-05-05"])

    def test_centroid_prediction_prefers_nearest(self) -> None:
        from v2.state import SymbolState
        from v2.state_reconstruction import predict_centroid

        got = predict_centroid(
            (0.0,) * 9,
            {
                SymbolState.NOISE: (0.0,) * 9,
                SymbolState.CONFIRMED_TREND: (10.0,) * 9,
            },
        )
        self.assertEqual(got, SymbolState.NOISE)

    def test_scaler_uses_train_statistics(self) -> None:
        from v2.state import SymbolState
        from v2.state_reconstruction import ReconstructionRow, fit_scaler, scale_features

        rows = [
            ReconstructionRow("AAA", "2026-05-01", 1, (1.0,) * 9, SymbolState.NOISE, 1.0),
            ReconstructionRow("AAA", "2026-05-01", 2, (3.0,) * 9, SymbolState.NOISE, 1.0),
        ]
        means, stds = fit_scaler(rows)
        self.assertEqual(scale_features((2.0,) * 9, means, stds), (0.0,) * 9)
