from __future__ import annotations

import unittest

from v2.state import SymbolState
from v2.state_reconstruction import ReconstructionRow
from v2.v1_structural_projection import project_v1_structural_features


class V1StructuralProjectionTest(unittest.TestCase):
    def test_projection_emits_dense_v1_style_features(self) -> None:
        row = ReconstructionRow(
            symbol="TESTUSDT",
            local_day="2026-05-18",
            ts_ms=1_700_000_000_000,
            features=(1.0, 2.0, 1.5, 0.5, 24.0, 62.0, 2.0, 0.1, 5.0),
            label=SymbolState.EMERGING_MOVE,
            confidence=0.5,
        )
        projected = project_v1_structural_features(row, today_change_pct=3.0)
        self.assertGreater(projected.projected_forecast_proxy_pct, 0.0)
        self.assertGreater(projected.projected_leader_score_trend, projected.projected_candidate_score_trend)
        self.assertEqual(projected.rsi, 62.0)


if __name__ == "__main__":
    unittest.main()
