from __future__ import annotations

import unittest

from test_v2_offline_env import _frame
from v2.entry_admission_dataset import V1StructuralFeatures, build_row
from v2.offline_env import DecisionFrame
from v2.state import SymbolState


class _Row:
    symbol = "TESTUSDT"
    local_day = "2026-05-18"
    ts_ms = 1_700_000_000_000
    label = SymbolState.EMERGING_MOVE


class _Filtered:
    def __init__(self) -> None:
        frame: DecisionFrame = _frame(0, SymbolState.EMERGING_MOVE, 101.0, 101.2)
        self.row = _Row()
        self.belief = frame.belief
        self.prediction = SymbolState.EMERGING_MOVE


class EntryAdmissionDatasetTest(unittest.TestCase):
    def test_build_row_keeps_belief_and_v1_features(self) -> None:
        row = build_row(_Filtered(), structural=V1StructuralFeatures(candidate_score=42.0))
        self.assertEqual(row.symbol, "TESTUSDT")
        self.assertEqual(row.v1_structural.candidate_score, 42.0)
        self.assertAlmostEqual(sum(row.belief.values()), 1.0)
        self.assertGreater(row.belief_entropy, 0.0)


if __name__ == "__main__":
    unittest.main()
