from __future__ import annotations

import unittest


class TestV2BeliefFilter(unittest.TestCase):
    def test_transition_matrix_is_normalized(self) -> None:
        from v2.belief_filter import transition_matrix

        for row in transition_matrix().values():
            self.assertAlmostEqual(sum(row.values()), 1.0)

    def test_filter_outputs_normalized_beliefs(self) -> None:
        from v2.belief_filter import filter_rows
        from v2.state import SymbolState
        from v2.state_reconstruction import ReconstructionRow

        rows = [
            ReconstructionRow("AAA", "2026-05-18", i, (0.0,) * 9, SymbolState.NOISE, 1.0)
            for i in range(3)
        ]
        centroids = {state: ((0.0,) * 9 if state == SymbolState.NOISE else (10.0,) * 9) for state in SymbolState}
        filtered = filter_rows(rows, centroids)
        self.assertEqual(len(filtered), 3)
        for item in filtered:
            self.assertAlmostEqual(sum(item.belief.probabilities.values()), 1.0)
