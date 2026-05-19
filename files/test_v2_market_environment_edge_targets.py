from __future__ import annotations

import unittest

from audit_v2_market_environment_edge_targets import _edge_label


class MarketEnvironmentEdgeTargetsTest(unittest.TestCase):
    def test_edge_label_has_no_edge_zone(self) -> None:
        self.assertEqual(_edge_label(2.0, 1.0), "candidate_edge")
        self.assertEqual(_edge_label(-2.0, 1.0), "base_edge")
        self.assertEqual(_edge_label(0.2, 1.0), "no_edge")


if __name__ == "__main__":
    unittest.main()
