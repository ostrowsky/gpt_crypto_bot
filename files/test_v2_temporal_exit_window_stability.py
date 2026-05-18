from __future__ import annotations

import unittest

from audit_v2_temporal_exit_window_stability import _split_windows


class TemporalExitWindowStabilityTest(unittest.TestCase):
    def test_split_windows_preserves_order_and_balance(self) -> None:
        windows = _split_windows(list(range(10)), parts=4)
        self.assertEqual([len(window) for window in windows], [3, 3, 2, 2])
        self.assertEqual(sum(windows, []), list(range(10)))


if __name__ == "__main__":
    unittest.main()
