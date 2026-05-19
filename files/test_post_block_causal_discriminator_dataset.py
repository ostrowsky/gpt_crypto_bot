from __future__ import annotations

import unittest

from build_post_block_causal_discriminator_dataset import _first_bar_at_or_after, _ret_pct, _range_pct


class PostBlockCausalDiscriminatorDatasetTest(unittest.TestCase):
    def test_first_bar_at_or_after(self) -> None:
        bars = [{"open_ts_ms": 1000}, {"open_ts_ms": 2000}, {"open_ts_ms": 3000}]
        self.assertEqual(_first_bar_at_or_after(bars, 500), 0)
        self.assertEqual(_first_bar_at_or_after(bars, 2000), 1)
        self.assertEqual(_first_bar_at_or_after(bars, 2500), 2)
        self.assertIsNone(_first_bar_at_or_after(bars, 4000))

    def test_ret_and_range_pct(self) -> None:
        self.assertAlmostEqual(_ret_pct(110.0, 100.0), 10.0)
        self.assertEqual(_ret_pct(110.0, 0.0), 0.0)
        self.assertAlmostEqual(_range_pct({"high": 110.0, "low": 90.0, "close": 100.0}), 20.0)


if __name__ == "__main__":
    unittest.main()
