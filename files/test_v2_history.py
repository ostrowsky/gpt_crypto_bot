from __future__ import annotations

import unittest


def _row(ts: int) -> dict:
    return {
        "open_ts_ms": ts,
        "open": 1.0,
        "high": 1.1,
        "low": 0.9,
        "close": 1.05,
        "volume": 100.0,
    }


class TestV2CanonicalHistory(unittest.TestCase):
    def test_contiguous_slice_is_reported(self) -> None:
        from v2.history import build_history_slice

        base = 1_779_019_200_000
        slice_ = build_history_slice(
            "AAAUSDT",
            "15m",
            [_row(base), _row(base + 15 * 60 * 1000)],
            source="test",
        )
        self.assertTrue(slice_.continuity.is_contiguous)
        self.assertEqual(slice_.start_ts_ms, base)

    def test_missing_intervals_are_explicit(self) -> None:
        from v2.history import build_history_slice

        base = 1_779_019_200_000
        slice_ = build_history_slice(
            "AAAUSDT",
            "15m",
            [_row(base), _row(base + 45 * 60 * 1000)],
            source="test",
        )
        self.assertFalse(slice_.continuity.is_contiguous)
        self.assertEqual(slice_.continuity.missing_intervals[0].missing_bars, 2)


if __name__ == "__main__":
    unittest.main()
