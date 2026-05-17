from __future__ import annotations

import unittest


def _row(sym: str, tf: str, bar_ts: int) -> dict:
    return {
        "sym": sym,
        "tf": tf,
        "bar_ts": bar_ts,
        "f": {"rsi": 55.0, "adx": 20.0},
    }


class TestV2SequenceDataset(unittest.TestCase):
    def test_dedupes_and_builds_transitions_for_contiguous_rows(self) -> None:
        from v2.sequence_dataset import build_from_rows

        base = 1_700_000_000_000
        rows = [
            _row("AAAUSDT", "15m", base),
            _row("AAAUSDT", "15m", base),  # duplicate
            _row("AAAUSDT", "15m", base + 15 * 60 * 1000),
        ]
        result = build_from_rows(rows)
        self.assertEqual(result.summary["duplicates_removed"], 1)
        self.assertEqual(result.summary["sequences_built"], 1)
        self.assertEqual(result.summary["transitions_built"], 1)

    def test_splits_sequences_on_time_gaps(self) -> None:
        from v2.sequence_dataset import build_from_rows

        base = 1_700_000_000_000
        rows = [
            _row("AAAUSDT", "15m", base),
            _row("AAAUSDT", "15m", base + 15 * 60 * 1000),
            _row("AAAUSDT", "15m", base + 45 * 60 * 1000),
        ]
        result = build_from_rows(rows)
        self.assertEqual(result.summary["gap_breaks"], 1)
        self.assertEqual(result.summary["sequences_built"], 2)
        self.assertEqual(result.summary["transitions_built"], 1)

    def test_rejects_bad_rows(self) -> None:
        from v2.sequence_dataset import build_from_rows

        result = build_from_rows([{"sym": "", "tf": "15m", "bar_ts": None, "f": {}}])
        self.assertEqual(result.summary["rows_rejected"], 1)
        self.assertEqual(result.summary["coverage_status"], "insufficient")


if __name__ == "__main__":
    unittest.main()
