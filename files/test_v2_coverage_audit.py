from __future__ import annotations

import unittest


def _row(sym: str, tf: str, bar_ts: int) -> dict:
    return {
        "sym": sym,
        "tf": tf,
        "bar_ts": bar_ts,
        "f": {"rsi": 55.0},
    }


class TestV2CoverageAudit(unittest.TestCase):
    def test_reports_longest_sequence_and_fragmentation(self) -> None:
        from v2.coverage import build_coverage_audit
        from v2.sequence_dataset import build_from_rows

        base = 1_779_019_200_000
        rows = [
            _row("AAAUSDT", "15m", base),
            _row("AAAUSDT", "15m", base + 15 * 60 * 1000),
            _row("AAAUSDT", "15m", base + 30 * 60 * 1000),
            _row("AAAUSDT", "15m", base + 60 * 60 * 1000),
        ]
        audit = build_coverage_audit(build_from_rows(rows))
        self.assertEqual(audit.summary["longest_sequence_bars"], 3)
        self.assertEqual(audit.summary["longest_sequence_minutes"], 45)
        self.assertEqual(audit.fragmented_slices[0]["segments"], 2)


if __name__ == "__main__":
    unittest.main()
