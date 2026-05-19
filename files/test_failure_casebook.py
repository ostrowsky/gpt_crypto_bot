from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from build_failure_casebook import build


class FailureCasebookTest(unittest.TestCase):
    def test_builds_compact_casebook_from_reports(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "signal_quality_2026-05-01_final.json").write_text(
                json.dumps(
                    {
                        "late_entries": [
                            {
                                "sym": "AAAUSDT",
                                "tf": "15m",
                                "pnl_pct": -2.0,
                                "max_favorable_pct": 4.0,
                                "giveback_pct": 6.0,
                                "exit_reason": "late",
                            }
                        ],
                        "false_positive_buys": [
                            {"sym": "BBBUSDT", "mode": "impulse", "pnl_pct": -3.0, "giveback_pct": 1.0}
                        ],
                    }
                ),
                encoding="utf-8",
            )
            (root / "top_gainer_critic_2026-05-01_final.json").write_text(
                json.dumps(
                    {
                        "target_day_local": "2026-05-01",
                        "watchlist_top_gainers": [
                            {
                                "symbol": "CCCUSDT",
                                "status": "missed",
                                "day_change_pct": 5.0,
                                "first_block_time": "08:00",
                                "first_block_reason_code": "score_gate",
                                "opportunity_from_first_block_pct": 4.5,
                                "blocked_count": 9,
                                "blocked_reason_counts": {"score_gate": 9},
                            },
                            {
                                "symbol": "DDDUSDT",
                                "status": "bought",
                                "capture_ratio_at_entry": 0.1,
                                "opportunity_from_entry_pct": -1.5,
                                "blocked_reason_counts": {"score_gate": 3},
                            },
                        ],
                        "blocked_reason_harm": [
                            {"reason_code": "score_gate", "blocked_events": 9, "missed_opportunity_pct": 4.5, "missed_symbols": ["CCCUSDT"]}
                        ],
                    }
                ),
                encoding="utf-8",
            )
            payload = build(root, limit=5)
            self.assertEqual(payload["source_counts"]["signal_quality_reports"], 1)
            self.assertEqual(payload["source_counts"]["top_gainer_critic_reports"], 1)
            self.assertEqual(payload["worst_exit_cases"][0]["symbol"], "AAAUSDT")
            self.assertEqual(payload["missed_or_blocked_winner_cases"][0]["symbol"], "CCCUSDT")
            self.assertTrue(payload["hypotheses"])


if __name__ == "__main__":
    unittest.main()
