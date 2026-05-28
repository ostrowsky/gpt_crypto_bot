from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import report_exit_quality


class ExitQualityAuditorTest(unittest.TestCase):
    def _write_report(self, root: Path, day: str, payload: dict) -> None:
        (root / f"signal_quality_{day}_final.json").write_text(json.dumps(payload), encoding="utf-8")

    def test_build_aggregates_tags_and_worst_cases(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            base_case = {
                "sym": "ARUSDT",
                "tf": "15m",
                "source": "agent",
                "mode": "impulse_speed",
                "entry_ts": "2026-05-27T12:30:00Z",
                "exit_ts": "2026-05-27T13:30:00Z",
                "exit_reason": "Price below EMA20",
                "pnl_pct": -2.5,
                "max_favorable_pct": 1.0,
                "future_favorable_pct": 5.0,
                "exit_efficiency": -2.5,
                "giveback_pct": 3.5,
                "entry_timing": "late",
                "exit_timing": "late",
                "trend": {"top_mover_rank": 3, "top_mover_change_pct": 8.0},
            }
            duplicate = dict(base_case)
            good_case = {
                "sym": "ICPUSDT",
                "tf": "15m",
                "source": "bot",
                "mode": "trend",
                "entry_ts": "2026-05-27T07:00:00Z",
                "exit_ts": "2026-05-27T11:00:00Z",
                "exit_reason": "ATR trail broken",
                "pnl_pct": 3.0,
                "max_favorable_pct": 4.0,
                "future_favorable_pct": 4.0,
                "exit_efficiency": 0.75,
                "giveback_pct": 0.8,
                "entry_timing": "ok",
                "exit_timing": "ok",
                "trend": {"top_mover_rank": 2, "top_mover_change_pct": 9.0},
            }
            self._write_report(
                root,
                "2026-05-27",
                {
                    "coverage": {"status": "complete", "paired_trades": 2, "reasons": []},
                    "summary": {
                        "closed_trades": 2,
                        "early_exits": 0,
                        "late_exits": 1,
                        "false_positive_buys": 0,
                        "late_entries": 1,
                        "exit_efficiency": {"median": -0.875, "avg": -0.875},
                        "giveback_pct": {"median": 2.25, "avg": 2.25},
                        "pnl_pct": {"median": 0.25, "avg": 0.25},
                        "realized_capture_ratio": {"median": 0.1},
                    },
                    "late_entries": [base_case, duplicate],
                    "early_exits": [],
                    "false_positive_buys": [],
                    "trades": [good_case],
                },
            )

            report = report_exit_quality.build(days=14, reports_dir=root)

            self.assertEqual(report["summary"]["days_loaded"], 1)
            self.assertEqual(report["summary"]["closed_trades_total"], 2)
            self.assertEqual(report["summary"]["case_rows_loaded"], 2)
            self.assertEqual(report["summary"]["top_mover_exit_failure_count"], 1)
            self.assertEqual(report["summary"]["negative_after_mfe_count"], 1)
            self.assertEqual(report["worst_cases"][0]["sym"], "ARUSDT")
            self.assertIn("top_mover_exit_failure", report["worst_cases"][0]["tags"])
            self.assertIn("post_exit_continuation", report["worst_cases"][0]["tags"])
            self.assertEqual(report["exit_reason_buckets"][0]["exit_reason_bucket"], "ema_break")

    def test_empty_reports_are_marked_empty(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            report = report_exit_quality.build(days=14, reports_dir=Path(td))
            self.assertEqual(report["status"], "empty")
            self.assertEqual(report["summary"]["days_loaded"], 0)
            self.assertEqual(report["summary"]["case_coverage_status"], "empty")


if __name__ == "__main__":
    unittest.main()
