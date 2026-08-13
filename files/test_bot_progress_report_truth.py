from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from datetime import date, timedelta
from pathlib import Path


SCRIPT = Path(__file__).resolve().parent.parent / "skills" / "bot-progress-report" / "scripts" / "build_progress_report.py"
SPEC = importlib.util.spec_from_file_location("truthful_progress_report", SCRIPT)
assert SPEC and SPEC.loader
progress = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(progress)


def _write_day(root: Path, day: date, *, top: int, bought: int, early: int, quality: str = "complete") -> None:
    ds = day.isoformat()
    (root / f"top_gainer_critic_{ds}_final.json").write_text(json.dumps({
        "phase": "final",
        "summary": {
            "watchlist_top_count": top,
            "watchlist_top_bought": bought,
            "watchlist_top_early_captured": early,
            "bot_unique_buys": 10,
            "bot_false_positive_buys": 2,
        },
    }), encoding="utf-8")
    (root / f"watchlist_top_gainer_goal_{ds}_22h.json").write_text(json.dumps({
        "summary": {
            "watchlist_top_count": top,
            "watchlist_top_bought": bought,
            "bot_unique_buys": 10,
            "bot_false_positive_buys": 2,
        },
    }), encoding="utf-8")
    (root / f"signal_quality_{ds}_final.json").write_text(json.dumps({
        "coverage": {"status": quality},
        "summary": {
            "trend_episodes_total": 10,
            "missed_trends": 5,
            "buys_total": 10,
            "false_positive_buys": 2,
            "capture_ratio_at_entry": {"median": 0.4},
            "exit_efficiency": {"median": 0.2},
            "giveback_pct": {"median": 0.5},
        },
    }), encoding="utf-8")


class TruthfulProgressReportTest(unittest.TestCase):
    def test_zero_denominator_is_unknown(self) -> None:
        self.assertIsNone(progress._pct(0, 0))

    def test_missing_goal_day_is_excluded_from_objective_ratio(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            reports = Path(td)
            end = date(2026, 8, 12)
            _write_day(reports, end - timedelta(days=1), top=2, bought=1, early=1)
            _write_day(reports, end, top=100, bought=100, early=100)
            (reports / "watchlist_top_gainer_goal_2026-08-12_22h.json").unlink()
            old = progress.REPORTS
            try:
                progress.REPORTS = reports
                payload = progress.build(2, end)
            finally:
                progress.REPORTS = old

        self.assertEqual(payload["eligible_days"], ["2026-08-11"])
        self.assertEqual(payload["scout"]["early_capture"]["numerator"], 1.0)
        self.assertEqual(payload["scout"]["early_capture"]["denominator"], 2.0)

    def test_partial_quality_preserves_capture_but_blocks_verdict(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            reports = Path(td)
            end = date(2026, 8, 12)
            for offset in range(6):
                early = 1 if offset < 3 else 3
                _write_day(
                    reports,
                    end - timedelta(days=5 - offset),
                    top=5,
                    bought=4,
                    early=early,
                    quality="partial",
                )
            old = progress.REPORTS
            try:
                progress.REPORTS = reports
                payload = progress.build(6, end)
            finally:
                progress.REPORTS = old

        self.assertEqual(payload["scout"]["early_capture"]["denominator"], 30.0)
        self.assertEqual(payload["quality_eligible_days"], [])
        self.assertEqual(payload["comparison"]["status"], "quality_guardrails_incomplete")
        self.assertEqual(payload["verdict"]["toward_goal"], "inconclusive")

    def test_high_absolute_rates_do_not_fake_improvement(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            reports = Path(td)
            end = date(2026, 8, 12)
            for offset in range(6):
                _write_day(reports, end - timedelta(days=5 - offset), top=5, bought=4, early=3)
            old = progress.REPORTS
            try:
                progress.REPORTS = reports
                payload = progress.build(6, end)
            finally:
                progress.REPORTS = old

        self.assertEqual(payload["scout"]["watchlist_top_bought_rate_pct"], 80.0)
        self.assertEqual(payload["verdict"]["toward_goal"], "flat")

    def test_small_comparison_denominator_is_inconclusive(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            reports = Path(td)
            end = date(2026, 8, 12)
            for offset in range(6):
                _write_day(reports, end - timedelta(days=5 - offset), top=1, bought=1, early=1)
            old = progress.REPORTS
            try:
                progress.REPORTS = reports
                payload = progress.build(6, end)
            finally:
                progress.REPORTS = old

        self.assertEqual(payload["comparison"]["status"], "small_denominator")
        self.assertEqual(payload["verdict"]["toward_goal"], "inconclusive")

    def test_guardrailed_early_capture_gain_can_be_improving(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            reports = Path(td)
            end = date(2026, 8, 12)
            for offset in range(3):
                _write_day(reports, end - timedelta(days=5 - offset), top=5, bought=4, early=1)
            for offset in range(3):
                _write_day(reports, end - timedelta(days=2 - offset), top=5, bought=4, early=3)
            old = progress.REPORTS
            try:
                progress.REPORTS = reports
                payload = progress.build(6, end)
            finally:
                progress.REPORTS = old

        self.assertGreater(payload["comparison"]["early_capture_delta_pp"], 5.0)
        self.assertEqual(payload["verdict"]["toward_goal"], "improving")


if __name__ == "__main__":
    unittest.main()
