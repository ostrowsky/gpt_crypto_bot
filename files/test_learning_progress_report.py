from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import learning_progress_report as lpr


class LearningProgressReportTest(unittest.TestCase):
    def test_safe_partial_coverage_does_not_override_developing_verdict(self) -> None:
        latest = lpr.DayMetrics(
            day="2026-05-30",
            early_pct=100.0,
            coverage_status="partial",
            coverage_reasons=("partial candle coverage 186/206",),
            coverage_assessment="partial_safe_inactive_symbols_only",
            missing_series_count=20,
            missing_symbol_status_counts={"BREAK": 20},
        )
        previous = [
            lpr.DayMetrics(day=f"2026-05-2{i}", early_pct=40.0, coverage_status="complete")
            for i in range(3, 10)
        ]
        older = [
            lpr.DayMetrics(day=f"2026-05-1{i}", early_pct=20.0, coverage_status="complete")
            for i in range(6, 23)
        ]

        verdict = lpr._verdict(latest, previous, older, {"training": {"last_finished_at": "2026-05-31T06:00:00Z"}})
        alerts = lpr._alerts(latest, {}, {}, [], {})

        self.assertEqual(verdict["label"], "РАЗВИВАЕТСЯ ПО ЦЕЛЕВОЙ МЕТРИКЕ")
        self.assertEqual(alerts[0]["severity"], "warn")
        self.assertIn("partial_safe_inactive_symbols_only", alerts[0]["text"])

    def test_build_report_flags_stale_training_and_low_early_capture(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            reports = root / "reports"
            reports.mkdir()
            (reports / "top_gainer_critic_2026-05-19_final.json").write_text(json.dumps({
                "target_day_local": "2026-05-19",
                "summary": {
                    "watchlist_top_count": 15,
                    "watchlist_top_bought": 9,
                    "watchlist_top_missed": 6,
                    "watchlist_top_early_captured": 2,
                    "watchlist_top_capture_rate_pct": 60.0,
                    "watchlist_top_early_capture_rate_pct": 13.33,
                    "bot_false_positive_buys": 6,
                    "blocked_winner_count": 6,
                    "blocked_winner_symbols": ["ARUSDT"],
                    "blocked_winner_reason_counts": {"agent_mode_disabled": 3},
                },
                "watchlist_top_gainers": [{"symbol": "ARUSDT", "status": "blocked_rule", "reason": "agent mode disabled", "entries_count": 0, "blocked_count": 10}],
            }), encoding="utf-8")
            (reports / "signal_quality_2026-05-19_final.json").write_text(json.dumps({
                "summary": {
                    "miss_rate": 0.9,
                    "false_positive_rate": 0.05,
                    "capture_ratio_at_entry": {"median": 0.1},
                    "exit_efficiency": {"median": -2.0},
                    "giveback_pct": {"median": 1.5},
                },
                "coverage": {"status": "partial", "reasons": ["partial candle coverage"]},
            }), encoding="utf-8")
            status = root / "status.json"
            status.write_text(json.dumps({
                "training": {"last_finished_at": "2026-05-08T00:00:00Z", "last_rows_total": 100},
                "top_gainer_critic": {"last_target_day_local": "2026-05-19"},
                "signal_quality_evaluator": {"last_target_day_local": "2026-05-19"},
            }), encoding="utf-8")
            feedback = root / "feedback.json"
            feedback.write_text(json.dumps({"apply_cooldown_relaxation": True, "reason": "miss pressure"}), encoding="utf-8")
            report = lpr.build_report(reports, status, feedback, ["AR/USDT"], root / "out.json", root / "out.txt")
            text = lpr.render_text(report)
            self.assertIn("Бот — 2026-05-19", text)
            self.assertIn("shadow re-entry", text)
            self.assertTrue(any(a["severity"] == "serious" for a in report["alerts"]))
            self.assertEqual(report["learning_components"]["ranker_training"]["status"], "stale")
            self.assertEqual(report["focus_symbols"][0]["symbol"], "ARUSDT")

    def test_shadow_reentry_summary_is_rendered_and_actionable(self) -> None:
        scorecard = {
            "status": "complete",
            "summary": {
                "alerts_total": 12,
                "labeled_ret5": 10,
                "pending": 2,
                "avg_ret5": 0.42,
                "ret5_positive_rate": 0.6,
            },
        }
        summary = lpr._shadow_reentry_summary(scorecard)
        self.assertEqual(summary["status"], "complete")
        self.assertIn("avg_ret5=+0.42%", summary["detail"])

        actions = lpr._next_actions(
            lpr.DayMetrics(day="2026-05-31", early_pct=30.0, coverage_status="complete"),
            {"training": {"last_finished_at": "2026-06-01T00:00:00Z"}},
            {},
            [],
            scorecard,
        )
        self.assertTrue(any("Shadow re-entry выглядит promising" in x for x in actions))

    def test_measurement_is_ok_when_critic_is_newer_than_report_day(self) -> None:
        components = lpr._learning_components(
            {
                "training": {"last_finished_at": "2026-05-31T06:00:00Z"},
                "top_gainer_critic": {"last_target_day_local": "2026-05-31"},
                "signal_quality_evaluator": {"last_target_day_local": "2026-05-30"},
            },
            {},
            "2026-05-30",
        )
        self.assertEqual(components["measurement"]["status"], "ok")


class LearningProgressWorkerIntegrationTest(unittest.TestCase):
    def test_rl_worker_has_0900_learning_progress_task(self) -> None:
        src = Path("rl_headless_worker.py").read_text(encoding="utf-8")
        self.assertIn("def _scheduled_learning_progress_slot", src)
        self.assertIn("LEARNING_PROGRESS_DAILY_REPORT_HOUR_LOCAL", src)
        self.assertIn("asyncio.create_task(_learning_progress_loop(state)", src)
        cfg = Path("config.py").read_text(encoding="utf-8")
        self.assertIn("LEARNING_PROGRESS_DAILY_REPORT_HOUR_LOCAL: int = 9", cfg)
        self.assertIn("LEARNING_PROGRESS_DAILY_REPORT_TELEGRAM_ENABLED: bool = True", cfg)

    def test_worker_state_has_learning_progress_fields(self) -> None:
        import rl_headless_worker as worker
        state = worker.WorkerState(train_interval_sec=60, status_interval_sec=60, min_rows=1, min_new_rows=1, collector_enabled=False)
        self.assertTrue(hasattr(state, "learning_progress_enabled"))
        self.assertTrue(hasattr(state, "learning_progress_last_verdict"))


if __name__ == "__main__":
    unittest.main()
