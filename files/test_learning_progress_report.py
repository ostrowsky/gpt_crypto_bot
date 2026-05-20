from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import learning_progress_report as lpr


class LearningProgressReportTest(unittest.TestCase):
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
            self.assertTrue(any(a["severity"] == "serious" for a in report["alerts"]))
            self.assertEqual(report["learning_components"]["ranker_training"]["status"], "stale")
            self.assertEqual(report["focus_symbols"][0]["symbol"], "ARUSDT")


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
