from __future__ import annotations

import json
import unittest
import tempfile
from pathlib import Path

import config
import rl_headless_worker
from rl_headless_worker import (
    WorkerState,
    _claim_learning_progress_telegram_slot,
    build_status_snapshot,
    _render_top_gainer_telegram,
    _should_send_top_gainer_telegram,
    should_train,
)


class TestRLHeadlessWorkerTrainingGate(unittest.TestCase):
    def test_retrains_when_dataset_content_changes_without_new_rows(self) -> None:
        self.assertTrue(
            should_train(
                rows_total=3255,
                min_rows=500,
                last_trained_rows=3255,
                min_new_rows=50,
                dataset_mtime=200.0,
                last_dataset_mtime=100.0,
                force_first_train=True,
            )
        )

    def test_retrains_when_dataset_rolls_back_but_is_newer(self) -> None:
        self.assertTrue(
            should_train(
                rows_total=3200,
                min_rows=500,
                last_trained_rows=3255,
                min_new_rows=50,
                dataset_mtime=200.0,
                last_dataset_mtime=100.0,
                force_first_train=True,
            )
        )

    def test_does_not_retrain_when_dataset_is_older_or_unchanged(self) -> None:
        self.assertFalse(
            should_train(
                rows_total=3200,
                min_rows=500,
                last_trained_rows=3255,
                min_new_rows=50,
                dataset_mtime=100.0,
                last_dataset_mtime=100.0,
                force_first_train=True,
            )
        )
        self.assertFalse(
            should_train(
                rows_total=3255,
                min_rows=500,
                last_trained_rows=3255,
                min_new_rows=50,
                dataset_mtime=100.0,
                last_dataset_mtime=100.0,
                force_first_train=True,
            )
        )


class TestTopGainerTelegramReport(unittest.TestCase):
    def test_final_report_uses_compact_daily_format(self) -> None:
        text = _render_top_gainer_telegram(
            {
                "phase": "final",
                "target_day_local": "2026-04-25",
                "summary": {
                    "watchlist_top_bought": 10,
                    "watchlist_top_count": 15,
                    "watchlist_top_capture_rate_pct": 66.67,
                    "watchlist_top_early_captured": 2,
                    "watchlist_top_early_capture_rate_pct": 13.33,
                    "bot_false_positive_buys": 12,
                    "bot_unique_buys": 22,
                },
                "watchlist_top_gainers": [
                    {"symbol": "AXSUSDT", "status": "bought", "first_entry_mode": "trend"},
                    {"symbol": "ALGOUSDT", "status": "bought", "first_entry_mode": "impulse_speed"},
                    {"symbol": "GMTUSDT", "status": "bought", "first_entry_mode": "impulse_speed"},
                    {"symbol": "ILVUSDT", "status": "blocked_rule"},
                    {"symbol": "TRUUSDT", "status": "blocked_rule"},
                    {"symbol": "RUNEUSDT", "status": "no_signal"},
                ],
                "bot_false_positive_symbols": [
                    "AXLUSDT",
                    "C98USDT",
                    "CELRUSDT",
                    "DYDXUSDT",
                    "EGLDUSDT",
                    "EXTRAUSDT",
                ],
            }
        )

        self.assertIn("Top gainer critic final", text)
        self.assertIn("day: 2026-04-25", text)
        self.assertIn("watchlist top bought: 10/15 (66.67%)", text)
        self.assertIn("early captures: 2/15 (13.33%)", text)
        self.assertIn("false-positive buys: 12/22", text)
        self.assertIn("bought: AXSUSDT trend, ALGOUSDT impulse_speed, GMTUSDT impulse_speed", text)
        self.assertIn("missed: ILVUSDT blocked_rule, TRUUSDT blocked_rule, RUNEUSDT no_signal", text)
        self.assertIn("false positives: AXLUSDT, C98USDT, CELRUSDT, DYDXUSDT, EGLDUSDT", text)
        self.assertNotIn("EXTRAUSDT", text)

    def test_top_gainer_telegram_is_final_only_when_configured(self) -> None:
        old_enabled = config.TOP_GAINER_CRITIC_TELEGRAM_REPORTS_ENABLED
        old_final_only = config.TOP_GAINER_CRITIC_TELEGRAM_FINAL_ONLY
        try:
            config.TOP_GAINER_CRITIC_TELEGRAM_REPORTS_ENABLED = True
            config.TOP_GAINER_CRITIC_TELEGRAM_FINAL_ONLY = True

            self.assertTrue(_should_send_top_gainer_telegram("final"))
            self.assertFalse(_should_send_top_gainer_telegram("midday"))
        finally:
            config.TOP_GAINER_CRITIC_TELEGRAM_REPORTS_ENABLED = old_enabled
            config.TOP_GAINER_CRITIC_TELEGRAM_FINAL_ONLY = old_final_only


class TestLearningProgressTelegramIdempotency(unittest.TestCase):
    def test_learning_progress_slot_claim_is_persistent(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            marker_dir = Path(td)
            self.assertTrue(_claim_learning_progress_telegram_slot("2026-06-03::learning_progress", marker_dir))
            self.assertFalse(_claim_learning_progress_telegram_slot("2026-06-03::learning_progress", marker_dir))
            self.assertTrue(_claim_learning_progress_telegram_slot("2026-06-04::learning_progress", marker_dir))


class TestResearchUniverseShadowStatus(unittest.TestCase):
    def test_status_snapshot_exposes_research_universe_shadow_collector(self) -> None:
        state = WorkerState(
            train_interval_sec=60,
            status_interval_sec=60,
            min_rows=1,
            min_new_rows=1,
            collector_enabled=False,
        )
        state.research_universe_shadow_runs_total = 1
        state.research_universe_shadow_runs_ok = 1
        state.research_universe_shadow_last_symbols_scanned = 123
        state.research_universe_shadow_last_rows_written = 45

        snapshot = build_status_snapshot(state, critic_report={}, ml_rows_total=0)
        section = snapshot["research_universe_shadow"]

        self.assertTrue(section["enabled"])
        self.assertEqual(section["runs_total"], 1)
        self.assertEqual(section["last_symbols_scanned"], 123)
        self.assertEqual(section["last_rows_written"], 45)
        self.assertIn("research_universe_shadow.jsonl", section["dataset_file"])

    def test_status_snapshot_overlays_in_progress_collector_status(self) -> None:
        state = WorkerState(
            train_interval_sec=60,
            status_interval_sec=60,
            min_rows=1,
            min_new_rows=1,
            collector_enabled=False,
        )
        with tempfile.TemporaryDirectory() as td:
            status_file = Path(td) / "research_status.json"
            status_file.write_text(
                json.dumps(
                    {
                        "running": True,
                        "started_at": "2026-06-04T11:18:03Z",
                        "finished_at": None,
                        "symbols_total": 80,
                        "symbols_scanned": 80,
                        "pairs_scanned": 12,
                        "rows_written": 72,
                        "labels_updated": 3,
                        "last_error": None,
                    }
                ),
                encoding="utf-8",
            )
            old_status_file = rl_headless_worker.research_universe_shadow_collector.STATUS_FILE
            try:
                rl_headless_worker.research_universe_shadow_collector.STATUS_FILE = status_file
                snapshot = build_status_snapshot(state, critic_report={}, ml_rows_total=0)
            finally:
                rl_headless_worker.research_universe_shadow_collector.STATUS_FILE = old_status_file

        section = snapshot["research_universe_shadow"]
        self.assertTrue(section["running"])
        self.assertEqual(section["last_started_at"], "2026-06-04T11:18:03Z")
        self.assertEqual(section["last_symbols_scanned"], 80)
        self.assertEqual(section["last_pairs_scanned"], 12)
        self.assertEqual(section["last_rows_written"], 72)
        self.assertEqual(section["last_labels_updated"], 3)
        self.assertEqual(section["cycle_status"]["symbols_total"], 80)


if __name__ == "__main__":
    unittest.main()
