from __future__ import annotations

import json
import unittest
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfo

import config
import report_critic_dataset
import rl_headless_worker
from rl_headless_worker import (
    WorkerState,
    _claim_learning_progress_telegram_slot,
    _build_training_readiness_session_report,
    _release_learning_progress_telegram_slot,
    build_status_snapshot,
    _render_top_gainer_telegram,
    _run_trend_lifecycle_attribution_report,
    _restore_daily_report_state,
    _restore_latest_top_gainer_artifact,
    _scheduled_top_gainer_slot,
    _scheduled_watchlist_goal_slot,
    _should_send_top_gainer_telegram,
    should_train,
)


class TestDailyCriticSchedulerRecovery(unittest.TestCase):
    def test_supervisor_rejects_accidental_collector_disable(self) -> None:
        launcher = (Path(__file__).resolve().parents[1] / "headless_loop.ps1").read_text(
            encoding="utf-8-sig"
        )
        self.assertIn('"--disable-collector"', launcher)
        self.assertIn("GPT_BOT_ALLOW_UNLABELED_COLLECTION", launcher)
        self.assertIn("collector_disable_rejected", launcher)

    def test_online_ranker_artifacts_are_isolated_from_production_model(self) -> None:
        self.assertIn(".runtime", str(rl_headless_worker.MODEL_FILE))
        self.assertIn("online_shadow", rl_headless_worker.MODEL_FILE.name)
        self.assertEqual(rl_headless_worker.DEFAULT_MIN_ROWS, 120)
        self.assertEqual(rl_headless_worker.DEFAULT_MIN_NEW_ROWS, 20)

    def test_training_readiness_session_is_non_achievement_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            dataset = Path(td) / "critic.jsonl"
            dataset.write_text(
                json.dumps(
                    {
                        "ts_signal": "2026-08-03T10:00:00Z",
                        "signal_type": "trend",
                        "labels": {"ret_5": 1.0},
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            state = WorkerState(
                train_interval_sec=3600,
                status_interval_sec=60,
                min_rows=500,
                min_new_rows=10,
                collector_enabled=False,
            )
            report = _build_training_readiness_session_report(
                state=state,
                dataset_path=dataset,
                critic_rows_total=1,
                ml_rows_total=0,
            )

        self.assertEqual(report["evidence_status"], "blocked_insufficient_provenance")
        self.assertFalse(report["achievement_claimed"])
        self.assertFalse(report["runtime_eligible"])
        self.assertEqual(report["critic_rows_total"], 1)
        self.assertEqual(report["data_provenance"]["verified_rows"], 0)

    def test_trend_attribution_runner_enforces_bounded_positive_lookback(self) -> None:
        with patch.object(
            rl_headless_worker.report_trend_lifecycle_attribution,
            "build",
            return_value={"status": "complete"},
        ) as build:
            result = _run_trend_lifecycle_attribution_report(0)

        self.assertEqual(result["status"], "complete")
        build.assert_called_once_with(lookback_days=1)

    def test_status_dataset_scans_do_not_read_whole_jsonl_files(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            dataset = Path(td) / "critic.jsonl"
            dataset.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "ts_signal": "2026-08-03T10:00:00Z",
                                "signal_type": "trend",
                                "decision": {"action": "take"},
                                "labels": {"ret_3": 1.0, "ret_5": 1.0, "trade_taken": True},
                            }
                        ),
                        "not-json",
                        json.dumps(
                            {
                                "ts_signal": "2026-08-01T10:00:00Z",
                                "signal_type": "none",
                                "decision": {"action": "blocked"},
                                "labels": {},
                            }
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            old_critic_file = report_critic_dataset.critic_dataset.CRITIC_FILE
            try:
                report_critic_dataset.critic_dataset.CRITIC_FILE = dataset
                with patch.object(Path, "read_text", side_effect=AssertionError("whole-file read")):
                    report = report_critic_dataset.build_report(datetime(2026, 8, 3, 12, 0))
                    rows_total = rl_headless_worker._count_jsonl_rows(dataset)
                    ranker_rows = rl_headless_worker._count_ranker_rows(dataset)
            finally:
                report_critic_dataset.critic_dataset.CRITIC_FILE = old_critic_file

        self.assertEqual(report["rows_total"], 2)
        self.assertEqual(report["rows_last_24h"], 1)
        self.assertEqual(report["actions"], {"take": 1, "blocked": 1})
        self.assertEqual(rows_total, 3)
        # Status uses the same fail-closed provenance population as training.
        # The legacy labeled row is visible to the dataset report but is not a
        # provenance-verified ranker row.
        self.assertEqual(ranker_rows, 0)

    def test_missing_previous_final_is_due_after_nominal_window(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            slot = _scheduled_top_gainer_slot(
                datetime(2026, 8, 3, 10, 30, tzinfo=ZoneInfo("Europe/Budapest")),
                Path(td),
            )

        self.assertEqual(slot, ("final", datetime(2026, 8, 2).date(), "2026-08-02::final"))

    def test_existing_final_allows_independent_midday_slot(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            reports = Path(td)
            previous = datetime(2026, 8, 2).date()
            for offset in range(7):
                day = previous - timedelta(days=offset)
                (reports / f"top_gainer_critic_{day.isoformat()}_final.json").write_text("{}", encoding="utf-8")
            slot = _scheduled_top_gainer_slot(
                datetime(2026, 8, 3, 12, 5, tzinfo=ZoneInfo("Europe/Budapest")),
                reports,
            )

        self.assertEqual(slot, ("midday", datetime(2026, 8, 3).date(), "2026-08-03::midday"))

    def test_goal_slot_is_scheduled_separately(self) -> None:
        slot = _scheduled_watchlist_goal_slot(
            datetime(2026, 8, 3, 22, 5, tzinfo=ZoneInfo("Europe/Budapest"))
        )
        self.assertEqual(slot, (datetime(2026, 8, 3).date(), "2026-08-03::goal_22h"))

    def test_report_state_survives_restart_snapshot(self) -> None:
        state = WorkerState(60, 60, 1, 1, False)
        with tempfile.TemporaryDirectory() as td:
            status_file = Path(td) / "status.json"
            status_file.write_text(
                json.dumps(
                    {
                        "top_gainer_critic": {
                            "runs_total": 7,
                            "runs_ok": 6,
                            "last_slot_key": "2026-08-02::final",
                            "last_target_day_local": "2026-08-02",
                        },
                        "watchlist_top_gainer_goal": {
                            "runs_total": 4,
                            "last_slot_key": "2026-08-02::goal_22h",
                        },
                    }
                ),
                encoding="utf-8",
            )
            original = rl_headless_worker.STATUS_FILE
            try:
                rl_headless_worker.STATUS_FILE = status_file
                self.assertTrue(_restore_daily_report_state(state, Path(td) / "reports"))
            finally:
                rl_headless_worker.STATUS_FILE = original

        self.assertEqual(state.top_gainer_runs_total, 7)
        self.assertEqual(state.top_gainer_last_slot_key, "2026-08-02::final")
        self.assertEqual(state.watchlist_goal_last_slot_key, "2026-08-02::goal_22h")

    def test_latest_final_artifact_overrides_stale_midday_status(self) -> None:
        state = WorkerState(60, 60, 1, 1, False)
        state.top_gainer_last_target_day_local = "2026-08-02"
        state.top_gainer_last_phase = "midday"
        state.top_gainer_last_slot_key = "2026-08-02::midday"
        with tempfile.TemporaryDirectory() as td:
            reports = Path(td)
            report_path = reports / "top_gainer_critic_2026-08-02_final.json"
            report_path.write_text(
                json.dumps(
                    {
                        "target_day_local": "2026-08-02",
                        "phase": "final",
                        "summary": {
                            "watchlist_top_capture_rate_pct": 75.0,
                            "watchlist_top_early_capture_rate_pct": 50.0,
                        },
                    }
                ),
                encoding="utf-8",
            )

            self.assertTrue(_restore_latest_top_gainer_artifact(state, reports))

        self.assertEqual(state.top_gainer_last_slot_key, "2026-08-02::final")
        self.assertEqual(state.top_gainer_last_phase, "final")
        self.assertEqual(state.top_gainer_last_capture_rate_pct, 75.0)
        self.assertEqual(state.top_gainer_last_early_capture_rate_pct, 50.0)


class TestLearningProgressTelegramSlot(unittest.TestCase):
    def test_failed_delivery_can_release_claim_for_retry(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            marker_dir = Path(td)
            self.assertTrue(_claim_learning_progress_telegram_slot("2026-06-30::learning_progress", marker_dir))
            self.assertFalse(_claim_learning_progress_telegram_slot("2026-06-30::learning_progress", marker_dir))

            _release_learning_progress_telegram_slot("2026-06-30::learning_progress", marker_dir)

            self.assertTrue(_claim_learning_progress_telegram_slot("2026-06-30::learning_progress", marker_dir))


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


class TestStaticTargetTop50ShadowStatus(unittest.TestCase):
    def test_status_snapshot_exposes_shadow_without_claiming_production_effect(self) -> None:
        state = WorkerState(60, 60, 1, 1, False)
        state.static_target_shadow_runs_total = 3
        state.static_target_shadow_runs_ok = 2
        state.static_target_shadow_last_action = "finalize"
        state.static_target_shadow_last_target_day_local = "2026-08-22"
        state.static_target_shadow_last_status = "complete"

        snapshot = build_status_snapshot(state, critic_report={}, ml_rows_total=0)
        section = snapshot["static_target_top50_shadow"]

        self.assertTrue(section["enabled"])
        self.assertEqual(section["runs_total"], 3)
        self.assertEqual(section["last_action"], "finalize")
        self.assertEqual(section["last_target_day_local"], "2026-08-22")
        self.assertEqual(section["production_effect"], "none_shadow_only")


if __name__ == "__main__":
    unittest.main()
