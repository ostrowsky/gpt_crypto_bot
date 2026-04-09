from __future__ import annotations

import json
import tempfile
import unittest
from datetime import date, datetime, timezone
from pathlib import Path

import ml_dataset
import top_gainer_critic
import watchlist_top_gainer_goal


def _utc_ms(ts_text: str) -> int:
    return int(datetime.fromisoformat(ts_text.replace("Z", "+00:00")).timestamp() * 1000)


class TestWatchlistTopGainerGoal(unittest.TestCase):
    def test_goal_report_annotates_ml_dataset_and_computes_daily_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            bot_events = tmp / "bot_events.jsonl"
            watchlist_file = tmp / "watchlist.json"
            ml_file = tmp / "ml_dataset.jsonl"

            watchlist_file.write_text(
                json.dumps(["AAAUSDT", "BBBUSDT", "DDDUSDT", "XXXUSDT"], ensure_ascii=False),
                encoding="utf-8",
            )

            events = [
                {
                    "event": "entry",
                    "sym": "AAAUSDT",
                    "mode": "trend",
                    "price": 1.2,
                    "ts": "2026-03-31T00:30:00Z",
                },
                {
                    "event": "entry",
                    "sym": "BBBUSDT",
                    "mode": "retest",
                    "price": 1.1,
                    "ts": "2026-03-31T09:15:00Z",
                },
                {
                    "event": "blocked",
                    "sym": "DDDUSDT",
                    "reason": "hard veto",
                    "ts": "2026-03-31T07:30:00Z",
                },
                {
                    "event": "entry",
                    "sym": "XXXUSDT",
                    "mode": "impulse_speed",
                    "price": 0.8,
                    "ts": "2026-03-31T12:30:00Z",
                },
            ]
            bot_events.write_text(
                "\n".join(json.dumps(item, ensure_ascii=False) for item in events) + "\n",
                encoding="utf-8",
            )

            ml_rows = [
                {
                    "id": "AAAUSDT_15m_20260331T001500Z",
                    "sym": "AAAUSDT",
                    "tf": "15m",
                    "ts_signal": "2026-03-31T00:15:00Z",
                    "bar_ts": _utc_ms("2026-03-31T00:15:00Z"),
                    "signal_type": "none",
                    "labels": {},
                },
                {
                    "id": "BBBUSDT_15m_20260331T091500Z",
                    "sym": "BBBUSDT",
                    "tf": "15m",
                    "ts_signal": "2026-03-31T09:15:00Z",
                    "bar_ts": _utc_ms("2026-03-31T09:15:00Z"),
                    "signal_type": "none",
                    "labels": {},
                },
                {
                    "id": "XXXUSDT_15m_20260331T121500Z",
                    "sym": "XXXUSDT",
                    "tf": "15m",
                    "ts_signal": "2026-03-31T12:15:00Z",
                    "bar_ts": _utc_ms("2026-03-31T12:15:00Z"),
                    "signal_type": "none",
                    "labels": {},
                },
                {
                    "id": "OLDUSDT_15m_20260330T221500Z",
                    "sym": "OLDUSDT",
                    "tf": "15m",
                    "ts_signal": "2026-03-30T22:15:00Z",
                    "bar_ts": _utc_ms("2026-03-30T22:15:00Z"),
                    "signal_type": "none",
                    "labels": {},
                },
            ]
            ml_file.write_text(
                "\n".join(json.dumps(item, ensure_ascii=False) for item in ml_rows) + "\n",
                encoding="utf-8",
            )

            perf = [
                top_gainer_critic.DayPerformance(
                    symbol="ZZZUSDT",
                    day_open=1.0,
                    day_close=1.5,
                    day_high=1.5,
                    day_low=1.0,
                    day_change_pct=50.0,
                    quote_volume_24h=5_000_000.0,
                    in_watchlist=False,
                ),
                top_gainer_critic.DayPerformance(
                    symbol="AAAUSDT",
                    day_open=1.0,
                    day_close=1.4,
                    day_high=1.4,
                    day_low=1.0,
                    day_change_pct=40.0,
                    quote_volume_24h=5_000_000.0,
                    in_watchlist=True,
                ),
                top_gainer_critic.DayPerformance(
                    symbol="BBBUSDT",
                    day_open=1.0,
                    day_close=1.35,
                    day_high=1.35,
                    day_low=1.0,
                    day_change_pct=35.0,
                    quote_volume_24h=5_000_000.0,
                    in_watchlist=True,
                ),
                top_gainer_critic.DayPerformance(
                    symbol="DDDUSDT",
                    day_open=1.0,
                    day_close=1.3,
                    day_high=1.3,
                    day_low=1.0,
                    day_change_pct=30.0,
                    quote_volume_24h=5_000_000.0,
                    in_watchlist=True,
                ),
                top_gainer_critic.DayPerformance(
                    symbol="XXXUSDT",
                    day_open=1.0,
                    day_close=1.2,
                    day_high=1.2,
                    day_low=1.0,
                    day_change_pct=20.0,
                    quote_volume_24h=5_000_000.0,
                    in_watchlist=True,
                ),
            ]

            old_goal_bot = watchlist_top_gainer_goal.BOT_EVENTS_FILE
            old_base_bot = top_gainer_critic.BOT_EVENTS_FILE
            old_watchlist = top_gainer_critic.WATCHLIST_FILE
            old_ml = ml_dataset.ML_FILE
            try:
                watchlist_top_gainer_goal.BOT_EVENTS_FILE = bot_events
                top_gainer_critic.BOT_EVENTS_FILE = bot_events
                top_gainer_critic.WATCHLIST_FILE = watchlist_file
                ml_dataset.ML_FILE = ml_file
                report = watchlist_top_gainer_goal.build_goal_report(
                    target_day=date(2026, 3, 31),
                    cutoff_hour=22,
                    timezone_name="Europe/Budapest",
                    top_n=3,
                    min_quote_volume=0.0,
                    day_performance=perf,
                )
            finally:
                watchlist_top_gainer_goal.BOT_EVENTS_FILE = old_goal_bot
                top_gainer_critic.BOT_EVENTS_FILE = old_base_bot
                top_gainer_critic.WATCHLIST_FILE = old_watchlist
                ml_dataset.ML_FILE = old_ml

            self.assertEqual(report["summary"]["watchlist_top_count"], 3)
            self.assertEqual(report["summary"]["watchlist_top_bought"], 2)
            self.assertEqual(report["summary"]["recall_at_cutoff_pct"], 66.67)
            self.assertEqual(report["summary"]["mandatory_positive_coverage_pct"], 66.67)
            self.assertEqual(report["teacher_annotation"]["rows_annotated"], 3)
            self.assertEqual(report["teacher_annotation"]["positive_symbols_missing_rows"], ["DDDUSDT"])

            checkpoints = {
                int(item["hour_local"]): item["recall_pct"]
                for item in report["early_recall_checkpoints"]
            }
            self.assertEqual(checkpoints[4], 33.33)
            self.assertEqual(checkpoints[12], 66.67)
            self.assertEqual(checkpoints[22], 66.67)

            precision = {
                int(item["top_n"]): item
                for item in report["precision_first_n"]
            }
            self.assertEqual(precision[5]["alerts_considered"], 3)
            self.assertEqual(precision[5]["hits"], 2)
            self.assertEqual(precision[5]["precision_pct"], 66.67)

            rows = [
                json.loads(line)
                for line in ml_file.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            rows_by_sym = {row["sym"]: row for row in rows}
            teacher_key = "watchlist_top_gainer_22h"

            self.assertTrue(rows_by_sym["AAAUSDT"]["teacher"][teacher_key]["watchlist_top_gainer"])
            self.assertTrue(rows_by_sym["BBBUSDT"]["teacher"][teacher_key]["watchlist_top_gainer"])
            self.assertFalse(rows_by_sym["XXXUSDT"]["teacher"][teacher_key]["watchlist_top_gainer"])
            self.assertNotIn("teacher", rows_by_sym["OLDUSDT"])


if __name__ == "__main__":
    unittest.main()
