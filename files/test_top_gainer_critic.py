from __future__ import annotations

import json
import tempfile
import unittest
from datetime import date, datetime
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfo

import critic_dataset
import top_gainer_critic
from rl_headless_worker import _scheduled_top_gainer_slot, _scheduled_watchlist_goal_slot


class TestTopGainerCritic(unittest.TestCase):
    def test_iter_jsonl_streams_instead_of_reading_the_whole_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            event_file = Path(tmpdir) / "events.jsonl"
            event_file.write_text(
                '{"event":"entry","sym":"BTCUSDT"}\nnot-json\n[]\n'
                '{"event":"entry","sym":"ETHUSDT"}\n'
                '{"event":"exit","sym":"BTCUSDT"}\n',
                encoding="utf-8",
            )

            with patch.object(Path, "read_text", side_effect=AssertionError("whole-file read")):
                rows = list(
                    top_gainer_critic._iter_jsonl(
                        event_file,
                        raw_required_tokens=("BTCUSDT",),
                    )
                )

        self.assertEqual([row["event"] for row in rows], ["entry", "exit"])

    def test_watchlist_top_denominator_filters_exchange_top_before_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            bot_events = tmp / "bot_events.jsonl"
            agent_events = tmp / "agent_events.jsonl"
            bot_events.write_text(
                "\n".join(
                    json.dumps(x, ensure_ascii=False)
                    for x in [
                        {"event": "entry", "sym": "W1USDT", "mode": "trend", "price": 1.0, "ts": "2026-03-31T00:30:00Z"},
                        {"event": "entry", "sym": "W3USDT", "mode": "trend", "price": 1.0, "ts": "2026-03-31T01:30:00Z"},
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            agent_events.write_text("", encoding="utf-8")

            old_bot = top_gainer_critic.BOT_EVENTS_FILE
            old_agent = top_gainer_critic.AGENT_EVENTS_FILE
            try:
                top_gainer_critic.BOT_EVENTS_FILE = bot_events
                top_gainer_critic.AGENT_EVENTS_FILE = agent_events
                perf = []
                # Exchange top-10 contains 4 watchlist symbols. The bot bought 2.
                for idx in range(10):
                    symbol = {1: "W1USDT", 3: "W2USDT", 6: "W3USDT", 8: "W4USDT"}.get(idx, f"X{idx}USDT")
                    perf.append(
                        top_gainer_critic.DayPerformance(
                            symbol=symbol,
                            day_open=1.0,
                            day_close=1.2 - idx * 0.01,
                            day_high=1.25 - idx * 0.01,
                            day_low=1.0,
                            day_change_pct=20.0 - idx,
                            quote_volume_24h=5_000_000.0,
                            in_watchlist=symbol.startswith("W"),
                        )
                    )
                # This symbol is in the watchlist and would have entered the old
                # "top N within watchlist" denominator, but it is not in exchange top-10.
                perf.append(
                    top_gainer_critic.DayPerformance(
                        symbol="W5USDT",
                        day_open=1.0,
                        day_close=0.95,
                        day_high=1.02,
                        day_low=0.94,
                        day_change_pct=-5.0,
                        quote_volume_24h=5_000_000.0,
                        in_watchlist=True,
                    )
                )
                report = top_gainer_critic.build_report(
                    target_day=date(2026, 3, 31),
                    phase="final",
                    timezone_name="Europe/Budapest",
                    top_n=10,
                    min_quote_volume=0.0,
                    day_performance=perf,
                )
            finally:
                top_gainer_critic.BOT_EVENTS_FILE = old_bot
                top_gainer_critic.AGENT_EVENTS_FILE = old_agent

        summary = report["summary"]
        self.assertEqual(summary["exchange_top_count"], 10)
        self.assertEqual(summary["exchange_top_in_watchlist"], 4)
        self.assertEqual(summary["watchlist_top_denominator"], "exchange_top_filtered_to_watchlist")
        self.assertEqual(summary["watchlist_top_count"], 4)
        self.assertEqual(summary["watchlist_top_bought"], 2)
        self.assertEqual(summary["watchlist_top_missed"], 2)
        self.assertEqual(summary["watchlist_top_capture_rate_pct"], 50.0)
        self.assertEqual(summary["watchlist_universe_top_count"], 5)
        self.assertEqual([row["symbol"] for row in report["watchlist_top_gainers"]], ["W1USDT", "W2USDT", "W3USDT", "W4USDT"])
        self.assertIn("W5USDT", [row["symbol"] for row in report["watchlist_universe_top_gainers"]])

    def test_build_report_summarizes_top_gainers_and_false_positives(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            bot_events = tmp / "bot_events.jsonl"
            agent_events = tmp / "agent_events.jsonl"

            events = [
                {
                    "event": "entry",
                    "sym": "AAAUSDT",
                    "mode": "trend",
                    "price": 1.2,
                    "ts": "2026-03-31T00:30:00Z",
                },
                {
                    "event": "blocked",
                    "sym": "BBBUSDT",
                    "reason": "блок: портфель полон: 10/10 позиций",
                    "ts": "2026-03-31T01:30:00Z",
                },
                {
                    "event": "entry",
                    "sym": "XXXUSDT",
                    "mode": "impulse_speed",
                    "price": 2.5,
                    "ts": "2026-03-31T02:30:00Z",
                },
            ]
            bot_events.write_text(
                "\n".join(json.dumps(x, ensure_ascii=False) for x in events) + "\n",
                encoding="utf-8",
            )
            agent_events.write_text("", encoding="utf-8")

            old_bot = top_gainer_critic.BOT_EVENTS_FILE
            old_agent = top_gainer_critic.AGENT_EVENTS_FILE
            try:
                top_gainer_critic.BOT_EVENTS_FILE = bot_events
                top_gainer_critic.AGENT_EVENTS_FILE = agent_events
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
                        day_close=1.3,
                        day_high=1.3,
                        day_low=1.0,
                        day_change_pct=30.0,
                        quote_volume_24h=5_000_000.0,
                        in_watchlist=True,
                    ),
                ]
                report = top_gainer_critic.build_report(
                    target_day=date(2026, 3, 31),
                    phase="final",
                    timezone_name="Europe/Budapest",
                    top_n=3,
                    min_quote_volume=0.0,
                    day_performance=perf,
                )
            finally:
                top_gainer_critic.BOT_EVENTS_FILE = old_bot
                top_gainer_critic.AGENT_EVENTS_FILE = old_agent

            self.assertEqual(report["summary"]["exchange_top_in_watchlist"], 2)
            self.assertEqual(report["summary"]["watchlist_top_count"], 2)
            self.assertEqual(report["summary"]["watchlist_top_bought"], 1)
            self.assertEqual(report["summary"]["watchlist_top_missed"], 1)
            self.assertEqual(report["summary"]["blocked_winner_count"], 1)
            self.assertEqual(report["summary"]["missed_reason_counts"]["portfolio_full"], 1)
            self.assertIn("XXXUSDT", report["bot_false_positive_symbols"])

            rows = {item["symbol"]: item for item in report["watchlist_top_gainers"]}
            self.assertEqual(rows["AAAUSDT"]["status"], "bought")
            self.assertEqual(rows["BBBUSDT"]["status"], "blocked_portfolio")
            self.assertEqual(rows["BBBUSDT"]["missed_reason_code"], "portfolio_full")
            self.assertGreater(rows["AAAUSDT"]["capture_ratio"], 0.0)

    def test_build_report_reads_agent_events_and_filter_harm(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            bot_events = tmp / "bot_events.jsonl"
            agent_events = tmp / "agent_events.jsonl"

            bot_events.write_text("", encoding="utf-8")
            agent_events.write_text(
                "\n".join(
                    json.dumps(x, ensure_ascii=False)
                    for x in [
                        {
                            "event": "entry",
                            "sym": "AAAUSDT",
                            "mode": "4h_leader_watch",
                            "price": 1.1,
                            "ts": "2026-03-31T00:20:00Z",
                            "source": "market_agent",
                        },
                        {
                            "event": "blocked",
                            "sym": "BBBUSDT",
                            "signal_type": "top_gainer_score_gate",
                            "reason": "top_gainer_score_gate score 0.52 < 0.60",
                            "price": 1.2,
                            "ts": "2026-03-31T01:20:00Z",
                            "source": "market_agent",
                        },
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            old_bot = top_gainer_critic.BOT_EVENTS_FILE
            old_agent = top_gainer_critic.AGENT_EVENTS_FILE
            try:
                top_gainer_critic.BOT_EVENTS_FILE = bot_events
                top_gainer_critic.AGENT_EVENTS_FILE = agent_events
                report = top_gainer_critic.build_report(
                    target_day=date(2026, 3, 31),
                    phase="final",
                    timezone_name="Europe/Budapest",
                    top_n=2,
                    min_quote_volume=0.0,
                    day_performance=[
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
                            day_close=1.3,
                            day_high=1.3,
                            day_low=1.0,
                            day_change_pct=30.0,
                            quote_volume_24h=5_000_000.0,
                            in_watchlist=True,
                        ),
                    ],
                )
            finally:
                top_gainer_critic.BOT_EVENTS_FILE = old_bot
                top_gainer_critic.AGENT_EVENTS_FILE = old_agent

            self.assertEqual(report["summary"]["watchlist_top_bought"], 1)
            self.assertEqual(report["summary"]["entry_source_counts"]["market_agent"], 1)
            self.assertEqual(report["summary"]["blocked_winner_count"], 1)
            self.assertEqual(report["summary"]["missed_reason_counts"]["top_gainer_score_gate"], 1)

            rows = {item["symbol"]: item for item in report["watchlist_top_gainers"]}
            self.assertEqual(rows["AAAUSDT"]["first_entry_source"], "market_agent")
            self.assertEqual(rows["BBBUSDT"]["missed_reason_code"], "top_gainer_score_gate")
            self.assertEqual(report["blocked_reason_harm"][0]["reason_code"], "top_gainer_score_gate")
            self.assertEqual(report["blocked_reason_harm"][0]["missed_symbols"], ["BBBUSDT"])

    def test_scheduled_slots_map_to_midday_and_previous_day_final(self) -> None:
        tz = ZoneInfo("Europe/Budapest")
        slot_goal = _scheduled_watchlist_goal_slot(datetime(2026, 4, 1, 22, 4, tzinfo=tz))
        self.assertEqual(slot_goal[0].isoformat(), "2026-04-01")

        with tempfile.TemporaryDirectory() as tmpdir:
            reports_dir = Path(tmpdir)
            for day in range(25, 32):
                (reports_dir / f"top_gainer_critic_2026-03-{day:02d}_final.json").write_text("{}")
            slot_midday = _scheduled_top_gainer_slot(
                datetime(2026, 4, 1, 12, 4, tzinfo=tz), reports_dir
            )
            slot_none = _scheduled_top_gainer_slot(
                datetime(2026, 4, 1, 13, 0, tzinfo=tz), reports_dir
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            slot_final = _scheduled_top_gainer_slot(
                datetime(2026, 4, 1, 0, 5, tzinfo=tz), Path(tmpdir)
            )

        self.assertEqual(slot_midday[0], "midday")
        self.assertEqual(slot_midday[1].isoformat(), "2026-04-01")
        self.assertEqual(slot_final[0], "final")
        self.assertEqual(slot_final[1].isoformat(), "2026-03-31")
        self.assertIsNone(slot_none)

    def test_report_teacher_labels_are_written_back_to_critic_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            bot_events = tmp / "bot_events.jsonl"
            agent_events = tmp / "agent_events.jsonl"
            critic_file = tmp / "critic_dataset.jsonl"

            events = [
                {
                    "event": "entry",
                    "sym": "AAAUSDT",
                    "mode": "trend",
                    "price": 1.2,
                    "ts": "2026-03-31T00:30:00Z",
                },
                {
                    "event": "blocked",
                    "sym": "BBBUSDT",
                    "reason": "блок: портфель полон: 10/10 позиций",
                    "ts": "2026-03-31T01:30:00Z",
                },
                {
                    "event": "entry",
                    "sym": "XXXUSDT",
                    "mode": "impulse_speed",
                    "price": 2.5,
                    "ts": "2026-03-31T02:30:00Z",
                },
            ]
            bot_events.write_text(
                "\n".join(json.dumps(x, ensure_ascii=False) for x in events) + "\n",
                encoding="utf-8",
            )
            agent_events.write_text("", encoding="utf-8")

            critic_rows = [
                {
                    "id": "AAAUSDT_15m_20260331T001500Z",
                    "sym": "AAAUSDT",
                    "tf": "15m",
                    "ts_signal": "2026-03-31T00:15:00Z",
                    "bar_ts": 1774916100000,
                    "labels": {},
                },
                {
                    "id": "BBBUSDT_15m_20260331T011500Z",
                    "sym": "BBBUSDT",
                    "tf": "15m",
                    "ts_signal": "2026-03-31T01:15:00Z",
                    "bar_ts": 1774919700000,
                    "labels": {},
                },
                {
                    "id": "XXXUSDT_15m_20260331T021500Z",
                    "sym": "XXXUSDT",
                    "tf": "15m",
                    "ts_signal": "2026-03-31T02:15:00Z",
                    "bar_ts": 1774923300000,
                    "labels": {},
                },
                {
                    "id": "OLDUSDT_15m_20260330T221500Z",
                    "sym": "OLDUSDT",
                    "tf": "15m",
                    "ts_signal": "2026-03-30T20:15:00Z",
                    "bar_ts": 1774901700000,
                    "labels": {},
                },
            ]
            critic_file.write_text(
                "\n".join(json.dumps(x, ensure_ascii=False) for x in critic_rows) + "\n",
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
                    day_close=1.3,
                    day_high=1.3,
                    day_low=1.0,
                    day_change_pct=30.0,
                    quote_volume_24h=5_000_000.0,
                    in_watchlist=True,
                ),
            ]

            old_bot = top_gainer_critic.BOT_EVENTS_FILE
            old_agent = top_gainer_critic.AGENT_EVENTS_FILE
            old_critic = critic_dataset.CRITIC_FILE
            try:
                top_gainer_critic.BOT_EVENTS_FILE = bot_events
                top_gainer_critic.AGENT_EVENTS_FILE = agent_events
                critic_dataset.CRITIC_FILE = critic_file
                report = top_gainer_critic.build_report(
                    target_day=date(2026, 3, 31),
                    phase="final",
                    timezone_name="Europe/Budapest",
                    top_n=3,
                    min_quote_volume=0.0,
                    day_performance=perf,
                )
                with patch.object(Path, "read_text", side_effect=AssertionError("whole-file read")):
                    summary = critic_dataset.annotate_top_gainer_teacher(report)
            finally:
                top_gainer_critic.BOT_EVENTS_FILE = old_bot
                top_gainer_critic.AGENT_EVENTS_FILE = old_agent
                critic_dataset.CRITIC_FILE = old_critic

            self.assertEqual(summary["rows_annotated"], 3)
            rows = [
                json.loads(line)
                for line in critic_file.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            rows_by_sym = {row["sym"]: row for row in rows}

            aaa = rows_by_sym["AAAUSDT"]["teacher"]["final"]
            self.assertTrue(aaa["watchlist_top_gainer"])
            self.assertEqual(aaa["status"], "bought")
            self.assertFalse(aaa["bot_false_positive_buy"])
            self.assertTrue(aaa["early_capture"])

            bbb = rows_by_sym["BBBUSDT"]["teacher"]["final"]
            self.assertTrue(bbb["watchlist_top_gainer"])
            self.assertEqual(bbb["status"], "blocked_portfolio")

            xxx = rows_by_sym["XXXUSDT"]["teacher"]["final"]
            self.assertFalse(xxx["watchlist_top_gainer"])
            self.assertTrue(xxx["bot_false_positive_buy"])

            self.assertNotIn("teacher", rows_by_sym["OLDUSDT"])


if __name__ == "__main__":
    unittest.main()
