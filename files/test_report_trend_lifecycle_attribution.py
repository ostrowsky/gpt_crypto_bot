from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import report_trend_lifecycle_attribution as report


DAY = "2026-08-02"


def _missed(symbol: str) -> dict:
    return {
        "sym": symbol,
        "tf": "15m",
        "start_ts": "2026-08-02T10:00:00Z",
        "peak_ts": "2026-08-02T12:00:00Z",
        "end_ts": "2026-08-02T12:30:00Z",
        "start_price": 1.0,
        "peak_price": 1.2,
        "end_price": 1.1,
        "move_pct": 20.0,
        "duration_bars": 10,
        "top_mover_rank": None,
    }


def _event(**values: object) -> str:
    return json.dumps(values) + "\n"


class TrendLifecycleAttributionTests(unittest.TestCase):
    def test_decomposes_missed_stages_and_ranks_post_decision_opportunity(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            files = root / "files"
            reports = root / "reports"
            files.mkdir()
            reports.mkdir()
            db = root / "runtime" / "cohorts.sqlite3"
            missed = [_missed(symbol) for symbol in ("AAAUSDT", "BBBUSDT", "CCCUSDT", "DDDUSDT")]
            late = {
                "sym": "EEEUSDT", "tf": "15m", "entry_ts": "2026-08-02T11:00:00Z",
                "entry_price": 1.1, "exit_ts": "2026-08-02T12:10:00Z", "exit_price": 1.15,
                "trend": _missed("EEEUSDT"),
            }
            early = {
                "sym": "FFFUSDT", "tf": "15m", "entry_ts": "2026-08-02T10:10:00Z",
                "entry_price": 1.0, "exit_ts": "2026-08-02T11:00:00Z", "exit_price": 1.1,
                "future_favorable_pct": 9.0, "trend": _missed("FFFUSDT"),
            }
            signal = {
                "target_day_local": DAY,
                "summary": {"missed_trends": 4, "late_entries": 1, "early_exits": 1, "false_positive_buys": 0},
                "detail_coverage": {
                    "missed_trends": {"total": 4, "exported": 4, "complete": True},
                    "late_entries": {"total": 1, "exported": 1, "complete": True},
                    "early_exits": {"total": 1, "exported": 1, "complete": True},
                    "false_positive_buys": {"total": 0, "exported": 0, "complete": True},
                },
                "missed_trends": missed,
                "late_entries": [late],
                "early_exits": [early],
                "false_positive_buys": [],
            }
            (reports / f"signal_quality_{DAY}_final.json").write_text(json.dumps(signal), encoding="utf-8")
            critic = {
                "target_day_local": DAY,
                "summary": {"watchlist_top_count": 2, "watchlist_top_bought": 1, "watchlist_top_early_captured": 1},
            }
            (reports / f"top_gainer_critic_{DAY}_final.json").write_text(json.dumps(critic), encoding="utf-8")
            research_row = {
                "sym": "BBBUSDT", "bar_ts": report._ts_ms("2026-08-02T10:15:00Z"),
                "ts_utc": "2026-08-02T10:15:00Z", "in_trade_watchlist": True, "rule_signal": "none",
            }
            (files / "research_universe_shadow.jsonl").write_text(_event(**research_row), encoding="utf-8")
            (files / "v2_shadow_events.jsonl").write_text("", encoding="utf-8")
            (files / "bot_events.jsonl").write_text(
                _event(event="blocked", sym="CCCUSDT", signal_type="accuracy_gate", reason="accuracy < floor", price=1.05, ts="2026-08-02T10:30:00Z")
                + _event(event="blocked", sym="DDDUSDT", signal_type="portfolio_full", reason="portfolio full", price=1.02, ts="2026-08-02T10:30:00Z"),
                encoding="utf-8",
            )
            (files / "agent_events.jsonl").write_text("", encoding="utf-8")

            payload = report.build(files_dir=files, reports_dir=reports, cohort_db=db, save=False)

        counts = payload["attribution"]["missed_stage_counts"]
        self.assertEqual(counts["not_observed"], 1)
        self.assertEqual(counts["observed_but_not_signaled"], 1)
        self.assertEqual(counts["signaled_but_rejected"], 1)
        self.assertEqual(counts["blocked_by_portfolio_capacity"], 1)
        self.assertEqual(payload["attribution"]["entered_late"], 1)
        self.assertEqual(payload["attribution"]["exited_early"], 1)
        self.assertTrue(payload["latest_day_complete"])
        self.assertGreater(payload["casebook"][0]["realizable_opportunity_net_pct"], 0)
        self.assertNotIn("AAAUSDT", [row["symbol"] for row in payload["casebook"]])

    def test_old_truncated_report_is_marked_partial(self) -> None:
        coverage = report._detail_coverage([{
            "day": DAY,
            "data": {"summary": {"missed_trends": 140}, "missed_trends": [{}] * 100},
        }])

        self.assertFalse(coverage["all_reports_complete"])
        self.assertEqual(coverage["partial_reports"], 1)

    def test_pre_observer_episode_is_not_claimed_as_not_observed(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            files = Path(td)
            (files / "research_universe_shadow.jsonl").write_text(
                _event(
                    sym="OTHERUSDT",
                    bar_ts=report._ts_ms("2026-08-02T10:15:00Z"),
                    ts_utc="2026-08-02T10:15:00Z",
                    rule_signal="none",
                ),
                encoding="utf-8",
            )
            index = report.ObservationIndex(files)
            row = _missed("AAAUSDT")
            row["start_ts"] = "2026-07-01T10:00:00Z"
            row["peak_ts"] = "2026-07-01T12:00:00Z"

            attributed = report._attribute_missed(
                row,
                report.BlockedIntervalIndex([]),
                index,
                [],
                [],
            )

        self.assertEqual(attributed["stage"], "observation_coverage_unavailable")

    def test_lookback_keeps_only_latest_signal_days(self) -> None:
        signal_reports = [
            {"day": f"2026-08-0{day}", "path": "unused", "data": {}, "mtime": day}
            for day in range(1, 4)
        ]
        critic_reports = [
            {"day": f"2026-08-0{day}", "path": "unused", "data": {}, "mtime": day}
            for day in range(1, 5)
        ]

        selected_signal, selected_critic = report._apply_lookback(signal_reports, critic_reports, 2)

        self.assertEqual([row["day"] for row in selected_signal], ["2026-08-02", "2026-08-03"])
        self.assertEqual([row["day"] for row in selected_critic], ["2026-08-02", "2026-08-03"])

    def test_casebook_deduplicates_day_symbol_stage_by_remaining_opportunity(self) -> None:
        rows = [
            {"day": DAY, "symbol": "AAAUSDT", "stage": "signaled_but_rejected", "realizable_opportunity_net_pct": 2.0},
            {"day": DAY, "symbol": "AAAUSDT", "stage": "signaled_but_rejected", "realizable_opportunity_net_pct": 5.0},
        ]

        ranked = report._rank_casebook(rows)

        self.assertEqual(len(ranked), 1)
        self.assertEqual(ranked[0]["realizable_opportunity_net_pct"], 5.0)


if __name__ == "__main__":
    unittest.main()
