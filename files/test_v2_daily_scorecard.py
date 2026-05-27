from __future__ import annotations

import json
import tempfile
import unittest
from datetime import date
from pathlib import Path


class TestV2DailyScorecard(unittest.TestCase):
    def test_joins_v2_upside_to_top_mover_outcomes(self) -> None:
        import report_v2_daily_scorecard as mod

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            events = root / "events.jsonl"
            reports = root / "reports"
            reports.mkdir()
            rows = [
                {"ts": "2026-05-26T08:00:00Z", "sym": "AAAUSDT", "state": "emerging_move", "action": "elevate_priority", "bootstrap": False},
                {"ts": "2026-05-26T08:15:00Z", "sym": "AAAUSDT", "state": "confirmed_trend", "action": "watch", "bootstrap": False},
                {"ts": "2026-05-26T09:00:00Z", "sym": "CCCUSDT", "state": "emerging_move", "action": "elevate_priority", "bootstrap": False},
                {"ts": "2026-05-26T09:15:00Z", "sym": "DDDUSDT", "state": "noise", "action": "watch", "bootstrap": False},
            ]
            events.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
            top = {
                "target_day_local": "2026-05-26",
                "watchlist_top_gainers": [
                    {"symbol": "AAAUSDT", "status": "bought"},
                    {"symbol": "BBBUSDT", "status": "missed"},
                ],
            }
            (reports / "top_gainer_critic_2026-05-26_final.json").write_text(json.dumps(top), encoding="utf-8")

            payload = mod.build_scorecard(
                date(2026, 5, 26),
                events_file=events,
                reports_dir=reports,
                output_json=root / "out.json",
                output_txt=root / "out.txt",
            )

        latest = payload["latest"]
        self.assertEqual(payload["status"], "complete")
        self.assertEqual(latest["top_count"], 2)
        self.assertEqual(latest["top_with_v2_upside"], 1)
        self.assertEqual(latest["top_with_v2_upside_bought"], 1)
        self.assertEqual(latest["v2_false_favorable_symbols"], 1)
        self.assertEqual(latest["v2_top_recall_pct"], 50.0)
        self.assertEqual(latest["v2_top_precision_pct"], 50.0)
        self.assertEqual(latest["v2_handoff_bought_pct"], 100.0)
        self.assertIn("BBBUSDT", latest["top_symbols_missed_by_v2"])

    def test_progress_has_day_and_week_deltas(self) -> None:
        import report_v2_daily_scorecard as mod

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            events = root / "events.jsonl"
            reports = root / "reports"
            reports.mkdir()
            lines = []
            for day in range(10, 25):
                date_s = f"2026-05-{day:02d}"
                # one top seen every day; target day has two seen, so DoD improves.
                syms = ["AAAUSDT"] if day < 24 else ["AAAUSDT", "BBBUSDT"]
                for sym in syms:
                    lines.append(json.dumps({"ts": f"{date_s}T08:00:00Z", "sym": sym, "state": "emerging_move", "bootstrap": False}))
                top = {
                    "target_day_local": date_s,
                    "watchlist_top_gainers": [
                        {"symbol": "AAAUSDT", "status": "bought"},
                        {"symbol": "BBBUSDT", "status": "missed"},
                    ],
                }
                (reports / f"top_gainer_critic_{date_s}_final.json").write_text(json.dumps(top), encoding="utf-8")
            events.write_text("\n".join(lines), encoding="utf-8")

            payload = mod.build_scorecard(
                date(2026, 5, 24),
                events_file=events,
                reports_dir=reports,
                output_json=root / "out.json",
                output_txt=root / "out.txt",
            )

        dod = payload["progress"]["day_over_day"]["v2_top_recall_pct"]
        wow = payload["progress"]["week_over_week"]
        self.assertEqual(dod["previous"], 50.0)
        self.assertEqual(dod["current"], 100.0)
        self.assertIsNotNone(wow["v2_top_recall_pct"]["current"])
        self.assertIn("День ко дню", mod.render_text(payload))

    def test_missing_outcome_report_is_partial(self) -> None:
        import report_v2_daily_scorecard as mod

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            events = root / "events.jsonl"
            events.write_text(json.dumps({"ts": "2026-05-26T08:00:00Z", "sym": "AAAUSDT", "state": "emerging_move"}), encoding="utf-8")
            payload = mod.build_scorecard(
                date(2026, 5, 26),
                events_file=events,
                reports_dir=root / "reports",
                output_json=root / "out.json",
                output_txt=root / "out.txt",
            )
        self.assertEqual(payload["status"], "partial")
        self.assertIn("missing top-gainer outcome report", payload["coverage_reasons"])


if __name__ == "__main__":
    unittest.main()
