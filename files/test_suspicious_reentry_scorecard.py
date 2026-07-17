from __future__ import annotations

import json
import tempfile
import unittest
from datetime import date, datetime, timezone
from pathlib import Path

import report_suspicious_reentry_scorecard as scorecard


BASE_TS_MS = int(datetime(2026, 5, 30, 10, 0, tzinfo=timezone.utc).timestamp() * 1000)
BAR_MS = 15 * 60 * 1000


def _event(sym: str = "AAAUSDT", *, price: float = 100.0) -> dict:
    return {
        "ts": "2026-05-30T10:00:00Z",
        "event": "suspicious_reentry_shadow",
        "sym": sym,
        "tf": "15m",
        "mode": "trend",
        "price": price,
        "candidate_score": 42.0,
        "exit_score": 0.72,
        "exit_reason": "WEAK: RSI divergence",
        "exit_pnl_pct": 1.0,
        "mfe_pct": 4.0,
        "bars_since_exit": 1,
        "cooldown_bars_left": 3,
    }


def _candles(closes: list[float]) -> list[list]:
    rows = []
    for i, close in enumerate(closes):
        open_time = BASE_TS_MS - BAR_MS + i * BAR_MS
        high = close * 1.01
        low = close * 0.99
        rows.append([open_time, str(close), str(high), str(low), str(close), "1000", open_time + BAR_MS - 1])
    return rows


class SuspiciousReentryScorecardTests(unittest.TestCase):
    def test_positive_shadow_reentry_is_labeled(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "bot_events.jsonl"
            path.write_text(json.dumps(_event()) + "\n", encoding="utf-8")

            def loader(sym: str, tf: str, start_ms: int, end_ms: int) -> list[list]:
                return _candles([99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111])

            payload = scorecard.build_scorecard(
                date(2026, 5, 30),
                events_file=path,
                reports_dir=Path(td),
                output_json=Path(td) / "latest.json",
                output_txt=Path(td) / "latest.txt",
                save=False,
                kline_loader=loader,
                now_utc=datetime(2026, 5, 30, 14, 0, tzinfo=timezone.utc),
            )
            self.assertEqual(payload["summary"]["alerts_total"], 1)
            self.assertEqual(payload["summary"]["labeled_ret5"], 1)
            self.assertGreater(payload["summary"]["avg_ret5"], 0)
            self.assertEqual(payload["summary"]["ret5_positive_rate"], 1.0)

    def test_pending_when_forward_horizon_not_mature(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "bot_events.jsonl"
            path.write_text(json.dumps(_event()) + "\n", encoding="utf-8")
            payload = scorecard.build_scorecard(
                date(2026, 5, 30),
                events_file=path,
                reports_dir=Path(td),
                output_json=Path(td) / "latest.json",
                output_txt=Path(td) / "latest.txt",
                save=False,
                kline_loader=lambda *_: [],
                now_utc=datetime(2026, 5, 30, 10, 30, tzinfo=timezone.utc),
            )
            self.assertEqual(payload["status"], "partial")
            self.assertEqual(payload["summary"]["pending"], 1)
            self.assertIn("pending labels", ";".join(payload["coverage_reasons"]))

    def test_negative_shadow_reentry_is_visible(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "bot_events.jsonl"
            path.write_text(json.dumps(_event("BADUSDT")) + "\n", encoding="utf-8")

            def loader(sym: str, tf: str, start_ms: int, end_ms: int) -> list[list]:
                return _candles([99, 100, 99, 98, 97, 96, 95, 94, 93, 92, 91, 90, 89])

            payload = scorecard.build_scorecard(
                date(2026, 5, 30),
                events_file=path,
                reports_dir=Path(td),
                output_json=Path(td) / "latest.json",
                output_txt=Path(td) / "latest.txt",
                save=False,
                kline_loader=loader,
                now_utc=datetime(2026, 5, 30, 14, 0, tzinfo=timezone.utc),
            )
            self.assertLess(payload["summary"]["avg_ret5"], 0)
            self.assertEqual(payload["summary"]["ret5_positive_rate"], 0.0)
            self.assertIn("шум", payload["interpretation"])

    def test_zero_alert_day_exposes_upstream_watch_funnel(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "bot_events.jsonl"
            rows = [
                {
                    "ts": "2026-05-30T08:00:00Z",
                    "event": "suspicious_reentry_watch_decision",
                    "sym": "AAAUSDT",
                    "tf": "15m",
                    "decision": "rejected_exit_score",
                    "exit_score": 0.5,
                    "mfe_pct": 2.0,
                },
                {
                    "ts": "2026-05-30T09:00:00Z",
                    "event": "suspicious_reentry_watch_decision",
                    "sym": "BBBUSDT",
                    "tf": "1h",
                    "decision": "rejected_mfe",
                    "exit_score": 0.72,
                    "mfe_pct": 0.5,
                },
                {
                    "ts": "2026-05-30T10:00:00Z",
                    "event": "suspicious_reentry_watch_decision",
                    "sym": "CCCUSDT",
                    "tf": "15m",
                    "decision": "registered",
                    "exit_score": 0.8,
                    "mfe_pct": 3.0,
                },
            ]
            path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
            payload = scorecard.build_scorecard(
                date(2026, 5, 30),
                events_file=path,
                reports_dir=Path(td),
                output_json=Path(td) / "latest.json",
                output_txt=Path(td) / "latest.txt",
                save=False,
            )

        summary = payload["summary"]
        self.assertEqual(payload["status"], "complete")
        self.assertEqual(summary["alerts_total"], 0)
        self.assertEqual(summary["watch_decisions_total"], 3)
        self.assertEqual(summary["watch_registered"], 1)
        self.assertEqual(summary["watch_rejected_exit_score"], 1)
        self.assertEqual(summary["watch_rejected_mfe"], 1)
        self.assertEqual(summary["same_day_alert_per_registered_ratio"], 0.0)
        self.assertIn("final candidate confirmation", payload["interpretation"])

    def test_open_registered_watch_marks_scorecard_partial(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "bot_events.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "ts": "2026-05-30T10:00:00Z",
                        "event": "suspicious_reentry_watch_decision",
                        "sym": "AAAUSDT",
                        "tf": "15m",
                        "decision": "registered",
                        "exit_score": 0.8,
                        "mfe_pct": 3.0,
                    }
                ),
                encoding="utf-8",
            )
            payload = scorecard.build_scorecard(
                date(2026, 5, 30),
                events_file=path,
                reports_dir=Path(td),
                output_json=Path(td) / "latest.json",
                output_txt=Path(td) / "latest.txt",
                save=False,
                now_utc=datetime(2026, 5, 30, 10, 30, tzinfo=timezone.utc),
            )

        self.assertEqual(payload["status"], "partial")
        self.assertEqual(payload["summary"]["watch_pending"], 1)
        self.assertIn("pending registered watch windows", payload["interpretation"])

    def test_zero_alert_and_zero_watch_day_is_partial(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "bot_events.jsonl"
            path.write_text("", encoding="utf-8")
            payload = scorecard.build_scorecard(
                date(2026, 5, 30),
                events_file=path,
                reports_dir=Path(td),
                output_json=Path(td) / "latest.json",
                output_txt=Path(td) / "latest.txt",
                save=False,
            )

        self.assertEqual(payload["status"], "partial")
        self.assertTrue(any("no re-entry watch decisions" in reason for reason in payload["coverage_reasons"]))


if __name__ == "__main__":
    unittest.main()
