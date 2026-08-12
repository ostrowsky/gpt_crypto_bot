from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

import why_no_signal_report as report


NOW = datetime(2026, 8, 12, 10, 0, tzinfo=timezone.utc)


class WhyNoSignalReportTests(unittest.TestCase):
    def _write(self, path: Path, rows: list[dict]) -> None:
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    def test_runtime_event_fallback_reports_c98_block(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            critic = root / "critic.jsonl"
            events = root / "events.jsonl"
            self._write(critic, [])
            self._write(events, [{
                "event": "blocked",
                "sym": "C98USDT",
                "tf": "15m",
                "signal_type": "top_gainer_score_gate",
                "reason_code": "top_gainer_score_gate",
                "gate": "quality",
                "reason": "top-gainer score gate: score 28.48 < 34.00 for 15m breakout",
                "candidate_score": 126.8034,
                "score_floor": 0.0,
                "ts": "2026-08-12T09:49:48Z",
            }])

            payload = report.build_report(
                "c98usdt", days=2, critic_file=critic, event_file=events, now=NOW,
            )

        self.assertEqual(payload["blocked_events"], 1)
        self.assertEqual(payload["reason_counts"], {"top_gainer_score_gate": 1})
        self.assertEqual(payload["trace"][0]["signal_type"], "breakout")
        self.assertEqual(payload["trace"][0]["score_floor"], 34.0)
        self.assertEqual(payload["trace"][0]["sources"], ["bot_events"])

    def test_reverse_window_reader_stops_before_older_history(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "events.jsonl"
            self._write(path, [
                {"event": "blocked", "ts": "2026-01-01T00:00:00Z"},
                {"event": "blocked", "ts": "2026-08-11T09:00:00Z"},
                {"event": "blocked", "ts": "2026-08-12T09:00:00Z"},
            ])

            rows = list(report._iter_window_jsonl(
                path,
                start=datetime(2026, 8, 11, 0, 0, tzinfo=timezone.utc),
                end=NOW,
                ts_fields=("ts",),
            ))

        self.assertEqual([row["ts"] for row in rows], [
            "2026-08-12T09:00:00Z",
            "2026-08-11T09:00:00Z",
        ])

    def test_repeated_runtime_poll_blocks_are_collapsed_per_bar(self) -> None:
        base = {
            "event": "blocked",
            "sym": "C98USDT",
            "tf": "15m",
            "signal_type": "top_gainer_score_gate",
            "reason_code": "top_gainer_score_gate",
            "reason": "top-gainer score gate: score 28.48 < 34.00 for 15m breakout",
            "candidate_score": 126.8034,
        }
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            critic = root / "critic.jsonl"
            events = root / "events.jsonl"
            self._write(critic, [])
            self._write(events, [
                {**base, "ts": "2026-08-12T09:49:48Z"},
                {**base, "ts": "2026-08-12T09:50:57Z"},
                {**base, "ts": "2026-08-12T09:52:05Z"},
            ])

            payload = report.build_report(
                "C98USDT", days=2, critic_file=critic, event_file=events, now=NOW,
            )

        self.assertEqual(payload["blocked_events_raw"], 3)
        self.assertEqual(payload["blocked_events"], 1)
        self.assertEqual(payload["trace"][0]["repeat_count"], 3)
        self.assertEqual(payload["trace"][0]["last_ts"], "2026-08-12T09:52:05Z")

    def test_critic_and_runtime_rows_merge_and_keep_richer_critic_context(self) -> None:
        reason = "top-gainer score gate: score 33.20 < 34.00 for 15m trend"
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            critic = root / "critic.jsonl"
            events = root / "events.jsonl"
            self._write(critic, [{
                "sym": "BTCUSDT",
                "tf": "15m",
                "signal_type": "trend",
                "ts_signal": "2026-08-12T09:00:00Z",
                "decision": {
                    "action": "blocked",
                    "reason_code": "top_gainer_score_gate",
                    "reason": reason,
                    "stage": "quality_floor",
                    "candidate_score": 120.0,
                    "score_floor": 60.0,
                    "signal_flags": {"trend": True},
                },
            }])
            self._write(events, [{
                "event": "blocked",
                "sym": "BTCUSDT",
                "tf": "15m",
                "signal_type": "top_gainer_score_gate",
                "reason_code": "top_gainer_score_gate",
                "reason": reason,
                "candidate_score": 120.0,
                "score_floor": 60.0,
                "ts": "2026-08-12T09:00:02Z",
            }])

            payload = report.build_report(
                "BTCUSDT", days=2, critic_file=critic, event_file=events, now=NOW,
            )

        self.assertEqual(payload["blocked_events_raw"], 2)
        self.assertEqual(payload["blocked_events"], 1)
        self.assertEqual(payload["trace"][0]["sources"], ["bot_events", "critic_dataset"])
        self.assertEqual(payload["trace"][0]["stage"], "quality_floor")
        self.assertEqual(payload["trace"][0]["score_floor"], 34.0)
        self.assertEqual(payload["trace"][0]["signal_flags"], {"trend": True})


if __name__ == "__main__":
    unittest.main()
