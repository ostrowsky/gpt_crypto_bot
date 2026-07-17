from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from v2.btc_trend_watch import (
    event_key,
    find_latest_btc_watch_event,
    find_recent_main_block,
    format_watch_message,
    is_btc_early_trend_event,
    mark_sent,
    was_sent,
)


def _v2_event(**updates) -> dict:
    event = {
        "event": "v2_shadow_signal",
        "ts": "2026-07-17T17:46:47Z",
        "sym": "BTCUSDT",
        "tf": "15m",
        "bar_ts": 1784309400000,
        "previous_state": "noise",
        "state": "emerging_move",
        "action": "elevate_priority",
        "reason": "early positive structure",
        "bootstrap": False,
        "features": {
            "price": 64029.99,
            "slope": 0.4016,
            "rsi": 69.47,
            "adx": 19.21,
            "vol_x": 1.074,
        },
    }
    event.update(updates)
    return event


def _block(ts: str, score: float, **updates) -> dict:
    row = {
        "event": "blocked",
        "sym": "BTCUSDT",
        "tf": "15m",
        "reason_code": "top_gainer_score_gate",
        "candidate_score": score,
        "ts": ts,
    }
    row.update(updates)
    return row


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


class TestBtcTrendWatch(unittest.TestCase):
    def test_event_profile_is_exact_and_rejects_bootstrap(self) -> None:
        self.assertTrue(is_btc_early_trend_event(_v2_event()))
        self.assertFalse(is_btc_early_trend_event(_v2_event(sym="ETHUSDT")))
        self.assertFalse(is_btc_early_trend_event(_v2_event(tf="1h")))
        self.assertFalse(is_btc_early_trend_event(_v2_event(bootstrap=True)))
        self.assertFalse(is_btc_early_trend_event(_v2_event(reason="trend strength confirmed")))

    def test_recent_main_block_selects_highest_raw_score(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "events.jsonl"
            _write_jsonl(
                path,
                [
                    _block("2026-07-17T17:10:00Z", 99.0),
                    _block("2026-07-17T17:25:00Z", 78.1),
                    _block("2026-07-17T17:29:55Z", 82.102),
                    _block("2026-07-17T17:35:00Z", 120.0, sym="ETHUSDT"),
                ],
            )
            found = find_recent_main_block(
                path,
                signal_ts="2026-07-17T17:46:47Z",
                lookback_minutes=30,
                min_candidate_score=60.0,
                max_bytes=1_000_000,
            )
        self.assertIsNotNone(found)
        self.assertEqual(found["candidate_score"], 82.102)

    def test_recent_main_block_rejects_stale_low_and_future_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "events.jsonl"
            _write_jsonl(
                path,
                [
                    _block("2026-07-17T17:15:00Z", 90.0),
                    _block("2026-07-17T17:30:00Z", 59.99),
                    _block("2026-07-17T17:47:00Z", 95.0),
                ],
            )
            found = find_recent_main_block(
                path,
                signal_ts="2026-07-17T17:46:47Z",
                lookback_minutes=30,
                min_candidate_score=60.0,
                max_bytes=1_000_000,
            )
        self.assertIsNone(found)

    def test_catchup_finds_only_fresh_exact_profile(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "v2.jsonl"
            _write_jsonl(
                path,
                [
                    _v2_event(ts="2026-07-17T16:00:00Z"),
                    _v2_event(ts="2026-07-17T17:40:00Z", sym="ETHUSDT"),
                    _v2_event(ts="2026-07-17T17:46:47Z"),
                ],
            )
            found = find_latest_btc_watch_event(
                path,
                now=datetime(2026, 7, 17, 18, 0, tzinfo=timezone.utc),
                max_age_minutes=90,
                max_bytes=1_000_000,
            )
        self.assertIsNotNone(found)
        self.assertEqual(found["ts"], "2026-07-17T17:46:47Z")

    def test_persistent_dedupe_uses_exact_event_key(self) -> None:
        event = _v2_event()
        with tempfile.TemporaryDirectory() as tmp:
            state_path = Path(tmp) / "state.json"
            self.assertFalse(was_sent(state_path, event))
            mark_sent(state_path, event, sent_at="2026-07-17T18:00:00Z")
            self.assertTrue(was_sent(state_path, event))
            self.assertFalse(was_sent(state_path, _v2_event(ts="2026-07-17T18:01:00Z")))
            state = json.loads(state_path.read_text(encoding="utf-8"))
        self.assertEqual(state["last_event_key"], event_key(event))

    def test_operator_message_is_explicitly_watch_not_buy(self) -> None:
        text = format_watch_message(
            _v2_event(),
            _block("2026-07-17T17:29:55Z", 82.102),
        )
        self.assertIn("WATCH", text)
        self.assertIn("не BUY", text)
        self.assertIn("V1 raw score 82.10", text)
        self.assertIn("slope +0.40%", text)


if __name__ == "__main__":
    unittest.main()
