from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import research_event_cohort_store as store


def _line(**values: object) -> str:
    return json.dumps(values, ensure_ascii=False) + "\n"


class ResearchEventCohortStoreTests(unittest.TestCase):
    def test_initial_sync_append_and_idempotent_reload(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            files = root / "files"
            files.mkdir()
            db = root / "runtime" / "cohorts.sqlite3"
            bot = files / "bot_events.jsonl"
            agent = files / "agent_events.jsonl"
            bot.write_text(
                _line(event="blocked", sym="AAAUSDT", signal_type="top_gainer_score_gate", reason="score", ts="2026-08-02T10:00:00Z")
                + _line(event="blocked", sym="AAAUSDT", signal_type="top_gainer_score_gate", reason="score", ts="2026-08-02T10:05:00Z")
                + _line(event="entry", sym="AAAUSDT", price=10, mode="trend", ts="2026-08-02T11:00:00Z")
                + _line(event="exit", sym="AAAUSDT", pnl_pct=1.5, reason="done", ts="2026-08-02T12:00:00Z"),
                encoding="utf-8",
            )
            agent.write_text(
                _line(event="blocked", sym="AAAUSDT", signal_type="top_gainer_score_gate", reason="score", ts="2026-08-02T09:00:00Z"),
                encoding="utf-8",
            )

            first = store.sync_event_cohorts(files, db)
            blocked, entries, _ = store.load_replay_inputs(
                files_dir=files,
                allowed_days={"2026-08-02"},
                db_path=db,
                sync=False,
            )

            self.assertGreater(first["bytes_processed"], 0)
            self.assertEqual(len(blocked), 1)
            self.assertEqual(blocked[0]["block_count"], 3)
            self.assertEqual(blocked[0]["ts"], "2026-08-02T09:00:00Z")
            self.assertEqual(entries[("2026-08-02", "AAAUSDT")][0]["price"], 10.0)

            with bot.open("a", encoding="utf-8") as handle:
                handle.write(_line(event="blocked", sym="AAAUSDT", signal_type="chase_guard", reason="late", ts="2026-08-02T13:00:00Z"))
            second = store.sync_event_cohorts(files, db)
            third = store.sync_event_cohorts(files, db)
            blocked, _, _ = store.load_replay_inputs(
                files_dir=files,
                allowed_days={"2026-08-02"},
                db_path=db,
                sync=False,
            )

            self.assertGreater(second["bytes_processed"], 0)
            self.assertEqual(third["bytes_processed"], 0)
            self.assertEqual(sum(row["block_count"] for row in blocked), 4)

    def test_truncated_source_is_rebuilt_without_duplicate_counts(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            files = root / "files"
            files.mkdir()
            db = root / "runtime" / "cohorts.sqlite3"
            bot = files / "bot_events.jsonl"
            (files / "agent_events.jsonl").write_text("", encoding="utf-8")
            bot.write_text(
                _line(event="blocked", sym="AAAUSDT", signal_type="chase_guard", reason="x", ts="2026-08-02T10:00:00Z")
                + _line(event="blocked", sym="AAAUSDT", signal_type="chase_guard", reason="x", ts="2026-08-02T10:01:00Z"),
                encoding="utf-8",
            )
            store.sync_event_cohorts(files, db)
            bot.write_text(
                _line(event="blocked", sym="AAAUSDT", signal_type="chase_guard", reason="x", ts="2026-08-02T10:02:00Z"),
                encoding="utf-8",
            )

            result = store.sync_event_cohorts(files, db)
            blocked, _, _ = store.load_replay_inputs(
                files_dir=files,
                allowed_days={"2026-08-02"},
                db_path=db,
                sync=False,
            )

            self.assertTrue(result["sources"][0]["reset"])
            self.assertEqual(blocked[0]["block_count"], 1)


if __name__ == "__main__":
    unittest.main()
