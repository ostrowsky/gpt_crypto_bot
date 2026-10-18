import concurrent.futures
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import critic_dataset as critic
import numpy as np
from v2 import trace_index
from v2.shadow_observer import append_decision_trace


def event(bar, sym="AAA"):
    return {"sym": sym, "tf": "15m", "bar_ts": bar, "state": "noise"}


class TraceIndexTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "trace.jsonl"

    def rows(self):
        return [json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines()]

    def test_history_duplicates_restart_and_incremental_decode(self):
        self.path.write_text("".join(json.dumps(event(i)) + "\n" for i in range(100)), encoding="utf-8")
        with patch.object(trace_index.json, "loads", wraps=json.loads) as decode:
            append_decision_trace(self.path, event(0))
            self.assertEqual(decode.call_count, 100)
            decode.reset_mock()
            append_decision_trace(self.path, event(100))
            append_decision_trace(self.path, event(99))
            self.assertEqual(decode.call_count, 0)
            with self.path.open("a", encoding="utf-8") as output:
                output.write(json.dumps(event(101)) + "\n")
            append_decision_trace(self.path, event(101))
            self.assertEqual(decode.call_count, 1)
        self.assertEqual(len(self.rows()), 102)
        # Each call opens a fresh connection; no process memory is needed.
        append_decision_trace(self.path, event(0))
        self.assertEqual(len(self.rows()), 102)

    def test_rebuild_missing_index_and_full_history_not_tail_only(self):
        append_decision_trace(self.path, event(1))
        append_decision_trace(self.path, event(2))
        trace_index.index_path_for(self.path).unlink()
        append_decision_trace(self.path, event(1))
        self.assertEqual(len(self.rows()), 2)

    def test_replacement_truncation_and_same_size_edit(self):
        append_decision_trace(self.path, event(1))
        replacement = self.path.with_suffix(".tmp")
        replacement.write_text(json.dumps(event(2)) + "\n", encoding="utf-8")
        replacement.replace(self.path)
        append_decision_trace(self.path, event(1))
        self.assertEqual([r["bar_ts"] for r in self.rows()], [2, 1])
        self.path.write_text("", encoding="utf-8")
        append_decision_trace(self.path, event(1))
        before = self.path.stat()
        self.path.write_text(json.dumps(event(3)) + "\n", encoding="utf-8")
        import os
        os.utime(self.path, ns=(before.st_atime_ns, before.st_mtime_ns + 1_000_000))
        append_decision_trace(self.path, event(1))
        self.assertEqual([r["bar_ts"] for r in self.rows()], [3, 1])

    def test_malformed_nonobjects_and_unterminated_last_row(self):
        self.path.write_text('bad\n[]\n' + json.dumps(event(1)), encoding="utf-8")
        append_decision_trace(self.path, event(1))
        append_decision_trace(self.path, event(2))
        lines = self.path.read_text(encoding="utf-8").splitlines()
        self.assertEqual([json.loads(row)["bar_ts"] for row in lines[2:]], [1, 2])

    def test_external_unterminated_tail_is_rechecked_when_completed(self):
        self.path.write_text('{"sym":', encoding="utf-8")
        # Build an index without writing because the requested key exists earlier.
        self.path.write_text(json.dumps(event(1)) + '\n{"sym":', encoding="utf-8")
        append_decision_trace(self.path, event(1))
        with self.path.open("a", encoding="utf-8") as output:
            output.write('"AAA","tf":"15m","bar_ts":2}\n')
        append_decision_trace(self.path, event(2))
        self.assertEqual(len(self.rows()), 2)

    def test_append_before_index_commit_is_recovered(self):
        append_decision_trace(self.path, event(1))
        with patch.object(trace_index, "_save_position", side_effect=OSError("commit failed")):
            with self.assertRaises(OSError):
                append_decision_trace(self.path, event(2))
        append_decision_trace(self.path, event(2))
        self.assertEqual(len(self.rows()), 2)

    def test_concurrent_participating_writers_append_once(self):
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            list(executor.map(lambda _: append_decision_trace(self.path, event(1)), range(12)))
        self.assertEqual(len(self.rows()), 1)


class CriticScopedSerializationTests(unittest.TestCase):
    def rewrite(self, raw, mutate, scope, *, fail=False):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "critic.jsonl"
            path.write_text(raw, encoding="utf-8")
            with patch.object(critic, "CRITIC_FILE", path):
                if fail:
                    with patch.object(critic, "_atomic_replace_with_retry", side_effect=OSError("failed")):
                        with self.assertRaises(critic.DatasetIntegrityError):
                            critic._rewrite_records(mutate, strict=True, mutation_scope=scope)
                    self.assertEqual(path.read_text(encoding="utf-8"), raw)
                    self.assertFalse(list(path.parent.glob("*.tmp")))
                    return
                result = critic._rewrite_records(mutate, strict=True, mutation_scope=scope)
                return result, path.read_text(encoding="utf-8")

    def test_scope_only_encodes_matching_records_and_preserves_provenance_side_effects(self):
        raw = '{"id":"a","sym":"AAA"}\n{"id":"b","sym":"AAA"}\n {"id":"c", "sym":"BBB"}'
        def mutate(row):
            self.assertEqual(row["sym"], "AAA")
            row["provenance"] = "side effect"
            return row["id"] == "a"
        with patch.object(critic.json, "dumps", wraps=json.dumps) as encode:
            changed, output = self.rewrite(raw, mutate, lambda row: row.get("sym") == "AAA")
        self.assertTrue(changed)
        self.assertEqual(encode.call_count, 2)
        self.assertTrue(all(json.loads(line)["provenance"] == "side effect" for line in output.splitlines()[:2]))
        self.assertIn(' {"id":"c", "sym":"BBB"}\n', output)

    def test_scoped_and_legacy_outputs_match_with_decoded_duplicate_ids_and_bad_rows(self):
        raw = '{"id":"a","sym":"AAA"}\n{"id":"\\u0061","sym":"AAA"}\nbad\n[]\n{"id":"b","sym":"BBB"}\n'
        def mutate(row):
            if row.get("sym") != "AAA":
                return False
            row["label"] = 2
            return True
        old = self.rewrite(raw, mutate, None)
        new = self.rewrite(raw, mutate, lambda row: row.get("sym") == "AAA")
        self.assertEqual(old[0], new[0])
        self.assertEqual([json.loads(line) for line in old[1].splitlines()],
                         [json.loads(line) for line in new[1].splitlines()])

    def test_nochange_badrow_cleanup_and_publication_failure(self):
        raw = '{"id":"a"}\n'
        self.assertEqual(self.rewrite(raw, lambda row: False, lambda row: True), (False, raw))
        self.assertEqual(self.rewrite(raw + 'bad\n', lambda row: False, lambda row: False), (True, raw))
        def mutate(row):
            row["label"] = 2
            return True
        self.rewrite(raw, mutate, lambda row: True, fail=True)

    def test_label_entrypoints_supply_exact_pair_scope(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(critic, "CRITIC_FILE", Path(directory) / "rows"):
            critic.CRITIC_FILE.touch()
            with patch.object(critic, "_rewrite_records") as rewrite:
                critic.fill_pending_from_data("AAA", "15m", [], [], 900000)
                scope = rewrite.call_args.kwargs["mutation_scope"]
                self.assertTrue(scope({"sym": "AAA", "tf": "15m"}))
                self.assertFalse(scope({"sym": "AAA", "tf": "1h"}))
                complete = {"sym": "AAA", "tf": "15m", "labels": {"ret_3": 0, "ret_5": 0, "ret_10": 0}}
                self.assertFalse(scope(complete))
                critic.fill_pending_batch([{"sym": "AAA", "tf": "15m", "t_arr": [], "c_arr": [], "bar_ms": 900000}], strict=True)
                scope = rewrite.call_args.kwargs["mutation_scope"]
                self.assertTrue(scope({"sym": "AAA", "tf": "15m"}))
                self.assertFalse(scope({"sym": "BBB", "tf": "15m"}))
                self.assertFalse(scope(complete))
                self.assertTrue(rewrite.call_args.kwargs["strict"])

    def test_actual_forward_labels_match_legacy_including_zero_and_provenance(self):
        from datetime import datetime, timezone
        bars = np.arange(13, dtype=np.int64) * 900000 + 1767225600000
        closes = np.arange(13, dtype=float) + 100
        rows = [
            {"id": "pending", "sym": "AAA", "tf": "15m", "bar_ts": int(bars[0]), "labels": {}},
            {"id": "complete", "sym": "AAA", "tf": "15m", "bar_ts": int(bars[0]),
             "labels": {"ret_3": 0, "ret_5": 0, "ret_10": 0}},
            {"id": "other", "sym": "BBB", "tf": "15m", "bar_ts": int(bars[0]), "labels": {}},
        ]
        raw = "".join(json.dumps(row) + "\n" for row in rows)
        original = critic.policy_provenance.build_label_provenance
        def fixed_provenance(**kwargs):
            return original(**kwargs, recorded_at=datetime(2026, 10, 18, tzinfo=timezone.utc))
        def legacy_mutate(row):
            if row.get("sym") != "AAA" or row.get("tf") != "15m":
                return False
            return critic._fill_pending_record(row, t_arr=bars, c_arr=closes, bar_ms=900000)
        with patch.object(critic.policy_provenance, "build_label_provenance", side_effect=fixed_provenance):
            legacy = self.rewrite(raw, legacy_mutate, None)[1]
            with tempfile.TemporaryDirectory() as directory, patch.object(critic, "CRITIC_FILE", Path(directory) / "rows"):
                critic.CRITIC_FILE.write_text(raw, encoding="utf-8")
                with patch.object(critic.json, "dumps", wraps=json.dumps) as encode:
                    critic.fill_pending_batch([{"sym": "AAA", "tf": "15m", "t_arr": bars, "c_arr": closes, "bar_ms": 900000}], strict=True)
                actual = critic.CRITIC_FILE.read_text(encoding="utf-8")
        self.assertEqual(encode.call_count, 1)
        self.assertEqual([json.loads(line) for line in legacy.splitlines()],
                         [json.loads(line) for line in actual.splitlines()])


if __name__ == "__main__":
    unittest.main()
