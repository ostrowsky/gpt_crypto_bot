"""P2B2-08 attribution contracts, written before implementation."""
import importlib
import unittest
from unittest.mock import patch
from _p2_support import Fixture, NOW, metadata, book, candle


class WriterAttributionTests(Fixture, unittest.TestCase):
    def setUp(self):
        self.setup_market()
        self.api = importlib.import_module("binance_demo_bot.market_sqlite")

    def test_P2B208_cache_stats_actual_write_and_spill(self):
        connection = self.store._con
        before = connection.cache_stats()
        self.assertEqual(set(before), {"hits_n", "misses_n", "writes_n", "spills_n", "used_bytes_n"})
        self.assertTrue(all(type(value) is int and value >= 0 for value in before.values()))
        connection.execute("PRAGMA cache_size=10")
        self.store.append("FEATURE", {"probe": "x"*524288},
            received_at_ms=NOW, received_monotonic_ns=self.mono)
        after = connection.cache_stats()
        self.assertGreater(after["writes_n"], before["writes_n"])
        self.assertGreater(after["spills_n"], before["spills_n"])

    def test_P2B208_cache_stats_no_sql_reset_or_shared_mutation(self):
        connection = self.store._con
        before = connection.cache_stats()
        traced = []
        connection.set_trace_callback(traced.append)
        try:
            snapshot = connection.cache_stats()
            snapshot["writes_n"] = -1
            after = connection.cache_stats()
        finally:
            connection.set_trace_callback(None)
        self.assertEqual(traced, [])
        self.assertEqual(after, before)

    def test_P2B208_closed_cache_stats_fail(self):
        connection = self.api.connect(":memory:")
        connection.close()
        with self.assertRaises(self.api.ProgrammingError): connection.cache_stats()

    def prepared(self, count):
        pending = self.store.prepared_batch()
        pending.__enter__()
        values = [pending.append("FEATURE", {"index": index}, received_at_ms=NOW,
            received_monotonic_ns=self.mono) for index in range(count)]
        return pending, values

    def test_P2B208_eight_rows_use_one_actual_insert(self):
        pending, values = self.prepared(8)
        traced = []
        self.store._con.set_trace_callback(traced.append)
        try: pending.__exit__(None, None, None)
        finally: self.store._con.set_trace_callback(None)
        inserts = [sql for sql in traced if sql.startswith("INSERT OR IGNORE INTO snapshots")]
        self.assertEqual(len(inserts), 1)
        self.assertEqual(self.store.records("FEATURE"), values)
        self.assertEqual(self.store.sqlite_settings()["synchronous"], 2)

    def test_P2B208_bulk_chunks_bounded_and_keep_order(self):
        pending, values = self.prepared(65)
        traced = []
        self.store._con.set_trace_callback(traced.append)
        try: pending.__exit__(None, None, None)
        finally: self.store._con.set_trace_callback(None)
        inserts = [sql for sql in traced if sql.startswith("INSERT OR IGNORE INTO snapshots")]
        self.assertEqual(len(inserts), 2)
        self.assertEqual([sql.count("),(")+1 for sql in inserts], [64, 1])
        self.assertEqual(self.store.records("FEATURE"), values)

    def test_P2B208_later_bulk_chunk_failure_rolls_back_every_row(self):
        pending, values = self.prepared(65)
        self.store._con.execute("CREATE TEMP TRIGGER reject_last BEFORE INSERT ON snapshots "
            "WHEN json_extract(NEW.payload_json,'$.data.index')=64 "
            "BEGIN SELECT RAISE(ABORT,'test chunk failure'); END")
        with self.assertRaises(self.api.IntegrityError): pending.__exit__(None, None, None)
        self.assertEqual(self.store.records("FEATURE"), [])
        self.assertFalse(self.store._con.in_transaction)
        self.assertFalse(self.store._own_batch())

    def test_P2B208_concurrent_prepared_retry_keeps_first_primary(self):
        first, values = self.prepared(8)
        second, copies = self.prepared(8)
        first.__exit__(None, None, None)
        second.__exit__(None, None, None)
        self.assertEqual(self.store.records("FEATURE"), values)
        self.assertEqual([value["snapshot_id"] for value in copies], [value["snapshot_id"] for value in values])

    def test_P2B208_standalone_return_select_never_holds_writer(self):
        import threading
        original = self.store._con.execute
        def checked(statement, *arguments):
            if statement.startswith("SELECT payload_json"):
                self.assertNotEqual(self.store._lock._owner, threading.get_ident())
            return original(statement, *arguments)
        with patch.object(self.store._con, "execute", checked):
            saved = self.store.append("FEATURE", {"probe": True},
                received_at_ms=NOW, received_monotonic_ns=self.mono)
        self.assertEqual(self.store.get(saved["snapshot_id"]), saved)

    def test_P2B208_committed_dependency_read_once_per_prepared_group(self):
        source = self.store.append("FEATURE", {"source": True},
            received_at_ms=NOW+5, received_monotonic_ns=self.mono)
        traced = []
        self.store._read_con.set_trace_callback(traced.append)
        try:
            with self.store.prepared_batch() as pending:
                values = [pending.append("FEATURE", {"index": index}, received_at_ms=NOW,
                    received_monotonic_ns=self.mono, dependencies=[source["snapshot_id"]])
                    for index in range(8)]
        finally: self.store._read_con.set_trace_callback(None)
        queries = [sql for sql in traced if sql.startswith("SELECT available_ms")]
        self.assertEqual(len(queries), 1)
        self.assertEqual(sum(sql=="BEGIN" for sql in traced), 9)
        self.assertTrue(all(value["available_at_ms"]==NOW+5 for value in values))
        self.assertEqual(self.store.records("FEATURE", cutoff_ms=NOW), [])

    def test_P2B208_missing_dependency_not_cached_and_staged_availability_kept(self):
        module = importlib.import_module("binance_demo_bot.market_store")
        _, row = module._prepared_row(self.store.identity, "FEATURE", {"later": True},
            NOW+5, self.mono, [], NOW+5)
        identifier = row[0]
        with self.store.prepared_batch() as pending:
            self.assert_reason("DEPENDENCY_MISSING", pending.append, "FEATURE", {},
                received_at_ms=NOW, received_monotonic_ns=self.mono, dependencies=[identifier])
            source = pending.append("FEATURE", {"later": True}, received_at_ms=NOW+5,
                received_monotonic_ns=self.mono)
            saved = pending.append("FEATURE", {"child": True}, received_at_ms=NOW,
                received_monotonic_ns=self.mono, dependencies=[identifier])
            self.assertEqual(source["snapshot_id"], identifier)
            self.assertEqual(saved["available_at_ms"], NOW+5)
        self.assertEqual(self.store.get(saved["snapshot_id"]), saved)

    def test_P2B208_append_return_preserves_older_pinned_read_and_retry(self):
        original = self.store.append("FEATURE", {"first": True}, received_at_ms=NOW,
            received_monotonic_ns=self.mono)
        with self.store._reading():
            self.assertEqual(self.store.records("FEATURE"), [original])
            added = self.store.append("FEATURE", {"later": True}, received_at_ms=NOW+1,
                received_monotonic_ns=self.mono+1)
            self.assertIsNotNone(added)
            retried = self.store.append("FEATURE", {"later": True}, received_at_ms=NOW+1,
                received_monotonic_ns=self.mono+1)
            self.assertEqual(retried, added)
            self.assertEqual(self.store.records("FEATURE"), [original])
        self.assertEqual(self.store.records("FEATURE"), [original, added])


class RuntimeClosedWarmupTests(Fixture, unittest.TestCase):
    def setUp(self): self.setup_market()

    def test_P2B208_latest_candles_include_open_bar_but_warm_three_closed(self):
        from test_market_runtime import FakeSupervisor
        api = importlib.import_module("binance_demo_bot.market_runtime")
        supervisor_patch = patch.object(api, "MarketStreamSupervisor", FakeSupervisor)
        supervisor_patch.start(); self.addCleanup(supervisor_patch.stop)
        network_patch = patch("binance_demo_bot.market_stream.ws_connect",
            side_effect=AssertionError("Unit tests cannot open network connections"))
        forbidden = network_patch.start(); self.addCleanup(network_patch.stop)
        self.addCleanup(forbidden.assert_not_called)
        runtime = api.MarketDataRuntime(self.store, self.client)
        self.addCleanup(runtime.stop)
        self.replies.append(metadata()); runtime.start()
        self.replies.append([book()])
        latest = [candle(NOW-offset*60000) for offset in (3, 2, 1, 0)]
        original = self.client.read_klines
        requested = []
        def read_latest(symbol, **parameters):
            requested.append(parameters["limit"])
            self.replies.append(latest[-parameters["limit"]:])
            return original(symbol, **parameters)
        with patch.object(self.client, "read_klines", read_latest): runtime.tick()
        self.assertEqual(requested, [4])
        self.assertEqual(runtime.health()["closed_history_ready_n"], 1)
        rows = self.store.candle_index(NOW, "AAAUSDT")
        self.assertEqual(sum(row[0]["closed"] for row in rows.values()), 3)
        self.assertFalse(rows[NOW][0]["closed"])
