"""P2B2-07 contracts; intended missing interfaces RED before code."""
import importlib
import sqlite3 as stdlib_sqlite
import threading
import unittest
from unittest.mock import patch

from _p2_support import Fixture,NOW,metadata
from test_market_runtime import FakeSupervisor


class SqliteBridgeTests(unittest.TestCase):
    def bridge(self): return importlib.import_module("binance_demo_bot.market_sqlite")

    def test_P2B207_patched_pinned_engine(self):
        api=self.bridge()
        self.assertEqual(api.sqlite_version_info,(3,53,4))
        self.assertEqual(api.backend_version,"3.53.4.0")

    def test_P2B207_row_transactions_and_rollback_parity(self):
        api=self.bridge();connection=api.connect(":memory:",isolation_level=None,check_same_thread=False)
        self.addCleanup(connection.close);connection.row_factory=api.Row
        connection.execute("CREATE TABLE example (id INTEGER PRIMARY KEY,value TEXT)")
        connection.execute("BEGIN IMMEDIATE");self.assertTrue(connection.in_transaction)
        connection.execute("INSERT INTO example VALUES (?,?)",(1,"one"));connection.rollback()
        self.assertEqual(connection.execute("SELECT * FROM example").fetchall(),[])
        connection.execute("BEGIN");connection.executemany("INSERT INTO example VALUES (?,?)",[(1,"one"),(2,"two")]);connection.commit()
        rows=connection.execute("SELECT * FROM example ORDER BY id").fetchall()
        self.assertEqual(rows[0]["value"],"one");self.assertEqual(rows[0][0],1)
        self.assertEqual(tuple(rows[1]),(2,"two"));self.assertEqual(len(rows[0]),2)
        self.assertEqual(rows[0].keys(),["id","value"])
        connection.commit();connection.rollback();self.assertFalse(connection.in_transaction)
        with self.assertRaises(api.IntegrityError):connection.execute("INSERT INTO example VALUES (1,'duplicate')")
        connection.close();connection.close()
        with self.assertRaises(api.ProgrammingError):connection.execute("SELECT 1")

    def test_P2B207_expanded_trace_not_placeholder_sql(self):
        api=self.bridge();connection=api.connect(":memory:");self.addCleanup(connection.close)
        seen=[];connection.set_trace_callback(seen.append)
        row=connection.execute("SELECT ? AS symbol",("AAAUSDT",)).fetchone()
        self.assertEqual(row[0],"AAAUSDT")
        self.assertIn("'AAAUSDT'",seen[0]);self.assertNotIn("?",seen[0])
        connection.set_trace_callback(None)


class CheckpointTests(Fixture,unittest.TestCase):
    def setUp(self): self.setup_market()

    def worker(self):
        api=importlib.import_module("binance_demo_bot.market_checkpoint")
        worker=api.MarketCheckpointer(self.store)
        self.addCleanup(worker.stop)
        return worker

    def append(self):
        return self.store.append("FEATURE",{},received_at_ms=NOW,received_monotonic_ns=self.mono)

    def test_P2B207_no_thread_on_store_construction(self):
        self.assertIsNone(self.store._checkpoint)

    def test_P2B207_start_full_version_disable_restore_and_restart(self):
        worker=self.worker();original=self.store._con.execute("PRAGMA wal_autocheckpoint").fetchone()[0]
        worker.start()
        self.assertTrue(worker.health()["running"])
        self.assertEqual(self.store._con.execute("PRAGMA wal_autocheckpoint").fetchone()[0],0)
        self.assertEqual(self.store.sqlite_settings()["synchronous"],2)
        worker.stop();worker.stop()
        self.assertFalse(worker._thread.is_alive())
        self.assertEqual(self.store._con.execute("PRAGMA wal_autocheckpoint").fetchone()[0],original)
        worker.start();worker.stop()

    def test_P2B207_unsafe_engine_blocked_before_worker_or_policy_change(self):
        worker=self.worker();original=self.store._con.execute("PRAGMA wal_autocheckpoint").fetchone()[0]
        with patch.object(worker.api,"sqlite_version_info",(3,50,4)):
            self.assert_reason("SQLITE_VERSION_UNSAFE",worker.start)
        self.assertIsNone(worker._thread)
        self.assertEqual(self.store._con.execute("PRAGMA wal_autocheckpoint").fetchone()[0],original)

    def test_P2B207_owner_batch_cannot_start_worker(self):
        worker=self.worker()
        with self.store.batch():self.assert_reason("NESTED_MARKET_BATCH",worker.start)

    def test_P2B207_maintenance_cannot_write_economic_rows_or_attach(self):
        worker=self.worker();worker.start()
        for query in ("INSERT INTO meta VALUES ('x','y')","DELETE FROM snapshots","CREATE TABLE forbidden (x)","ATTACH ':memory:' AS forbidden"):
            with self.assertRaises(worker.api.Error):worker._connection.execute(query)
        self.append();self.assertEqual(len(self.store.records("FEATURE")),1)

    def test_P2B207_checkpoint_wait_never_holds_writer(self):
        worker=self.worker();worker.start();entered=threading.Event();release=threading.Event()
        original=worker._passive
        def delayed():entered.set();release.wait(2);return original()
        with patch.object(worker,"_passive",delayed):
            thread=threading.Thread(target=worker.run_once);thread.start()
            self.assertTrue(entered.wait(1))
            try:
                result=self.append();self.assertIsNotNone(self.store.get(result["snapshot_id"]))
            finally:release.set();thread.join(2)
        self.assertFalse(thread.is_alive())

    def test_P2B207_busy_partial_error_and_healthy_recovery(self):
        worker=self.worker();worker.start()
        with patch.object(worker,"_passive",return_value=(10,3)):worker.run_once()
        health=worker.health();self.assertEqual(health["state"],"PARTIAL")
        self.assertEqual((health["log_frames_n"],health["checkpointed_frames_n"]),(10,3))
        with patch.object(worker,"_passive",side_effect=worker.api.OperationalError("busy")):worker.run_once()
        self.assertEqual(worker.health()["state"],"CHECKPOINT_FAILED")
        self.assertEqual(worker.health()["attempts_n"],2)
        with patch.object(worker,"_passive",return_value=(10,10)):worker.run_once()
        self.assertEqual(worker.health()["state"],"READY")

    def test_P2B207_wal_bound_blocks_append_and_prepared_commit(self):
        worker=self.worker();worker.start()
        with patch.object(worker,"wal_bytes",return_value=worker.HARD_BYTES):
            self.assert_reason("MARKET_STORAGE_BACKPRESSURE",self.append)
            with self.assertRaises(Exception) as caught:
                with self.store.prepared_batch() as pending:
                    pending.append("FEATURE",{},received_at_ms=NOW,received_monotonic_ns=self.mono)
            self.assertEqual(caught.exception.reason,"MARKET_STORAGE_BACKPRESSURE")
            self.assertEqual(worker.health()["state"],"STORAGE_BACKPRESSURE")
        self.assertEqual(self.store.records(),[])

    def test_P2B207_store_close_joins_owned_worker(self):
        self.store.start_checkpointing();worker=self.store._checkpoint
        self.store.close();self.store.close()
        self.assertFalse(worker._thread.is_alive())

    def test_P2B207_stdlib_file_and_snapshot_reopen_compatible(self):
        before=self.append();database=self.store.path
        self.store.close()
        connection=stdlib_sqlite.connect(database)
        try:self.assertEqual(connection.execute("SELECT id FROM snapshots").fetchone()[0],before["snapshot_id"])
        finally:connection.close()
        reopened=self.market.MarketSnapshotStore(self.paths,account_alias="demo-one",episode_id="episode-one",scope_id="whole-account")
        self.addCleanup(reopened.close)
        self.assertEqual(reopened.get(before["snapshot_id"]),before)

    def test_P2B207_maintenance_denies_unsafe_pragmas(self):
        worker=self.worker();worker.start()
        for query in ("PRAGMA wal_checkpoint(TRUNCATE)","PRAGMA wal_checkpoint(FULL)",
                      "PRAGMA wal_checkpoint(RESTART)","PRAGMA synchronous=OFF",
                      "PRAGMA journal_mode=DELETE","PRAGMA wal_autocheckpoint=1"):
            with self.assertRaises(worker.api.Error):worker._connection.execute(query)
        self.assertEqual(worker._connection.execute("PRAGMA synchronous").fetchone()[0],2)
        worker.run_once();self.assertGreater(worker.health()["attempts_n"],0)

    def test_P2B207_thread_start_failure_cleans_resources(self):
        worker=self.worker();original=self.store._con.execute("PRAGMA wal_autocheckpoint").fetchone()[0]
        with patch.object(threading.Thread,"start",side_effect=RuntimeError("start failed")):
            with self.assertRaises(RuntimeError):worker.start()
        self.assertFalse(worker.running);self.assertIsNone(worker._connection)
        self.assertIsNone(worker._thread)
        self.assertEqual(self.store._con.execute("PRAGMA wal_autocheckpoint").fetchone()[0],original)

    def test_P2B207_unexpected_checkpoint_exception_is_visible(self):
        worker=self.worker();worker.start()
        with patch.object(worker,"_passive",side_effect=RuntimeError("failed")):worker.run_once()
        self.assertEqual(worker.health()["state"],"CHECKPOINT_FAILED")
        self.assertEqual(worker.health()["failures_n"],1)

    def test_P2B207_wal_stat_failure_visible_and_blocks_writes(self):
        worker=self.worker();worker.start()
        with patch.object(worker,"wal_bytes",side_effect=OSError("stat failed")):
            health=worker.health()
            self.assertEqual(health["state"],"CHECKPOINT_FAILED")
            self.assertIsNone(health["wal_physical_bytes_n"])
            self.assert_reason("MARKET_STORAGE_UNAVAILABLE",self.append)
            with self.assertRaises(Exception) as caught:
                with self.store.prepared_batch() as pending:
                    pending.append("FEATURE",{},received_at_ms=NOW,received_monotonic_ns=self.mono)
            self.assertEqual(caught.exception.reason,"MARKET_STORAGE_UNAVAILABLE")
        self.assertEqual(self.store.records(),[])

    def test_P2B207_wal_bound_rechecked_after_waiting_for_writer(self):
        worker=self.worker();worker.start()
        with patch.object(worker,"wal_bytes",side_effect=[0,worker.HARD_BYTES]):
            self.assert_reason("MARKET_STORAGE_BACKPRESSURE",self.append)
        with patch.object(worker,"wal_bytes",side_effect=[0,worker.HARD_BYTES]):
            with self.assertRaises(Exception) as caught:
                with self.store.prepared_batch() as pending:
                    pending.append("FEATURE",{},received_at_ms=NOW,received_monotonic_ns=self.mono)
            self.assertEqual(caught.exception.reason,"MARKET_STORAGE_BACKPRESSURE")
        self.assertEqual(self.store.records(),[])

    def test_P2B207_real_busy_category_and_counter(self):
        worker=self.worker();worker.start()
        contender=worker.api.connect(self.store.path,timeout=0);self.addCleanup(contender.close)
        self.store._con.execute("BEGIN IMMEDIATE")
        try:
            with self.assertRaises(worker.api.BusyError):contender.execute("BEGIN IMMEDIATE")
        finally:self.store._con.rollback()
        with patch.object(worker,"_passive",side_effect=worker.api.BusyError("busy")):worker.run_once()
        self.assertEqual(worker.health()["busy_n"],1)
        self.assertEqual(worker.health()["failures_n"],1)
        self.assertEqual(worker.health()["state"],"CHECKPOINT_FAILED")


class RuntimeCheckpointTests(Fixture,unittest.TestCase):
    def setUp(self):
        self.setup_market();self.api=importlib.import_module("binance_demo_bot.market_runtime")
        patcher=patch.object(self.api,"MarketStreamSupervisor",FakeSupervisor)
        patcher.start();self.addCleanup(patcher.stop)

    def runtime(self,**options):
        runtime=self.api.MarketDataRuntime(self.store,self.client,**options)
        self.addCleanup(runtime.stop);return runtime

    def test_P2B207_runtime_optout_and_disabled_no_worker(self):
        runtime=self.runtime(checkpoint_background=False);self.replies.append(metadata());runtime.start()
        self.assertIsNone(self.store._checkpoint)
        runtime.stop();disabled=self.runtime(enabled=False)
        self.assert_reason("MARKET_DATA_DISABLED",disabled.start)
        self.assertIsNone(self.store._checkpoint)

    def test_P2B207_runtime_owns_worker_and_degrades_on_failure(self):
        runtime=self.runtime();self.replies.append(metadata());runtime.start()
        worker=self.store._checkpoint;self.assertTrue(worker.health()["running"])
        with patch.object(worker,"_passive",side_effect=worker.api.OperationalError("failed")):worker.run_once()
        self.assertEqual(runtime.health()["pipeline_state"],"STORAGE_DEGRADED")
        self.assertFalse(runtime.health()["production_qualified"])
        runtime.stop();self.assertFalse(worker._thread.is_alive())

    def test_P2B207_supervisor_stop_failure_still_joins_worker(self):
        runtime=self.runtime();self.replies.append(metadata());runtime.start()
        worker=self.store._checkpoint
        with patch.object(runtime.supervisor,"stop",side_effect=RuntimeError("stop failed")):
            with self.assertRaises(RuntimeError):runtime.stop()
        self.assertFalse(worker.running);self.assertFalse(worker._thread.is_alive())
