"""P2B2-06 WAL reader/writer contracts, recorded RED before implementation."""

import importlib
import threading
import unittest
from unittest.mock import patch

from _p2_support import Fixture, NOW, book, candle


class StoreConcurrencyTests(Fixture, unittest.TestCase):
    def setUp(self):
        self.setup_market()
        self.api = importlib.import_module("binance_demo_bot.market_store")

    def _append(self, kind="FEATURE", data=None):
        return self.store.append(kind, {"probe": True} if data is None else data,
            received_at_ms=self.wall, received_monotonic_ns=self.mono)

    def _read_does_not_block_writer(self, operation, *, serializer="loads"):
        original = getattr(self.api.json, serializer)
        owner = threading.get_ident()
        workers, errors = [], []
        performed = False
        def paused(*args, **kwargs):
            nonlocal performed
            if threading.get_ident() == owner and not performed:
                performed = True
                finished = threading.Event()
                def write():
                    try:
                        self._append()
                    except BaseException as error:
                        errors.append(error)
                    finally:
                        finished.set()
                worker = threading.Thread(target=write)
                workers.append(worker); worker.start()
                self.assertTrue(finished.wait(1), "read parsing/hashing holds the writer lock")
            return original(*args, **kwargs)
        try:
            with patch.object(self.api.json, serializer, paused):
                result = operation()
            self.assertTrue(performed, "test did not reach the read parse/hash boundary")
            self.assertEqual(errors, [])
            return result
        finally:
            for worker in workers:
                worker.join(2)

    def test_P2B206_get_parsing_does_not_block_writer(self):
        source = self.refresh()
        self.assertEqual(self._read_does_not_block_writer(lambda:self.store.get(source["snapshot_id"])), source)

    def test_P2B206_records_parsing_does_not_block_writer(self):
        self.refresh()
        self.assertEqual(len(self._read_does_not_block_writer(lambda:self.store.records("UNIVERSE"))), 1)

    def test_P2B206_as_of_parsing_does_not_block_writer(self):
        source = self.refresh()
        self.assertEqual(self._read_does_not_block_writer(lambda:self.store.as_of("UNIVERSE", NOW)), source)

    def test_P2B206_quote_index_parsing_does_not_block_writer(self):
        source = self.refresh(); self.quotes(universe=source)
        result = self._read_does_not_block_writer(lambda:self.store.quote_index(NOW))
        self.assertEqual(result["AAAUSDT"][0]["state"], "QUOTE_READY")

    def test_P2B206_candle_index_parsing_does_not_block_writer(self):
        self.bars()
        result = self._read_does_not_block_writer(lambda:self.store.candle_index(NOW))
        self.assertEqual(len(result["AAAUSDT"]), 3)

    def test_P2B206_full_envelope_hashing_does_not_block_writer(self):
        source = self.refresh()
        self._read_does_not_block_writer(lambda:self.store.validate_snapshot(source, "UNIVERSE", NOW), serializer="dumps")
        source["data"]["rows"][0]["base"] = "TAMPERED"
        self.assert_reason("UNIVERSE_SNAPSHOT_INVALID", self.store.validate_snapshot, source, "UNIVERSE", NOW)

    def _foreign_read_in_batch(self, operation):
        results, errors = [], []
        finished = threading.Event()
        def read():
            try: results.append(operation())
            except BaseException as error: errors.append(error)
            finally: finished.set()
        worker = threading.Thread(target=read)
        worker.start()
        return worker, finished, results, errors

    def test_P2B206_foreign_reader_does_not_wait_or_see_uncommitted_row(self):
        committed = self._append()
        worker = None
        try:
            with self.store.batch():
                staged = self._append(data={"staged": True})
                self.assertEqual(self.store.get(staged["snapshot_id"]), staged)
                worker, finished, results, errors = self._foreign_read_in_batch(lambda:self.store.records("FEATURE"))
                self.assertTrue(finished.wait(1), "reader waits for writer instead of using committed WAL snapshot")
                self.assertEqual(errors, [])
                self.assertEqual(results, [[committed]])
            self.assertEqual(len(self.store.records("FEATURE")), 2)
        finally:
            if worker: worker.join(2)

    def test_P2B206_rollback_cannot_enter_foreign_quote_cache(self):
        source = self.refresh(); self.quotes(universe=source)
        original = self.store.quote_index(NOW)
        worker = None
        try:
            with self.assertRaisesRegex(RuntimeError, "rollback"):
                with self.store.batch():
                    self._append("QUOTES", {"rows":[{"symbol":"AAAUSDT", "state":"QUOTE_INVALID", "quote":None, "reasons":["probe"]}]})
                    worker, finished, results, errors = self._foreign_read_in_batch(lambda:self.store.quote_index(NOW))
                    self.assertTrue(finished.wait(1), "foreign cache waits for uncommitted writer")
                    self.assertEqual(errors, [])
                    self.assertEqual(results, [original])
                    raise RuntimeError("rollback")
            self.assertEqual(self.store.quote_index(NOW), original)
        finally:
            if worker: worker.join(2)

    def test_P2B206_reader_connection_is_read_only(self):
        self.assertEqual(self.store._read_con.execute("PRAGMA query_only").fetchone()[0], 1)
        with self.assertRaises(self.api.sqlite3.OperationalError):
            self.store._read_con.execute("INSERT INTO meta VALUES ('forbidden','write')")

    def test_P2B206_owner_candle_revision_sees_its_staged_predecessor(self):
        from test_market_runtime import kline
        with self.store.batch():
            first = self.service.record_closed_kline(kline(), received_at_ms=NOW, received_monotonic_ns=self.mono)
            changed = self.service.record_closed_kline(kline(c="103"), received_at_ms=NOW, received_monotonic_ns=self.mono)
            self.assertTrue(first["data"]["rows"][0]["valid"])
            self.assertFalse(changed["data"]["rows"][0]["valid"])
        self.assertFalse(self.store.candle_index(NOW)["AAAUSDT"][NOW-60000][0]["valid"])

    def test_P2B206_candle_symbol_query_is_indexed_and_keeps_prefix(self):
        self.bars(); self.bars(symbol="BBBUSDT")
        with self.store.batch():
            traced=[]
            self.store._con.set_trace_callback(traced.append)
            try: rows = self.store.candle_index(NOW, "AAAUSDT")
            finally: self.store._con.set_trace_callback(None)
            queries=[q for q in traced if q.startswith("SELECT payload_json") and "kind='CANDLES'" in q]
            self.assertEqual(len(queries),1)
            plan = self.store._con.execute("EXPLAIN QUERY PLAN "+queries[0]).fetchall()
            self.assertTrue(any("candle_symbol_seq" in row[3] for row in plan), str([row[3] for row in plan]))
            self.assertEqual(len(rows), 3)
            self.assertEqual(self.store.candle_index(NOW-1, "AAAUSDT"), {})

    def test_P2B206_reopen_adds_derived_indexes_without_rewriting_evidence(self):
        original = self._append()
        database = self.store.path
        self.store.close()
        connection = self.api.sqlite3.connect(database)
        try:
            for name in ("snapshot_kind_seq", "candle_symbol_seq"):
                connection.execute("DROP INDEX IF EXISTS " + name)
            connection.commit()
        finally: connection.close()
        reopened = self.api.MarketSnapshotStore(self.paths, account_alias="demo-one", episode_id="episode-one", scope_id="whole-account")
        self.addCleanup(reopened.close)
        names = {row[1] for row in reopened._con.execute("PRAGMA index_list(snapshots)")}
        self.assertTrue({"snapshot_kind_seq", "candle_symbol_seq"} <= names)
        self.assertEqual(reopened.get(original["snapshot_id"]), original)
        self.assertEqual(reopened.sqlite_settings()["synchronous"], 2)
        reopened.close(); reopened.close()

    def _reader(self):
        from test_market_stream import FakeStream
        api=importlib.import_module("binance_demo_bot.market_stream")
        transport=FakeStream()
        reader=api.MarketStreamReader(self.store,self.client,self.refresh(),["aaausdt@bookTicker"],transport=transport,background=False)
        self.addCleanup(reader.stop)
        reader.start()
        return reader,transport,api

    def test_P2B206_small_groups_preserve_every_frame_and_receipt(self):
        from test_market_stream import tick
        reader,transport,_=self._reader()
        transport.frames.extend(tick(u=i) for i in range(1,21))
        reader.read_once()
        valid=lambda:[r for r in self.store.records("QUOTES") if "wire" in r["data"]]
        self.assertEqual(len(valid()),8)
        while transport.frames:reader.read_once()
        rows=valid()
        self.assertEqual([r["data"]["wire"]["u"] for r in rows],list(range(1,21)))
        self.assertTrue(all(r["received_at_ms"]==NOW and r["received_monotonic_ns"]==self.mono for r in rows))

    def test_P2B206_recv_time_cap_independent_of_frame_count(self):
        from test_market_stream import tick
        reader,transport,api=self._reader()
        transport.frames.extend(tick(u=i) for i in range(1,5))
        times=iter([0,0.003])
        with patch.object(api.time,"monotonic",side_effect=lambda:next(times,0.003)):
            self.assertTrue(reader.read_once())
        rows=[r for r in self.store.records("QUOTES") if "wire" in r["data"]]
        self.assertEqual(len(rows),1)
        self.assertEqual(len(transport.frames),3)

    def test_P2B206_slow_reader_does_not_block_another_reader(self):
        source=self.refresh()
        owner=threading.get_ident()
        original=self.api.json.loads
        workers=[]
        once=False
        def paused(value,*args,**kwargs):
            nonlocal once
            if threading.get_ident()==owner and not once:
                once=True
                worker,finished,results,errors=self._foreign_read_in_batch(lambda:self.store.get(source["snapshot_id"]))
                workers.append(worker)
                self.assertTrue(finished.wait(1),"diagnostic reader blocks receiver source checks")
                self.assertEqual(errors,[]);self.assertEqual(results,[source])
            return original(value,*args,**kwargs)
        try:
            with patch.object(self.api.json,"loads",paused):self.store.get(source["snapshot_id"])
        finally:
            for worker in workers:worker.join(2)

    def test_P2B206_read_snapshot_stays_pinned_across_foreign_commit(self):
        original=self._append()
        with self.store._reading() as connection:
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM snapshots").fetchone()[0],1)
            worker=threading.Thread(target=lambda:self._append(data={"later":True}))
            worker.start();worker.join(2)
            self.assertFalse(worker.is_alive())
            self.assertEqual(self.store.records("FEATURE"),[original])
        self.assertEqual(len(self.store.records("FEATURE")),2)

    def test_P2B206_prepared_serialization_never_holds_writer(self):
        original=self.api._json
        owner=threading.get_ident()
        workers=[]
        def paused(value):
            if value=={"prepared":True} and threading.get_ident()==owner:
                finished=threading.Event()
                worker=threading.Thread(target=lambda:(self._append(),finished.set()))
                workers.append(worker);worker.start()
                self.assertTrue(finished.wait(1),"prepared serialization holds writer")
            return original(value)
        try:
            with patch.object(self.api,"_json",paused):
                with self.store.prepared_batch() as pending:
                    pending.append("FEATURE",{"prepared":True},received_at_ms=NOW,received_monotonic_ns=self.mono)
            self.assertEqual(len(self.store.records("FEATURE")),2)
        finally:
            for worker in workers:worker.join(2)

    def test_P2B206_prepared_hash_dependencies_retry_and_detached_values(self):
        from binance_demo_bot.journal import _digest
        from decimal import Decimal
        data={"amount":Decimal("1.000")}
        original=self._append(data=data)
        with self.store.prepared_batch() as pending:
            retry=pending.append("FEATURE",data,received_at_ms=NOW,received_monotonic_ns=self.mono)
            self.assertEqual(retry,original)
            first=pending.append("QUOTES",{"rows":[]},received_at_ms=NOW,received_monotonic_ns=self.mono)
            second=pending.append("FEATURE",{"prepared":True},received_at_ms=NOW,received_monotonic_ns=self.mono,dependencies=[first["snapshot_id"]])
            basis={k:v for k,v in second.items() if k not in {"event_id","snapshot_id","recorded_at_ms"}}
            self.assertEqual(second["snapshot_id"],_digest(basis))
            second["data"]["prepared"]="TAMPERED"
            self.assertIsNone(self.store.get(first["snapshot_id"]))
        saved=self.store.get(second["snapshot_id"])
        self.assertIs(saved["data"]["prepared"],True)
        self.assertEqual(saved["dependencies"],[first["snapshot_id"]])
        self.assertEqual(len(self.store.records()),3)

    def test_P2B206_prepared_failure_publishes_no_partial_rows(self):
        with self.assertRaisesRegex(RuntimeError,"rollback"):
            with self.store.prepared_batch() as pending:
                pending.append("FEATURE",{},received_at_ms=NOW,received_monotonic_ns=self.mono)
                raise RuntimeError("rollback")
        self.assertEqual(self.store.records(),[])
        with self.store.batch():
            self.assert_reason("NESTED_MARKET_BATCH",self.store.prepared_batch().__enter__)

    def test_P2B206_prepared_candles_overlay_and_guard_concurrent_revision(self):
        from test_market_runtime import kline
        with self.store.prepared_batch() as pending:
            service=self.market.MarketDataService(pending,self.client)
            service.record_closed_kline(kline(),received_at_ms=NOW,received_monotonic_ns=self.mono)
            changed=service.record_closed_kline(kline(c="103"),received_at_ms=NOW,received_monotonic_ns=self.mono)
            self.assertFalse(changed["data"]["rows"][0]["valid"])
        before=self.store.records("CANDLES")
        with self.assertRaises(Exception) as caught:
            with self.store.prepared_batch() as pending:
                self.market.MarketDataService(pending,self.client).record_closed_kline(kline(),received_at_ms=NOW,received_monotonic_ns=self.mono)
                self.bars([candle(NOW-60000,"104")])
        self.assertEqual(caught.exception.reason,"CANDLE_SOURCE_CHANGED")
        self.assertEqual(len(self.store.records("CANDLES")),len(before)+1)

    def test_P2B206_stream_validation_runs_outside_writer_batch(self):
        from test_market_stream import tick
        reader,transport,_=self._reader()
        transport.frames.append(tick())
        original=reader._process
        def processing(*args):
            self.assertFalse(self.store._own_batch(),"stream CPU work holds writer")
            return original(*args)
        with patch.object(reader,"_process",processing):self.assertTrue(reader.read_once())
        self.assertTrue(reader.connected)

    def test_P2B206_all_thread_reader_connections_close(self):
        foreign=[]
        def reading():
            with self.store._reading() as connection:foreign.append(connection)
        worker=threading.Thread(target=reading);worker.start();worker.join(2)
        self.assertFalse(worker.is_alive())
        self.assertEqual(len(foreign),1)
        own=self.store._read_con
        self.assertIsNot(foreign[0],own)
        self.store.close();self.store.close()
        for connection in [own,*foreign]:
            with self.assertRaises(self.api.sqlite3.ProgrammingError):connection.execute("SELECT 1")

    def test_P2B206_cache_updates_cannot_interleave_two_pinned_reads(self):
        source=self.refresh();self.quotes(universe=source)
        owner=threading.get_ident()
        original=self.api.json.loads
        workers=[];once=False
        def paused(value,*args,**kwargs):
            nonlocal once
            if threading.get_ident()==owner and not once:
                once=True
                worker,finished,results,errors=self._foreign_read_in_batch(lambda:self.store.quote_index(NOW))
                workers.append(worker)
                self.assertFalse(finished.wait(1),"two readers mutate the same index cache concurrently")
            return original(value,*args,**kwargs)
        try:
            with patch.object(self.api.json,"loads",paused):self.store.quote_index(NOW)
        finally:
            for worker in workers:worker.join(2)

    def test_P2B206_stream_candle_guard_records_exact_failure_reason(self):
        from test_market_runtime import kline
        from test_market_stream import FakeStream
        api=importlib.import_module("binance_demo_bot.market_stream")
        transport=FakeStream()
        reader=api.MarketStreamReader(self.store,self.client,self.refresh(),["aaausdt@kline_1m"],transport=transport,background=False)
        self.addCleanup(reader.stop);reader.start()
        transport.frames.append({"stream":"aaausdt@kline_1m","data":kline()})
        original=reader._process
        def competing(*args):
            original(*args)
            self.bars([candle(NOW-60000,"104")])
        with patch.object(reader,"_process",competing):self.assertFalse(reader.read_once())
        self.assertEqual(reader.error,"CANDLE_SOURCE_CHANGED")
        failures=[r for r in self.store.records("INCIDENT") if r["data"].get("operation")=="CANDLE_COLLECTION_FAILED"]
        self.assertEqual(failures[-1]["data"]["reason"],"CANDLE_SOURCE_CHANGED")
        self.assertEqual(len(self.store.records("CANDLES")),1)

    def test_P2B206_release_thread_reader_preserves_other_owners(self):
        source=self.refresh()
        connections=[];errors=[]
        def worker():
            try:
                with self.store._reading() as connection:connections.append(connection)
                self.store.release_thread_reader();self.store.release_thread_reader()
            except BaseException as error:errors.append(error)
        thread=threading.Thread(target=worker);thread.start();thread.join(2)
        self.assertFalse(thread.is_alive());self.assertEqual(errors,[])
        self.assertEqual(len(self.store._readers),1)
        self.assertEqual(self.store.get(source["snapshot_id"]),source)
        with self.assertRaises(self.api.sqlite3.ProgrammingError):connections[0].execute("SELECT 1")

    def test_P2B206_real_reader_stop_releases_receiver_connection(self):
        from test_market_stream import FakeStream,tick
        api=importlib.import_module("binance_demo_bot.market_stream")
        transport=FakeStream();processed=threading.Event()
        reader=api.MarketStreamReader(self.store,self.client,self.refresh(),["aaausdt@bookTicker"],transport=transport,background=True)
        self.addCleanup(reader.stop)
        original=reader._process
        def processing(*args):
            result=original(*args);processed.set();return result
        with patch.object(reader,"_process",processing):
            reader.start();transport.frames.append(tick())
            self.assertTrue(processed.wait(2))
            reader.stop()
        self.assertFalse(reader._thread.is_alive())
        self.assertEqual(len(self.store._readers),1)
