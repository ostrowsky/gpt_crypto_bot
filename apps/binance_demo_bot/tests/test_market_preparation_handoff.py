"""P2B2-10 actual SQL/read-prefix contracts, written before implementation."""
import threading
import unittest
from unittest.mock import patch
from _p2_support import Fixture, NOW, candle


class PreparationHandoffTests(Fixture, unittest.TestCase):
    def setUp(self): self.setup_market()

    def append(self, data, **options):
        return self.store.append('FEATURE',data,received_at_ms=NOW,
            received_monotonic_ns=self.mono,**options)

    def staged(self,pending,data,**options):
        return pending.append('FEATURE',data,received_at_ms=NOW,
            received_monotonic_ns=self.mono,**options)

    def foreign(self,operation):
        result,errors=[],[]
        def run():
            try: result.append(operation())
            except BaseException as error: errors.append(error)
            finally: self.store.release_thread_reader()
        worker=threading.Thread(target=run)
        worker.start();worker.join(2)
        self.assertFalse(worker.is_alive(),'preparation blocks foreign writer')
        self.assertEqual(errors,[])
        return result[0]

    def test_P2B210_01_eight_rows_one_read_transaction(self):
        dependency=self.append({'dependency':True})
        reader=self.store._thread_reader()['connection']
        traced=[];reader.set_trace_callback(traced.append)
        try:
            with self.store.prepared_batch() as pending:
                values=[self.staged(pending,{'i':i},dependencies=[dependency['snapshot_id']]) for i in range(8)]
            self.assertFalse(reader.in_transaction)
        finally: reader.set_trace_callback(None)
        self.assertEqual(traced.count('BEGIN'),1)
        self.assertEqual(traced.count('COMMIT'),1)
        self.assertEqual(self.store.records('FEATURE'),[dependency,*values])

    def test_P2B210_02_read_closed_before_writer_acquisition(self):
        reader=self.store._thread_reader()['connection']
        original=self.store._lock.acquire
        def acquire():
            self.assertFalse(reader.in_transaction,'preparation pins WAL while waiting for writer')
            return original()
        with patch.object(self.store._lock,'acquire',acquire):
            with self.store.prepared_batch() as pending:
                self.staged(pending,{'probe':True})
                self.assertTrue(reader.in_transaction)

    def test_P2B210_02_empty_group_releases_read(self):
        reader=self.store._thread_reader()['connection']
        with self.store.prepared_batch():
            self.assertTrue(reader.in_transaction)
        self.assertFalse(reader.in_transaction)
        self.assertEqual(self.store.records(),[])

    def test_P2B210_02_preparation_failure_releases_read_and_writes_nothing(self):
        reader=self.store._thread_reader()['connection']
        with self.assertRaisesRegex(RuntimeError,'preparation error'):
            with self.store.prepared_batch() as pending:
                self.staged(pending,{'uncommitted':True})
                self.assertTrue(reader.in_transaction)
                raise RuntimeError('preparation error')
        self.assertFalse(reader.in_transaction)
        self.assertEqual(self.store.records(),[])
        self.assertEqual(self.append({'next':True})['data'],{'next':True})

    def test_P2B210_03_foreign_dependency_excluded_until_new_cut(self):
        with self.store.prepared_batch() as pending:
            self.staged(pending,{'pins_read':True})
            later=self.foreign(lambda:self.append({'later':True}))
            self.assert_reason('DEPENDENCY_MISSING',self.staged,pending,{'child':True},dependencies=[later['snapshot_id']])
        with self.store.prepared_batch() as pending:
            child=self.staged(pending,{'child':True},dependencies=[later['snapshot_id']])
        self.assertEqual(self.store.get(child['snapshot_id']),child)

    def test_P2B210_03_candle_guard_rechecks_foreign_revision(self):
        self.bars()
        previous=self.store.records('CANDLES')
        with self.assertRaises(Exception) as caught:
            with self.store.prepared_batch() as pending:
                self.assertEqual(len(pending.candle_index(NOW,'AAAUSDT')),3)
                self.staged(pending,{'guarded':True})
                changed=self.foreign(lambda:self.bars([candle(NOW-60000,'104')]))
                self.assertEqual(len(pending.candle_index(NOW,'AAAUSDT')),3)
        self.assertEqual(caught.exception.reason,'CANDLE_SOURCE_CHANGED')
        self.assertEqual(self.store.records('FEATURE'),[])
        self.assertEqual(self.store.records('CANDLES'),[*previous,changed])
        self.assertFalse(self.store._thread_reader()['connection'].in_transaction)

    def test_P2B210_04_caller_pin_preserved_without_extra_transaction(self):
        original=self.append({'before':True})
        reader=self.store._thread_reader()['connection']
        traced=[];reader.set_trace_callback(traced.append)
        try:
            with self.store._reading() as connection:
                self.assertEqual(connection.execute('SELECT COUNT(*) FROM snapshots').fetchone()[0],1)
                with self.store.prepared_batch() as pending:
                    self.staged(pending,{'prepared':True})
                    self.foreign(lambda:self.append({'foreign':True}))
                self.assertTrue(connection.in_transaction)
                self.assertEqual(self.store.records('FEATURE'),[original])
            self.assertFalse(reader.in_transaction)
        finally:reader.set_trace_callback(None)
        self.assertEqual(traced.count('BEGIN'),1)
        self.assertEqual(traced.count('COMMIT'),1)
        self.assertEqual(len(self.store.records('FEATURE')),3)

    def test_P2B210_02_writer_failure_does_not_leak_read_transaction(self):
        self.store._con.execute("CREATE TEMP TRIGGER refuse BEFORE INSERT ON snapshots "
            "WHEN NEW.kind='FEATURE' BEGIN SELECT RAISE(ABORT,'test writer failure'); END")
        with self.assertRaises(Exception):
            with self.store.prepared_batch() as pending:self.staged(pending,{'fails':True})
        self.assertFalse(self.store._thread_reader()['connection'].in_transaction)
        self.assertFalse(self.store._con.in_transaction)
        self.assertEqual(self.store.records(),[])

    def test_P2B210_05_non_lifo_preparation_release_preserves_other_cut(self):
        reader=self.store._thread_reader()['connection']
        first=self.store.prepared_batch();first.__enter__()
        self.staged(first,{'first':True})
        second=self.store.prepared_batch();second.__enter__()
        try:
            self.staged(second,{'second':True})
            first.__exit__(None,None,None)
            self.assertTrue(reader.in_transaction,'first release closes another active preparation cut')
            later=self.foreign(lambda:self.append({'foreign':True}))
            self.assert_reason('DEPENDENCY_MISSING',self.staged,second,{'child':True},dependencies=[later['snapshot_id']])
        finally:
            if first.active:first.__exit__(RuntimeError,RuntimeError('cleanup'),None)
            second.__exit__(None,None,None)
        self.assertFalse(reader.in_transaction)
        self.assertEqual(len(self.store.records('FEATURE')),3)

    def test_P2B210_05_failed_lease_rolls_back_on_last_release(self):
        reader=self.store._thread_reader()['connection']
        traced=[];reader.set_trace_callback(traced.append)
        first=self.store.prepared_batch();first.__enter__()
        self.staged(first,{'discarded':True})
        second=self.store.prepared_batch();second.__enter__()
        try:
            first.__exit__(RuntimeError,RuntimeError('abort preparation'),None)
            self.assertTrue(reader.in_transaction,'failed lease closes a shared read cut early')
        finally:
            if first.active:first.__exit__(RuntimeError,RuntimeError('cleanup'),None)
            second.__exit__(None,None,None)
            reader.set_trace_callback(None)
        self.assertFalse(reader.in_transaction)
        self.assertEqual(traced.count('BEGIN'),1)
        self.assertEqual(traced.count('ROLLBACK'),1)
        self.assertEqual(traced.count('COMMIT'),0)
        self.assertEqual(self.store.records(),[])
