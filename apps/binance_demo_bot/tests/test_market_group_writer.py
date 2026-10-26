"""P2B2-12 spec-first real SQLite/worker ACK and bounded admission contracts."""
import importlib
import threading
import unittest
from unittest.mock import patch
from _p2_support import Fixture, NOW, metadata, candle
from test_market_runtime import FakeSupervisor
from test_market_stream import FakeStream,tick
from test_market_delivery_latency import RawStream
import json


class GroupWriterTests(Fixture,unittest.TestCase):
    def setUp(self):
        self.setup_market()
        self.workers=[]
        self.addCleanup(self.join_all)

    def join_all(self):
        for worker,_,_,_ in self.workers:
            worker.join(3)
            self.assertFalse(worker.is_alive(),'own submission worker leaked')

    def start(self,**options): return self.store.start_group_writer(**options)

    def producer(self,index,rows=1,prepare=None):
        results,errors=[],[]
        done=threading.Event()
        def run():
            try:
                with self.store.prepared_batch(monotonic_ns=lambda:self.mono) as pending:
                    if prepare: prepare(pending)
                    values=[pending.append('FEATURE',{'group':index,'i':i},received_at_ms=NOW,
                        received_monotonic_ns=self.mono) for i in range(rows)]
                results.append((values,pending.timing))
            except BaseException as error:errors.append(error)
            finally:self.store.release_thread_reader();done.set()
        worker=threading.Thread(target=run);self.workers.append((worker,done,results,errors));worker.start()
        return worker,done,results,errors

    def admitted(self,service,count):
        with service._condition:
            self.assertTrue(service._condition.wait_for(lambda:service.health()['accepted_groups_n']>=count,timeout=2),'submission was not admitted')

    def paused(self,**options):
        service=self.start(**options)
        entered,released=threading.Event(),threading.Event()
        original=service._take_batch
        def paused():
            entered.set()
            self.assertTrue(released.wait(3),'test gate not released')
            return original()
        # Finish current idle wait, then hold the first requested drain.
        patcher=patch.object(service,'_take_batch',paused)
        patcher.start();self.addCleanup(patcher.stop);self.addCleanup(released.set)
        return service,released

    def finish(self,submissions):
        for worker,done,_,_ in submissions:
            self.assertTrue(done.wait(3),'commit ACK waiter did not resolve');worker.join(1)

    def test_GW01_ack_after_real_commit_not_insert(self):
        self.start();entered,released=threading.Event(),threading.Event()
        original=self.store._con.commit
        def held():entered.set();self.assertTrue(released.wait(3));return original()
        with patch.object(self.store._con,'commit',held):
            submission=self.producer(0)
            try:
                self.assertTrue(entered.wait(2))
                self.assertFalse(submission[1].is_set())
                self.assertEqual(self.store.records('FEATURE'),[])
            finally:released.set();self.finish([submission])
        self.assertEqual(submission[3],[])
        self.assertEqual(submission[2][0][1]['state'],'COMMITTED')

    def test_GW02_six_groups_share_one_commit_and_bulk_insert(self):
        service,released=self.paused()
        # Gate the writer lock so every accepted group is waiting before collection.
        with self.store._lock:
            submissions=[]
            for i in range(6):submissions.append(self.producer(i,8));self.admitted(service,i+1)
            traced=[];self.store._con.set_trace_callback(traced.append)
            released.set()
        try:self.finish(submissions)
        finally:self.store._con.set_trace_callback(None)
        self.assertTrue(all(not s[3] for s in submissions))
        self.assertEqual(traced.count('COMMIT'),1)
        self.assertEqual(sum(sql.startswith('INSERT OR IGNORE INTO snapshots') for sql in traced),1)
        self.assertEqual([r['data']['group'] for r in self.store.records('FEATURE')],[i for i in range(6) for _ in range(8)])
        self.assertEqual(service.health()['committed_groups_n'],6)

    def test_GW01_queue_capacity_rejects_without_silent_loss(self):
        service=self.start(capacity=2)
        with self.store._lock:
            first=self.producer(0);self.admitted(service,1)
            second=self.producer(1);self.admitted(service,2)
            self.assert_reason('MARKET_WRITER_BACKPRESSURE',self.store.append,'FEATURE',{'overflow':True},received_at_ms=NOW,received_monotonic_ns=self.mono)
        self.finish([first,second])
        self.assertEqual(len(self.store.records('FEATURE')),2)
        health=service.health();self.assertEqual(health['rejected_groups_n'],1)
        self.assertEqual(health['pending_groups_n'],0)
        self.assertEqual(health['submitted_groups_n'],health['accepted_groups_n']+health['rejected_groups_n'])

    def test_GW01_byte_bound_and_detached_health(self):
        service=self.start(byte_limit=1024)
        self.assert_reason('MARKET_WRITER_BACKPRESSURE',self.store.append,'FEATURE',{'large':'x'*2048},received_at_ms=NOW,received_monotonic_ns=self.mono)
        health=service.health();health['pending_bytes_n']=-1
        self.assertEqual(service.health()['pending_bytes_n'],0)
        self.assertEqual(self.store.records('FEATURE'),[])
        self.assertNotIn('large',str(service.health()))

    def test_GW01_config_validation(self):
        for options in ({'capacity':True},{'capacity':9},{'byte_limit':1023},{'byte_limit':True},{'collect_ms':3},{'collect_ms':0.5}):
            self.assert_reason('INVALID_PARAMETERS',self.start,**options)

    def test_GW02_large_head_chunks_and_retry_preserve_primary(self):
        self.start();traced=[];self.store._con.set_trace_callback(traced.append)
        submission=self.producer(0,65);self.finish([submission]);self.store._con.set_trace_callback(None)
        self.assertEqual(submission[3],[])
        self.assertEqual(traced.count('COMMIT'),1)
        self.assertEqual(sum(sql.startswith('INSERT OR IGNORE INTO snapshots') for sql in traced),2)
        value=submission[2][0][0][0]
        saved=self.store.append('FEATURE',value['data'],received_at_ms=NOW,received_monotonic_ns=self.mono)
        self.assertEqual(saved,value)
        self.assertEqual(self.store.sqlite_settings()['synchronous'],2)

    def test_GW03_constraint_fallback_isolates_bad_group(self):
        service=self.start()
        self.store._con.execute("CREATE TEMP TRIGGER bad BEFORE INSERT ON snapshots WHEN NEW.kind='FEATURE' AND json_extract(NEW.payload_json,'$.data.group')=1 BEGIN SELECT RAISE(ABORT,'private detail'); END")
        with self.store._lock:
            submissions=[]
            for i in range(3):submissions.append(self.producer(i));self.admitted(service,i+1)
        self.finish(submissions)
        self.assertEqual([r['data']['group'] for r in self.store.records('FEATURE')],[0,2])
        self.assertEqual(submissions[0][3],[]);self.assertEqual(submissions[2][3],[])
        self.assertEqual(submissions[1][3][0].reason,'MARKET_WRITER_FAILED')
        self.assertNotIn('private detail',str(service.health()))

    def test_GW03_current_candle_guard_isolates_group(self):
        self.bars();service=self.start()
        with self.store._lock:
            good=self.producer(0);self.admitted(service,1)
            stale=self.producer(1,prepare=lambda p:p.candle_index(NOW,'AAAUSDT'));self.admitted(service,2)
            with self.store.batch():self.bars([candle(NOW-60000,'104')])
            last=self.producer(2);self.admitted(service,3)
        self.finish([good,stale,last])
        self.assertEqual(stale[3][0].reason,'CANDLE_SOURCE_CHANGED')
        self.assertEqual([r['data']['group'] for r in self.store.records('FEATURE')],[0,2])

    def test_GW03_commit_unknown_no_blind_retry_or_false_ack(self):
        service=self.start();original=self.store._con.commit
        def ambiguous():original();raise RuntimeError('unconfirmed private detail')
        with patch.object(self.store._con,'commit',ambiguous):
            submission=self.producer(0);self.finish([submission])
        self.assertEqual(submission[3][0].reason,'MARKET_WRITER_COMMIT_UNKNOWN')
        self.assertEqual(len(self.store.records('FEATURE')),1)
        health=service.health();self.assertEqual(health['committed_groups_n'],0)
        self.assertEqual(health['unconfirmed_groups_n'],1)
        self.assertEqual(health['state'],'FAILED')
        self.assert_reason('MARKET_WRITER_UNAVAILABLE',self.store.append,'FEATURE',{},received_at_ms=NOW,received_monotonic_ns=self.mono)

    def test_GW03_later_fallback_unknown_cannot_erase_earlier_commit_ack(self):
        service=self.start();original=self.store._con.commit;calls=[]
        self.store._con.execute("CREATE TEMP TRIGGER bad BEFORE INSERT ON snapshots WHEN NEW.kind='FEATURE' AND json_extract(NEW.payload_json,'$.data.group')=1 BEGIN SELECT RAISE(ABORT,'bad group'); END")
        def commits():
            original();calls.append(1)
            if len(calls)==2:raise RuntimeError('later ACK unknown')
        with patch.object(self.store._con,'commit',commits):
            with self.store._lock:
                submissions=[]
                for i in range(3):submissions.append(self.producer(i));self.admitted(service,i+1)
            self.finish(submissions)
        self.assertEqual(submissions[0][3],[])
        self.assertEqual(submissions[1][3][0].reason,'MARKET_WRITER_FAILED')
        self.assertEqual(submissions[2][3][0].reason,'MARKET_WRITER_COMMIT_UNKNOWN')
        self.assertEqual(service.health()['committed_groups_n'],1)
        self.assertEqual(service.health()['unconfirmed_groups_n'],1)
        self.assertEqual([r['data']['group'] for r in self.store.records('FEATURE')],[0,2])

    def test_GW05_stop_drains_accepted_and_prevents_bypass(self):
        service=self.start();stopped=threading.Event()
        with self.store._lock:
            submission=self.producer(0);self.admitted(service,1)
            stopper=threading.Thread(target=lambda:(self.store.stop_group_writer(),stopped.set()))
            stopper.start()
            with service._condition:self.assertTrue(service._condition.wait_for(lambda:not service.accepting,timeout=2))
            self.assertFalse(stopped.is_set())
            self.assert_reason('MARKET_WRITER_UNAVAILABLE',self.producer_inline)
        self.finish([submission]);stopper.join(3);self.assertTrue(stopped.is_set())
        self.assertFalse(service._thread.is_alive());self.assertEqual(len(self.store.records('FEATURE')),1)
        self.store.start_group_writer();self.producer_inline()

    def producer_inline(self):
        return self.store.append('FEATURE',{'inline':True},received_at_ms=NOW,received_monotonic_ns=self.mono)

    def test_GW01_caller_read_pin_and_exact_retry_survive_ack(self):
        original=self.producer_inline();self.start()
        with self.store._reading():
            self.assertEqual(self.store.records('FEATURE'),[original])
            added=self.store.append('FEATURE',{'later':True},received_at_ms=NOW+1,received_monotonic_ns=self.mono+1)
            retry=self.store.append('FEATURE',{'later':True},received_at_ms=NOW+1,received_monotonic_ns=self.mono+1)
            self.assertEqual(added,retry);self.assertEqual(self.store.records('FEATURE'),[original])

    def test_GW05_thread_start_failure_cleans_owned_service(self):
        with patch.object(threading.Thread,'start',side_effect=RuntimeError('start failed')):
            with self.assertRaises(Exception):self.start()
        self.assertFalse(self.store.group_writer_health()['running'])
        self.store.close()

    def test_GW01_row_bound_and_empty_group_not_enqueued(self):
        service=self.start()
        with self.store.prepared_batch():pass
        self.assertEqual(service.health()['submitted_groups_n'],0)
        submission=self.producer(0,257);self.finish([submission])
        self.assertEqual(submission[3][0].reason,'MARKET_WRITER_BACKPRESSURE')
        self.assertEqual(self.store.records('FEATURE'),[])

    def test_GW05_unexpected_worker_failure_wakes_waiter(self):
        service=self.start()
        with patch.object(service,'_take_batch',side_effect=RuntimeError('private crash')):
            submission=self.producer(0);self.finish([submission])
        self.assertEqual(submission[3][0].reason,'MARKET_WRITER_UNAVAILABLE')
        self.assertEqual(service.health()['state'],'FAILED')
        self.assertEqual(service.health()['pending_groups_n'],0)
        self.assertEqual(self.store.records(),[])

    def test_GW05_shutdown_timeout_keeps_database_and_lease_open(self):
        service=self.start()
        with self.store._lock:
            submission=self.producer(0);self.admitted(service,1)
            with patch.object(service._thread,'join',return_value=None):
                self.assert_reason('MARKET_WRITER_STOP_TIMEOUT',self.store.close)
            self.assertIsNotNone(self.store._con)
            self.assertFalse(self.store._lease.handle.closed)
        self.finish([submission]);self.store.stop_group_writer()

    def test_GW05_stop_never_clears_sticky_failure(self):
        service=self.start()
        with service._condition:service.state='FAILED';service.accepting=False
        self.store.stop_group_writer()
        self.assertEqual(service.health()['state'],'FAILED')
        self.assert_reason('MARKET_WRITER_UNAVAILABLE',self.store.start_group_writer)

    def test_GW05_diagnostic_clock_failure_cannot_strand_waiter(self):
        service=self.start();done=threading.Event();errors=[]
        def clock():
            if threading.current_thread().name=='binance-demo-market-writer':raise RuntimeError('diagnostic clock failed')
            return self.mono
        def run():
            try:
                with self.store.prepared_batch(monotonic_ns=clock) as pending:
                    pending.append('FEATURE',{},received_at_ms=NOW,received_monotonic_ns=self.mono)
            except BaseException as error:errors.append(error)
            finally:self.store.release_thread_reader();done.set()
        worker=threading.Thread(target=run,daemon=True);worker.start()
        try:
            self.assertTrue(done.wait(1),'writer clock failure stranded accepted request')
            self.assertEqual(errors[0].reason,'MARKET_WRITER_UNAVAILABLE')
            self.assertEqual(service.health()['pending_groups_n'],0)
        finally:
            # Only teardown a failed pre-repair probe; no passing assertion relies on it.
            for request in list(service._pending):request.error=RuntimeError('probe cleanup');request.done.set()
            worker.join(2)


class ReaderWriterEvidenceTests(Fixture,unittest.TestCase):
    def setUp(self):
        self.setup_market();self.api=importlib.import_module('binance_demo_bot.market_stream')
        deny=patch.object(self.api,'ws_connect',side_effect=AssertionError('network forbidden'))
        probe=deny.start();self.addCleanup(deny.stop);self.addCleanup(probe.assert_not_called)
        self.transport=RawStream()
        self.reader=self.api.MarketStreamReader(self.store,self.client,self.refresh(),['aaausdt@bookTicker'],transport=self.transport,background=False)
        self.addCleanup(self.reader.stop);self.reader.start()
        self.wire=json.dumps(tick());self.transport.frames.append(self.wire)

    def test_GW06_pressure_keeps_exact_frames_when_evidence_cannot_commit(self):
        service=self.store.start_group_writer(capacity=1)
        admitted=threading.Event();done=threading.Event()
        def occupies():
            try:
                with self.store.prepared_batch() as pending:
                    pending.append('FEATURE',{},received_at_ms=NOW,received_monotonic_ns=self.mono)
                    admitted.set()
            finally:self.store.release_thread_reader();done.set()
        worker=threading.Thread(target=occupies)
        with self.store._lock:
            worker.start();self.assertTrue(admitted.wait(2))
            with service._condition:self.assertTrue(service._condition.wait_for(lambda:service.health()['accepted_groups_n']==1,timeout=2))
            self.assertFalse(self.reader.read_once())
            self.assertEqual(self.reader.error,'MARKET_WRITER_BACKPRESSURE')
            self.assertEqual(self.reader._lag_evidence_buffer['frames'][0]['raw_utf8'],self.wire)
            self.assertFalse(self.reader._lag_evidence_buffer['committed'])
        self.assertTrue(done.wait(3));worker.join(1)
        self.assertTrue(self.transport.closed)

    def test_GW06_unknown_commit_retains_candidates_not_false_uncommitted_claim(self):
        self.store.start_group_writer();original=self.store._con.commit
        def ambiguous():original();raise RuntimeError('private ACK failure')
        with patch.object(self.store._con,'commit',ambiguous):self.assertFalse(self.reader.read_once())
        evidence=self.reader._lag_evidence_buffer
        self.assertIsNotNone(evidence)
        self.assertIsNone(evidence['committed'])
        self.assertEqual(evidence['operation'],'MARKET_GROUP_WRITE_EVIDENCE')
        self.assertEqual(evidence['reason'],'MARKET_WRITER_COMMIT_UNKNOWN')
        self.assertEqual(evidence['committed_snapshot_ids'],[])
        self.assertEqual(len(evidence['candidate_snapshot_ids']),1)
        self.assertIsNotNone(self.store.get(evidence['candidate_snapshot_ids'][0]))
        self.assertTrue(self.transport.closed)

    def test_GW02_shared_ack_latency_still_enforces_250ms_and_one_failure_commit(self):
        service=self.store.start_group_writer();original=self.store._con.commit
        def delayed():original();self.advance(251)
        with patch.object(self.store._con,'commit',delayed):self.assertFalse(self.reader.read_once())
        self.assertEqual(self.reader.error,'WS_PROCESSING_LAG')
        latency=self.reader.health()['latency']
        self.assertEqual(latency['attempted_groups_n'],1)
        self.assertEqual(latency['durable_groups_n'],1)
        self.assertEqual(latency['last_group']['writer_attempts_n'],1)
        self.assertEqual(service.health()['commit_acks_n'],2)
        evidence=[r for r in self.store.records('INCIDENT') if r['data'].get('operation')=='MARKET_GROUP_LAG_EVIDENCE']
        self.assertEqual(evidence[0]['data']['frames'][0]['received_at_ms'],NOW)
        self.assertTrue(evidence[0]['data']['committed'])


class RuntimeWriterTests(Fixture,unittest.TestCase):
    def setUp(self):
        self.setup_market();self.api=importlib.import_module('binance_demo_bot.market_runtime')
        patcher=patch.object(self.api,'MarketStreamSupervisor',FakeSupervisor);patcher.start();self.addCleanup(patcher.stop)

    def test_GW05_runtime_owns_writer_and_stop_failure_still_joins(self):
        runtime=self.api.MarketDataRuntime(self.store,self.client);self.addCleanup(runtime.stop)
        self.replies.append(metadata());runtime.start()
        self.assertTrue(self.store.group_writer_health()['running'])
        with patch.object(runtime.supervisor,'stop',side_effect=RuntimeError('stop failed')):
            with self.assertRaises(RuntimeError):runtime.stop()
        self.assertFalse(self.store.group_writer_health()['running'])

    def test_GW05_optout_disabled_and_parameter_validation(self):
        runtime=self.api.MarketDataRuntime(self.store,self.client,group_writer_enabled=False)
        self.addCleanup(runtime.stop);self.replies.append(metadata());runtime.start()
        self.assertEqual(self.store.group_writer_health()['state'],'NOT_STARTED')
        runtime.stop()
        self.assert_reason('INVALID_PARAMETERS',self.api.MarketDataRuntime,self.store,self.client,group_writer_enabled=1)

    def test_GW05_coordinator_thread_start_failure_joins_storage_workers(self):
        runtime=self.api.MarketDataRuntime(self.store,self.client,background=True);self.addCleanup(runtime.stop)
        self.replies.append(metadata());original=threading.Thread.start
        def start(worker):
            if worker.name=='binance-demo-market-runtime':raise RuntimeError('coordinator start failed')
            return original(worker)
        with patch.object(threading.Thread,'start',start):
            with self.assertRaises(RuntimeError):runtime.start()
        self.assertFalse(runtime.running)
        self.assertFalse(self.store.group_writer_health()['running'])
        self.assertFalse(self.store._checkpoint.running)
