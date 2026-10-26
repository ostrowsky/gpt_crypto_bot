"""AUD-B2 spec-first decisions and frozen-prefix coverage; synthetic only."""
import importlib
import unittest
import threading
from types import SimpleNamespace
from unittest.mock import patch

from _p2_support import Fixture, NOW, metadata, pair, book
from _p1_support import FixtureEnvironment
from test_market_runtime import FakeSupervisor
from binance_demo_bot.adapter import AdapterError
from binance_demo_bot.interaction_audit import AuditJournal


class RuntimeAuditTests(Fixture,unittest.TestCase):
    def setUp(self):
        self.api=importlib.import_module('binance_demo_bot.operational_audit')
        self.setup_market()
        self.audit=AuditJournal(self.paths,account_alias='demo-one',episode_id='episode-one',scope_id='whole-account')
        self.addCleanup(self.audit.close)
        self.client.client.audit=self.audit

    def rows(self):return [r for r in self.audit.read(limit=1000) if r['kind']=='DECISION']

    def helper(self):return self.api.OperationalAudit(self.audit,self.clock,self.store,component='MARKET_RUNTIME')

    def runtime(self):
        module=importlib.import_module('binance_demo_bot.market_runtime')
        p=patch.object(module,'MarketStreamSupervisor',FakeSupervisor);p.start();self.addCleanup(p.stop)
        runtime=module.MarketDataRuntime(self.store,self.client,checkpoint_background=False)
        self.addCleanup(runtime.stop)
        return runtime

    def started(self,rows=None):
        result=self.runtime();self.replies.append(metadata(rows));result.start();return result

    def test_b201_helper_counts_no_action_and_causal_store_reference(self):
        universe=self.refresh();helper=self.helper()
        identifier=helper.decide('BACKFILL','NO_ACTION','HISTORY_READY',source_ids=[universe['snapshot_id']],
            population=['AAAUSDT'],context={'observed_n':3})
        row=self.rows()[-1];data=row['data']
        self.assertEqual(row['event_id'],identifier)
        self.assertEqual(data['component'],'MARKET_RUNTIME')
        self.assertEqual(data['sources'],[dict(snapshot_id=universe['snapshot_id'],available_at_ms=universe['available_at_ms'])])
        self.assertEqual(data['decision_type'],'OPERATIONAL');self.assertIsNone(data['model_version'])
        self.assertEqual(data['candidate_population'],['AAAUSDT'])
        self.assertEqual(helper.health()['expected_N'],1);self.assertEqual(helper.health()['recorded_n'],1)

    def test_b202_future_or_missing_real_source_is_denied(self):
        helper=self.helper()
        future=self.store.append('FEATURE',{'x':1},received_at_ms=self.wall+1,received_monotonic_ns=self.mono)
        for source,reason in ((future['snapshot_id'],'AUDIT_FUTURE_SOURCE'),('missing','AUDIT_SOURCE_INVALID')):
            with self.assertRaises(AdapterError) as caught:
                helper.decide('BACKFILL','BACKFILL','HISTORY_NOT_READY',source_ids=[source],population=['AAAUSDT'])
            self.assertEqual(caught.exception.reason,reason)
        self.assertEqual(self.rows(),[])

    def test_b202_foreign_namespace_denied_at_constructor(self):
        self.audit.identity=dict(self.audit.identity,account_alias='foreign')
        try:
            with self.assertRaises(AdapterError) as caught:self.helper()
            self.assertEqual(caught.exception.reason,'AUDIT_IDENTITY_MISMATCH')
        finally:self.audit.identity=dict(self.audit.identity,account_alias='demo-one')

    def test_b201_start_metadata_and_not_due_wait_are_logged(self):
        runtime=self.started()
        runtime._quote_due_ns=self.mono+1000000000
        self.bars()
        runtime.tick()
        decisions=self.rows();reasons={r['data']['reasons'][0] for r in decisions}
        self.assertTrue({'START_REQUESTED','METADATA_NOT_DUE','CLOCK_SAMPLE_FRESH','QUOTES_NOT_DUE','HISTORY_READY'}<=reasons)
        initial=next(r['data'] for r in decisions if r['data']['reasons']==['START_REQUESTED'])
        self.assertEqual(initial['population_state'],'NOT_DISCOVERED')
        history=next(r['data'] for r in decisions if r['data']['reasons']==['HISTORY_READY'])
        self.assertEqual(history['candidate_population'],['AAAUSDT'])
        self.assertTrue(any(self.store.get(s['snapshot_id'])['kind']=='CANDLES' for s in history['sources']))

    def test_b201_backfill_budget_and_cooldown_record_population(self):
        runtime=self.started([pair(),pair('BBB'),pair('CCC')])
        self.replies.extend([[book()],[],[]]);runtime.tick()
        data=[r['data'] for r in self.rows()]
        backfills=[r for r in data if r['action']=='BACKFILL']
        self.assertEqual([r['candidate_population'] for r in backfills],[['AAAUSDT'],['BBBUSDT']])
        self.assertTrue(any(r['reasons']==['BACKFILL_TICK_BUDGET'] for r in data))
        self.replies.append([]);runtime.tick()
        self.assertTrue(any(r['data']['reasons']==['BACKFILL_COOLDOWN'] for r in self.rows()))

    def test_b203_start_decision_abort_prevents_any_network_or_writer(self):
        runtime=self.runtime()
        self.audit._con.execute("CREATE TRIGGER fail_decision BEFORE INSERT ON records WHEN NEW.kind='DECISION' BEGIN SELECT RAISE(ABORT,'private'); END")
        self.assert_reason('AUDIT_WRITE_FAILED',runtime.start)
        self.assertEqual(self.calls,[]);self.assertFalse(runtime.running)
        self.assertEqual(runtime.health()['pipeline_state'],'AUDIT_FAILED')
        self.assertNotEqual(self.store.group_writer_health()['state'],'READY')

    def test_b203_quote_decision_is_durable_before_side_effect(self):
        runtime=self.started();seen=[]
        def poll(source):
            seen.append(self.rows()[-1]['data']['operation'])
            raise AdapterError('API_REJECTED')
        with patch.object(runtime.service,'_poll_quotes',side_effect=poll):
            with self.assertRaises(AdapterError):runtime.tick()
        self.assertEqual(seen,['QUOTE_REFRESH'])

    def test_b203_tick_failure_stops_owned_supervisor_and_writer(self):
        runtime=self.started();supervisor=runtime.supervisor
        self.audit._con.execute("CREATE TRIGGER fail_tick BEFORE INSERT ON records WHEN NEW.kind='DECISION' BEGIN SELECT RAISE(ABORT,'bad'); END")
        self.assert_reason('AUDIT_WRITE_FAILED',runtime.tick)
        self.assertFalse(supervisor.running);self.assertFalse(runtime.running)
        self.assertEqual(runtime.health()['pipeline_state'],'AUDIT_FAILED')
        self.assertNotEqual(self.store.group_writer_health()['state'],'READY')
        self.assertEqual(runtime.health()['audit_error'],'AUDIT_WRITE_FAILED')

    def test_b206_health_is_read_only_and_unconfigured_is_explicit(self):
        runtime=self.started();before=self.audit.manifest()
        first=runtime.health();runtime.health()
        self.assertEqual(self.audit.manifest(),before)
        self.assertEqual(first['audit']['component'],'MARKET_RUNTIME')
        helper=self.api.OperationalAudit(None,self.clock,self.store,component='MARKET_RUNTIME')
        helper.decide('START','WAIT','NOT_STARTED')
        self.assertEqual(helper.health()['state'],'UNCONFIGURED')
        self.assertFalse(helper.health()['full_audit_complete'])

    def test_b201_supervisor_connect_quota_and_backoff_decisions(self):
        from binance_demo_bot.market_stream import MarketStreamSupervisor
        from test_market_stream import FakeStream
        universe=self.refresh()
        supervisor=MarketStreamSupervisor(self.store,self.client,universe,transport_factory=FakeStream,background=False)
        self.addCleanup(supervisor.stop)
        supervisor.attempts.extend([self.mono]*20);supervisor.start()
        self.assertTrue(any(r['data']['reasons']==['CONNECT_QUOTA'] for r in self.rows()))
        self.assertEqual(supervisor.readers,{})
        supervisor.attempts.clear()
        for slot in supervisor.slots.values():slot['next_ns']=self.mono+1000000000
        supervisor.tick()
        self.assertTrue(any(r['data']['reasons']==['RECONNECT_BACKOFF'] for r in self.rows()))

    def test_b203_supervisor_failed_decision_never_creates_transport(self):
        from binance_demo_bot.market_stream import MarketStreamSupervisor
        universe=self.refresh();created=[]
        def factory():created.append(True);raise AssertionError('unexpected transport')
        supervisor=MarketStreamSupervisor(self.store,self.client,universe,transport_factory=factory,background=False)
        self.addCleanup(supervisor.stop)
        self.audit._con.execute("CREATE TRIGGER fail_supervisor BEFORE INSERT ON records WHEN NEW.kind='DECISION' BEGIN SELECT RAISE(ABORT,'bad'); END")
        self.assert_reason('AUDIT_WRITE_FAILED',supervisor.start)
        self.assertEqual(created,[]);self.assertFalse(supervisor.running)

    def test_b203_background_does_not_mask_audit_failure(self):
        runtime=self.started()
        with patch.object(runtime._stop,'wait',return_value=False),patch.object(runtime,'tick',side_effect=AdapterError('AUDIT_WRITE_FAILED')):
            runtime._run()
        self.assertEqual(runtime.pipeline_state,'AUDIT_FAILED')
        self.assertFalse(runtime.running);self.assertFalse(runtime.supervisor.running)

    def test_b203_rest_audit_terminal_failure_aborts_foreground_runtime(self):
        runtime=self.started();self.replies.append([book()])
        self.audit._con.execute("CREATE TRIGGER fail_rest_end BEFORE INSERT ON records WHEN NEW.kind='REST_END' BEGIN SELECT RAISE(ABORT,'bad'); END")
        self.assert_reason('AUDIT_WRITE_FAILED',runtime.tick)
        self.assertFalse(runtime.running);self.assertFalse(runtime.supervisor.running)
        self.assertEqual(runtime.health()['pipeline_state'],'AUDIT_FAILED')
        self.assertEqual(runtime.health()['audit_error'],'AUDIT_WRITE_FAILED')

    def test_b206_external_journal_failure_is_not_healthy_before_next_tick(self):
        runtime=self.started()
        self.audit._con.execute("CREATE TRIGGER fail_probe BEFORE INSERT ON records WHEN NEW.kind='DECISION' BEGIN SELECT RAISE(ABORT,'bad'); END")
        self.assert_reason('AUDIT_WRITE_FAILED',self.audit.append,'DECISION',{'action':'WAIT'})
        self.assertEqual(runtime.health()['pipeline_state'],'AUDIT_FAILED')

    def test_b202_source_read_failure_has_safe_reason_and_independent_gap(self):
        from binance_demo_bot import market_sqlite
        helper=self.helper()
        with patch.object(self.store,'get',side_effect=market_sqlite.OperationalError('secret SQL text')):
            self.assert_reason('AUDIT_SOURCE_UNAVAILABLE',helper.decide,'REFRESH','REFRESH','DATA_DUE',source_ids=['source'])
        self.assertEqual(helper.health()['gap_n'],1)

    def test_b201_connected_supervisor_l2_ready_and_buffer_wait_are_logged(self):
        from binance_demo_bot.market_stream import MarketStreamSupervisor
        universe=self.refresh()
        supervisor=MarketStreamSupervisor(self.store,self.client,universe,['AAAUSDT'],transport_factory=lambda:None,background=False)
        key=supervisor.plan['shards'][0]['shard_id']
        view={'state':'READY','last_update_id':1,'buffer_n':0}
        reader=SimpleNamespace(connected=True,started_ns=self.mono,books={'AAAUSDT':SimpleNamespace(view=lambda:dict(view))},
            health=lambda:{'needs_reconnect':False},stop=lambda:None)
        supervisor.readers[key]=reader;supervisor.running=True
        self.addCleanup(supervisor.stop)
        supervisor.tick()
        self.assertTrue(any(r['data']['reasons']==['L2_READY'] for r in self.rows()))
        self.assertTrue(any(r['data']['reasons']==['SESSION_CONNECTED'] for r in self.rows()))
        view.update(state='BUFFERING',last_update_id=None)
        supervisor.tick()
        self.assertTrue(any(r['data']['reasons']==['L2_DIFF_REQUIRED'] for r in self.rows()))

    def test_b203_abort_under_runtime_lock_does_not_skip_owned_cleanup(self):
        runtime=self.started();entered=threading.Event()
        def coordinator():
            entered.set()
            with runtime._lock:pass
        worker=threading.Thread(target=coordinator,daemon=True)
        runtime._thread=worker
        original_join=worker.join
        try:
            with runtime._lock:
                worker.start();self.assertTrue(entered.wait(1))
                self.audit._con.execute("CREATE TRIGGER fail_locked BEFORE INSERT ON records WHEN NEW.kind='DECISION' BEGIN SELECT RAISE(ABORT,'bad'); END")
                # Arrange a short real join deadline; never fake worker completion.
                with patch.object(worker,'join',side_effect=lambda timeout=None:original_join(min(timeout or 1,.05))):
                    self.assert_reason('AUDIT_WRITE_FAILED',runtime.tick)
                self.assertFalse(runtime.supervisor.running)
                self.assertNotEqual(self.store.group_writer_health()['state'],'READY')
        finally:
            original_join(2)
            self.assertFalse(worker.is_alive())

    def test_b203_supervisor_abort_under_lock_still_stops_readers(self):
        from binance_demo_bot.market_stream import MarketStreamSupervisor
        universe=self.refresh();entered=threading.Event();stopped=[]
        supervisor=MarketStreamSupervisor(self.store,self.client,universe,transport_factory=lambda:None,background=False)
        self.addCleanup(supervisor.stop)
        key=supervisor.plan['shards'][0]['shard_id']
        supervisor.readers[key]=SimpleNamespace(connected=False,session_id='old',stop=lambda:stopped.append(True))
        supervisor.running=True
        def coordinator():
            entered.set()
            with supervisor._lock:pass
        worker=threading.Thread(target=coordinator,daemon=True);supervisor._thread=worker
        original_join=worker.join
        try:
            with supervisor._lock:
                worker.start();self.assertTrue(entered.wait(1))
                self.audit._con.execute("CREATE TRIGGER fail_locked_supervisor BEFORE INSERT ON records WHEN NEW.kind='DECISION' BEGIN SELECT RAISE(ABORT,'bad'); END")
                with patch.object(worker,'join',side_effect=lambda timeout=None:original_join(min(timeout or 1,.05))):
                    self.assert_reason('AUDIT_WRITE_FAILED',supervisor.tick)
                self.assertTrue(stopped)
        finally:
            original_join(2);self.assertFalse(worker.is_alive())


class ProcessingReportTests(Fixture,unittest.TestCase):
    def setUp(self):
        self.setup_market()
        self.audit=AuditJournal(self.paths,account_alias='demo-one',episode_id='episode-one',scope_id='whole-account')
        self.addCleanup(self.audit.close)

    def receipt(self,connection='c1'):
        parent=self.audit.append('WS_BEGIN',dict(operation='RECV',operation_id='op'+str(len(self.audit.read(limit=1000))),connection_id=connection))
        begin=self.audit.read(limit=1000)[-1]
        return self.audit.append('WS_END',dict(operation='RECV',operation_id=begin['data']['operation_id'],
            parent_event_id=parent,connection_id=connection,outcome='COMPLETED',payload={'representation':'OMITTED'}))

    def marker(self,identifier,outcome='PROCESSED',connection='c1'):
        return self.audit.append('WS_PROCESSING',dict(receipt_event_ids=[identifier],connection_id=connection,outcome=outcome,snapshot_ids=[]))

    def test_b205_zero_is_unknown_not_perfect_coverage(self):
        report=self.audit.processing_report()
        self.assertEqual(report['recorded_receipts_N'],0);self.assertEqual(report['state'],'UNKNOWN')
        self.assertEqual(report['capture_coverage'],'UNKNOWN');self.assertFalse(report['full_audit_complete'])

    def test_b205_exact_partition_and_frozen_prefix(self):
        ids=[self.receipt() for _ in range(4)]
        self.marker(ids[0]);self.marker(ids[1],'REJECTED');self.marker(ids[2],'CONTROL_ACK')
        cutoff=self.audit.manifest()['last_seq'];before=self.audit.processing_report(through_seq=cutoff)
        self.assertEqual([before[k] for k in ('recorded_receipts_N','processed_n','rejected_n','control_ack_n','unknown_n','conflict_n')],[4,1,1,1,1,0])
        self.marker(ids[3])
        self.assertEqual(self.audit.processing_report(through_seq=cutoff),before)
        after=self.audit.processing_report()
        self.assertEqual(after['unknown_n'],0);self.assertEqual(after['state'],'RESOLVED_RECORDED_PREFIX')
        self.assertEqual(after['denominator_basis'],'DURABLE_COMPLETED_RECV')
        self.assertEqual(after['capture_coverage'],'UNKNOWN')

    def test_b205_duplicate_foreign_or_unavailable_reference_is_failure(self):
        identifier=self.receipt();self.marker(identifier);self.marker(identifier,'REJECTED')
        self.marker('missing');self.marker(identifier,connection='foreign')
        report=self.audit.processing_report()
        self.assertEqual(report['state'],'FAIL');self.assertEqual(report['conflict_n'],1)
        self.assertGreaterEqual(report['invalid_marker_n'],1)
        self.assertEqual(report['processed_n'],0)

    def test_b205_prefix_survives_reopen_and_rejects_future_cutoff(self):
        identifier=self.receipt();self.marker(identifier)
        cutoff=self.audit.manifest()['last_seq'];before=self.audit.processing_report(through_seq=cutoff)
        self.audit.close()
        reopened=AuditJournal(self.paths,account_alias='demo-one',episode_id='episode-one',scope_id='whole-account')
        self.addCleanup(reopened.close)
        self.assertEqual(reopened.processing_report(through_seq=cutoff),before)
        with self.assertRaises(AdapterError):reopened.processing_report(through_seq=1000000)

    def test_b205_non_array_marker_cannot_manufacture_success(self):
        identifier=self.receipt()
        self.audit.append('WS_PROCESSING',dict(receipt_event_ids=identifier,connection_id='c1',outcome='PROCESSED'))
        report=self.audit.processing_report()
        self.assertEqual(report['state'],'FAIL');self.assertGreater(report['invalid_marker_n'],0)
        self.assertEqual(report['processed_n'],0);self.assertEqual(report['conflict_n'],1)


class ReaderDispositionTests(Fixture,unittest.TestCase):
    def setUp(self):
        self.setup_market()
        self.audit=AuditJournal(self.paths,account_alias='demo-one',episode_id='episode-one',scope_id='whole-account')
        self.addCleanup(self.audit.close)
        universe=self.refresh();self.client.client.audit=self.audit
        from binance_demo_bot.market_stream import MarketStreamReader
        from test_ws_audit import RawStream
        self.raw=RawStream()
        self.reader=MarketStreamReader(self.store,self.client,universe,['aaausdt@kline_1m'],transport=self.raw,background=False)
        self.addCleanup(self.reader.stop)
        self.reader.start()

    def test_b204_ack_open_and_malformed_have_distinct_dispositions(self):
        from test_market_runtime import kline
        self.raw.frames.append(dict(stream='aaausdt@kline_1m',data=kline(False)))
        self.assertTrue(self.reader.read_once())
        self.raw.frames.append(b'{');self.assertFalse(self.reader.read_once())
        report=self.audit.processing_report()
        self.assertEqual([report[k] for k in ('recorded_receipts_N','processed_n','rejected_n','control_ack_n','unknown_n')],[3,1,1,1,0])

    def test_b204_marker_abort_remains_unknown_not_rejected(self):
        from test_market_runtime import kline
        self.audit._con.execute("CREATE TRIGGER fail_marker BEFORE INSERT ON records WHEN NEW.kind='WS_PROCESSING' BEGIN SELECT RAISE(ABORT,'bad'); END")
        self.raw.frames.append(dict(stream='aaausdt@kline_1m',data=kline(False)))
        self.assertFalse(self.reader.read_once())
        report=self.audit.processing_report()
        self.assertEqual(report['unknown_n'],1);self.assertEqual(report['rejected_n'],0)
        self.assertEqual(self.audit.health()['channels']['WS_PROCESSING']['gap_n'],1)


class PrivateDispositionTests(FixtureEnvironment,unittest.TestCase):
    def setUp(self):
        from test_ws_audit import PrivateWsAuditTests
        # Reuse its synthetic environment/lifecycle, without inheriting its tests.
        PrivateWsAuditTests.setUp(self)

    def test_b204_wrong_subscription_is_rejected_with_known_receipt(self):
        from _p1_support import execution
        self.reader.start();self.raw.queue.append({'subscriptionId':99,'event':execution()})
        with self.assertRaises(AdapterError):self.reader.poll_once()
        report=self.audit.processing_report()
        self.assertEqual([report[k] for k in ('recorded_receipts_N','rejected_n','control_ack_n','unknown_n')],[2,1,1,0])

    def test_b204_invalid_buffered_event_before_ack_is_rejected(self):
        from _p1_support import execution
        self.raw.queue.append({'subscriptionId':99,'event':execution()})
        with self.assertRaises(AdapterError):self.reader.start()
        report=self.audit.processing_report()
        self.assertEqual(report['rejected_n'],1);self.assertEqual(report['unknown_n'],0)

    def test_b204_invalid_subscription_id_in_ack_is_rejected(self):
        self.raw.ack={'status':200,'result':{'subscriptionId':'invalid'}}
        with self.assertRaises(AdapterError):self.reader.start()
        report=self.audit.processing_report()
        self.assertEqual(report['rejected_n'],1);self.assertEqual(report['unknown_n'],0)
