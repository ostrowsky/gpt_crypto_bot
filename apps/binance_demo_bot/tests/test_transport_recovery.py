"""TRN-01..04 actual own pilot journal; synthetic transport, no network/keys."""
import importlib,unittest
from copy import deepcopy
from decimal import Decimal
from types import SimpleNamespace
import _support
import test_usdt_service as fixture
from binance_demo_bot.adapter import AdapterError

class Reader:
    process_id='process-1'
    def __init__(self,healthy=False,reason='WS_HEARTBEAT_FAILED',stuck=False):
        self.healthy,self.reason,self.stuck=healthy,reason,stuck;self.closes=0;self.starts=0
    def health(self):return dict(connected=self.healthy,transport_healthy=self.healthy,
        reader_alive=self.healthy or self.stuck,error_reason=self.reason)
    def close(self):self.closes+=1;self.healthy=False
    def start(self):self.starts+=1;self.healthy=True;self.reason=None;return self.health()

class TransportRecoveryTests(unittest.TestCase):
    setUp=fixture.UsdtServiceTests.setUp
    def runtime(self):return importlib.import_module('binance_demo_bot.pilot_runtime')
    def broken_snapshot(self,reason='CLOCK_QUALITY'):
        original=self.exchange.snapshot
        def fail():raise AdapterError(reason)
        self.exchange.snapshot=fail;return original
    def make_clock_client(self):
        calls=[]
        def sync():calls.append('clock');return dict(server_time_ms=self.exchange.now,received_at_ms=self.exchange.now,clock_uncertainty_ms=Decimal('1'))
        self.exchange.client=SimpleNamespace(sync_clock=sync,clock=SimpleNamespace(timestamp_ms=lambda:self.exchange.now,
            monotonic_ns=lambda:self.exchange.now*1000000,wall_ms=lambda:self.exchange.now),read_open_orders=lambda:{'data':[]},read_open_order_lists=lambda:{'data':[]})
        self.exchange.reader=Reader(healthy=True,reason=None);self.exchange.evidence=object()
        return calls
    def halt(self,prior_kill=False):
        state=self.book.state();state['entry_killed']=prior_kill;self.book.save_state(state)
        state.update(phase='HALTED',recovery_phase='IDLE',entry_killed=True,reason='CLOCK_QUALITY',protection_action=None)
        self.book.save_state(state)
    def test_trn01_pre_send_clock_failure_waits_without_permanent_kill(self):
        self.broken_snapshot();state=self.service.tick()
        self.assertEqual(state['phase'],'IDLE');self.assertFalse(state['entry_killed'])
        self.assertEqual(state['data_wait_reason'],'CLOCK_QUALITY');self.assertEqual(self.exchange.calls,[])
    def test_trn01_preexisting_kill_is_never_cleared_by_clock_wait(self):
        state=self.book.state();state['entry_killed']=True;self.book.save_state(state)
        self.broken_snapshot();state=self.service.tick();self.assertTrue(state['entry_killed']);self.assertEqual(state['phase'],'IDLE')
    def test_trn01_next_good_snapshot_clears_only_data_wait(self):
        original=self.broken_snapshot();self.service.tick();self.exchange.snapshot=original
        self.exchange.stream=False
        original=self.exchange.snapshot
        def no_stream():value=original();value['stream_ok']=False;return value
        self.exchange.snapshot=no_stream;state=self.service.tick()
        self.assertNotIn('data_wait_reason',state);self.assertEqual(state['phase'],'IDLE');self.assertEqual(self.exchange.calls,[])
    def test_trn01_open_clock_failure_keeps_recovery_and_kill(self):
        self.service.tick();self.broken_snapshot();state=self.service.tick()
        self.assertEqual(state['phase'],'RECOVERY_REQUIRED');self.assertTrue(state['entry_killed'])
    def test_trn02_healthy_reader_is_not_replaced(self):
        self.make_clock_client();made=[]
        supervisor=self.runtime().PrivateReaderSupervisor(self.exchange,factory=lambda *a,**k:made.append(k))
        self.assertIs(supervisor.maintain(),self.exchange.reader);self.assertEqual(made,[])
    def test_trn02_failed_reader_is_closed_joined_and_replaced(self):
        self.make_clock_client();old=Reader();self.exchange.reader=old;made=[]
        def factory(*a,**kw):made.append(kw);return Reader()
        supervisor=self.runtime().PrivateReaderSupervisor(self.exchange,factory=factory)
        new=supervisor.maintain();self.assertEqual(old.closes,1);self.assertEqual(new.starts,1)
        self.assertIs(self.exchange.reader,new);self.assertEqual(made[0]['process_instance_id'],'process-1')
    def test_trn02_failed_reconnect_has_one_minute_backoff(self):
        self.make_clock_client();self.exchange.reader=Reader();attempts=[]
        def factory(*a,**kw):attempts.append(1);raise AdapterError('WS_UNAVAILABLE')
        supervisor=self.runtime().PrivateReaderSupervisor(self.exchange,factory=factory)
        for _ in range(2):
            with self.assertRaises(AdapterError):supervisor.maintain()
        self.assertEqual(attempts,[1]);self.exchange.now+=60001
        with self.assertRaises(AdapterError):supervisor.maintain()
        self.assertEqual(attempts,[1,1])
    def test_trn02_auth_error_and_stuck_reader_do_not_create_second_reader(self):
        self.make_clock_client();made=[]
        for reader in (Reader(reason='WS_AUTH_DENIED'),Reader(stuck=True)):
            self.exchange.reader=reader
            supervisor=self.runtime().PrivateReaderSupervisor(self.exchange,factory=lambda *a,**k:made.append(k))
            with self.assertRaises(AdapterError):supervisor.maintain()
        self.assertEqual(made,[])
    def test_trn02_active_position_is_never_delayed_by_reconnect(self):
        self.make_clock_client();state=self.book.state();state['phase']='OPEN';self.book.save_state(state)
        self.exchange.book=self.book;self.exchange.reader=Reader();made=[]
        supervisor=self.runtime().PrivateReaderSupervisor(self.exchange,factory=lambda *a,**k:made.append(k),book=self.book)
        supervisor.maintain();self.assertEqual(made,[])
    def test_trn03_reviewed_safe_flat_resume_preserves_binding_and_counters(self):
        self.halt();calls=self.make_clock_client();before=deepcopy(self.book._binding)
        state=self.runtime().resume_clock_halt(self.service,reviewed=True)
        self.assertEqual(state['phase'],'IDLE');self.assertFalse(state['entry_killed']);self.assertEqual(state['cycles'],0)
        self.assertEqual(calls,['clock']*3);self.assertEqual(self.book._binding,before);self.assertEqual(self.exchange.calls,[])
    def test_trn03_review_cannot_clear_prior_independent_kill(self):
        self.halt(prior_kill=True);self.make_clock_client()
        with self.assertRaises(AdapterError):self.runtime().resume_clock_halt(self.service,reviewed=True)
        self.assertTrue(self.book.state()['entry_killed'])
    def test_trn03_requires_explicit_review_and_known_all_actions(self):
        self.halt();self.make_clock_client()
        with self.assertRaises(AdapterError):self.runtime().resume_clock_halt(self.service)
        original=self.book.actions;self.book.actions=lambda:{'unknown':{'state':'QUERY_REQUIRED'}}
        with self.assertRaises(AdapterError):self.runtime().resume_clock_halt(self.service,reviewed=True)
        self.book.actions=original
    def test_trn03_bad_clock_never_relaxes_uncertainty(self):
        self.halt();self.make_clock_client();self.exchange.client.sync_clock=lambda:dict(clock_uncertainty_ms=Decimal('251'))
        with self.assertRaises(AdapterError):self.runtime().resume_clock_halt(self.service,reviewed=True)
        self.assertEqual(self.book.state()['phase'],'HALTED')
    def test_trn03_source_staleness_during_review_does_not_reenable_entries(self):
        self.halt();self.make_clock_client();original=self.exchange.snapshot
        def delayed():value=original();self.exchange.now+=61000;return value
        self.exchange.snapshot=delayed
        with self.assertRaises(AdapterError) as caught:self.runtime().resume_clock_halt(self.service,reviewed=True)
        self.assertEqual(caught.exception.reason,'PILOT_REVIEW_SOURCE_STALE')
        self.assertEqual(self.book.state()['phase'],'HALTED')
    def test_trn03_expiry_foreign_order_and_balance_mismatch_remain_blocked(self):
        self.halt();self.make_clock_client();self.exchange.client.read_open_orders=lambda:{'data':[{'order_id':999}]}
        with self.assertRaises(AdapterError):self.runtime().resume_clock_halt(self.service,reviewed=True)
        self.exchange.client.read_open_orders=lambda:{'data':[]};self.exchange.now=86402000
        with self.assertRaises(AdapterError):self.runtime().resume_clock_halt(self.service,reviewed=True)
    def test_trn04_fresh_quotes_do_not_hide_halted_trading(self):
        state=dict(phase='HALTED',entry_killed=True,reason='CLOCK_QUALITY')
        report=self.api.operational_report(state,dict(quotes_received_at_ms=1000,selected_symbol='ETHUSDT'),pid=1,now_ms=1100,
            expires_ms=10000,actual_outcomes_n=2,stream=dict(transport_healthy=True))
        self.assertEqual(report['operational_status'],'BLOCKED');self.assertEqual(report['monitoring_status'],'RUNNING')
        self.assertFalse(report['entry_enabled'])
    def test_trn04_dead_private_reader_blocks_entry_even_with_fresh_quotes(self):
        report=self.api.operational_report(dict(phase='IDLE',entry_killed=False),dict(quotes_received_at_ms=1000,selected_symbol='ETHUSDT'),
            pid=1,now_ms=1100,expires_ms=10000,actual_outcomes_n=2,stream=dict(transport_healthy=False))
        self.assertEqual(report['trading_status'],'WAIT_PRIVATE');self.assertFalse(report['entry_enabled'])
    def test_trn04_dead_thread_is_not_entry_ready_despite_recent_pong(self):
        report=self.api.operational_report(dict(phase='IDLE',entry_killed=False),dict(quotes_received_at_ms=1000,selected_symbol='ETHUSDT'),
            pid=1,now_ms=1100,expires_ms=10000,actual_outcomes_n=2,stream=dict(connected=True,transport_healthy=True,reader_alive=False))
        self.assertFalse(report['entry_enabled']);self.assertEqual(report['trading_status'],'WAIT_PRIVATE')
    def test_trn02_audit_failure_closes_new_reader_without_leaking_thread(self):
        self.make_clock_client();self.exchange.reader=Reader();new=Reader()
        supervisor=self.runtime().PrivateReaderSupervisor(self.exchange,factory=lambda *a,**k:new,book=self.book)
        def fault(phase):
            if phase=='before_commit':raise OSError('synthetic')
        self.book._audit.fault_hook=fault
        with self.assertRaises(AdapterError):supervisor.maintain()
        self.assertEqual(new.closes,1);self.assertFalse(new.health()['connected'])



class ReviewedStartupTests(unittest.TestCase):
    setUp=TransportRecoveryTests.setUp
    runtime=TransportRecoveryTests.runtime
    make_clock_client=TransportRecoveryTests.make_clock_client
    halt=TransportRecoveryTests.halt
    def test_trn03_reader_supervised_after_warmup_before_review_probes(self):
        self.halt();clock_calls=self.make_clock_client();events=[]
        self.exchange.refresh_catalog=lambda:events.append('catalog')
        self.exchange.poll_quotes=lambda:events.append('quotes')
        self.exchange.authorize=lambda:events.append('authorize')
        self.exchange.reader=Reader()
        def factory(*a,**kw):events.append('reader');return Reader()
        supervisor=self.runtime().PrivateReaderSupervisor(self.exchange,factory=factory,book=self.book)
        reader=self.runtime().review_flat_clock_startup(self.service,supervisor,sleep=lambda _:None)
        self.assertEqual(events,['catalog','quotes','authorize','reader']);self.assertEqual(clock_calls,['clock']*3)
        self.assertTrue(reader.health()['reader_alive']);self.assertEqual(self.book.state()['phase'],'IDLE')
    def test_trn03_warmup_failure_never_clears_halt(self):
        self.halt();self.make_clock_client()
        def failure():raise AdapterError('MARKET_INTEGRITY_FAILED')
        self.exchange.refresh_catalog=failure
        supervisor=self.runtime().PrivateReaderSupervisor(self.exchange,book=self.book)
        with self.assertRaises(AdapterError):self.runtime().review_flat_clock_startup(self.service,supervisor,sleep=lambda _:None)
        self.assertEqual(self.book.state()['phase'],'HALTED')

if __name__=='__main__':unittest.main()
