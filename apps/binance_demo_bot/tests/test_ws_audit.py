"""AUD-B1 contracts written before code; own SQLite and synthetic transports."""
import base64
import importlib
import json
import unittest

from _p2_support import Fixture
from _p1_support import FixtureEnvironment, SyntheticWs, execution
from test_market_stream import FakeStream, tick
from test_market_runtime import kline
from binance_demo_bot.adapter import AdapterError
from binance_demo_bot.interaction_audit import AuditJournal


class RawStream(FakeStream):
    def recv(self,timeout_s=1):
        if not self.frames:raise TimeoutError()
        value=self.frames.pop(0)
        if isinstance(value,BaseException):raise value
        return value if isinstance(value,(bytes,str)) else json.dumps(value)


class WsAuditTests(Fixture,unittest.TestCase):
    def setUp(self):
        self.api=importlib.import_module('binance_demo_bot.ws_audit')
        self.setup_market()
        identity=self.store.identity
        self.audit=AuditJournal(self.paths,account_alias=identity['account_alias'],
            episode_id=identity['account_episode_id'],scope_id=identity['portfolio_scope_id'])
        self.addCleanup(self.audit.close)

    def rows(self,kind):return [r for r in self.audit.read(limit=1000) if r['kind']==kind]

    def wrap(self,raw=None):
        raw=raw or RawStream()
        result=self.api.AuditedWsTransport(raw,self.audit,self.clock,private=False)
        self.addCleanup(result.close)
        return result,raw

    def reason(self,reason,call,*args,**kwargs):
        with self.assertRaises(AdapterError) as caught:call(*args,**kwargs)
        self.assertEqual(caught.exception.reason,reason)

    def reader(self,topics=None,frames=None):
        from binance_demo_bot.market_stream import MarketStreamReader
        universe=self.refresh()
        self.client.client.audit=self.audit
        raw=RawStream(frames)
        reader=MarketStreamReader(self.store,self.client,universe,topics or ['aaausdt@bookTicker'],
            transport=raw,background=False)
        self.addCleanup(reader.stop)
        reader.start()
        return reader,raw

    def test_b01_all_operations_and_independent_frame_counts(self):
        wrapper,raw=self.wrap()
        wrapper.connect();wrapper.send(dict(id='x',method='SUBSCRIBE',params=['aaausdt@bookTicker']))
        wrapper.recv(1);self.assertTrue(wrapper.ping(1));wrapper.close();wrapper.close()
        begins,ends=self.rows('WS_BEGIN'),self.rows('WS_END')
        self.assertEqual([r['data']['operation'] for r in begins],['CONNECT','SEND','RECV','PING','CLOSE'])
        self.assertEqual(len(ends),5)
        for begin,end in zip(begins,ends):
            self.assertEqual(end['data']['parent_event_id'],begin['event_id'])
            self.assertEqual(end['data']['operation_id'],begin['data']['operation_id'])
        h=wrapper.health();self.assertEqual(h['received_frames_n'],1)
        self.assertEqual(h['operations']['RECV']['attempted_n'],1)
        self.assertFalse(h['protocol_controls_observed']);self.assertFalse(h['full_audit_complete'])
        self.assertTrue(raw.closed)

    def test_b01_reconnect_gets_distinct_connection_id(self):
        wrapper,raw=self.wrap()
        wrapper.connect();wrapper.close();wrapper.connect()
        ids=[r['data']['connection_id'] for r in self.rows('WS_BEGIN') if r['data']['operation']=='CONNECT']
        self.assertEqual(len(set(ids)),2)

    def test_b02_exact_malformed_public_message_retained(self):
        wrapper,raw=self.wrap();wrapper.connect();raw.frames.append(b'{')
        self.assertEqual(wrapper.recv(1),b'{')
        end=self.rows('WS_END')[-1]
        self.assertEqual(base64.b64decode(end['data']['payload']['body_base64']),b'{')
        self.assertEqual(wrapper.last_receipt['event_id'],end['event_id'])

    def test_b03_receipt_precedes_audit_commit_delay(self):
        wrapper,raw=self.wrap();wrapper.connect();raw.frames.append(b'{}')
        original=self.wall
        self.audit.fault_hook=lambda phase:self.advance(80) if phase=='after_commit' else None
        wrapper.recv(1)
        self.assertEqual(wrapper.last_receipt['wall_ms'],original+80) # BEGIN commit before recv.
        self.assertEqual(self.wall,original+160)
        self.assertEqual(self.rows('WS_END')[-1]['data']['received_at_ms'],original+80)

    def test_b04_failed_intent_never_calls_transport_and_closes(self):
        wrapper,raw=self.wrap()
        self.audit._con.execute("CREATE TRIGGER reject_ws BEFORE INSERT ON records WHEN NEW.kind='WS_BEGIN' BEGIN SELECT RAISE(ABORT,'private'); END")
        self.reason('AUDIT_WRITE_FAILED',wrapper.connect)
        self.assertFalse(raw.connected);self.assertTrue(raw.closed)
        self.assertEqual(wrapper.health()['operations']['CONNECT']['attempted_n'],0)
        self.assertEqual(self.audit.health()['channels']['WS_BEGIN']['gap_n'],1)

    def test_b04_terminal_failure_keeps_received_counter_but_no_return(self):
        wrapper,raw=self.wrap();wrapper.connect();raw.frames.append(b'{}')
        self.audit._con.execute("CREATE TRIGGER reject_ws_end BEFORE INSERT ON records WHEN NEW.kind='WS_END' BEGIN SELECT RAISE(ABORT,'private'); END")
        self.reason('AUDIT_WRITE_FAILED',wrapper.recv,1)
        self.assertTrue(raw.closed);self.assertIsNone(wrapper.last_receipt)
        self.assertEqual(wrapper.health()['received_frames_n'],1)
        self.assertEqual(self.audit.health()['unresolved_ws_operations_n'],1)

    def test_b01_timeout_error_and_false_pong_are_explicit(self):
        wrapper,raw=self.wrap();wrapper.connect()
        with self.assertRaises(TimeoutError):wrapper.recv(1)
        raw.ping=lambda timeout_s:False
        self.assertFalse(wrapper.ping(1))
        outcomes=[r['data']['outcome'] for r in self.rows('WS_END')]
        self.assertEqual(outcomes,['COMPLETED','TIMEOUT','NEGATIVE'])
        self.assertEqual(wrapper.health()['operations']['RECV']['errors_n'],1)

    def test_b04_transport_exception_text_is_never_audit_payload(self):
        wrapper,raw=self.wrap()
        raw.connect=lambda:(_ for _ in ()).throw(OSError('raw-secret-cookie'))
        self.reason('WS_UNAVAILABLE',wrapper.connect)
        self.assertNotIn('raw-secret-cookie',json.dumps(self.audit.read(limit=1000)))
        self.assertEqual(self.rows('WS_END')[-1]['data']['outcome'],'TRANSPORT_ERROR')

    def test_b02_open_candle_and_subscribe_ack_retained_without_snapshots(self):
        reader,raw=self.reader(['aaausdt@kline_1m'])
        raw.frames.append(dict(stream='aaausdt@kline_1m',data=kline(False)))
        self.assertTrue(reader.read_once())
        self.assertEqual(reader.open_candle_packets_n,1)
        incoming=[r for r in self.rows('WS_END') if r['data']['operation']=='RECV' and r['data']['outcome']=='COMPLETED']
        self.assertEqual(len(incoming),2) # Subscribe ACK + open candle.
        marker=self.rows('WS_PROCESSING')[-1]['data']
        self.assertEqual(marker['receipt_event_ids'],[incoming[-1]['event_id']])
        self.assertEqual(marker['snapshot_ids'],[])
        self.assertEqual(marker['outcome'],'PROCESSED')

    def test_b05_market_snapshot_link_after_actual_commit(self):
        reader,raw=self.reader();raw.frames.append(tick())
        self.assertTrue(reader.read_once())
        marker=self.rows('WS_PROCESSING')[-1]['data']
        self.assertTrue(marker['snapshot_ids'])
        for identifier in marker['snapshot_ids']:self.assertIsNotNone(self.store.get(identifier))
        self.assertEqual(marker['outcome'],'PROCESSED')
        self.assertEqual(reader.health()['audit']['received_frames_n'],2)

    def test_b05_snapshot_abort_has_no_confirmed_ids(self):
        reader,raw=self.reader();raw.frames.append(tick())
        self.store._con.execute("CREATE TRIGGER reject_quote BEFORE INSERT ON snapshots WHEN NEW.kind='QUOTES' BEGIN SELECT RAISE(ABORT,'bad'); END")
        self.assertFalse(reader.read_once())
        marker=self.rows('WS_PROCESSING')[-1]['data']
        self.assertEqual(marker['outcome'],'REJECTED');self.assertEqual(marker['snapshot_ids'],[])
        self.assertFalse(reader.connected)

    def test_b04_processing_marker_failure_invalidates_committed_snapshot(self):
        reader,raw=self.reader();raw.frames.append(tick())
        self.audit._con.execute("CREATE TRIGGER reject_processing BEFORE INSERT ON records WHEN NEW.kind='WS_PROCESSING' BEGIN SELECT RAISE(ABORT,'bad'); END")
        self.assertFalse(reader.read_once());self.assertFalse(reader.connected)
        self.assertTrue(raw.closed);self.assertEqual(reader.error,'AUDIT_WRITE_FAILED')
        self.assertEqual(self.audit.health()['channels']['WS_PROCESSING']['gap_n'],1)

    def test_b02_malformed_reader_message_is_saved_before_parse(self):
        reader,raw=self.reader();raw.frames.append(b'{')
        self.assertFalse(reader.read_once());self.assertFalse(reader.connected)
        completed=[r for r in self.rows('WS_END') if r['data']['operation']=='RECV' and r['data']['outcome']=='COMPLETED']
        self.assertEqual(base64.b64decode(completed[-1]['data']['payload']['body_base64']),b'{')

    def test_b03_reader_keeps_original_receipt_and_enforces_250ms(self):
        reader,raw=self.reader();raw.frames.append(tick())
        self.audit.fault_hook=lambda phase:self.advance(260) if phase=='after_commit' else None
        self.assertFalse(reader.read_once());self.assertFalse(reader.connected)
        self.assertEqual(reader.error,'WS_PROCESSING_LAG')

    def test_b01_foreign_journal_store_identity_rejected_before_connect(self):
        from binance_demo_bot.market_stream import MarketStreamReader
        universe=self.refresh();self.client.client.audit=self.audit
        self.audit.identity=dict(self.audit.identity,account_alias='foreign')
        raw=RawStream()
        self.reason('AUDIT_IDENTITY_MISMATCH',MarketStreamReader,self.store,self.client,universe,
                    ['aaausdt@bookTicker'],transport=raw,background=False)
        self.assertFalse(raw.connected)
        self.audit.identity=dict(self.audit.identity,account_alias=self.store.identity['account_alias'])

    def test_b05_startup_buffer_has_original_raw_and_durable_snapshot_link(self):
        reader,raw=self.reader(frames=[tick()])
        marker=self.rows('WS_PROCESSING')[-1]['data']
        self.assertTrue(marker['receipt_event_ids']);self.assertTrue(marker['snapshot_ids'])
        self.assertEqual(marker['outcome'],'PROCESSED')
        for identifier in marker['snapshot_ids']:self.assertIsNotNone(self.store.get(identifier))
        self.assertTrue(reader.connected)

    def test_b05_closed_candle_snapshot_is_linked(self):
        reader,raw=self.reader(['aaausdt@kline_1m'])
        raw.frames.append(dict(stream='aaausdt@kline_1m',data=kline(True)))
        self.assertTrue(reader.read_once())
        marker=self.rows('WS_PROCESSING')[-1]['data']
        self.assertEqual(len(marker['snapshot_ids']),1)
        self.assertEqual(self.store.get(marker['snapshot_ids'][0])['kind'],'CANDLES')

    def test_b04_unresolved_ws_survives_journal_reopen(self):
        wrapper,raw=self.wrap();wrapper.connect();raw.frames.append(b'{}')
        self.audit._con.execute("CREATE TRIGGER fail_end BEFORE INSERT ON records WHEN NEW.kind='WS_END' BEGIN SELECT RAISE(ABORT,'bad'); END")
        self.reason('AUDIT_WRITE_FAILED',wrapper.recv,1)
        identity=dict(self.audit.identity)
        self.audit.close()
        recovered=AuditJournal(self.paths,account_alias=identity['account_alias'],
            episode_id=identity['account_episode_id'],scope_id=identity['portfolio_scope_id'])
        self.addCleanup(recovered.close)
        self.assertEqual(recovered.health()['unresolved_ws_operations_n'],1)
        self.assertEqual(recovered.health()['incomplete_prior_runs_n'],1)
        self.assertFalse(recovered.health()['full_audit_complete'])

    def test_b04_public_timeout_heartbeat_preserves_audit_failure(self):
        reader,raw=self.reader()
        def timeout(timeout_s=1):
            self.advance(21000)
            raise TimeoutError()
        raw.recv=timeout
        self.audit._con.execute("CREATE TRIGGER fail_public_ping BEFORE INSERT ON records WHEN NEW.kind='WS_BEGIN' AND json_extract(NEW.payload_json,'$.data.operation')='PING' BEGIN SELECT RAISE(ABORT,'bad'); END")
        self.assertFalse(reader.read_once())
        self.assertEqual(reader.error,'AUDIT_WRITE_FAILED')
        self.assertTrue(raw.closed)


class PrivateWsAuditTests(FixtureEnvironment,unittest.TestCase):
    def setUp(self):
        self.api=importlib.import_module('binance_demo_bot.ws_audit')
        self.evidence=importlib.import_module('binance_demo_bot.account_evidence')
        self.set_fixture()
        self.audit=AuditJournal(self.paths,account_alias='demo-one',episode_id='episode-one',scope_id='whole-account')
        self.addCleanup(self.audit.close)
        self.adapter.audit=self.audit
        from binance_demo_bot.private_stream import PrivateEventReader
        self.raw=SyntheticWs()
        self.reader=PrivateEventReader(self.adapter,self.store,transport=self.raw,process_instance_id='p1',background=False)
        self.addCleanup(self.reader.close)

    def rows(self,kind):return [r for r in self.audit.read(limit=1000) if r['kind']==kind]

    def test_b02_signed_subscribe_and_echo_redacted(self):
        self.reader.start()
        signature=self.raw.sent[0]['params']['signature']
        self.raw.queue.append({'msg':signature,'apiKey':'synthetic-key'})
        with self.assertRaises(AdapterError):self.reader.poll_once()
        saved=json.dumps(self.audit.read(limit=1000))
        for value in (signature,'synthetic-key','synthetic-secret'):self.assertNotIn(value,saved)
        sent=[r for r in self.rows('WS_BEGIN') if r['data']['operation']=='SEND'][0]
        self.assertEqual(sent['data']['request_method'],'SUBSCRIBE_USER_DATA')

    def test_b05_private_fill_links_to_confirmed_record(self):
        self.reader.start();self.raw.queue.append({'subscriptionId':7,'event':execution()})
        seq=self.reader.poll_once()
        marker=self.rows('WS_PROCESSING')[-1]['data']
        self.assertEqual(marker['normalized_record_seq'],seq)
        self.assertEqual(marker['store_identity'],self.store.identity)
        self.assertEqual(marker['outcome'],'PROCESSED')
        self.assertTrue(marker['receipt_event_ids'])
        self.assertEqual(len(self.store.fills()),1)

    def test_b04_private_terminal_failure_closes_reader_and_preserves_reason(self):
        self.reader.start();self.raw.queue.append({'subscriptionId':7,'event':execution()})
        self.audit._con.execute("CREATE TRIGGER reject_private_end BEFORE INSERT ON records WHEN NEW.kind='WS_END' BEGIN SELECT RAISE(ABORT,'bad'); END")
        with self.assertRaises(AdapterError) as caught:self.reader.poll_once()
        self.assertEqual(caught.exception.reason,'AUDIT_WRITE_FAILED')
        self.assertFalse(self.reader.health()['connected']);self.assertTrue(self.raw.closed)
        self.assertEqual(self.store.fills(),[])

    def test_b03_private_receipt_is_not_shifted_by_audit_commit(self):
        self.reader.start();self.raw.queue.append({'subscriptionId':7,'event':execution()})
        before=self.time.wall
        self.audit.fault_hook=lambda phase:self.time.advance(80) if phase=='after_commit' else None
        self.reader.poll_once()
        self.assertEqual(self.store.events()[0]['received_at_ms'],before+80)

    def test_b04_private_heartbeat_preserves_audit_failure_reason(self):
        self.reader.start()
        self.audit._con.execute("CREATE TRIGGER fail_ping BEFORE INSERT ON records WHEN NEW.kind='WS_BEGIN' BEGIN SELECT RAISE(ABORT,'bad'); END")
        with self.assertRaises(AdapterError) as caught:self.reader.heartbeat()
        self.assertEqual(caught.exception.reason,'AUDIT_WRITE_FAILED')
        self.assertEqual(self.reader.health()['error_reason'],'AUDIT_WRITE_FAILED')
        self.assertTrue(self.raw.closed)

    def test_b05_private_processing_failure_keeps_durable_fill_but_invalidates(self):
        self.reader.start();self.raw.queue.append({'subscriptionId':7,'event':execution()})
        self.audit._con.execute("CREATE TRIGGER reject_private_processing BEFORE INSERT ON records WHEN NEW.kind='WS_PROCESSING' BEGIN SELECT RAISE(ABORT,'bad'); END")
        with self.assertRaises(AdapterError) as caught:self.reader.poll_once()
        self.assertEqual(caught.exception.reason,'AUDIT_WRITE_FAILED')
        self.assertEqual(len(self.store.fills()),1)
        self.assertFalse(self.reader.health()['connected'])
