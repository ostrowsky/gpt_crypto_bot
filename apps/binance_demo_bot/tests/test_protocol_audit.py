"""AUD-B3 spec-first native protocol/owned loopback checks; no Binance."""
import base64
import importlib
import json
import logging
import socket
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from _p2_support import Fixture
from binance_demo_bot.interaction_audit import AuditJournal
from binance_demo_bot.adapter import AdapterError
from websockets.client import ClientProtocol
from websockets.protocol import OPEN, Protocol, SERVER
from websockets.frames import Frame, PING, PONG, CLOSE, TEXT
from websockets.uri import parse_uri


class RawSocket:
    def __init__(self):self.closed=False;self.writes=[]
    def shutdown(self,how):self.closed=True
    def close(self):self.closed=True
    def sendall(self,data):self.writes.append(data)


class ProtocolAuditTests(Fixture,unittest.TestCase):
    def setUp(self):
        self.api=importlib.import_module('binance_demo_bot.protocol_audit')
        self.setup_market()
        self.audit=AuditJournal(self.paths,account_alias='demo-one',episode_id='episode-one',scope_id='whole-account')
        self.addCleanup(self.audit.close)
        self.logger=logging.Logger('audit-test');self.logger.disabled=True;self.logger.propagate=False

    def observer(self,**changes):
        args=dict(private=False)
        args.update(changes)
        return self.api.ProtocolControlAudit(self.audit,self.clock,'connection-one',**args)

    def protocol(self):
        return ClientProtocol(parse_uri('wss://demo-stream.binance.com/stream'),state=OPEN,logger=self.logger)

    def installed(self,**changes):
        observer=self.observer(**changes);protocol=self.protocol();sock=RawSocket()
        observer.install(protocol,sock)
        return observer,protocol,sock

    def rows(self):return [r for r in self.audit.read(limit=1000) if r['kind']=='WS_CONTROL']

    def test_b301_automatic_pong_is_captured_before_native_queue(self):
        observer,protocol,sock=self.installed()
        protocol.receive_data(Frame(PING,b'peer-clock').serialize(mask=False))
        rows=self.rows()
        self.assertEqual([(r['data']['phase'],r['data']['opcode']) for r in rows],
            [('RECEIVED','PING'),('INTENT','PONG'),('QUEUED','PONG')])
        self.assertEqual(rows[1]['data']['parent_control_event_id'],rows[0]['event_id'])
        self.assertEqual(base64.b64decode(rows[0]['data']['payload']['body_base64']),b'peer-clock')
        self.assertEqual(observer.health()['pending_n'],1)
        observer.write_pending(lambda:[sock.sendall(data) for data in protocol.data_to_send()])
        server=Protocol(SERVER,state=OPEN,logger=self.logger);server.receive_data(b''.join(sock.writes))
        self.assertEqual(server.events_received()[0].opcode,PONG)
        self.assertEqual(server.events_received(),[])
        self.assertEqual(self.rows()[-1]['data']['phase'],'WRITE_RETURNED')
        self.assertEqual(observer.health()['pending_n'],0)

    def test_b301_peer_close_echo_and_client_ping_are_separate_controls(self):
        observer,protocol,sock=self.installed()
        protocol.send_ping(b'client')
        protocol.receive_data(Frame(PONG,b'client').serialize(mask=False))
        protocol.receive_data(Frame(CLOSE,b'\x03\xe8bye').serialize(mask=False))
        rows=self.rows()
        self.assertEqual([(r['data']['phase'],r['data']['opcode']) for r in rows],
            [('INTENT','PING'),('QUEUED','PING'),('RECEIVED','PONG'),('RECEIVED','CLOSE'),('INTENT','CLOSE'),('QUEUED','CLOSE')])
        self.assertEqual(observer.health()['inbound_n'],2);self.assertEqual(observer.health()['outbound_n'],2)

    def test_b301_text_is_not_duplicated_by_control_hook(self):
        observer,protocol,sock=self.installed()
        protocol.receive_data(Frame(TEXT,b'public data').serialize(mask=False))
        self.assertEqual(protocol.events_received()[0].data,b'public data')
        protocol.send_text(b'outgoing data')
        observer.write_pending(lambda:protocol.data_to_send())
        self.assertEqual(self.rows(),[])

    def test_b302_socket_failure_is_unknown_not_remote_ack(self):
        observer,protocol,sock=self.installed();protocol.send_ping(b'ping')
        with self.assertRaises(OSError):observer.write_pending(lambda:(_ for _ in ()).throw(OSError('private text')))
        last=self.rows()[-1]['data']
        self.assertEqual(last['phase'],'WRITE_UNKNOWN')
        self.assertNotIn('private text',json.dumps(self.rows()))
        self.assertEqual(observer.health()['write_batches_unknown_n'],1)
        self.assertEqual(observer.health()['write_batches_returned_n'],0)

    def test_b302_pending_controls_are_bounded(self):
        observer,protocol,sock=self.installed(max_pending=2)
        protocol.send_ping(b'1');protocol.send_ping(b'2');protocol.send_ping(b'3')
        self.assertLessEqual(observer.health()['pending_n'],2)
        self.assertEqual(observer.health()['state'],'FAILED')
        self.assertEqual(observer.health()['error'],'AUDIT_CONTROL_BACKPRESSURE')
        self.assertTrue(sock.closed)

    def test_b301_private_control_payload_never_exact_or_leaks(self):
        observer,protocol,sock=self.installed(private=True,secrets=lambda:('synthetic-key','synthetic-secret'))
        protocol.receive_data(Frame(PING,b'synthetic-secret').serialize(mask=False))
        text=json.dumps(self.rows())
        self.assertNotIn('synthetic-key',text);self.assertNotIn('synthetic-secret',text)
        for row in self.rows():
            if 'payload' in row['data']:
                self.assertNotEqual(row['data']['payload']['representation'],'EXACT_PUBLIC_BYTES')

    def test_b301_public_close_reason_is_omitted_on_receipt_and_echo(self):
        observer,protocol,sock=self.installed()
        # Valid application close code3072 also forms valid UTF-8; unlike1000,
        # generic raw-payload sanitization alone cannot omit its plain reason.
        protocol.receive_data(Frame(CLOSE,b'\x0c\x00internal-session-description').serialize(mask=False))
        payloads=[r['data']['payload'] for r in self.rows() if 'payload' in r['data']]
        self.assertEqual(len(payloads),2)
        for payload in payloads:
            self.assertEqual(payload,dict(representation='OMITTED',bytes_n=30))

    def test_b303_original_clock_and_frozen_connection_identity(self):
        observer,protocol,sock=self.installed();before=self.wall
        self.audit.fault_hook=lambda phase:self.advance(50) if phase=='after_commit' else None
        protocol.receive_data(Frame(PING,b'x').serialize(mask=False))
        self.assertEqual(self.rows()[0]['wall_ms'],before)
        self.assertEqual({r['data']['connection_id'] for r in self.rows()},{'connection-one'})
        self.assertTrue(observer.health()['installed'])
        self.assertFalse(observer.health()['full_audit_complete'])

    def test_b304_intent_abort_prevents_queueing_and_closes_raw_socket(self):
        observer,protocol,sock=self.installed()
        self.audit._con.execute("CREATE TRIGGER fail_control BEFORE INSERT ON records WHEN NEW.kind='WS_CONTROL' BEGIN SELECT RAISE(ABORT,'bad'); END")
        protocol.send_ping(b'x')
        self.assertEqual(protocol.data_to_send(),[])
        self.assertTrue(sock.closed)
        self.assertEqual(observer.health()['error'],'AUDIT_WRITE_FAILED')
        called=[];observer.write_pending(lambda:called.append(True))
        self.assertEqual(called,[])

    def test_b305_invalid_control_content_has_rejected_marker(self):
        observer,protocol,sock=self.installed()
        protocol.receive_data(Frame(CLOSE,b'\x00\x00').serialize(mask=False))
        self.assertTrue(any(r['data']['phase']=='REJECTED' for r in self.rows()))
        self.assertEqual(self.rows()[0]['data']['opcode'],'CLOSE')

    def test_b305_duplicate_protocol_ownership_is_rejected(self):
        observer,protocol,sock=self.installed()
        with self.assertRaises(AdapterError):self.observer().install(protocol,sock)

    def native(self):
        observer=self.observer()
        # Windows socketpair fallback has an unbounded accept. Bound both sides
        # of this equivalent own loopback pair so sandbox failures cannot hang.
        listener=socket.create_server(('127.0.0.1',0));listener.settimeout(2)
        client=socket.socket();client.settimeout(2)
        try:
            client.connect(listener.getsockname())
            peer,_=listener.accept()
        except BaseException:
            client.close();raise
        finally:listener.close()
        client.settimeout(None);peer.settimeout(2)
        self.addCleanup(peer.close)
        connection=observer.connection_factory()(client,self.protocol(),ping_interval=None,close_timeout=.05)
        connection.response=SimpleNamespace(status_code=101)
        def cleanup():
            connection.close_socket();connection.recv_events_thread.join(2)
            self.assertFalse(connection.recv_events_thread.is_alive())
        self.addCleanup(cleanup)
        return observer,connection,peer

    def test_b304_native_thread_terminates_after_audit_failure_under_mutex(self):
        observer,connection,peer=self.native()
        self.audit._con.execute("CREATE TRIGGER fail_native BEFORE INSERT ON records WHEN NEW.kind='WS_CONTROL' BEGIN SELECT RAISE(ABORT,'bad'); END")
        peer.sendall(Frame(PING,b'peer').serialize(mask=False))
        connection.recv_events_thread.join(2)
        self.assertFalse(connection.recv_events_thread.is_alive())
        self.assertEqual(observer.health()['error'],'AUDIT_WRITE_FAILED')
        self.assertEqual(observer.health()['record_gap_n'],1)

    def test_b301_native_automatic_pong_passes_real_socket_send(self):
        observer,connection,peer=self.native()
        peer.sendall(Frame(PING,b'peer').serialize(mask=False))
        wire=peer.recv(4096)
        server=Protocol(SERVER,state=OPEN,logger=self.logger);server.receive_data(wire)
        frames=server.events_received()
        self.assertEqual(frames[0].opcode,PONG);self.assertEqual(frames[0].data,b'peer')
        # sendall can unblock the peer just before the terminal audit commit.
        with connection.protocol_mutex:pass
        self.assertEqual(observer.health()['write_batches_returned_n'],1)

    def test_b303_default_public_and_private_factories_receive_observer(self):
        from binance_demo_bot.market_stream import DemoMarketWsTransport
        from binance_demo_bot.private_stream import DemoWsTransport
        from binance_demo_bot.ws_audit import AuditedWsTransport
        for raw,private,module,path,host in ((DemoMarketWsTransport(),False,'market_stream','/stream','demo-stream.binance.com'),
                (DemoWsTransport(),True,'private_stream','/ws-api/v3','demo-ws-api.binance.com')):
            wrapper=AuditedWsTransport(raw,self.audit,self.clock,private=private)
            self.addCleanup(wrapper.close)
            conn=SimpleNamespace(request=SimpleNamespace(path=path,headers={'Host':host}),close=lambda:None)
            with patch('binance_demo_bot.'+module+'.ws_connect',return_value=conn) as connect:
                wrapper.connect()
                self.assertIn('create_connection',connect.call_args.kwargs)
                self.assertFalse(wrapper.health()['protocol_controls_observed']) # Factory wasn't called.

    def test_b303_injected_transport_without_hook_stays_unobserved(self):
        from test_ws_audit import RawStream
        from binance_demo_bot.ws_audit import AuditedWsTransport
        wrapper=AuditedWsTransport(RawStream(),self.audit,self.clock,private=False)
        self.addCleanup(wrapper.close);wrapper.connect()
        self.assertFalse(wrapper.health()['protocol_controls_observed'])

    def test_b305_no_audit_keeps_original_connection_factory(self):
        from binance_demo_bot.market_stream import DemoMarketWsTransport
        raw=DemoMarketWsTransport()
        conn=SimpleNamespace(request=SimpleNamespace(path='/stream',headers={'Host':'demo-stream.binance.com'}),close=lambda:None)
        self.addCleanup(raw.close)
        with patch('binance_demo_bot.market_stream.ws_connect',return_value=conn) as connect:
            raw.connect();self.assertNotIn('create_connection',connect.call_args.kwargs)
