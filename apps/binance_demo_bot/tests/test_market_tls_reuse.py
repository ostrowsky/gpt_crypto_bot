"""P2B2-11 TLS lifetime/verification contracts, written before implementation."""
import importlib
import ssl
import unittest
from unittest.mock import patch, MagicMock
from _p2_support import Fixture
from test_market_stream import FakeStream


def context(): return ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)


def connection():
    result=MagicMock()
    result.request.path='/stream'
    result.request.headers={'Host':'demo-stream.binance.com'}
    return result


class TlsReuseTests(Fixture, unittest.TestCase):
    def setUp(self):
        self.setup_market()
        self.api=importlib.import_module('binance_demo_bot.market_stream')
        self.source=self.refresh()
        deny=patch.object(self.api,'ws_connect',side_effect=AssertionError('network forbidden'))
        blocked=deny.start();self.addCleanup(deny.stop)
        self.addCleanup(blocked.assert_not_called)

    def supervisor(self,**options):
        result=self.api.MarketStreamSupervisor(self.store,self.client,self.source,**options)
        self.addCleanup(result.stop)
        return result

    def test_P2B211_01_default_owner_shares_one_context_across_transports(self):
        verified=context()
        with patch.object(self.api.ssl,'create_default_context',return_value=verified) as builder:
            owner=self.supervisor()
            builder.assert_not_called()
            first,second=owner.transport_factory(),owner.transport_factory()
            with patch.object(self.api,'ws_connect',side_effect=[connection(),connection()]) as connect:
                first.connect();second.connect()
                self.assertIs(connect.call_args_list[0].kwargs['ssl'],verified)
                self.assertIs(connect.call_args_list[1].kwargs['ssl'],verified)
            first.close();second.close()
            builder.assert_called_once_with()

    def test_P2B211_01_independent_owners_do_not_share_global_context(self):
        left,right=context(),context()
        with patch.object(self.api.ssl,'create_default_context',side_effect=[left,right]) as builder:
            owners=[self.supervisor(),self.supervisor()]
            transports=[owner.transport_factory() for owner in owners]
            with patch.object(self.api,'ws_connect',side_effect=[connection(),connection()]) as connect:
                for transport in transports:transport.connect();transport.close()
                self.assertIs(connect.call_args_list[0].kwargs['ssl'],left)
                self.assertIs(connect.call_args_list[1].kwargs['ssl'],right)
            self.assertEqual(builder.call_count,2)

    def test_P2B211_01_custom_factory_never_builds_default_tls(self):
        with patch.object(self.api.ssl,'create_default_context') as builder:
            owner=self.supervisor(transport_factory=FakeStream)
            owner.start();self.assertEqual(owner.health()['connected_shards_n'],1)
            owner.stop();builder.assert_not_called()

    def test_P2B211_01_disabled_start_zero_tls_and_attempts(self):
        with patch.object(self.api.ssl,'create_default_context') as builder:
            owner=self.supervisor(enabled=False)
            self.assert_reason('MARKET_DATA_DISABLED',owner.start)
            builder.assert_not_called()
            self.assertEqual(owner.health()['connection_attempts_n'],0)

    def test_P2B211_01_factory_validation(self):
        for invalid in (False,1,'factory'):
            self.assert_reason('INVALID_PARAMETERS',self.supervisor,transport_factory=invalid)

    def test_P2B211_02_standalone_reconnect_reuses_verified_context(self):
        verified=context()
        with patch.object(self.api.ssl,'create_default_context',return_value=verified) as builder:
            transport=self.api.DemoMarketWsTransport()
            with patch.object(self.api,'ws_connect',side_effect=[connection(),connection()]) as connect:
                transport.connect();transport.close();transport.connect();transport.close()
                self.assertTrue(all(call.kwargs['ssl'] is verified for call in connect.call_args_list))
            builder.assert_called_once_with()

    def test_P2B211_02_unsafe_contexts_refused_before_network(self):
        no_hostname=context();no_hostname.check_hostname=False
        unverified=context();unverified.check_hostname=False;unverified.verify_mode=ssl.CERT_NONE
        old_tls=context();old_tls.minimum_version=ssl.TLSVersion.TLSv1_1
        server=ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        for invalid in (object(),True,no_hostname,unverified,old_tls,server):
            self.assert_reason('TLS_UNSAFE',self.api.DemoMarketWsTransport,ssl_context=invalid)

    def test_P2B211_02_mutation_after_construction_refused(self):
        verified=context()
        transport=self.api.DemoMarketWsTransport(ssl_context=verified)
        verified.check_hostname=False
        self.assert_reason('TLS_UNSAFE',transport.connect)
        transport.close()

    def test_P2B211_02_creation_failure_stops_owner_before_attempt(self):
        owner=self.supervisor()
        with patch.object(self.api.ssl,'create_default_context',side_effect=RuntimeError('private TLS detail')):
            self.assert_reason('WS_UNAVAILABLE',owner.start)
        health=owner.health()
        self.assertFalse(health['running'])
        self.assertEqual(health['connection_attempts_n'],0)
        self.assertEqual(health['shards'],{})
        self.assertNotIn('private TLS detail',str(health))

    def test_P2B211_03_demo_handshake_guards_remain(self):
        verified=context()
        transport=self.api.DemoMarketWsTransport(ssl_context=verified)
        with patch.object(self.api,'ws_connect',return_value=connection()) as connect:
            transport.connect();kwargs=connect.call_args.kwargs
            self.assertEqual(connect.call_args.args[0],self.api.WS_URL)
            self.assertIsNone(kwargs['proxy']);self.assertIsNone(kwargs['compression'])
            self.assertEqual(kwargs['max_queue'],16);self.assertEqual(kwargs['max_size'],1024*1024)
            self.assertTrue(kwargs['ssl'].check_hostname)
            self.assertEqual(kwargs['ssl'].verify_mode,ssl.CERT_REQUIRED)
            self.assertTrue(kwargs['logger'].disabled)
        transport.close()
        self.assert_reason('FORBIDDEN_ENDPOINT',self.api.DemoMarketWsTransport,'wss://stream.binance.com/stream',ssl_context=verified)
