"""VLD contracts: synthetic credentials, native own SQL, no real exchange I/O."""
from contextlib import closing
from decimal import Decimal
import hashlib
import hmac
import importlib
import json
from pathlib import Path
import sqlite3
import ssl
import tempfile
import unittest
from unittest.mock import patch
from urllib.parse import parse_qs

import _support
from binance_demo_bot import adapter
from binance_demo_bot.interaction_audit import AuditJournal
from binance_demo_bot.isolation import AppPaths, DemoCredentials
import test_adapter as fixtures


class ValidatorTests(unittest.TestCase):
    def setUp(self):
        self.api=importlib.import_module('binance_demo_bot.order_validation')
        base=Path(__file__).resolve().parents[1]/'.runtime'/'unit-fixtures';base.mkdir(parents=True,exist_ok=True)
        temporary=tempfile.TemporaryDirectory(dir=base);self.addCleanup(temporary.cleanup)
        self.paths=AppPaths(Path(temporary.name).resolve())
        self.audit=AuditJournal(self.paths,account_alias='demo-one',episode_id='episode-one',scope_id='whole-account')
        self.addCleanup(self.audit.close)
        self.time=fixtures.TimeSource()
        self.clock=adapter.DemoClock(wall_ms=lambda:self.time.wall,monotonic_ns=lambda:self.time.mono)
        self.clock.observe(self.time.wall,self.time.wall,self.time.wall,self.time.mono,self.time.mono)
        self.read_calls=[];self.calls=[];self.replies=[]
        outer=self
        class Reads:
            def request(self,method,path,query,headers,timeout):
                outer.read_calls.append((method,path));outer.time.advance(10)
                data={'serverTime':outer.time.wall} if path=='/api/v3/time' else fixtures.account()
                return adapter.HttpResponse(200,json.dumps(data).encode(),{})
        class Posts:
            def request(self,method,path,query,headers,timeout):
                with closing(sqlite3.connect(outer.audit.path)) as con:
                    begins=con.execute("SELECT COUNT(*) FROM records WHERE kind='REST_BEGIN' "
                        "AND json_extract(payload_json,'$.data.method')='POST'").fetchone()[0]
                outer.calls.append((method,path,query,dict(headers),timeout,begins))
                outer.time.advance(10)
                reply=outer.replies.pop(0)
                if isinstance(reply,BaseException):raise reply
                status,data,response_headers=reply
                return adapter.HttpResponse(status,data if isinstance(data,bytes) else json.dumps(data).encode(),response_headers)
            def close(self):outer.post_closed=True
        self.transport=Posts();self.post_closed=False
        self.client=adapter.BinanceDemoAdapter(DemoCredentials('synthetic-key','synthetic-secret'),
            account_alias='demo-one',credential_generation='g1',episode_id='episode-one',scope_id='whole-account',
            identity_key=b'i'*32,transport=Reads(),clock=self.clock,audit=self.audit)
        self.addCleanup(self.client.close)
        self.client.read_account()
        self.validator=self.make();self.addCleanup(self.validator.close)

    def make(self,**changes):
        values=dict(allow_order_validation=True,transport=self.transport);values.update(changes)
        return self.api.DemoTradeValidator(self.client,**values)

    def validate(self,**changes):
        params=dict(side='BUY',quantity='0.001',price='10000');params.update(changes)
        return self.validator.validate_limit('BTCUSDT',**params)

    def reply(self,data=None,status=200,headers=None):self.replies.append((status,{} if data is None else data,headers or {}))
    def reason(self,code,call,*args,**kwargs):
        with self.assertRaises(adapter.AdapterError) as caught:call(*args,**kwargs)
        self.assertEqual(caught.exception.reason,code)
    def capability(self):return self.client.capabilities()['KEY_TRADE_VALIDATED']['state']
    def post_rows(self,kind):
        return [r for r in self.audit.read(limit=1000) if r['kind']==kind and r['data'].get('method')=='POST']

    def test_vld01_default_disabled_without_post(self):
        self.validator=self.api.DemoTradeValidator(self.client,transport=self.transport)
        self.reason('ORDER_VALIDATION_DISABLED',self.validate)
        self.assertEqual(self.calls,[]);self.assertEqual(self.capability(),'UNVERIFIED')

    def test_pil01_native_audited_matching_attempt_requires_explicit_flag(self):
        self.reply(data={'orderId':123})
        params=dict(symbol='BTCUSDT',side='BUY',type='MARKET',quoteOrderQty='30')
        self.reason('FORBIDDEN_ENDPOINT',self.client._transport_attempt,'/api/v3/order','',{},params,True,
            self.time.mono+15000000000,'default-attempt',0,transport=self.transport,method='POST')
        response,*_=self.client._transport_attempt('/api/v3/order','',{},params,True,
            self.time.mono+15000000000,'pilot-attempt',0,transport=self.transport,method='POST',matching_engine=True)
        self.assertEqual(response.status,200);self.assertEqual(len(self.post_rows('REST_BEGIN')),1)
        self.assertEqual(self.calls[0][-1],1)

    def test_vld01_requires_configured_same_namespace_audit(self):
        saved=self.client.audit;self.client.audit=None
        self.reason('AUDIT_UNAVAILABLE',self.make)
        self.client.audit=saved;self.client.identity['account_alias']='different'
        self.reason('AUDIT_IDENTITY_MISMATCH',self.make)

    def test_vld01_constructor_requires_boolean_enablement(self):
        self.reason('INVALID_PARAMETERS',self.make,allow_order_validation=1)

    def test_vld02_exact_signed_limit_and_scoped_result(self):
        self.reply();result=self.validate(quantity=Decimal('0.001'),price=Decimal('10000'))
        method,path,query,headers,_,begins=self.calls[0]
        self.assertEqual((method,path,begins),('POST','/api/v3/order/test',1))
        unsigned,signature=query.rsplit('&signature=',1)
        self.assertEqual(signature,hmac.new(b'synthetic-secret',unsigned.encode(),hashlib.sha256).hexdigest())
        params={k:v[0] for k,v in parse_qs(unsigned).items()}
        self.assertEqual((params['type'],params['timeInForce'],params['quantity'],params['price']),('LIMIT','GTC','0.001','10000'))
        self.assertRegex(params['newClientOrderId'],r'^bdv_[0-9a-f]{32}$')
        self.assertEqual(headers,{'X-MBX-APIKEY':'synthetic-key'})
        self.assertEqual(result['state'],'VALIDATED_NOT_SUBMITTED')
        self.assertFalse(result['submitted']);self.assertFalse(result['order_send_enabled'])
        self.assertEqual(result['source_method'],'POST')
        self.assertEqual(self.capability(),'SUPPORTED')
        self.assertEqual(self.client.capabilities()['ORDER_SEND']['state'],'DENIED')
        evidence=self.client.capabilities()['KEY_TRADE_VALIDATED']
        self.assertEqual(evidence['source'],'P1_ORDER_TEST')
        self.assertEqual(evidence['tested_symbol'],'BTCUSDT')
        self.assertEqual(evidence['credential_generation'],'g1')

    def test_vld02_invalid_numeric_values_never_reach_post(self):
        for value in (True,0,1.25,'0','-1','NaN','Infinity','1e-19','1e19'):
            with self.subTest(value=value):self.reason('INVALID_PARAMETERS',self.validate,quantity=value)
        self.assertEqual(self.calls,[])

    def test_vld02_invalid_pair_side_or_price(self):
        self.reason('INVALID_PARAMETERS',self.validator.validate_limit,'BTCBTC',side='BUY',quantity='1',price='10')
        self.reason('INVALID_PARAMETERS',self.validate,side='LONG')
        self.reason('INVALID_PARAMETERS',self.validate,price=10.0)
        self.assertEqual(self.calls,[])

    def test_vld04_terminal_audit_ack_and_redaction(self):
        self.reply();result=self.validate()
        self.assertEqual(len(self.post_rows('REST_BEGIN')),1);self.assertEqual(len(self.post_rows('REST_END')),1)
        self.assertEqual(result['audit_end_event_id'],self.post_rows('REST_END')[0]['event_id'])
        saved=json.dumps(self.audit.read(limit=100))
        for value in ('synthetic-key','synthetic-secret','987654321','signature'):self.assertNotIn(value,saved)
        self.assertEqual(self.client.audit_health()['transport']['attempted_n'],2)
        decisions=[r['data'] for r in self.audit.read(limit=100) if r['kind']=='DECISION']
        self.assertEqual([r['action'] for r in decisions],['WAIT','NO_ACTION'])
        self.assertFalse(decisions[-1]['intent']['submitted'])

    def test_vld05_auth_denial_is_not_support(self):
        self.reply({'code':-2015,'msg':'synthetic-key'},status=401)
        self.reason('AUTH_DENIED',self.validate);self.assertEqual(self.capability(),'DENIED')
        self.assertEqual(len(self.calls),1)

    def test_vld05_invalid_filter_remains_unverified(self):
        self.reply({'code':-1013,'msg':'Filter failure: NOTIONAL'},status=400)
        self.reason('ORDER_TEST_REJECTED',self.validate);self.assertEqual(self.capability(),'UNVERIFIED')

    def test_vld05_unsupported_endpoint_remains_unverified(self):
        self.reply({},status=404)
        self.reason('ORDER_TEST_REJECTED',self.validate);self.assertEqual(self.capability(),'UNVERIFIED')

    def test_vld05_waf_forbidden_is_not_key_denial(self):
        self.reply(b'Access denied',status=403)
        self.reason('ORDER_TEST_UNAVAILABLE',self.validate);self.assertEqual(self.capability(),'UNAVAILABLE')

    def test_vld05_only_exact_empty_200_is_valid(self):
        for data in ([],{'orderId':123},b'invalid',b'{"x":1,"x":2}'):
            with self.subTest(data=data):
                self.reply(data);self.reason('SCHEMA_INVALID',self.validate)
                self.assertEqual(self.capability(),'UNVERIFIED')

    def test_vld06_timeout_never_retries_post(self):
        self.replies.append(TimeoutError('synthetic-secret'))
        self.reason('ORDER_TEST_UNAVAILABLE',self.validate)
        self.assertEqual(len(self.calls),1);self.assertEqual(self.capability(),'UNAVAILABLE')
        self.assertEqual(self.post_rows('REST_END')[0]['data']['outcome'],'TIMEOUT')

    def test_vld06_server_failure_never_retries_post(self):
        self.reply({},status=503);self.reason('ORDER_TEST_UNAVAILABLE',self.validate)
        self.assertEqual(len(self.calls),1);self.assertEqual(self.capability(),'UNAVAILABLE')

    def test_vld06_timestamp_rejection_never_resends_or_resyncs(self):
        reads=len(self.read_calls)
        self.reply({'code':-1021},status=400);self.reason('TIMESTAMP_REJECTED',self.validate)
        self.assertEqual(len(self.calls),1);self.assertEqual(len(self.read_calls),reads)
        self.assertEqual(self.capability(),'UNAVAILABLE')

    def test_vld06_rate_limit_honors_cooldown(self):
        self.reply({},status=429,headers={'Retry-After':'7'})
        self.reason('RATE_LIMITED',self.validate)
        self.reason('RATE_LIMITED',self.validate)
        self.assertEqual(len(self.calls),1);self.assertGreater(self.client.cooldown_until_ns,self.time.mono)

    def test_vld06_ban_blocks_later_calls(self):
        self.reply({},status=418,headers={'Retry-After':'8'})
        self.reason('IP_BANNED',self.validate);self.reason('IP_BANNED',self.validate)
        self.assertEqual(len(self.calls),1)

    def test_vld06_shared_weight_prevents_another_call(self):
        self.reply({},headers={'X-MBX-USED-WEIGHT-1M':'1200'});self.validate()
        self.reason('LOCAL_RATE_BUDGET',self.validate)
        self.assertEqual(len(self.calls),1)

    def test_vld06_clock_jump_blocks_before_post(self):
        self.time.wall+=1000
        self.reason('CLOCK_JUMP',self.validate)
        self.assertEqual(self.calls,[]);self.assertNotEqual(self.capability(),'SUPPORTED')

    def test_vld06_missing_clock_sample_can_use_get_sync(self):
        self.clock.sample=None;self.reply();self.validate()
        self.assertIn(('GET','/api/v3/time'),self.read_calls);self.assertEqual(len(self.calls),1)

    def test_vld06_stale_account_blocks_even_after_time_sync(self):
        self.time.advance(31000)
        self.reason('ACCOUNT_NOT_READY',self.validate);self.assertEqual(self.calls,[])

    def test_vld06_late_receipt_cannot_become_supported(self):
        original=self.transport.request
        def delayed(*args):
            result=original(*args);self.time.advance(15000);return result
        self.transport.request=delayed;self.reply()
        self.reason('ORDER_TEST_UNAVAILABLE',self.validate);self.assertEqual(len(self.calls),1)
        self.assertEqual(self.capability(),'UNAVAILABLE')

    def test_vld06_cancellation_preserves_audit_and_revokes_evidence(self):
        self.replies.append(KeyboardInterrupt('synthetic-secret'))
        with self.assertRaises(KeyboardInterrupt):self.validate()
        self.assertEqual(self.capability(),'UNAVAILABLE')
        self.assertEqual(self.post_rows('REST_END')[0]['data']['outcome'],'CANCELLED')

    def test_vld07_disable_revokes_success_and_keeps_read_adapter_open(self):
        self.reply();self.validate();self.validator.disable()
        self.assertEqual(self.capability(),'UNVERIFIED')
        self.reason('ORDER_VALIDATION_DISABLED',self.validate)
        self.assertFalse(self.audit._closed);self.assertEqual(len(self.calls),1)

    def test_vld07_invalid_later_request_revokes_old_success(self):
        self.reply();self.validate();self.reason('INVALID_PARAMETERS',self.validate,quantity='0')
        self.assertEqual(self.capability(),'UNVERIFIED');self.assertEqual(len(self.calls),1)

    def test_vld07_success_expires_after_thirty_seconds(self):
        self.reply();self.validate();self.time.advance(30001)
        self.assertEqual(self.capability(),'UNVERIFIED')

    def test_vld08_native_begin_abort_prevents_network(self):
        self.audit._con.execute("CREATE TRIGGER fixture_fail BEFORE INSERT ON records WHEN NEW.kind='REST_BEGIN' "
            "AND json_extract(NEW.payload_json,'$.data.method')='POST' BEGIN SELECT RAISE(ABORT,'fixture'); END")
        self.reason('AUDIT_WRITE_FAILED',self.validate);self.assertEqual(self.calls,[])
        self.assertEqual(self.capability(),'UNAVAILABLE')

    def test_vld08_native_terminal_abort_never_proves_support(self):
        self.audit._con.execute("CREATE TRIGGER fixture_fail BEFORE INSERT ON records WHEN NEW.kind='REST_END' "
            "AND json_extract(NEW.payload_json,'$.data.method')='POST' BEGIN SELECT RAISE(ABORT,'fixture'); END")
        self.reply();self.reason('AUDIT_WRITE_FAILED',self.validate)
        self.assertEqual(len(self.calls),1);self.assertEqual(self.capability(),'UNAVAILABLE')
        self.assertEqual(self.audit.health()['unresolved_attempts_n'],1)

    def test_vld03_internal_attempt_cannot_target_real_orders(self):
        self.reason('FORBIDDEN_ENDPOINT',self.client._transport_attempt,'/api/v3/order','',{}, {},True,
            self.time.mono+15000000000,'request',0,method='POST',transport=self.transport)
        self.assertEqual(self.calls,[])

    def test_vld03_post_attempt_requires_true_boolean_signed_flag(self):
        for signed in (False,1):
            with self.subTest(signed=signed):
                self.reason('FORBIDDEN_ENDPOINT',self.client._transport_attempt,'/api/v3/order/test','',{}, {},signed,
                    self.time.mono+15000000000,'request',0,method='POST',transport=self.transport)
        self.assertEqual(self.calls,[])

    def test_vld08_final_decision_abort_cannot_prove_support(self):
        self.audit._con.execute("CREATE TRIGGER fixture_fail BEFORE INSERT ON records WHEN NEW.kind='DECISION' "
            "AND json_extract(NEW.payload_json,'$.data.action')='NO_ACTION' BEGIN SELECT RAISE(ABORT,'fixture'); END")
        self.reply();self.reason('AUDIT_WRITE_FAILED',self.validate)
        self.assertEqual(len(self.calls),1);self.assertEqual(self.capability(),'UNAVAILABLE')

    def test_vld06_deadline_includes_final_audit_ack(self):
        original=self.validator._decision
        def delayed(action,*args):
            result=original(action,*args)
            if action=='NO_ACTION':self.time.advance(15000)
            return result
        self.validator._decision=delayed;self.reply()
        self.reason('ORDER_TEST_UNAVAILABLE',self.validate)
        self.assertEqual(self.capability(),'UNAVAILABLE');self.assertEqual(len(self.calls),1)

    def test_vld07_disable_decision_is_durable(self):
        self.validator.disable()
        decisions=[r['data'] for r in self.audit.read(limit=100) if r['kind']=='DECISION']
        self.assertEqual(decisions[-1]['action'],'BLOCK')
        self.assertEqual(decisions[-1]['reasons'],['ORDER_VALIDATION_DISABLED'])

    def test_vld01_disabled_parent_prevents_post(self):
        self.client.enabled=False
        self.reason('ADAPTER_DISABLED',self.validate);self.assertEqual(self.calls,[])

    def test_vld06_dependency_error_text_is_never_exposed(self):
        self.replies.append(RuntimeError('synthetic-secret'))
        self.reason('ORDER_TEST_UNAVAILABLE',self.validate)
        self.assertNotIn('synthetic-secret',json.dumps(self.audit.read(limit=100)))


class TestOnlyTransportTests(unittest.TestCase):
    def setUp(self):self.api=importlib.import_module('binance_demo_bot.order_validation')
    def test_vld03_both_allowlists_remain_separate(self):
        test=self.api.DemoOrderTestTransport();read=adapter.DemoHttpTransport()
        with patch.object(adapter.http.client,'HTTPSConnection') as factory:
            for method,path in (('GET','/api/v3/account'),('POST','/api/v3/order'),('DELETE','/api/v3/order'),
                    ('POST','/api/v3/orderList/oco'),('POST','/api/v3/order/test?x=1'),
                    ('POST','/api/v3/order/test/../order'),('POST','https://api.binance.com/api/v3/order/test')):
                with self.subTest(method=method,path=path):
                    with self.assertRaises(adapter.AdapterError):test.request(method,path,'',{},15)
            with self.assertRaises(adapter.AdapterError):read.request('POST','/api/v3/order/test','',{},15)
            factory.assert_not_called()
        for url in ('https://api.binance.com/api/v3/','https://testnet.binance.vision/api/v3/',
                    'http://demo-api.binance.com/api/v3/'):
            with self.assertRaises(adapter.AdapterError):self.api.DemoOrderTestTransport(base_url=url)

    def test_vld03_exact_post_query_and_verified_tls(self):
        class Response:
            status=200
            def read(self,maximum):return b'{}'
            def getheaders(self):return []
        with patch.object(adapter.http.client,'HTTPSConnection') as factory:
            factory.return_value.getresponse.return_value=Response()
            transport=self.api.DemoOrderTestTransport();self.addCleanup(transport.close)
            response=transport.request('POST','/api/v3/order/test','a=1&signature=synthetic',{'X-MBX-APIKEY':'fake'},15)
            self.assertEqual(response.status,200)
            self.assertEqual(factory.call_args.args[0],'demo-api.binance.com')
            context=factory.call_args.kwargs['context']
            self.assertTrue(context.check_hostname);self.assertEqual(context.verify_mode,ssl.CERT_REQUIRED)
            self.assertEqual(factory.return_value.request.call_args.args[:2],
                             ('POST','/api/v3/order/test?a=1&signature=synthetic'))

    def test_vld03_common_tls_helper_cannot_bypass_method_path_fence(self):
        transport=self.api.DemoOrderTestTransport();self.addCleanup(transport.close)
        with patch.object(adapter.http.client,'HTTPSConnection') as factory:
            for method,path in (('POST','/api/v3/order'),('DELETE','/api/v3/order'),('GET','/api/v3/order/test')):
                with self.subTest(method=method,path=path):
                    with self.assertRaises(adapter.AdapterError):transport._request_https(method,path,'',{},15)
            factory.assert_not_called()

    def test_vld03_redirect_cleans_up_without_following(self):
        with patch.object(adapter.http.client,'HTTPSConnection') as factory:
            factory.return_value.getresponse.return_value.status=302
            transport=self.api.DemoOrderTestTransport()
            with self.assertRaises(adapter.AdapterError) as caught:transport.request('POST','/api/v3/order/test','',{},15)
            self.assertEqual(caught.exception.reason,'REDIRECT_BLOCKED')
            factory.return_value.close.assert_called_once()


if __name__=='__main__':unittest.main()
