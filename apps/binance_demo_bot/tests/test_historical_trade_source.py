"""HT02/03/08 test-first fixed research source; no network or credentials."""
import importlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import _support
from binance_demo_bot.adapter import HttpResponse
from binance_demo_bot.interaction_audit import AuditJournal
from binance_demo_bot.isolation import AppPaths


PARAMS=dict(symbol='币安人生USDT',startTime=900000,endTime=1799999,limit=1000)
ROWS=[dict(a=1,p='100',q='2',f=1,l=2,T=900001,m=False,M=True)]


class TradeSourceTests(unittest.TestCase):
    def setUp(self):
        self.api=importlib.import_module('binance_demo_bot.historical_trade_source')
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.paths=AppPaths(Path(self.tmp.name));self.folder=self.paths.runtime_path('trade-probe');self.folder.mkdir(parents=True)
        self.audit=AuditJournal(self.paths,account_alias='public-research',episode_id='historical-trade-test',
            scope_id='historical-trade-source',database='trade-audit.sqlite');self.addCleanup(self.audit.close)
        self.now=0.;self.wall=1800000000000;self.calls=[];self.replies=[];self.sleeps=[]
        outer=self
        class Transport:
            def request(self,method,path,query,headers,timeout_s):
                outer.calls.append(dict(method=method,path=path,query=query,headers=headers,timeout=timeout_s,at=outer.now))
                outer.now+=.01;outer.wall+=10
                reply=outer.replies.pop(0)
                if isinstance(reply,Exception):raise reply
                return reply
            def close(self):pass
        self.transport=Transport()

    def sleep(self,seconds):
        self.sleeps.append(seconds);self.now+=seconds;self.wall+=round(seconds*1000)

    def client(self,**kwargs):
        return self.api.HistoricalTradeClient(self.audit,transport=self.transport,sleep=self.sleep,
            monotonic=lambda:self.now,wall_ms=lambda:self.wall,**kwargs)

    def response(self,rows=ROWS,status=200,headers=None):
        return HttpResponse(status,json.dumps(rows,ensure_ascii=False).encode(),headers or {})

    def cache(self,rows,metadata,params=PARAMS):
        target=self.folder/'responses'/'one.json';target.parent.mkdir(exist_ok=True)
        target.write_text(json.dumps(dict(endpoint='https://demo-api.binance.com/api/v3/aggTrades',
            parameters=params,rows=rows,metadata=metadata)),encoding='utf-8')
        return target

    def test_HT02_valid_bounded_and_fromid_modes_unicode_no_signed_headers(self):
        self.replies=[self.response(),self.response()];client=self.client()
        rows,meta=client.read(dict(PARAMS));self.assertEqual(rows,ROWS)
        client.read(dict(symbol='BTCUSDT',fromId=2,limit=1));client.close()
        self.assertEqual(len(self.calls),2)
        for call in self.calls:
            self.assertEqual((call['method'],call['path'],call['headers']),('GET','/api/v3/aggTrades',{}))
        self.assertIn('%E5%B8%81',self.calls[0]['query'])
        self.assertEqual(meta['http_status'],200)
        self.assertEqual(meta['source_endpoint'],'/api/v3/aggTrades')
        self.assertGreaterEqual(self.calls[1]['at']-self.calls[0]['at'],.3-1e-9)

    def test_HT02_forbidden_parameters_rejected_before_any_transport(self):
        invalid=[dict(PARAMS,signature='x'),dict(PARAMS,fromId=1),dict(PARAMS,limit=0),dict(PARAMS,limit=1001),
            dict(PARAMS,startTime=True),dict(PARAMS,endTime=900000+900001),dict(PARAMS,startTime=-1),
            dict(symbol='BTCUSDT',fromId=-1,limit=1),dict(symbol='BTCUSDT',limit=1),dict(PARAMS,symbol='')]
        for params in invalid:
            with self.assertRaises(self.api.TradeSourceError):self.client().read(params)
        self.assertEqual(self.calls,[])

    def test_HT03_empty_success_is_actual_response_and_audit_links_match(self):
        self.replies=[self.response([])];client=self.client();rows,meta=client.read(dict(PARAMS));client.close()
        self.assertEqual(rows,[]);self.cache(rows,meta);self.audit.close()
        checked=self.api.verify_trade_audit(self.folder,self.audit.path)
        self.assertEqual(checked['state'],'VERIFIED_RECORDED_PUBLIC_ATTEMPTS')
        self.assertEqual(checked['attempts_n'],1);self.assertEqual(checked['linked_cached_responses_n'],1)

    def test_HT03_timeout_and503_retry_are_recorded_and_paced_separately(self):
        self.replies=[TimeoutError('must never be logged'),self.response({'code':-1000},503),self.response()]
        rows,meta=self.client().read(dict(PARAMS));self.cache(rows,meta);self.audit.close()
        checked=self.api.verify_trade_audit(self.folder,self.audit.path)
        self.assertEqual(checked['state'],'VERIFIED_RECORDED_PUBLIC_ATTEMPTS')
        self.assertEqual(checked['attempts_n'],3);self.assertEqual(checked['errors_n'],2)
        self.assertTrue(all(b['at']-a['at']>=.3-1e-9 for a,b in zip(self.calls,self.calls[1:])))

    def test_HT03_ip_ban_never_retries_and_preserves_status_and_code(self):
        self.replies=[self.response({'code':-1003,'msg':'banned'},418)]
        client=self.client()
        with self.assertRaises(self.api.TradeSourceError) as caught:client.read(dict(PARAMS))
        self.assertEqual(caught.exception.reason,'IP_BANNED');self.assertEqual(caught.exception.status,418)
        self.assertEqual(caught.exception.api_code,-1003);self.assertEqual(len(self.calls),1)
        with self.assertRaises(self.api.TradeSourceError):client.read(dict(PARAMS))
        self.assertEqual(len(self.calls),1)

    def test_HT03_rate_limit_retry_after_is_bounded(self):
        self.replies=[self.response({'code':-1003},429,{'Retry-After':'999999'}),self.response([])]
        client=self.client()
        with self.assertRaises(self.api.TradeSourceError):client.read(dict(PARAMS))
        with self.assertRaises(self.api.TradeSourceError):client.read(dict(PARAMS))
        self.assertEqual(len(self.calls),1)
        self.assertTrue(all(0<=s<=60 for s in self.sleeps))

    def test_HT03_short_cooldown_waits_full_duration_before_retry(self):
        self.replies=[self.response({'code':-1003},429,{'Retry-After':'2'}),self.response([])]
        self.client().read(dict(PARAMS))
        self.assertEqual(len(self.calls),2)
        self.assertGreaterEqual(self.calls[1]['at']-self.calls[0]['at'],2)

    def test_HT03_last_retry_cooldown_survives_next_logical_read(self):
        self.replies=[self.response({'code':-1003},429,{'Retry-After':'2'}) for _ in range(3)]+[self.response([])]
        client=self.client()
        with self.assertRaises(self.api.TradeSourceError):client.read(dict(PARAMS))
        client.read(dict(PARAMS))
        self.assertEqual(len(self.calls),4)
        self.assertGreaterEqual(self.calls[3]['at']-self.calls[2]['at'],2)

    def test_HT03_api_error_is_not_empty_history_and_is_not_retried(self):
        self.replies=[self.response({'code':-1121,'msg':'bad symbol'},400)]
        with self.assertRaises(self.api.TradeSourceError) as caught:self.client().read(dict(PARAMS))
        self.assertEqual(caught.exception.api_code,-1121);self.assertEqual(caught.exception.status,400)
        self.assertEqual(len(self.calls),1)

    def test_HT03_client_body_schema_and_size_limits_fail_closed(self):
        for response in (HttpResponse(200,b'not-json',{}),self.response({'unexpected':1}),
                         HttpResponse(200,b'x'*(8*1024*1024+1),{})):
            self.replies=[response]
            with self.assertRaises(self.api.TradeSourceError):self.client().read(dict(PARAMS))

    def test_HT03_invocation_attempt_budget_persists_across_reads(self):
        self.replies=[self.response([]),self.response([])];client=self.client(max_attempts=1)
        client.read(dict(PARAMS))
        with self.assertRaises(self.api.TradeSourceError):client.read(dict(PARAMS))
        self.assertEqual(len(self.calls),1)

    def test_HT03_deadline_caps_native_timeout_and_blocks_later_send(self):
        self.replies=[TimeoutError(),self.response([])];client=self.client(max_wall_s=.1)
        with self.assertRaises(self.api.TradeSourceError):client.read(dict(PARAMS))
        self.assertEqual(len(self.calls),1);self.assertLessEqual(self.calls[0]['timeout'],.1)
        self.assertLessEqual(self.now,.1+1e-9)

    def test_HT03_missing_audit_or_relaxed_budget_is_rejected(self):
        for kwargs in (dict(audit=None),dict(audit=self.audit,max_attempts=301),dict(audit=self.audit,max_wall_s=901)):
            with self.assertRaises(self.api.TradeSourceError):self.api.HistoricalTradeClient(**kwargs)

    def test_HT08_trade_audit_rejects_cache_receipt_or_content_tamper(self):
        self.replies=[self.response()];rows,meta=self.client().read(dict(PARAMS));target=self.cache(rows,meta);self.audit.close()
        value=json.loads(target.read_text());value['metadata']['received_at_ms']+=1;target.write_text(json.dumps(value))
        self.assertEqual(self.api.verify_trade_audit(self.folder,self.audit.path)['state'],'FAIL')

    def test_HT08_missing_provenance_and_open_run_are_unknown(self):
        self.replies=[self.response()];rows,meta=self.client().read(dict(PARAMS));self.cache(rows,meta)
        self.assertEqual(self.api.verify_trade_audit(self.folder,self.audit.path)['state'],'UNKNOWN')

    def test_HT02_transport_pins_host_tls_route_and_rejects_private_scope(self):
        transport=self.api.HistoricalTradeTransport()
        for method,path,query,headers,timeout in [('POST','/api/v3/aggTrades','',{},1),
                ('GET','/api/v3/order','',{},1),('GET','/api/v3/aggTrades','signature=x',{},1),
                ('GET','/api/v3/aggTrades','',{'X-MBX-APIKEY':'x'},1),
                ('GET','/api/v3/aggTrades','',{},16)]:
            with self.assertRaises(self.api.TradeSourceError):transport.request(method,path,query,headers,timeout)
        outer=self;calls=[]
        class Sock:
            def settimeout(self,value):outer.assertLessEqual(value,15)
        class Reply:
            status=200
            def read(self,n):return b'[]'
            def getheaders(self):return []
        class Connection:
            def __init__(self,host,port,timeout,context):
                calls.append((host,port,context.check_hostname));self.sock=Sock()
            def connect(self):pass
            def request(self,method,path,headers):calls.append((method,path,headers))
            def getresponse(self):return Reply()
            def close(self):pass
        with patch('http.client.HTTPSConnection',Connection):
            response=transport.request('GET','/api/v3/aggTrades','symbol=BTCUSDT&fromId=0&limit=1',{},1)
        self.assertEqual(response.body,b'[]');self.assertEqual(calls[0],('demo-api.binance.com',443,True))
        transport.close()


if __name__=='__main__':unittest.main()
