"""MR07..10 bounded native-data provenance, first RED before collector code."""
import json
from pathlib import Path
import tempfile
import unittest
import _support
from test_minute_replay_core import frame
from binance_demo_bot.minute_replay_data import query_parameters, accept_batch, validate_decision, scope_allows, verify_response_link
from binance_demo_bot.intrabar_data import canonical_sha256

class MinuteDataTests(unittest.TestCase):
    def test_MR08_scope_only_registered_past_demo_minutes(self):
        scope=dict(symbols=['AAAUSDT'],start=0,end=30*900000,expires_at_ms=999999999,attempt_limit=2000)
        p=query_parameters(scope,'AAAUSDT',0)
        self.assertEqual(p,dict(symbol='AAAUSDT',interval='1m',timeZone='0',startTime=0,endTime=16*900000-1,limit=240))
        self.assertEqual(query_parameters(scope,'AAAUSDT',29*900000)['limit'],15)
        for symbol,at in (('OTHERUSDT',0),('AAAUSDT',-900000),('AAAUSDT',30*900000),('AAAUSDT',1)):
            with self.assertRaises(ValueError):query_parameters(scope,symbol,at)
        with self.assertRaises(ValueError):scope_allows(scope,1998,now_ms=10)
        with self.assertRaises(ValueError):scope_allows(scope,0,now_ms=999999999)
        scope_allows(scope,1997,now_ms=10)

    def test_MR08_frozen_decision_hash_and_nonwait_required(self):
        d=dict(origin_ms=0,candidate='AAAUSDT',action=2,source='FROZEN_POLICY',state_sha256='a'*64)
        d['sha256']=canonical_sha256(d);validate_decision(d,0)
        d['action']=0
        with self.assertRaises(ValueError):validate_decision(d,0)
        with self.assertRaises(ValueError):validate_decision(dict(d,sha256=canonical_sha256({k:v for k,v in d.items() if k!='sha256'})),900000)

    def test_MR09_partial_future_does_not_invalidate_current_frame(self):
        f=frame(); rows=f['rows']+frame(900000)['rows'][:3]
        batches=accept_batch(rows,dict(startTime=0,endTime=2*900000-1,limit=30),received_at_ms=2*900000)
        self.assertEqual(len(batches[0]),15);self.assertEqual(len(batches[900000]),3)
        for bad in (rows[::-1],rows+[rows[-1]],[[99]+rows[0][1:]]):
            with self.assertRaises(ValueError):accept_batch(bad,dict(startTime=0,endTime=2*900000-1,limit=30),received_at_ms=2*900000)

    def test_MR09_wire_and_receipt_links_cannot_be_invented(self):
        import base64,hashlib
        rows=frame()['rows'];raw=json.dumps(rows).encode();meta=dict(audit_request_id='r',audit_attempt_id='a',received_at_ms=900000,raw_body_sha256=hashlib.sha256(raw).hexdigest())
        response=dict(rows=rows,metadata=meta,source_sha256=canonical_sha256(rows))
        terminal=dict(request_id='r',attempt_id='a',received_at_ms=900000,outcome='RESPONSE',status=200,
            payload=dict(representation='EXACT_PUBLIC_BYTES',body_base64=base64.b64encode(raw).decode(),bytes_n=len(raw),content_sha256=hashlib.sha256(raw).hexdigest()))
        verify_response_link(response,terminal)
        for key,value in (('received_at_ms',900001),('status',201)):
            bad=dict(terminal,**{key:value})
            with self.assertRaises(ValueError):verify_response_link(response,bad)
        response['rows'][0][1]='101'
        with self.assertRaises(ValueError):verify_response_link(response,terminal)


class FrameBindingTests(unittest.TestCase):
    def test_MR07_original_page_identity_is_preserved_in_accepted_frame(self):
        from binance_demo_bot.minute_replay_data import validated_frame
        f=frame();original=dict(row=f['original'],page_source_sha256='a'*64,page_blob_sha256='b'*64)
        r=dict(rows=f['rows'],metadata=dict(received_at_ms=900000))
        accepted=validated_frame('AAAUSDT',0,original,r,'sealed-response','c'*64,imported=True)
        self.assertEqual(accepted['original_page_sha256'],'a'*64)
        self.assertEqual(accepted['original'],f['original'])
        self.assertEqual(accepted['source_sha256'],'c'*64)
        r['rows']=r['rows'][:-1]
        with self.assertRaises(ValueError):validated_frame('AAAUSDT',0,original,r,'sealed-response','c'*64,imported=True)

class CollectorIntegrationTests(unittest.TestCase):
    def setUp(self):
        import tempfile,time
        from binance_demo_bot.minute_replay_data import APP
        from binance_demo_bot.intrabar_probe import _seal,_bytes_sha
        root=APP/'.runtime/minute-portfolio-replay';root.mkdir(parents=True,exist_ok=True)
        self.tmp=tempfile.TemporaryDirectory(dir=root);self.addCleanup(self.tmp.cleanup);self.folder=Path(self.tmp.name)
        scope=dict(symbols=['AAAUSDT'],start=0,end=2*900000,expires_at_ms=time.time_ns()//1000000+60000,attempt_limit=2000,files={})
        _seal(self.folder/'data-scope.json',scope);_seal(self.folder/'registration.json',dict(protocol='synthetic-no-network'))
        _seal(self.folder/'owner.json',dict(registration_sha256=_bytes_sha(self.folder/'registration.json')))
        self.d=dict(origin_ms=0,candidate='AAAUSDT',action=2,source='FROZEN_POLICY',state_sha256='a'*64)
        self.d['sha256']=canonical_sha256(self.d);self.dp=self.folder/'decisions'/(self.d['sha256']+'.json');_seal(self.dp,self.d)

    def capture(self):
        from unittest.mock import patch
        from binance_demo_bot.adapter import HttpResponse
        from binance_demo_bot.minute_replay_data import fetch
        class Transport:
            def request(s,method,path,query,headers,timeout_s):
                assert method=='GET' and path=='/api/v3/klines' and headers=={}
                return HttpResponse(200,json.dumps(frame()['rows']+frame(900000)['rows'][:3]).encode(),{})
            def close(s):pass
        with patch('binance_demo_bot.intrabar_probe.PacedReadTransport',return_value=Transport()):
            return fetch(self.folder,symbol='AAAUSDT',at=0,decision_path=self.dp)

    def test_MR10_real_journal_collects_public_bytes_and_audits_without_network(self):
        from binance_demo_bot.minute_replay_data import audit
        result=self.capture();self.assertEqual(result['rows_n'],18)
        proof=audit(self.folder);self.assertEqual(proof['native_attempts_n'],1);self.assertEqual(proof['linked_new_responses_n'],1)
        self.assertEqual(proof['state'],'VERIFIED_RECORDED_NATIVE_MINUTE_DATA');self.assertEqual(proof['errors_n'],0)

    def test_MR10_query_tamper_detected_after_capture(self):
        from binance_demo_bot.minute_replay_data import audit
        from binance_demo_bot.intrabar_probe import _write,_bytes_sha
        self.capture();p=next((self.folder/'queries').glob('*.json'));r=json.loads(p.read_text());r['parameters']['symbol']='OTHERUSDT'
        _write(p,r);p.with_suffix('.json.sha256').write_text(_bytes_sha(p),encoding='ascii')
        with self.assertRaises(ValueError):audit(self.folder)

    def test_MR10_unregistered_symbol_no_journal_or_transport(self):
        from binance_demo_bot.minute_replay_data import fetch
        with self.assertRaises(ValueError):fetch(self.folder,symbol='OTHERUSDT',at=0,decision_path=self.dp)
        self.assertFalse((self.folder/'rest-audit.sqlite').exists())


class InterruptedQueryTests(unittest.TestCase):
    setUp=CollectorIntegrationTests.setUp
    capture=CollectorIntegrationTests.capture
    def test_MR10_unfinished_preexisting_query_cannot_reset_logical_attempt_budget(self):
        from binance_demo_bot.minute_replay_data import fetch,query_parameters
        from binance_demo_bot.intrabar_probe import _seal,_bytes_sha
        params=query_parameters(dict(symbols=['AAAUSDT'],start=0,end=1800000),'AAAUSDT',0)
        plan=dict(parameters=params,endpoint='https://demo-api.binance.com/api/v3/klines',
            owner_registration_sha256=_bytes_sha(self.folder/'registration.json'),decision_path=str(self.dp),
            decision_sha256=_bytes_sha(self.dp),symbol='AAAUSDT',origin_ms=0)
        _seal(self.folder/'queries'/(canonical_sha256(params)+'.json'),plan)
        with self.assertRaisesRegex(ValueError,'PREVIOUS_QUERY_UNFINISHED'):
            self.capture()
        self.assertFalse((self.folder/'rest-audit.sqlite').exists())

if __name__=='__main__':unittest.main()
