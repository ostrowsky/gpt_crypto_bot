"""PR01..03: exact public query evidence survives credential redaction."""
import json
from pathlib import Path
import tempfile
import unittest

import _support
from binance_demo_bot.adapter import DemoReadClient, HttpResponse, _sha
from binance_demo_bot.isolation import AppPaths
from binance_demo_bot.intrabar_probe import _seal, _write, _bytes_sha
from binance_demo_bot.intrabar_audit import verify_probe_audit
from binance_demo_bot.public_research_audit import PublicResearchJournal, verify_committed_probe_audit


class Transport:
    def request(self,*args):
        rows=[[i*60000,'100','101','99','100','1',i*60000+59999,'100',1,'0','0','0'] for i in range(15)]
        return HttpResponse(200,json.dumps(rows).encode(),{})
    def close(self):pass


class PublicCommitmentTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]/'.runtime');self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.folder=self.root/'.runtime/probe';self.folder.mkdir(parents=True)
        self.audit=PublicResearchJournal(AppPaths(self.root),account_alias='public-research',episode_id='cookie-test',scope_id='proof',database='probe/audit.sqlite')
        self.addCleanup(self.audit.close)

    def capture(self):
        params=dict(symbol='COOKIEUSDT',interval='1m',timeZone='0',startTime=0,endTime=899999,limit=15)
        _seal(self.folder/'registration.json',dict(windows=[dict(symbol='COOKIEUSDT',origin_ms=0)]))
        client=DemoReadClient(audit=self.audit,transport=Transport())
        try:rows,meta=client.request_public('/api/v3/klines',params,weight=2)
        finally:client.close()
        path=self.folder/'responses'/(_sha(params)+'.json')
        _seal(path,dict(endpoint='https://demo-api.binance.com/api/v3/klines',parameters=params,rows=rows,metadata=meta,source_sha256=_sha(rows)))
        self.audit.close();return path

    def test_PR01_masking_retained_but_new_commitment_verifies(self):
        self.capture()
        legacy=verify_probe_audit(self.folder,self.folder/'audit.sqlite')
        self.assertEqual(legacy['state'],'FAIL')
        result=verify_committed_probe_audit(self.folder,self.folder/'audit.sqlite')
        self.assertEqual(result['state'],'VERIFIED_RECORDED_PUBLIC_ATTEMPTS')
        self.assertEqual(result['linked_cached_responses_n'],1)
        self.assertEqual(result['validation_profile'],'GET_DESCRIPTOR_COMMITMENT_V1')

    def test_PR02_resigned_cache_cannot_replace_request_symbol(self):
        path=self.capture();value=json.loads(path.read_text());value['parameters']['symbol']='BTCUSDT'
        _write(path,value);path.with_suffix('.json.sha256').write_text(_bytes_sha(path),encoding='ascii')
        self.assertEqual(verify_committed_probe_audit(self.folder,self.folder/'audit.sqlite')['state'],'FAIL')

    def test_PR02_resigned_cache_cannot_replace_wire_body(self):
        path=self.capture();value=json.loads(path.read_text());value['rows'][0][1]='200';value['source_sha256']=_sha(value['rows'])
        _write(path,value);path.with_suffix('.json.sha256').write_text(_bytes_sha(path),encoding='ascii')
        self.assertEqual(verify_committed_probe_audit(self.folder,self.folder/'audit.sqlite')['state'],'FAIL')

    def test_PR01_private_or_unknown_request_rejected(self):
        for method,endpoint,private in (('POST','/api/v3/klines',False),('GET','/api/v3/account',True),('GET','https://evil.invalid/a',False)):
            with self.subTest(endpoint=endpoint),self.assertRaises(ValueError):
                self.audit.append('REST_BEGIN',dict(method=method,endpoint=endpoint,private=private,parameters={}))


if __name__=='__main__':unittest.main()
