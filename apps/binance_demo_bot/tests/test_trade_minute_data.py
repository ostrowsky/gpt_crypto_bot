"""TA08/09 collector provenance acceptance, tests before implementation."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import _support
from test_trade_minute_core import activity_frame
from binance_demo_bot.trade_minute_data import validated_frame, verify_prior_cache, fetch, audit
from binance_demo_bot.minute_replay_data import APP, query_parameters, frame_key
from binance_demo_bot.intrabar_probe import _seal,_bytes_sha,_write
from binance_demo_bot.intrabar_data import canonical_sha256
from binance_demo_bot.adapter import HttpResponse

class TradeDataTests(unittest.TestCase):
    def setUp(self):
        import time
        root=APP/'.runtime/minute-portfolio-replay';root.mkdir(exist_ok=True)
        self.tmp=tempfile.TemporaryDirectory(dir=root);self.addCleanup(self.tmp.cleanup);self.f=Path(self.tmp.name)
        source=self.f/'source.py';source.write_text('frozen research source',encoding='utf-8')
        _seal(self.f/'registration.json',dict(protocol='MRP-1-NATIVE-MINUTE-PORTFOLIO',files={str(source):_bytes_sha(source)}))
        _write(self.f/'source-snapshot.json',{str(source):dict(path=str(source),registered_sha256=_bytes_sha(source),sha256=_bytes_sha(source))})
        _seal(self.f/'owner.json',dict(registration_sha256=_bytes_sha(self.f/'registration.json')))
        _seal(self.f/'data-scope.json',dict(symbols=['AAAUSDT'],start=0,end=2*900000,expires_at_ms=time.time_ns()//1000000+60000,attempt_limit=2000,files={}))
        self.d=dict(origin_ms=0,candidate='AAAUSDT',action=2,source='FROZEN_POLICY',state_sha256='a'*64)
        self.d['sha256']=canonical_sha256(self.d);self.dp=self.f/'decisions'/(self.d['sha256']+'.json');_seal(self.dp,self.d)

    def capture(self):
        from binance_demo_bot.minute_replay_data import fetch as old_fetch
        rows=activity_frame(active={3:(100,100.1,99.9,100)},carry=98)['rows']+activity_frame(900000)['rows'][:3]
        class Transport:
            def request(s,*args):return HttpResponse(200,json.dumps(rows).encode(),{})
            def close(s):pass
        with patch('binance_demo_bot.intrabar_probe.PacedReadTransport',return_value=Transport()):
            old_fetch(self.f,symbol='AAAUSDT',at=0,decision_path=self.dp)

    def test_TA09_prior_raw_wire_verified_without_requiring_future_batch_complete(self):
        self.capture();proof=verify_prior_cache(self.f)
        self.assertEqual(proof['attempts_n'],1);self.assertEqual(len(proof['raw_frames']),2)
        self.assertEqual(len(next(r['value']['rows'] for r in proof['raw_frames'] if r['value']['origin_ms']==900000)),3)

    def test_TA09_resigned_raw_cache_tamper_rejected(self):
        self.capture();path=next((self.f/'raw-frames').glob('*.json'));r=json.loads(path.read_text());r['rows'][0][1]='200'
        _write(path,r);path.with_suffix('.json.sha256').write_text(_bytes_sha(path),encoding='ascii')
        with self.assertRaises(ValueError):verify_prior_cache(self.f)

    def test_TA09_source_snapshot_tamper_rejected(self):
        self.capture();(self.f/'source.py').write_text('changed',encoding='utf-8')
        with self.assertRaises(ValueError):verify_prior_cache(self.f)

    def test_TA04_frame_preserves_old_gate_failure_and_new_semantics(self):
        f=activity_frame(active={3:(100,100.1,99.9,100)},carry=98)
        out=validated_frame('AAAUSDT',0,dict(row=f['original'],page_source_sha256='b'*64),
            dict(rows=f['rows'],metadata=dict(received_at_ms=900000)),'sealed','a'*64,imported=True)
        self.assertEqual(out['source_check']['state'],'MATCH_TRADE_AWARE')
        self.assertEqual(out['source_check']['strict']['state'],'MISMATCH')

if __name__=='__main__':unittest.main()
