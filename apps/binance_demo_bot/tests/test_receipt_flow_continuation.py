"""JR03/04: economic identity, complete-case reuse and module-local binding."""
from copy import deepcopy
from dataclasses import asdict
import gzip
import json
from pathlib import Path
import tempfile
import unittest
from binance_demo_bot import receipt_flow_continuation as continuation
from binance_demo_bot import receipt_flow_direction as io
from binance_demo_bot.receipt_flow_portfolio import Policy,simulate
from binance_demo_bot.receipt_flow_portfolio_audit import audit_events


class ContinuationContracts(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]/'.runtime')
        self.addCleanup(self.tmp.cleanup);self.base=Path(self.tmp.name)
        self.parent=self.base/'parent';self.parent.mkdir();self.child=self.base/'child';self.child.mkdir()
        self.case=dict(case_id='M0-h5-direction-base',arm='M0',horizon_seconds=5,admission='direction',profile='base')
        source=self.base/'manifest.json';io.write_json(source,{'source':'fixed'})
        self.reg=dict(cases=[self.case],inputs={str(source):io.file_sha(source)},profiles={'base':{'fee_bps':'10'}})
        for root in (self.parent,self.child):
            io.freeze(root/'registration.json',self.reg)
            io.freeze(root/'microprice-calibration.json',{'5':{'intercept':0.,'slope':1.}})
            io.freeze(root/'signals-manifest.json',{'source':'same'})
        self.policy=Policy();self.events=[]
        def quote(s,t):return dict(symbol=s,bid='100',ask='101',bid_qty='10',ask_qty='10',receipt_ms=t-1,
                                   session_id='s',chunk='fixture',chunk_sha256='a'*64,record_index=0)
        self.result=simulate({0:[dict(symbol='BTCUSDT',score_bps=2.,spread_bps=1.,source_row=0,feature_sha256='f'*64)]},
             start_ms=0,end_ms=9000,quote_at=quote,policy=self.policy,emit=self.events.append)
        out=self.parent/'cases'/self.case['case_id'];out.mkdir(parents=True)
        with gzip.open(out/'events.jsonl.gz','wt',encoding='utf-8') as stream:
            for e in self.events:stream.write(json.dumps(e,sort_keys=True,separators=(',',':'))+'\n')
        io.freeze(out/'result.json',self.result)
        io.freeze(out/'audit.json',audit_events(iter(self.events),self.result,self.policy))
        io.freeze(out/'manifest.json',dict(case=self.case,policy=asdict(self.policy),
            registration_sha256=io.file_sha(self.parent/'registration.json'),
            result_sha256=io.file_sha(out/'result.json'),events_sha256=io.file_sha(out/'events.jsonl.gz'),
            audit_sha256=io.file_sha(out/'audit.json'),microprice_calibration_sha256=io.file_sha(self.parent/'microprice-calibration.json')))

    def test_JR04_reuse_replays_exact_events_and_reaudits_economics(self):
        reuse=continuation.VerifiedParent(self.parent);received=[]
        result=reuse.replay(self.case['case_id'],self.child,self.policy,received.append)
        self.assertEqual(received,self.events);self.assertEqual(result,self.result)
        self.assertEqual(audit_events(iter(received),result,self.policy)['closed_cycles_n'],1)

    def test_JR03_changed_registration_or_policy_is_rejected(self):
        reuse=continuation.VerifiedParent(self.parent)
        with self.assertRaises(ValueError):reuse.replay(self.case['case_id'],self.child,Policy(fee_bps='0'),lambda e:None)
        other=self.base/'other';other.mkdir()
        changed=deepcopy(self.reg);changed['profiles']['base']['fee_bps']='0'
        io.freeze(other/'registration.json',changed)
        with self.assertRaises(ValueError):reuse.replay(self.case['case_id'],other,self.policy,lambda e:None)

    def test_JR04_corrupt_sealed_case_is_not_silently_recomputed(self):
        path=self.parent/'cases'/self.case['case_id']/'events.jsonl.gz'
        path.write_bytes(path.read_bytes()+b'x')
        with self.assertRaises(ValueError):continuation.VerifiedParent(self.parent)

    def test_JR05_read_mode_delegates_and_does_not_patch_global_gzip(self):
        path=self.parent/'cases'/self.case['case_id']/'events.jsonl.gz'
        original=gzip.open
        facade=continuation.JournalFacade(lambda case,writer:None)
        with facade.open(path,'rt',encoding='utf-8') as stream:self.assertIn('INITIAL_EQUITY',stream.readline())
        self.assertIs(gzip.open,original)


if __name__=='__main__':unittest.main()
