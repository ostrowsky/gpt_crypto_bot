"""R2A01/02/05/06 source gate: real protocol contracts, synthetic public data."""
import hashlib
from contextlib import closing
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
import zlib

from binance_demo_bot.adapter import AdapterError, _sha
from binance_demo_bot.intrabar_probe import register_probe_run, collect_probe_run
from binance_demo_bot import intrabar_probe as probe
from binance_demo_bot.journal import _WriterLease, WriterBusy


AMBIGUOUS = [('TSTUSDT',1749763800000),('NXPCUSDT',1749808800000),
             ('NXPCUSDT',1749809700000),('PROMUSDT',1750638600000),
             ('DEXEUSDT',1750797000000),('GMXUSDT',1752068700000)]


def bar(at, interval=900000, volume='15.00000000', trades=15):
    return [at,'100.00000000','101.00000000','99.00000000','100.00000000',
            volume,at+interval-1,volume,trades,'0.00000000','0.00000000','0']


class FakeClient:
    def __init__(self, *, error=None, empty=False, mismatch=False):
        self.calls=[]; self.error=error; self.empty=empty; self.mismatch=mismatch

    def request_public(self,path,params,*,weight):
        self.calls.append((path,params,weight))
        if self.error: raise AdapterError(self.error)
        at=params['startTime']; interval=60000 if params['interval']=='1m' else 1000
        rows=[bar(at+i*interval,interval, '1.00000000' if interval==60000 else '0.01666667',
                  1 if interval==60000 else int(i%60==0)) for i in range(900000//interval)]
        if self.mismatch: rows[0][2]='102.00000000'
        return ([] if self.empty else rows),dict(received_at_ms=2000000000000,
            response_started_at_ms=2000000000000-10,rate_headers={})


class IntrabarProbeTest(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]/'.runtime'); self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name); self.archive=self.root/'history.sqlite'
        self.rde=self.root/'RDE-1'; self.rde.mkdir(); self.folder=self.root/'RDE-2'
        self.spec=self.root/'spec.md'; self.spec.write_text('frozen source protocol',encoding='utf-8')
        symbols=sorted(set(s for s,_ in AMBIGUOUS)|{'BTCUSDT','ETHUSDT','SOLUSDT'})
        registration=dict(attempt='RDE-1-MAX519-20261029',symbols=symbols,
            splits=[['train',0,900000],['tune',900000,1749659400000],
                    ['test',1749659400000,1793197800000]],
            data_sha256='a'*64,source_sha256={},promotion_allowed=False)
        self.prepared=self.root/'prepared'; self.prepared.mkdir()
        entries={}
        with closing(sqlite3.connect(self.archive,isolation_level=None)) as con:
            con.execute('CREATE TABLE pages(symbol TEXT,start_ms INTEGER,rows_n INTEGER,first_ms INTEGER,last_ms INTEGER,source_sha TEXT,blob_sha TEXT,body BLOB,received_ms INTEGER)')
            for symbol in symbols:
                ats=sorted({a for s,a in AMBIGUOUS if s==symbol}|{0,1749659400000,1793196900000})
                rows=[bar(a) for a in ats]; blob=zlib.compress(json.dumps(rows).encode())
                con.execute('INSERT INTO pages VALUES(?,?,?,?,?,?,?,?,?)',
                    (symbol,0,len(rows),ats[0],ats[-1],_sha(rows),hashlib.sha256(blob).hexdigest(),blob,2000000000000))
                entries[symbol]=dict(source_pages_sha256=_sha([[0,len(rows),ats[0],ats[-1],_sha(rows),hashlib.sha256(blob).hexdigest()]]))
        prepared=dict(archive=str(self.archive.resolve()),entries=entries,symbols=symbols,cutoff_ms=1793197800000)
        prepared['data_sha256']=_sha(prepared); registration['data_sha256']=prepared['data_sha256']
        (self.prepared/'manifest.json').write_text(json.dumps(prepared),encoding='utf-8')
        (self.rde/'registration.json').write_text(json.dumps(registration),encoding='utf-8')
        (self.rde/'frozen-encoder.json').write_text(json.dumps(dict(data_sha256=prepared['data_sha256'],
            registration_sha256=hashlib.sha256((self.rde/'registration.json').read_bytes()).hexdigest())),encoding='utf-8')

    def register(self):
        return register_probe_run(self.folder,archive=self.archive,rde_folder=self.rde,spec_path=self.spec)

    def collect(self, client, **kwargs):
        return collect_probe_run(self.folder,client=client,sleep=lambda _:None,**kwargs)

    def test_R2A02_fixed_windows_and_immutable_registration(self):
        registered=self.register()
        windows={(r['symbol'],r['origin_ms']) for r in registered['windows']}
        self.assertEqual(len(windows),15)
        self.assertTrue(set(AMBIGUOUS)<=windows)
        self.assertEqual(registered['promotion_allowed'],False)
        self.assertEqual(registered['truth_harness_at_registration']['status'],'UNKNOWN')
        before=(self.folder/'registration.json').read_bytes()
        self.register(); self.assertEqual(before,(self.folder/'registration.json').read_bytes())
        self.spec.write_text('changed protocol',encoding='utf-8')
        with self.assertRaisesRegex(ValueError,'BINDING'): self.register()

    def test_R2A02_truth_report_is_bound_and_not_hardcoded(self):
        truth=self.root/'truth.json'
        truth.write_text(json.dumps(dict(profile='full',status='fail',findings=[{'invariant':'TH-11'}])),encoding='utf-8')
        registered=register_probe_run(self.folder,archive=self.archive,rde_folder=self.rde,
                                    spec_path=self.spec,truth_report=truth)
        self.assertEqual(registered['truth_harness_at_registration']['status'],'FAIL')
        self.assertEqual(registered['truth_harness_at_registration']['sha256'],hashlib.sha256(truth.read_bytes()).hexdigest())
        truth.write_text(json.dumps(dict(profile='full',status='pass')),encoding='utf-8')
        client=FakeClient()
        with self.assertRaisesRegex(ValueError,'BINDING'): self.collect(client)
        self.assertEqual(client.calls,[])

    def test_R2A01_GET_same_source_parameters_and_all_probes(self):
        self.register(); client=FakeClient()
        report=self.collect(client,include_seconds=False)
        self.assertEqual(report['state'],'MINUTES_ONLY')
        self.assertEqual(report['match_n'],15)
        self.assertEqual(len(client.calls),15)
        for path,p,w in client.calls:
            self.assertEqual(path,'/api/v3/klines'); self.assertEqual(w,2)
            self.assertEqual(set(p),{'symbol','interval','timeZone','startTime','endTime','limit'})
            self.assertEqual(p['interval'],'1m'); self.assertEqual(p['timeZone'],'0')
            self.assertEqual(p['limit'],15); self.assertEqual(p['endTime'],p['startTime']+899999)
        self.assertFalse(report['full_replay_complete'])

    def test_R2A05_success_resume_never_refetches(self):
        self.register(); client=FakeClient(); self.collect(client,include_seconds=False)
        report=self.collect(FakeClient(error='READ_UNAVAILABLE'),include_seconds=False)
        self.assertEqual(report['state'],'MINUTES_ONLY')
        self.assertEqual(report['fetched_n'],0)
        self.assertEqual(report['cached_n'],15)

    def test_R2A02_replacement_archive_rejected(self):
        replacement=self.root/'other.sqlite'; replacement.write_bytes(self.archive.read_bytes())
        with self.assertRaisesRegex(ValueError,'BINDING'):
            register_probe_run(self.folder,archive=replacement,rde_folder=self.rde,spec_path=self.spec)

    def test_R2A02_parent_data_sha_tamper_rejected(self):
        path=self.rde/'registration.json'; obj=json.loads(path.read_text()); obj['data_sha256']='b'*64
        path.write_text(json.dumps(obj),encoding='utf-8')
        with self.assertRaisesRegex(ValueError,'BINDING'): self.register()

    def test_R2A06_seconds_for_all_six_after_minute_success(self):
        self.register(); client=FakeClient(); report=self.collect(client)
        self.assertEqual(report['state'],'PASS_GATE_ONLY')
        self.assertEqual(report['seconds_requested_n'],6)
        seconds=[p for _,p,_ in client.calls if p['interval']=='1s']
        self.assertEqual({(p['symbol'],p['startTime']) for p in seconds},set(AMBIGUOUS))
        self.assertEqual(report['match_n'],21)
        self.assertTrue(all(p['limit']==900 for p in seconds))

    def test_R2A05_response_tamper_rejected_before_use(self):
        self.register(); self.collect(FakeClient(),include_seconds=False)
        path=next((self.folder/'responses').glob('*.json'))
        obj=json.loads(path.read_text()); obj['rows'][0][2]='10000'
        path.write_text(json.dumps(obj),encoding='utf-8')
        with self.assertRaisesRegex(ValueError,'INTEGRITY'):
            self.collect(FakeClient(),include_seconds=False)

    def test_R2A05_original_binding_change_rejected_before_network(self):
        self.register(); client=FakeClient()
        with closing(sqlite3.connect(self.archive,isolation_level=None)) as con: con.execute("UPDATE pages SET blob_sha='bad' WHERE symbol='BTCUSDT'")
        with self.assertRaises(ValueError): self.collect(client,include_seconds=False)
        self.assertEqual(client.calls,[])

    def test_R2A06_no_fallback_on_mismatch_and_no_seconds(self):
        self.register(); client=FakeClient(mismatch=True)
        report=self.collect(client)
        self.assertEqual(report['state'],'SOURCE_INCOMPATIBLE')
        self.assertEqual(report['mismatch_n'],15)
        self.assertEqual(len(client.calls),15)
        self.assertEqual(report['seconds_requested_n'],0)

    def test_R2A06_absence_is_incomplete_not_success(self):
        self.register(); report=self.collect(FakeClient(empty=True))
        self.assertEqual(report['state'],'INCOMPLETE')
        self.assertEqual(report['incomplete_n'],15)
        self.assertEqual(report['match_n'],0)

    def test_R2A05_unavailable_and_budget_are_unknown(self):
        self.register(); client=FakeClient(error='READ_UNAVAILABLE')
        report=self.collect(client,max_attempts=3)
        self.assertEqual(report['state'],'UNKNOWN')
        self.assertEqual(len(client.calls),1)  # client itself has <=3 transport attempts
        self.assertGreater(report['unavailable_n'],0)
        self.assertGreater(report['pending_n'],0)

    def test_R2A05_invalid_budgets_make_no_request(self):
        self.register(); client=FakeClient()
        for kwargs in ({'max_attempts':2001},{'max_attempts':True},{'max_wall_s':3601},{'max_wall_s':0}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError): self.collect(client,**kwargs)
        self.assertEqual(client.calls,[])

    def test_R2A05_exclusive_writer_before_network(self):
        self.register(); client=FakeClient(); lease=_WriterLease(self.folder/'probe.writer.lock')
        try:
            with self.assertRaises(WriterBusy): self.collect(client)
            self.assertEqual(client.calls,[])
        finally: lease.close()

    def test_R2A05_pace_every_native_attempt_and_deadline(self):
        now=[0.0]; seen=[]
        class Wire:
            def request(self,method,path,query,headers,timeout_s):
                seen.append((now[0],timeout_s)); return 'response'
            def close(self): pass
        def sleep(duration): now[0]+=duration
        paced=probe.PacedReadTransport(transport=Wire(),deadline=1.0,monotonic=lambda:now[0],sleep=sleep)
        for _ in range(3): paced.request('GET','/api/v3/klines','symbol=BTCUSDT',{},15)
        self.assertEqual([round(t,3) for t,_ in seen],[0,.3,.6])
        self.assertAlmostEqual(seen[-1][1],.4)
        with self.assertRaises(AdapterError): paced.request('POST','/api/v3/order','',{},15)
        now[0]=1.0
        with self.assertRaises(AdapterError): paced.request('GET','/api/v3/klines','',{},15)
        self.assertEqual(len(seen),3)

    def test_R2A05_finalization_keeps_source_and_audit_verdicts_separate(self):
        self.register(); self.collect(FakeClient())
        for audit_state in ('FAIL','UNKNOWN','VERIFIED_RECORDED_PUBLIC_ATTEMPTS'):
            with self.subTest(audit_state=audit_state):
                result=probe.finalize_probe_run(self.folder,self.root/'audit.sqlite',
                    verifier=lambda folder,path:dict(state=audit_state,reasons=[]))
                self.assertEqual(result['state'],'PASS_GATE_ONLY')
                self.assertEqual(result['native_audit_certification'],audit_state)
                self.assertEqual(result['overall_state'],'PASS_GATE_ONLY' if audit_state.startswith('VERIFIED') else 'UNKNOWN')
                self.assertFalse(result['full_replay_complete'])


if __name__=='__main__': unittest.main()
