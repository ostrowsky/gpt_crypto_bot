"""HT01/04/09: sealed source scope, pagination and honest availability reports."""
from contextlib import closing
import hashlib
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from urllib.parse import parse_qs
import zlib

import _support
import test_intrabar_probe as old_fixture
from binance_demo_bot.adapter import DemoClock, DemoReadClient, HttpResponse, _sha
from binance_demo_bot.interaction_audit import AuditJournal
from binance_demo_bot.isolation import AppPaths
from binance_demo_bot.intrabar_probe import register_probe_run, collect_probe_run, finalize_probe_run
from binance_demo_bot.historical_trade_probe import register_trade_run, collect_trade_run
from binance_demo_bot.historical_trade_source import TradeSourceError


WINDOWS=(('DEXEUSDT',1750797000000),('GMXUSDT',1752068700000),('NXPCUSDT',1749808800000))


def candles(at):
    return [[at+i*60000,'100','101','99','100','4.00000000',at+(i+1)*60000-1,
             '400.00000000',4,'4.00000000','400.00000000','0'] for i in range(15)]


def trade(i,at,price='100'):
    return dict(a=i,p=price,q='1.00000000',f=i,l=i,T=at,m=False,M=True)


class FakeTrades:
    def __init__(self,*,empty=False,error=False,gap=False):
        self.calls=[]; self.empty=empty; self.error=error; self.gap=gap
    def read(self,params):
        self.calls.append(dict(params))
        if self.error: raise TradeSourceError('READ_UNAVAILABLE')
        at=dict(WINDOWS)[params['symbol']]
        if self.empty: rows=[]
        elif 'startTime' in params:
            rows=[trade(1+i*4+j,at+i*60000+j*1000,price)
                  for i in range(15) for j,price in enumerate(('100','101','99','100'))]
            if self.gap: rows[2]['a']+=100
        elif params['fromId']==0: rows=[trade(0,at-1)]
        else: rows=[trade(61,at+900000)]
        return rows,dict(received_at_ms=2000000000000,status=200,
                         raw_body_sha256=hashlib.sha256(json.dumps(rows).encode()).hexdigest())


class HistoricalTradeProbeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # A complete synthetic parent with actual native journal links, no HTTP.
        base=old_fixture.IntrabarProbeTest('runTest'); base.setUp(); cls.base=base
        base.folder=base.root/'.runtime/parent'
        with closing(sqlite3.connect(base.archive,isolation_level=None)) as con:
            entries={}
            for symbol,start,body in con.execute('SELECT symbol,start_ms,body FROM pages').fetchall():
                rows=json.loads(zlib.decompress(body))
                for row in rows:
                    row[5]='60.00000000'; row[7]='6000.00000000'; row[8]=60
                    row[9]='60.00000000'; row[10]='6000.00000000'
                blob=zlib.compress(json.dumps(rows).encode()); bh=hashlib.sha256(blob).hexdigest(); sh=_sha(rows)
                con.execute('UPDATE pages SET body=?,source_sha=?,blob_sha=? WHERE symbol=? AND start_ms=?',
                            (blob,sh,bh,symbol,start))
                entries[symbol]=dict(source_pages_sha256=_sha([[start,len(rows),rows[0][0],rows[-1][0],sh,bh]]))
        mp=base.prepared/'manifest.json'; prepared=json.loads(mp.read_text()); prepared['entries']=entries
        prepared['data_sha256']=_sha({k:v for k,v in prepared.items() if k!='data_sha256'})
        mp.write_text(json.dumps(prepared),encoding='utf-8')
        rp=base.rde/'registration.json'; reg=json.loads(rp.read_text()); reg['data_sha256']=prepared['data_sha256']
        rp.write_text(json.dumps(reg),encoding='utf-8')
        (base.rde/'frozen-encoder.json').write_text(json.dumps(dict(data_sha256=prepared['data_sha256'],
            registration_sha256=hashlib.sha256(rp.read_bytes()).hexdigest())),encoding='utf-8')
        (base.rde/'portfolios').mkdir()
        for model in ('CatBoost','GRU'):
            trades=[dict(symbol=s,entry_ms=at,exit_ms=at,entry_price=100.05,reason='SL')
                    for s,at in WINDOWS if model=='CatBoost' or s=='NXPCUSDT']
            (base.rde/'portfolios'/('test-base-'+model+'.json')).write_text(json.dumps(dict(trades=trades)),encoding='utf-8')
        register_probe_run(base.folder,archive=base.archive,rde_folder=base.rde,spec_path=base.spec)
        paths=AppPaths(base.root)
        audit=AuditJournal(paths,account_alias='public-research',episode_id='parent',scope_id='native-intrabar-source',
                           database='parent/rest-audit.sqlite')
        class Wire:
            def request(self,method,path,query,headers,timeout_s):
                params={k:v[0] for k,v in parse_qs(query).items()}; at=int(params['startTime'])
                rows=candles(at) if params['interval']=='1m' else []
                return HttpResponse(200,json.dumps(rows).encode(),{})
            def close(self): pass
        client=DemoReadClient(audit=audit,transport=Wire(),sleep=lambda _:None,
            clock=DemoClock(wall_ms=lambda:2000000000000))
        try: collect_probe_run(base.folder,client=client,sleep=lambda _:None)
        finally: client.close(); audit.close()
        finalized=finalize_probe_run(base.folder,base.folder/'rest-audit.sqlite')
        assert finalized['native_audit_certification']=='VERIFIED_RECORDED_PUBLIC_ATTEMPTS'
        cls.spec=base.root/'trades-spec.md'; cls.spec.write_text('sealed HT01..09 protocol',encoding='utf-8')

    @classmethod
    def tearDownClass(cls): cls.base.doCleanups()

    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(dir=self.base.root); self.addCleanup(self.tmp.cleanup)
        self.folder=Path(self.tmp.name)/'child'
    def register(self):
        return register_trade_run(self.folder,parent_folder=self.base.folder,spec_path=self.spec)
    def collect(self,client,**kwargs): return collect_trade_run(self.folder,client=client,**kwargs)

    def test_HT01_exact_scope_four_bindings_and_immutable_registration(self):
        r=self.register()
        self.assertEqual({(w['symbol'],w['origin_ms']) for w in r['windows']},set(WINDOWS))
        self.assertEqual(sum(len(w['brackets']) for w in r['windows']),4)
        before=(self.folder/'registration.json').read_bytes(); self.register()
        self.assertEqual(before,(self.folder/'registration.json').read_bytes())
        self.assertFalse(r['promotion_allowed'])

    def test_HT01_spec_change_blocks_before_get(self):
        self.register(); before=self.spec.read_bytes(); self.spec.write_text('changed')
        c=FakeTrades()
        try:
            with self.assertRaisesRegex(ValueError,'BINDING'): self.collect(c)
        finally: self.spec.write_bytes(before)
        self.assertEqual(c.calls,[])

    def test_HT04_empty_success_is_three_incomplete_windows_not_flat_or_api_error(self):
        self.register(); c=FakeTrades(empty=True); r=self.collect(c)
        self.assertEqual(r['state'],'INCOMPLETE'); self.assertEqual(len(c.calls),3)
        self.assertEqual(r['history_empty_n'],3); self.assertEqual(r['api_error_n'],0)
        self.assertEqual(r['unresolved_bindings_n'],4)
        self.assertFalse(r['pnl_repriced']); self.assertFalse(r['full_portfolio_replayed'])

    def test_HT04_http_error_is_not_empty_history(self):
        self.register(); r=self.collect(FakeTrades(error=True))
        self.assertEqual(r['state'],'UNKNOWN'); self.assertEqual(r['history_empty_n'],0)
        self.assertEqual(r['api_error_n'],3); self.assertEqual(r['unresolved_bindings_n'],4)

    def test_HT04_short_initial_page_still_needs_both_guards(self):
        self.register(); c=FakeTrades(); r=self.collect(c)
        self.assertEqual(r['state'],'SOURCE_GATE_ONLY_COMPLETE')
        self.assertEqual(r['matched_windows_n'],3); self.assertEqual(r['resolved_bindings_n'],4)
        self.assertEqual(len(c.calls),9)
        initial=[p for p in c.calls if 'startTime' in p]
        self.assertEqual(len(initial),3)
        self.assertTrue(all(p['endTime']==p['startTime']+899999 for p in initial))
        self.assertEqual(sorted(p['fromId'] for p in c.calls if 'fromId' in p),[0,0,0,61,61,61])

    def test_HT04_page_budget_never_certifies_partial_path(self):
        self.register(); r=self.collect(FakeTrades(),max_pages=1)
        self.assertEqual(r['state'],'INCOMPLETE'); self.assertEqual(r['resolved_bindings_n'],0)

    def test_HT05_gapped_data_never_gets_order_verdict(self):
        self.register(); r=self.collect(FakeTrades(gap=True))
        self.assertEqual(r['state'],'INCOMPLETE'); self.assertEqual(r['resolved_bindings_n'],0)

    def test_HT01_cached_response_resume_no_get_and_tamper_is_rejected(self):
        self.register(); self.collect(FakeTrades(empty=True)); c=FakeTrades(error=True)
        r=self.collect(c); self.assertEqual(c.calls,[]); self.assertEqual(r['history_empty_n'],3)
        p=next((self.folder/'responses').glob('*.json')); d=json.loads(p.read_text()); d['rows']=[{}]
        p.write_text(json.dumps(d),encoding='utf-8')
        with self.assertRaisesRegex(ValueError,'INTEGRITY'): self.collect(FakeTrades())

    def test_HT01_parent_cache_change_blocks_before_get(self):
        self.register(); p=next((self.base.folder/'responses').glob('*.json')); raw=p.read_bytes()
        p.write_bytes(raw+b' '); c=FakeTrades()
        try:
            with self.assertRaises(ValueError): self.collect(c)
        finally: p.write_bytes(raw)
        self.assertEqual(c.calls,[])


if __name__=='__main__': unittest.main()
