"""MR01..06 lossless lazy signals, not a different trading policy."""
from copy import deepcopy
import unittest
from binance_demo_bot import receipt_flow_lazy as lazy
from binance_demo_bot import receipt_flow_portfolio_study as old
from binance_demo_bot.receipt_flow_portfolio import Policy,simulate,digest


class GuardedData(dict):
    def __getitem__(self,key):
        if key=='targets':raise AssertionError('Execution may not read future labels')
        return super().__getitem__(key)


def fixture():
    times=[0,0,5000,10000,15000];symbols=[0,1,0,1,0]
    data=GuardedData(times=times,symbols=symbols,features_F=[[1.,10000/100.5],[2.,10000/100.5],
        [-1.,10000/100.5],[.5,10000/100.5],[3.,10000/100.5]],mids=[100.5]*5,targets=None)
    for a in ('P','B','F'):
        for h in (5,10,30):data[f'{a}_{h}']=[2.,3.,-1.,4.,1.]
    rows=[];names=['BTCUSDT','ETHUSDT']
    for i,t in enumerate(times):
        symbol=names[symbols[i]];features=data['features_F'][i]
        rows.append(dict(at_ms=t,feature_end_ms=t-1,symbol=symbol,source_row=i,
            features_sha256=digest(dict(source_row=i,at_ms=t,symbol=symbol,features=features)),
            micro_bps=features[0],spread_bps=features[1],origin_mid=100.5,
            predictions={f'{a}_{h}':data[f'{a}_{h}'][i] for a in ('P','B','F') for h in (5,10,30)}))
    return data,rows,names


class LazySignalContracts(unittest.TestCase):
    def test_MR01_all15_arm_horizon_signals_match_original_without_labels(self):
        data,rows,names=fixture();source=lazy.LazySignals(data,names,['micro_bps','spread_bps'],0,15000)
        coeffs={str(h):dict(intercept=.1,slope=2.) for h in (5,10,30)}
        for a in ('M0','M1','P','B','F'):
            for h in (5,10,30):
                view=source.for_case(a,h,coeffs)
                expected=old.signal_rows(rows[:4],arm=a,horizon=h,coefficients=coeffs)
                got=[r for t in (0,5000,10000) for r in view.get(t,[])]
                self.assertEqual(got,expected)
        self.assertEqual(source.evidence()['retained_origin_dicts_n'],0)

    def test_MR02_full_cohort_provenance_is_identical_canonical_hash(self):
        data,rows,names=fixture();source=lazy.LazySignals(data,names,['micro_bps','spread_bps'],0,15000)
        expected=digest([(r['at_ms'],r['symbol'],r['source_row'],r['features_sha256']) for r in rows[:4]])
        self.assertEqual(source.evidence()['provenance_sha256'],expected)
        self.assertEqual(source.evidence()['issued_rows_n'],4)

    def test_MR02_ledger_and_all_events_match_dense_original(self):
        data,rows,names=fixture();source=lazy.LazySignals(data,names,['micro_bps','spread_bps'],0,15000)
        dense={}
        for r in old.signal_rows(rows[:4],arm='F',horizon=5,coefficients={}):dense.setdefault(r['at_ms'],[]).append(r)
        def q(s,t):return dict(symbol=s,bid='100',ask='101',bid_qty='10',ask_qty='10',receipt_ms=t-1,
                               session_id='s',chunk='fixture',chunk_sha256='a'*64,record_index=0)
        a=simulate(dense,start_ms=0,end_ms=15000,quote_at=q,policy=Policy())
        b=simulate(source.for_case('F',5,{}),start_ms=0,end_ms=15000,quote_at=q,policy=Policy())
        self.assertEqual(a,b)

    def test_MR01_future_suffix_and_targets_cannot_change_current_rows(self):
        data,rows,names=fixture();changed=deepcopy(data);changed['F_5'][-1]=999999.
        changed['features_F'][-1]=[9999.,1.]
        a=lazy.LazySignals(data,names,['micro_bps','spread_bps'],0,20000).for_case('F',5,{})
        b=lazy.LazySignals(changed,names,['micro_bps','spread_bps'],0,20000).for_case('F',5,{})
        self.assertEqual(a.get(0,[]),b.get(0,[]))
        self.assertEqual(len(a.get(15000,[])),1)

    def test_MR03_quote_cache_size_does_not_change_past_asof_answers(self):
        from pathlib import Path
        import tempfile
        from test_receipt_flow_quotes import chunks_at,raw_book
        from binance_demo_bot.public_flow_dataset import FlowValidator
        from binance_demo_bot.receipt_flow_quotes import ReceiptQuotes
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]/'.runtime') as folder:
            v=FlowValidator();rows=[v.begin_session('s',wall_ms=1000,mono_ns=1000)]
            for i in range(4):rows.append(v.ingest(raw_book(i+1,str(100+i),str(101+i)),
                wall_ms=1000+i*100,mono_ns=1000+i*100,session_id='s'))
            manifest,hashes=chunks_at(Path(folder),[rows[:2],rows[2:3],rows[3:4],rows[4:]])
            small=ReceiptQuotes(folder,manifest,hashes,cache_limit=1)
            large=ReceiptQuotes(folder,manifest,hashes,cache_limit=4)
            for t in (1301,1001,1201,1101,2000):self.assertEqual(small.quote_at('BTCUSDT',t),large.quote_at('BTCUSDT',t))


if __name__=='__main__':unittest.main()
