import unittest
import json
import tempfile
import asyncio
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import compare_price_volatility_bot as c

def market(n=96*15):
    t=np.arange(n)*c.STEP+1774130400000
    returns=.001*np.sin(np.arange(n)*.23)+.0002
    close=100*np.exp(np.cumsum(returns));open_=np.r_[close[0],close[:-1]]
    return dict(t=t,o=open_,h=np.maximum(open_,close)*1.001,l=np.minimum(open_,close)*.999,c=close,v=100+np.arange(n)%24)

def trade(sym,entry,exit,entry_price,exit_price,**kw):
    row=dict(sym=sym,entry_ts=entry,exit_ts=exit,entry_price=entry_price,exit_price=exit_price,
        partial_exit_taken=False,partial_exit_fraction=0,partial_exit_ts=0,partial_exit_price=0)
    row.update(kw);return SimpleNamespace(**row)

class ForecastTests(unittest.TestCase):
    def test_replacement_defaults_match_existing_evaluator_and_explicit_overrides(self):
        self.assertEqual(c.replacement_options(SimpleNamespace()),(True,8.0))
        self.assertEqual(c.replacement_options(SimpleNamespace(PORTFOLIO_REPLACE_ENABLED=False,PORTFOLIO_REPLACE_MIN_DELTA=12)),(False,12.0))

    def test_trusted_candidate_resume_rejects_hash_source_and_count_drift(self):
        with tempfile.TemporaryDirectory() as temp:
            source=Path(temp)/'source';source.mkdir();out=Path(temp)/'out';out.mkdir()
            manifest=dict(input_hashes={'X':'market-hash'},start_ms=0,end_ms=c.STEP)
            snapshot=({0:[SimpleNamespace(ts_ms=0)]},{0,c.STEP},1)
            path=source/'candidate_snapshot.pkl';path.write_bytes(c.pickle.dumps(snapshot))
            receipt=dict(market_hashes=manifest['input_hashes'],start_ms=0,end_ms=c.STEP,snapshot_sha256=c.digest(path),
                source_hashes={name:c.digest(Path(c.__file__).with_name(name)) for name in
                    ('replay_backtest.py','monitor.py','strategy.py','indicators.py','config.py','research_rocket_capture.py')})
            meta=source/'candidate_checkpoint_receipt.json';meta.write_text(json.dumps(receipt),encoding='utf-8')
            self.assertEqual(c.reuse_candidate_snapshot(source,manifest,out)[2],1)
            path.write_bytes(path.read_bytes()+b'tampered')
            with self.assertRaises(ValueError):c.reuse_candidate_snapshot(source,manifest,out)
            path.write_bytes(c.pickle.dumps((snapshot[0],snapshot[1],2)));receipt['snapshot_sha256']=c.digest(path)
            meta.write_text(json.dumps(receipt),encoding='utf-8')
            with self.assertRaises(ValueError):c.reuse_candidate_snapshot(source,manifest,out)
            receipt['source_hashes']['config.py']='stale';meta.write_text(json.dumps(receipt),encoding='utf-8')
            with self.assertRaises(ValueError):c.reuse_candidate_snapshot(source,manifest,out)
    def test_parallel_checkpoint_matches_causal_sequential_indicators_and_rejects_input_drift(self):
        import replay_backtest as rb
        raw=market(n=96*10)
        quarter=[{k:float(raw[k][i]) if k!='t' else int(raw[k][i]) for k in raw} for i in range(len(raw['t']))]
        hourly=[]
        for i in range(0,len(quarter),4):
            block=quarter[i:i+4]
            hourly.append(dict(t=block[0]['t'],o=block[0]['o'],h=max(r['h'] for r in block),
                l=min(r['l'] for r in block),c=block[-1]['c'],v=sum(r['v'] for r in block)))
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);source=root/'source';(source/'market').mkdir(parents=True)
            for tf,rows in [('15m',quarter),('1h',hourly)]:
                (source/'market'/f'X_{tf}.json').write_text(json.dumps(rows),encoding='utf-8')
            checksums={p.name:c.digest(p) for p in (source/'market').iterdir()}
            (source/'manifest.json').write_text(json.dumps(dict(input_hashes=checksums)),encoding='utf-8')
            out=root/'out';out.mkdir();c.frozen_market(source,out)
            (out/'features').mkdir();_,path,_=c.feature_worker(('X',str(out/'market'),str(out/'features')))
            with np.load(path,allow_pickle=False) as stored:
                for tf in ('15m','1h','4h'):
                    d=stored[tf+'__data'];expected=rb.compute_features(d['o'],d['h'],d['l'],d['c'],d['v'])
                    for key,value in expected.items():np.testing.assert_array_equal(stored[tf+'__'+key],value)
                    cut=len(d)*2//3;changed=d.copy()
                    for key in ('o','h','l','c','v'):changed[key][cut+1:]*=3
                    after=rb.compute_features(changed['o'],changed['h'],changed['l'],changed['c'],changed['v'])
                    for key,value in expected.items():np.testing.assert_array_equal(value[:cut+1],after[key][:cut+1])
            packs=c.read_feature_pack(path);context=rb._build_bull_day_context(packs['1h'][0])
            context_file=out/'context.npz';np.savez_compressed(context_file,t=context[0],bull=context[1],vs=context[2])
            (out/'candidates').mkdir()
            from research_rocket_capture import policy
            for tf in ('15m','1h'):
                with policy('baseline'):
                    sequential=asyncio.run(rb._build_candidates_for_symbol('X',tf,*packs[tf],{'X':packs['15m']},{'X':packs['4h']},context,variant='score_replace_cluster'))
                _,_,candidate_file,_,_=c.candidate_worker(('X',tf,str(out/'features'),str(context_file),str(out/'candidates')))
                self.assertEqual(json.loads(Path(candidate_file).read_bytes()),[asdict(row) for row in sequential])
            (source/'market'/'X_15m.json').write_text('[]',encoding='utf-8')
            with self.assertRaises(ValueError):c.frozen_market(source,root/'rejected')

    def test_future_changes_and_closed_prefix_do_not_change_features(self):
        raw=market();origin=700
        changed={k:v.copy() for k,v in raw.items()}
        for k in ('o','h','l','c','v'):changed[k][origin+1:]*=3
        before=c.market_frame(raw);after=c.market_frame(changed)
        prefix=c.market_frame({k:v[:origin+1] for k,v in raw.items()})
        pd.testing.assert_frame_equal(before.loc[:origin,c.DIR_FEATURES+['ewma']],after.loc[:origin,c.DIR_FEATURES+['ewma']])
        pd.testing.assert_frame_equal(before.loc[:origin,c.DIR_FEATURES+['ewma']],prefix[c.DIR_FEATURES+['ewma']])
        self.assertNotEqual(before.return_target.iloc[origin],after.return_target.iloc[origin])

    def test_target_is_exactly_four_future_returns_and_close_timing(self):
        raw=market();f=c.market_frame(raw);i=200
        returns=np.diff(np.log(raw['c']))[i:i+4]
        self.assertAlmostEqual(f.variance_target.iloc[i],float(np.sum(returns**2)))
        self.assertAlmostEqual(f.return_target.iloc[i],np.log(raw['c'][i+4]/raw['c'][i]))
        self.assertEqual(f.origin.iloc[i],raw['t'][i]+c.STEP)
        self.assertEqual(f.label_close.iloc[i],raw['t'][i]+5*c.STEP)
        self.assertTrue(f.variance_target.tail(4).isna().all())

    def test_parameters_scaling_and_smearing_use_only_prior_labels(self):
        raw=market(n=96*45);f=c.market_frame(raw)
        start=int(f.origin.iloc[96*10]);end=int(f.origin.iloc[-1])+c.STEP
        pred,folds=c.walk_forward({'BTCUSDT':f},start,end)
        changed={k:v.copy() for k,v in raw.items()}
        cutoff=96*12
        for k in ('o','h','l','c','v'):changed[k][cutoff+1:]*=2
        other,_=c.walk_forward({'BTCUSDT':c.market_frame(changed)},start,end)
        np.testing.assert_allclose(pred['BTCUSDT'][96*10:cutoff+1],other['BTCUSDT'][96*10:cutoff+1])
        self.assertTrue(all(r['last_training_label']<r['start'] for r in folds))
        self.assertTrue(np.isfinite(pred['BTCUSDT'][-1]).all())

    def test_lookup_rejects_unavailable_and_different_origin(self):
        f=c.market_frame(market());pair=np.ones((len(f),2))
        at=int(f.origin.iloc[200]);self.assertEqual(c.lookup({'X':f},{'X':pair},'X',at)[:2],(1,1))
        with self.assertRaises(ValueError):c.lookup({'X':f},{'X':pair},'X',at+1)
        pair[200]=np.nan
        with self.assertRaises(ValueError):c.lookup({'X':f},{'X':pair},'X',at)

    def test_exact_cost_break_even_and_no_leverage(self):
        r=c.break_even(10,5);net=np.exp(r)*(1-.0005)*(1-.001)/((1+.0005)*(1+.001))
        self.assertAlmostEqual(net,1)
        self.assertEqual(c.sizing(.000001),1)
        self.assertAlmostEqual(c.sizing(.0004),.5)
        for bad in (0,-1,float('nan')):
            with self.assertRaises(ValueError):c.sizing(bad)

class AccountTests(unittest.TestCase):
    def setUp(self):
        self.grid=list(range(1774994400000,1774994400000+8*c.STEP,c.STEP))
        self.series={'X':list(zip(self.grid,[100,103,106,104,108,110,111,112])),
                     'Y':list(zip(self.grid,[50,51,52,53,54,55,56,57]))}
        self.trades=[trade('X',self.grid[0],self.grid[4],100,108,partial_exit_taken=True,
                          partial_exit_fraction=.4,partial_exit_ts=self.grid[2],partial_exit_price=106),
                     trade('Y',self.grid[1],self.grid[5],51,55),
                     trade('X',self.grid[4],self.grid[6],108,111)]

    def test_unit_weights_match_canonical_partial_and_reentry_account(self):
        from portfolio_alpha import _simulate_account
        actual=c.weighted_account(self.trades,self.series,self.grid,[1]*3,7.5,5)
        expected=_simulate_account(self.trades,price_series_by_symbol=self.series,valuation_timestamps=self.grid,
            capacity=10,initial_capital=1,fee_bps=7.5,slippage_bps=5)
        self.assertEqual(expected.violations,[])
        np.testing.assert_allclose(actual['curve'],expected.equity_curve,rtol=1e-12,atol=1e-12)
        self.assertAlmostEqual(actual['net_return_pct'],expected.return_pct)

    def test_reducing_size_reduces_exposure_without_changing_admissions(self):
        full=c.weighted_account(self.trades,self.series,self.grid,[1]*3,7.5,5)
        small=c.weighted_account(self.trades,self.series,self.grid,[.5]*3,7.5,5)
        self.assertEqual(full['trades'],small['trades'])
        self.assertLess(small['average_gross_exposure_pct'],full['average_gross_exposure_pct'])
        self.assertLess(small['costs_initial_capital'],full['costs_initial_capital'])

    def test_costs_apply_and_cash_empty_arm_is_not_fake_trading(self):
        normal=c.weighted_account(self.trades,self.series,self.grid,[1]*3,7.5,5)
        stress=c.weighted_account(self.trades,self.series,self.grid,[1]*3,15,10)
        self.assertLess(stress['net_return_pct'],normal['net_return_pct'])
        empty=c.weighted_account([],self.series,self.grid,[],7.5,5)
        self.assertEqual(empty['trades'],0);self.assertEqual(empty['net_return_pct'],0)

    def test_missing_marks_duplicate_and_invalid_sizes_fail(self):
        series={**self.series,'X':[]}
        with self.assertRaises(ValueError):c.weighted_account(self.trades,series,self.grid,[1]*3,7.5,5)
        with self.assertRaises(ValueError):c.weighted_account(self.trades,self.series,self.grid,[2]*3,7.5,5)
        duplicate=self.trades+[trade('X',self.grid[1],self.grid[2],103,106)]
        with self.assertRaises(ValueError):c.weighted_account(duplicate,self.series,self.grid,[1]*4,7.5,5)

    def test_own_same_timestamp_boundary_liquidation(self):
        trades=[trade('X',self.grid[-1],self.grid[-1],112,112)]
        row=c.weighted_account(trades,self.series,self.grid,[1],7.5,5)
        self.assertLess(row['net_return_pct'],0)
        self.assertEqual(row['trades'],1)

    def test_drawdown_includes_costs_at_initial_origin(self):
        row=c.weighted_account([trade('X',self.grid[0],self.grid[0],100,100)],self.series,self.grid,[1],7.5,5)
        self.assertGreater(row['max_drawdown_pct'],0)
        self.assertAlmostEqual(row['max_drawdown_pct'],-row['net_return_pct'])

    def test_paired_daily_intervals_count_days_and_identical_returns(self):
        grid=list(range(1774994400000,1774994400000+45*c.DAY+c.STEP,c.STEP))
        curve=[(at,1+idx/10000) for idx,at in enumerate(grid)]
        accounts={name:dict(curve=curve) for name in c.ARMS}
        result=c.paired_intervals(accounts)
        self.assertEqual(result['combined']['n_days'],45)
        self.assertEqual(result['combined']['blocks']['3']['familywise95'],[0.,0.])

    def test_paired_intervals_include_opening_origin_costs(self):
        grid=list(range(1774994400000,1774994400000+45*c.DAY+c.STEP,c.STEP))
        accounts={name:dict(curve=[(at,.99 if name=='baseline' else 1.0) for at in grid]) for name in c.ARMS}
        row=c.paired_intervals(accounts)['direction']
        self.assertEqual(row['n_days'],45)
        self.assertAlmostEqual(row['blocks']['1']['mean_daily_log_gain_bps'],-np.log(.99)/45*10000)

if __name__=='__main__':unittest.main()
