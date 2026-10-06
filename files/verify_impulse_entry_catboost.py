"""Raw-market reconstruction, native prediction and portfolio contract audit."""
import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import pickle
import numpy as np
from catboost import CatBoostRegressor
import replay_backtest as rb
from historical_signal_evaluation import sha,freeze
from impulse_entry_catboost import BAR,HORIZON,DAY,fit_cohorts,gate
from portfolio_alpha import _simulate_account,_benchmark_result,closed_price_series
from run_turnover_economics import objective
from run_impulse_entry_catboost import single_interval


def raw_oracle(closes,at,tf,fee,slip):
    past=[closes.get(at-k*BAR) for k in range(16,-1,-1)]
    x=None;y=None
    if all(v is not None for v in past):
        a=np.array([[v[k] for k in ('o','h','l','c','v')] for v in past])
        if (np.isfinite(a).all() and (a[:,:4]>0).all() and (a[:,4]>=0).all()
            and (a[:,1]>=np.maximum(a[:,0],a[:,3])).all()
            and (a[:,2]<=np.minimum(a[:,0],a[:,3])).all()):
            c=a[:,3];lo=a[1:,2].min();hi=a[1:,1].max();volume=a[1:,4].mean()
            angle=2*np.pi*(at%DAY)/DAY
            x=np.array([100*(c[-1]/c[-1-k]-1) for k in (1,4,16)]+[
                100*np.std(np.diff(np.log(c))),a[-1,4]/volume if volume else 0,
                100*(hi/lo-1),(c[-1]-lo)/(hi-lo) if hi>lo else .5,
                float(tf=='1h'),np.sin(angle),np.cos(angle)])
    future=[closes.get(at+k*BAR) for k in range(6)]
    if all(v is not None and math.isfinite(v['c']) and v['c']>0 for v in future):
        f,s=fee/10000,slip/10000
        y=100*(future[-1]['c']/future[0]['c']*(1-f)*(1-s)/((1+f)*(1+s))-1)
    return x,y


def verify(directory):
    reg=json.loads((directory/'registration.json').read_bytes());r=json.loads((directory/'result.json').read_bytes())
    receipt=json.loads((directory/'receipt.json').read_bytes())
    for n,h in receipt.items():
        if Path(n).name!=n or sha(directory/n)!=h:raise ValueError('receipt drift')
    for n,h in reg['sources'].items():
        if Path(n).name!=n or sha(directory/'source_snapshot'/n)!=h:raise ValueError('source snapshot drift')
        if sha(Path(__file__).with_name(n))!=h:raise ValueError('current audit kernel differs')
    market=Path(reg['market']);candidates=Path(reg['candidates'])
    if not candidates.resolve().is_relative_to(Path(__file__).resolve().parent.parent/'.runtime'):
        raise ValueError('trusted runtime checkpoint required')
    if sha(market/'manifest.json')!=reg['market_sha256']:raise ValueError('manifest drift')
    m=json.loads((market/'manifest.json').read_bytes());cache={};oracle={}
    dtype=[(k,'i8' if k=='t' else 'f8') for k in ('t','o','h','l','c','v')]
    for sym in m['eligible_symbols']:
        for tf in ('15m','1h'):
            name=sym+'_'+tf+'.json';p=market/'market'/name
            if sha(p)!=m['input_hashes'][name]:raise ValueError('raw drift')
            rows=json.loads(p.read_bytes());d=np.array([tuple(v[k] for k in ('t','o','h','l','c','v')) for v in rows],dtype=dtype)
            cache[sym,tf]=(d,{})
            if tf=='15m':oracle[sym]={int(v['t'])+BAR:v for v in rows}
    cr=json.loads((candidates/'candidate_checkpoint_receipt.json').read_bytes())
    if sha(candidates/'candidate_snapshot.pkl')!=cr['snapshot_sha256']:raise ValueError('candidate drift')
    if cr!=json.loads((directory/'reused_candidate_checkpoint.json').read_bytes())['receipt']:
        raise ValueError('candidate receipt differs')
    with (candidates/'candidate_snapshot.pkl').open('rb') as f:raw,times,n=pickle.load(f)
    expected=[(at,j,c.sym,c.tf) for at in sorted(raw) for j,c in enumerate(raw[at]) if c.mode=='impulse_speed']
    with np.load(directory/'dataset.npz',allow_pickle=False) as payload:
        z={k:payload[k] for k in payload.files}
    ids=list(zip(z['clock'].tolist(),z['index'].tolist(),z['symbol'].tolist(),z['tf'].tolist()))
    if ids!=expected or n!=sum(map(len,raw.values())):raise ValueError('candidate population differs')
    for i,(at,j,sym,tf) in enumerate(ids):
        x,y=raw_oracle(oracle[sym],at,tf,reg['fee_bps'],reg['slippage_bps'])
        np.testing.assert_allclose(z['x'][i],np.full(10,np.nan) if x is None else x,rtol=1e-11,atol=1e-10,equal_nan=True)
        np.testing.assert_allclose(z['y'][i],np.nan if y is None else y,rtol=1e-11,atol=1e-10,equal_nan=True)
    folds=json.loads((directory/'folds.json').read_bytes());covered=np.zeros(len(ids),bool);native_n=0
    valid=np.isfinite(z['x']).all(axis=1)&np.isfinite(z['y'])
    for k,f in enumerate(folds):
        train,val,cut=fit_cohorts(z['clock'],valid,f['fit_at'],reg['start_ms'])
        if (train.sum(),val.sum(),cut)!=(f['train_n'],f['validation_n'],f['validation_boundary']):raise ValueError('fit cohort differs')
        issued=(z['clock']>=f['fit_at'])&(z['clock']<f['stop']);scored=issued&np.isfinite(z['x']).all(axis=1)
        if f['state']!='FROZEN':
            if train.sum()>=500 and val.sum()>=100:raise ValueError('invalid untrained passthrough')
            continue
        if np.any(covered & issued) or not np.all(z['fold'][issued]==k):raise ValueError('prediction fold differs')
        covered|=issued
        if not (f['train_max_label_at']==int((z['clock'][train]+HORIZON).max())<cut
                and f['validation_max_label_at']==int((z['clock'][val]+HORIZON).max())<f['fit_at']):
            raise ValueError('future fitting labels')
        if sha(directory/f['model'])!=f['model_sha256']:raise ValueError('model drift')
        model=CatBoostRegressor();model.load_model(str(directory/f['model']))
        native=model.predict(z['x'][scored]);np.testing.assert_allclose(native,z['prediction'][scored],atol=1e-12,rtol=0)
        if np.isfinite(z['prediction'][issued & ~scored]).any():raise ValueError('unknown feature scored')
        native_n+=len(native)
    if not np.array_equal(covered,z['fold']>=0):raise ValueError('missing OOS block')
    forecasts={}
    for name,mask in (('all_oos',covered),('test',covered & (z['clock']>=r['split_boundaries_ms'][1]))):
        known=mask & np.isfinite(z['prediction']) & np.isfinite(z['y'])
        p=z['prediction'][known];y=z['y'][known]
        forecasts[name]={'issued_n':int(mask.sum()),'label_known_n':int(known.sum()),
            'missing_target_or_feature_n':int((mask & ~known).sum()),
            'mae_pct':float(np.mean(abs(p-y))),'zero_mae_pct':float(np.mean(abs(y))),
            'rmse_pct':float(np.sqrt(np.mean((p-y)**2))),'accepted_n':int((p>0).sum()),
            'accepted_mean_net_proxy_pct':float(y[p>0].mean()) if (p>0).any() else None,
            'base_positive_n':int((y>0).sum()),'direction_correct_n':int(((p>0)==(y>0)).sum())}
    if forecasts!=r['forecast_metrics']:raise ValueError('forecast metric mismatch')
    allowed=set();accepted=0
    lookup={tuple(v[:2]):i for i,v in enumerate(ids)}
    for at,rows in raw.items():
        for j,c in enumerate(rows):
            i=lookup.get((at,j));ok=i is None or z['fold'][i]<0 or (np.isfinite(z['prediction'][i]) and z['prediction'][i]>0)
            if ok:accepted+=1;allowed.add((c.sym,c.tf,at))
    if accepted!=r['filter_audit']['retained_all_candidates']:raise ValueError('screen count differs')
    series={s:closed_price_series(cache[s,'15m'][0],bar_ms=BAR,start_ms=r['start_ms'],end_ms=r['end_ms']) for s in m['eligible_symbols']}
    grid=list(range(r['start_ms'],r['end_ms']+1,BAR));checked={}
    benchmark=_benchmark_result(series['BTCUSDT'],initial_capital=10000,fee_bps=reg['fee_bps'],slippage_bps=reg['slippage_bps'])
    if benchmark!=r['benchmark']:raise ValueError('benchmark differs')
    for name,a in r['accounts'].items():
        trades=[rb.ReplayTrade(**v) for v in json.loads((directory/f'trades_{name}.json').read_bytes())]
        if name=='catboost' and any((t.sym,t.tf,t.entry_ts) not in allowed for t in trades):raise ValueError('ineligible admission')
        ref=_simulate_account(trades,price_series_by_symbol=series,valuation_timestamps=grid,capacity=10,
                              initial_capital=10000,fee_bps=reg['fee_bps'],slippage_bps=reg['slippage_bps'])
        if ref.violations or ref.fully_valued_points!=len(grid):raise ValueError('canonical violation')
        np.testing.assert_allclose(a['curve'],ref.equity_curve,atol=1e-7,rtol=1e-10)
        np.testing.assert_allclose([a['totals']['fees'],a['totals']['slippage']],[ref.fees_quote,ref.slippage_quote],atol=1e-7,rtol=1e-10)
        ledger=json.loads((directory/f'ledger_{name}.json').read_bytes())
        if len(ledger)!=len(trades) or len(trades)!=a['trades']:raise ValueError('trade denominator')
        for row in ledger:
            if not math.isclose(row['raw_price_pnl']-row['fees']-row['slippage'],row['net_pnl'],abs_tol=1e-8):raise ValueError('cash attribution')
        for field,total in a['totals'].items():
            if not math.isclose(sum(v[field] for v in ledger),total,abs_tol=1e-7):raise ValueError('cash sum')
        if not math.isclose(sum(v['net_pnl'] for v in ledger),a['ending_cash']-10000,abs_tol=1e-7):raise ValueError('net reconciliation')
        for groups in r['attribution'][name].values():
            for field,total in a['totals'].items():
                if not math.isclose(sum(v[field] for v in groups.values()),total,abs_tol=1e-7):raise ValueError('group attribution mismatch')
        c=np.array(a['curve']);peak=np.maximum(10000,np.maximum.accumulate(c[:,1]));test=dict(a['curve'])[r['split_boundaries_ms'][1]]
        actual=[100*(c[-1,1]/10000-1),float(100*(1-c[:,1]/peak).max()),100*(c[-1,1]/test-1)]
        np.testing.assert_allclose(actual,[a['net_return_pct'],a['max_drawdown_pct'],a['test_return_pct']],atol=1e-8,rtol=0)
        if not math.isclose(a['alpha_pp'],a['net_return_pct']-benchmark['net_return_after_costs_pct'],abs_tol=1e-8):raise ValueError('alpha mismatch')
        if objective(trades,cache,m['eligible_symbols'],r['start_ms'],r['end_ms'])!=r['missions'][name]:raise ValueError('mission differs')
        cut=r['split_boundaries_ms'][1]
        if objective([t for t in trades if t.entry_ts>=cut],cache,m['eligible_symbols'],cut,r['end_ms'])!=r['test_missions'][name]:raise ValueError('TEST mission differs')
        checked[name]={'trades':len(trades),'equity_points':len(grid)}
    ci=single_interval(r['accounts']['control']['curve'],r['accounts']['catboost']['curve'],r['split_boundaries_ms'][1],r['end_ms'])
    v=gate(r['accounts']['control'],r['accounts']['catboost'],tuple(r['missions'][k] for k in ('control','catboost')),
           tuple(r['test_missions'][k] for k in ('control','catboost')),ci['interval'],ci['days'])
    if {**v,**ci}!=r['comparisons']['catboost']:raise ValueError('gate differs')
    import catboost,sys
    return {'status':'PASS','raw_impulse_rows':len(ids),'native_predictions':native_n,
            'all_candidates_checked':n,'retained':accepted,'accounts':checked,
            'result_sha256':sha(directory/'result.json'),'verifier_sha256':sha(Path(__file__)),
            'versions':{'python':sys.version,'numpy':np.__version__,'catboost':catboost.__version__},
            'scope':'numeric/timing audit, no live parity or deployment authority'}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True)
    a=p.parse_args();v=verify(a.run);freeze(a.run/'independent_verification.json',json.dumps(v,indent=2).encode());print(json.dumps(v,indent=2))
