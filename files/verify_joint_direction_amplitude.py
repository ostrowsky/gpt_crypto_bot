"""Independent raw labels, calibration, forecasts and full account reconciliation."""
import argparse
import json
import math
from pathlib import Path
import pickle
import numpy as np
from catboost import CatBoostClassifier,CatBoostRegressor
from sklearn.linear_model import LogisticRegression
from scipy.special import expit
import replay_backtest as rb
from historical_signal_evaluation import sha,freeze
from capacity_catboost import BAR,HORIZON
from impulse_entry_catboost import gate
from joint_direction_amplitude import cohorts,enough
from run_impulse_entry_catboost import single_interval
from run_turnover_economics import objective
from portfolio_alpha import _simulate_account,_benchmark_result,closed_price_series
from verify_impulse_entry_catboost import raw_oracle


def probability_oracle(p,truth):
    # Independent fixed-bin aggregation; no calls to research metric implementation.
    n=len(p);bins=[];error=0.
    for j in range(10):
        use=(p>=j/10)&((p<(j+1)/10) if j<9 else (p<=1))
        count=int(use.sum());ups=int(truth[use].sum())
        mean=float(p[use].mean()) if count else None;rate=ups/count if count else None
        bins.append(dict(bin=j,n=count,up_n=ups,mean_probability=mean,observed_up_rate=rate))
        if count:error+=count/n*abs(mean-rate)
    q=np.clip(p,1e-6,1-1e-6)
    return {'n':n,'up_n':int(truth.sum()),'direction_correct_n':int(((p>.5)==truth).sum()),
        'brier':float(np.mean((p-truth)**2)),'logloss':float(np.mean(np.where(truth,-np.log(q),-np.log1p(-q)))),
        'ece10':float(error),'bins':bins}


def assert_nested(actual,expected):
    if isinstance(actual,dict):
        if set(actual)!=set(expected):raise ValueError('metric field mismatch')
        for k in actual:assert_nested(actual[k],expected[k])
    elif isinstance(actual,list):
        if len(actual)!=len(expected):raise ValueError('metric length mismatch')
        for a,b in zip(actual,expected):assert_nested(a,b)
    elif isinstance(actual,float):
        if not math.isclose(actual,expected,rel_tol=1e-10,abs_tol=1e-10):raise ValueError('metric numeric mismatch')
    elif actual!=expected:raise ValueError('metric count/value mismatch')


def verify(directory):
    reg=json.loads((directory/'registration.json').read_bytes());r=json.loads((directory/'result.json').read_bytes())
    for n,h in json.loads((directory/'receipt.json').read_bytes()).items():
        if Path(n).name!=n or sha(directory/n)!=h:raise ValueError('receipt drift')
    for n,h in reg['sources'].items():
        if Path(n).name!=n or sha(directory/'source_snapshot'/n)!=h or sha(Path(__file__).with_name(n))!=h:raise ValueError('source drift')
    if sha(directory/'registered_spec.md')!=reg['registered_spec_sha256']:raise ValueError('registered spec drift')
    inheritance=json.loads((directory/'inheritance_receipt.json').read_bytes())
    for n,binding in inheritance['files'].items():
        if sha(directory/n)!=binding['sha256'] or sha(Path(binding['source']))!=binding['sha256']:raise ValueError('inherited file drift')
    parent=Path(reg['parent']);parent_reg=json.loads((directory/'parent_registration.json').read_bytes())
    parent_audit=json.loads((directory/'parent_audit.json').read_bytes())
    if parent_audit['status']!='PASS' or parent_audit['result_sha256']!=sha(directory/'parent_result.json'):raise ValueError('parent audit mismatch')
    if sha(Path(__file__).with_name('verify_impulse_entry_catboost.py'))!=parent_audit['verifier_sha256']:raise ValueError('parent auditor differs')
    if sha(parent/'receipt.json')!=inheritance['parent_receipt_sha256']:raise ValueError('parent receipt differs')
    market=Path(reg['market']);m=json.loads((market/'manifest.json').read_bytes());cache={};oracle={}
    if sha(market/'manifest.json')!=reg['market_sha256']:raise ValueError('market manifest drift')
    dtype=[(k,'i8' if k=='t' else 'f8') for k in ('t','o','h','l','c','v')]
    for sym in m['eligible_symbols']:
        for tf in ('15m','1h'):
            name=sym+'_'+tf+'.json';p=market/'market'/name
            if sha(p)!=m['input_hashes'][name]:raise ValueError('raw data drift')
            rows=json.loads(p.read_bytes());cache[sym,tf]=(np.array([tuple(v[k] for k in ('t','o','h','l','c','v')) for v in rows],dtype=dtype),{})
            if tf=='15m':oracle[sym]={int(v['t'])+BAR:v for v in rows}
    with np.load(directory/'dataset.npz',allow_pickle=False) as p:z={k:p[k] for k in p.files}
    ids=list(zip(z['clock'].tolist(),z['index'].tolist(),z['symbol'].tolist(),z['tf'].tolist()))
    candidates=Path(parent_reg['candidates'])
    if not candidates.resolve().is_relative_to(Path(__file__).resolve().parent.parent/'.runtime'):raise ValueError('trusted local pickle only')
    cr=json.loads((candidates/'candidate_checkpoint_receipt.json').read_bytes())
    if cr!=json.loads((directory/'reused_candidate_checkpoint.json').read_bytes())['receipt'] or sha(candidates/'candidate_snapshot.pkl')!=cr['snapshot_sha256']:raise ValueError('candidate checkpoint drift')
    with (candidates/'candidate_snapshot.pkl').open('rb') as f:raw,times,n=pickle.load(f)
    expected=[(at,j,c.sym,c.tf) for at in sorted(raw) for j,c in enumerate(raw[at]) if c.mode=='impulse_speed']
    if expected!=ids or n!=sum(map(len,raw.values())):raise ValueError('candidate population mismatch')
    fee,slip=reg['fee_bps'],reg['slippage_bps'];factor=(1-fee/10000)*(1-slip/10000)/((1+fee/10000)*(1+slip/10000))
    for i,(at,j,sym,tf) in enumerate(ids):
        x,y=raw_oracle(oracle[sym],at,tf,fee,slip)
        np.testing.assert_allclose(z['x'][i],np.full(10,np.nan) if x is None else x,rtol=1e-11,atol=1e-10,equal_nan=True)
        np.testing.assert_allclose(z['y'][i],np.nan if y is None else y,rtol=1e-11,atol=1e-10,equal_nan=True)
        future=[oracle[sym].get(at+k*BAR) for k in range(6)]
        gross=np.nan if any(v is None for v in future) else 100*(future[-1]['c']/future[0]['c']-1)
        np.testing.assert_allclose(gross,z['gross'][i],rtol=1e-11,atol=1e-10,equal_nan=True)
        if np.isfinite(gross) and ((gross>0)!=(z['gross'][i]>0)):raise ValueError('gross direction mismatch')
    folds=json.loads((directory/'folds.json').read_bytes());covered=np.zeros(len(ids),bool);native_n=0;frozen_folds=0
    valid=np.isfinite(z['gross'])&np.isfinite(z['x']).all(axis=1)
    for k,record in enumerate(folds):
        train,val,cal,cuts=cohorts(z['clock'],valid,record['fit_at'],reg['start_ms'])
        if (int(train.sum()),int(val.sum()),int(cal.sum()),cuts)!=(record['train_n'],record['validation_n'],record['calibration_n'],record['boundaries']):raise ValueError('cohort mismatch')
        issued=(z['clock']>=record['fit_at'])&(z['clock']<record['stop']);score=issued&np.isfinite(z['x']).all(axis=1)
        if record['state']!='FROZEN':
            if enough(z['gross'],train,val,cal):raise ValueError('invalid passthrough')
            continue
        if not (record['train_max_label_at']==int((z['clock'][train]+HORIZON).max())<cuts[0]
            and record['validation_max_label_at']==int((z['clock'][val]+HORIZON).max())<cuts[1]
            and record['calibration_max_label_at']==int((z['clock'][cal]+HORIZON).max())<record['fit_at']):raise ValueError('future fitting label')
        if np.any(covered&issued) or not np.all(z['fold'][issued]==k):raise ValueError('fold identity mismatch')
        covered|=issued;models={};frozen_folds+=1
        for name in ('direction','up','down'):
            filename=f'model_{k}_{name}.cbm'
            if sha(directory/filename)!=record['models'][filename]:raise ValueError('model hash drift')
            model=CatBoostClassifier() if name=='direction' else CatBoostRegressor()
            model.load_model(str(directory/filename));models[name]=model
        raw_p=models['direction'].predict_proba(z['x'][score])[:,1]
        calibration_raw=np.clip(models['direction'].predict_proba(z['x'][cal])[:,1],1e-6,1-1e-6)
        model=LogisticRegression(C=1.,solver='lbfgs',max_iter=1000)
        model.fit(np.log(calibration_raw/(1-calibration_raw))[:,None],(z['gross'][cal]>0).astype(int))
        np.testing.assert_allclose([record['platt_coef'],record['platt_intercept']],[model.coef_[0,0],model.intercept_[0]],rtol=0,atol=1e-12)
        clipped=np.clip(raw_p,1e-6,1-1e-6)
        p=expit(record['platt_coef']*np.log(clipped/(1-clipped))+record['platt_intercept'])
        up=np.maximum(0,models['up'].predict(z['x'][score]));down=np.maximum(0,models['down'].predict(z['x'][score]))
        net=100*((1+(p*up-(1-p)*down)/100)*factor-1)
        for name,expected in zip(('raw_p','p','up','down','prediction'),(raw_p,p,up,down,net)):
            np.testing.assert_allclose(z[name][score],expected,rtol=0,atol=1e-12)
        np.testing.assert_allclose(z['climatology'][score],np.mean(z['gross'][train]>0),rtol=0,atol=1e-12)
        np.testing.assert_allclose(z['train_mean_net'][score],100*((1+np.mean(z['gross'][train])/100)*factor-1),rtol=0,atol=1e-12)
        if np.isfinite(z['prediction'][issued&~score]).any():raise ValueError('unknown feature issued numeric forecast')
        native_n+=int(score.sum())
    if not np.array_equal(covered,z['fold']>=0):raise ValueError('missing OOS coverage')
    metrics={}
    for name,mask in (('all_oos',covered),('test',covered&(z['clock']>=r['split_boundaries_ms'][1]))):
        known=mask&np.isfinite(z['gross'])&np.isfinite(z['prediction']);g=z['gross'][known];y=z['y'][known];p=z['prediction'][known];accepted=p>0
        metrics[name]={'issued_n':int(mask.sum()),'known_n':int(known.sum()),'unknown_future_or_feature_n':int((mask&~known).sum()),
            'raw_probability':probability_oracle(z['raw_p'][known],g>0),'calibrated_probability':probability_oracle(z['p'][known],g>0),
            'climatology':probability_oracle(z['climatology'][known],g>0),'net_mae':float(np.mean(abs(p-y))),
            'net_rmse':float(np.sqrt(np.mean((p-y)**2))),'zero_net_mae':float(np.mean(abs(y))),
            'flat_price_net_mae':float(np.mean(abs(100*(factor-1)-y))),
            'train_mean_net_mae':float(np.mean(abs(z['train_mean_net'][known]-y))),
            'accepted_known_n':int(accepted.sum()),'accepted_realized_mean_net_pct':float(y[accepted].mean()) if accepted.any() else None}
    assert_nested(metrics,r['forecast_metrics'])
    lookup={tuple(v[:2]):i for i,v in enumerate(ids)};allowed=set();retained=0
    for at,rows in raw.items():
        for j,c in enumerate(rows):
            i=lookup.get((at,j));ok=i is None or z['fold'][i]<0 or (np.isfinite(z['prediction'][i]) and z['prediction'][i]>0)
            if ok:retained+=1;allowed.add((c.sym,c.tf,at))
    if retained!=r['filter_audit']['retained']:raise ValueError('filter count mismatch')
    series={s:closed_price_series(cache[s,'15m'][0],bar_ms=BAR,start_ms=r['start_ms'],end_ms=r['end_ms']) for s in m['eligible_symbols']}
    grid=list(range(r['start_ms'],r['end_ms']+1,BAR));checked={}
    benchmark=_benchmark_result(series['BTCUSDT'],initial_capital=10000,fee_bps=fee,slippage_bps=slip)
    assert_nested(benchmark,r['benchmark'])
    for name,a in r['accounts'].items():
        trades=[rb.ReplayTrade(**v) for v in json.loads((directory/f'trades_{name}.json').read_bytes())]
        if name=='joint' and any((t.sym,t.tf,t.entry_ts) not in allowed for t in trades):raise ValueError('ineligible joint entry')
        ref=_simulate_account(trades,price_series_by_symbol=series,valuation_timestamps=grid,capacity=10,initial_capital=10000,fee_bps=fee,slippage_bps=slip)
        if ref.violations or ref.fully_valued_points!=len(grid):raise ValueError('canonical violation')
        np.testing.assert_allclose(a['curve'],ref.equity_curve,atol=1e-7,rtol=1e-10)
        np.testing.assert_allclose([a['totals']['fees'],a['totals']['slippage']],[ref.fees_quote,ref.slippage_quote],atol=1e-7,rtol=1e-10)
        ledger=json.loads((directory/f'ledger_{name}.json').read_bytes())
        if len(ledger)!=len(trades) or len(trades)!=a['trades']:raise ValueError('trade denominator')
        for row in ledger:
            if not math.isclose(row['raw_price_pnl']-row['fees']-row['slippage'],row['net_pnl'],abs_tol=1e-8):raise ValueError('cash attribution mismatch')
        for field,total in a['totals'].items():
            if not math.isclose(sum(v[field] for v in ledger),total,abs_tol=1e-7):raise ValueError('cash sum mismatch')
            if not math.isclose(sum(v[field] for v in r['attribution'][name]['mode'].values()),total,abs_tol=1e-7):raise ValueError('group sum mismatch')
        if not math.isclose(a['totals']['net_pnl'],a['ending_cash']-10000,abs_tol=1e-7):raise ValueError('net cash mismatch')
        c=np.array(a['curve']);peak=np.maximum(10000,np.maximum.accumulate(c[:,1]));test=dict(a['curve'])[r['split_boundaries_ms'][1]]
        np.testing.assert_allclose([100*(c[-1,1]/10000-1),float(100*(1-c[:,1]/peak).max()),100*(c[-1,1]/test-1)],
            [a['net_return_pct'],a['max_drawdown_pct'],a['test_return_pct']],atol=1e-8,rtol=0)
        if not math.isclose(a['alpha_pp'],a['net_return_pct']-benchmark['net_return_after_costs_pct'],abs_tol=1e-8):raise ValueError('alpha mismatch')
        if objective(trades,cache,m['eligible_symbols'],r['start_ms'],r['end_ms'])!=r['missions'][name]:raise ValueError('mission differs')
        cut=r['split_boundaries_ms'][1]
        if objective([t for t in trades if t.entry_ts>=cut],cache,m['eligible_symbols'],cut,r['end_ms'])!=r['test_missions'][name]:raise ValueError('TEST mission differs')
        checked[name]={'trades':len(trades),'equity_points':len(grid)}
    ci=single_interval(r['accounts']['control']['curve'],r['accounts']['joint']['curve'],r['split_boundaries_ms'][1],r['end_ms'])
    v=gate(r['accounts']['control'],r['accounts']['joint'],tuple(r['missions'][k] for k in ('control','joint')),tuple(r['test_missions'][k] for k in ('control','joint')),ci['interval'],ci['days'])
    assert_nested({**v,**ci},r['comparisons']['joint'])
    import catboost,sklearn,sys
    return {'status':'PASS','raw_rows':len(ids),'native_prediction_rows':native_n,'native_models':3*frozen_folds,
        'reconstructed_calibrations':frozen_folds,'all_candidates':n,'retained':retained,'accounts':checked,
        'result_sha256':sha(directory/'result.json'),'verifier_sha256':sha(Path(__file__)),
        'raw_oracle_dependency_sha256':sha(Path(__file__).with_name('verify_impulse_entry_catboost.py')),
        'versions':{'python':sys.version,'numpy':np.__version__,'catboost':catboost.__version__,'sklearn':sklearn.__version__},
        'scope':'independent numeric/timing audit; no live parity or promotion authority'}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True)
    a=p.parse_args();v=verify(a.run);freeze(a.run/'independent_verification.json',json.dumps(v,indent=2).encode());print(json.dumps(v,indent=2))
