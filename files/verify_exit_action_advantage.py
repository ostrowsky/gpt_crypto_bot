"""Native-policy state reconstruction and independent SELL-action/account audit."""
import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import numpy as np
from catboost import CatBoostRegressor
import replay_backtest as rb
from historical_signal_evaluation import sha,freeze
from compare_price_volatility_bot import reuse_features
from impulse_entry_catboost import BAR,gate
from exit_action_advantage import position_features,label_action,cohorts,THRESHOLD,valid_origin
from run_exit_action_advantage import action_diagnostics
from run_impulse_entry_catboost import single_interval
from run_turnover_economics import objective
from research_rocket_capture import policy
from portfolio_alpha import closed_price_series,_simulate_account,_benchmark_result
from verify_joint_direction_amplitude import assert_nested
from render_turnover_economics import exit_diagnostics


def state_at(trade,cache,at,progress):
    """Replay original as-of position state from entry, no final extrema/stop input."""
    d,f=cache[trade.sym,trade.tf];i=trade.entry_i;atr=float(f['atr'][i]) if np.isfinite(f['atr'][i]) else 0.
    clone=rb.ReplayTrade(trade.sym,trade.tf,trade.mode,trade.entry_ts,trade.entry_price,i,
        trade.trail_k,trade.max_hold_bars,trade.entry_price-trade.trail_k*atr if atr>0 else 0.,
        entry_score=trade.entry_score,max_price_since_entry=trade.entry_price,min_price_since_entry=trade.entry_price)
    micro=cache.get((trade.sym,'15m')) if trade.tf=='1h' else None;last=None
    for clock in range(trade.entry_ts+BAR,at+1,BAR):
        idx=rb._find_last_closed_candle_index(d['t'],clock,rb.BAR_MS[trade.tf])
        if idx is None or idx<=i:continue
        last=progress(clone,d,f,idx,ts_ms=clock,micro_pack=micro)
        if last and clock<at:raise ValueError('earlier original SELL before recorded first soft decision')
    if not rb._is_weak_exit_reason(last):raise ValueError('recorded soft state not reproduced')
    idx=rb._find_last_closed_candle_index(d['t'],at,rb.BAR_MS[trade.tf])
    clone.exit_ts=at;clone.exit_price=float(d['c'][idx]);clone.exit_reason=last
    return clone


def verify(directory):
    reg=json.loads((directory/'registration.json').read_bytes());r=json.loads((directory/'result.json').read_bytes())
    for n,h in json.loads((directory/'receipt.json').read_bytes()).items():
        if Path(n).name!=n or sha(directory/n)!=h:raise ValueError('result receipt drift')
    for n,h in reg['sources'].items():
        if Path(n).name!=n or sha(directory/'source_snapshot'/n)!=h or sha(Path(__file__).with_name(n))!=h:raise ValueError('source drift')
    if sha(directory/'registered_spec.md')!=reg['spec_sha256']:raise ValueError('spec drift')
    inheritance=json.loads((directory/'inheritance_receipt.json').read_bytes())
    for n,b in inheritance['files'].items():
        if sha(directory/n)!=b['sha256'] or sha(Path(b['source']))!=b['sha256']:raise ValueError('inherited file drift')
    market=Path(reg['market']);parent_reg=json.loads((directory/'parent_registration.json').read_bytes())
    m=json.loads((market/'manifest.json').read_bytes())
    if sha(market/'manifest.json')!=reg['market_sha256']:raise ValueError('manifest drift')
    audit=json.loads((directory/'parent_audit.json').read_bytes())
    if audit['status']!='PASS' or audit['result_sha256']!=sha(directory/'parent_result.json'):raise ValueError('unaudited parent')
    features=Path(parent_reg['features']);audit_dir=directory/'verification_inputs';audit_dir.mkdir(exist_ok=False)
    cache,frames,hashes=reuse_features(features,m,audit_dir);del frames
    # Prices independently JSON-decoded; cached indicator arrays remain producer/source-hash bound.
    for sym in m['eligible_symbols']:
        for tf in ('15m','1h'):
            name=sym+'_'+tf+'.json';p=market/'market'/name
            if sha(p)!=m['input_hashes'][name]:raise ValueError('raw drift')
            rows=json.loads(p.read_bytes());native=cache[sym,tf][0]
            for field in ('t','o','h','l','c','v'):
                np.testing.assert_array_equal(native[field],np.array([v[field] for v in rows],dtype=native[field].dtype))
    cases=json.loads((directory/'cases.json').read_bytes())
    control=[rb.ReplayTrade(**v) for v in json.loads((directory/'trades_control.json').read_bytes())]
    model_trades=[rb.ReplayTrade(**v) for v in json.loads((directory/'trades_exit_model.json').read_bytes())]
    if [c['trade_id'] for c in cases]!=[i for i,t in enumerate(control) if rb._is_weak_exit_reason(t.exit_reason)]:raise ValueError('soft population mismatch')
    with np.load(directory/'dataset.npz',allow_pickle=False) as p:z={k:p[k] for k in p.files}
    progress=rb._update_trade_progress;fee,slip=reg['fee_bps'],reg['slippage_bps']
    with policy('baseline'):
        for i,case in enumerate(cases):
            recorded=control[case['trade_id']];replayed=state_at(recorded,cache,case['at'],progress)
            np.testing.assert_allclose(replayed.trail_stop,recorded.trail_stop,rtol=1e-10,atol=1e-10)
            d=cache[replayed.sym,replayed.tf][0];idx=rb._find_last_closed_candle_index(d['t'],case['at'],rb.BAR_MS[replayed.tf])
            x=(position_features(replayed,cache[replayed.sym,'15m'][0],case['at'],replayed.exit_price)
               if valid_origin(replayed,d,idx,case['at'],replayed.exit_price) else None)
            np.testing.assert_allclose(z['x'][i],np.full(22,np.nan) if x is None else x,rtol=1e-11,atol=1e-10,equal_nan=True)
            target=label_action(replayed,cache,r['end_ms'],fee,slip,progress) if x is not None else None
            assert_nested(target,case['target'])
            np.testing.assert_allclose(z['y'][i],np.nan if target is None else target['advantage'],rtol=1e-11,atol=1e-10,equal_nan=True)
            if z['clock'][i]!=case['at'] or z['available'][i]!=case['available_at']:raise ValueError('label clock mismatch')
    folds=json.loads((directory/'folds.json').read_bytes());models={};covered=np.zeros(len(cases),bool);native_rows=0
    valid=np.isfinite(z['x']).all(axis=1)&np.isfinite(z['y'])
    for k,f in enumerate(folds):
        train,val,cut=cohorts(z['clock'],z['available'],valid,f['fit_at'],reg['start_ms'])
        if (int(train.sum()),int(val.sum()),cut)!=(f['train_n'],f['validation_n'],f['validation_boundary']):raise ValueError('fit cohort mismatch')
        if f['state']!='FROZEN':
            if train.sum()>=300 and val.sum()>=50:raise ValueError('invalid untrained block')
            continue
        if not (f['train_max_label_at']==int(z['available'][train].max())<cut and
                f['validation_max_label_at']==int(z['available'][val].max())<f['fit_at']):raise ValueError('future fit label')
        issued=(z['clock']>=f['fit_at'])&(z['clock']<f['stop']);score=issued&np.isfinite(z['x']).all(axis=1)
        if np.any(covered&issued) or not np.all(z['fold'][issued]==k):raise ValueError('fold mismatch')
        covered|=issued
        if sha(directory/f['model'])!=f['model_sha256']:raise ValueError('model drift')
        model=CatBoostRegressor();model.load_model(str(directory/f['model']));models[k]=model
        np.testing.assert_allclose(model.predict(z['x'][score]),z['prediction'][score],rtol=0,atol=1e-12);native_rows+=int(score.sum())
    if not np.array_equal(covered,z['fold']>=0):raise ValueError('OOS coverage mismatch')
    diagnostics=action_diagnostics(z,r['split_boundaries_ms'][1]);assert_nested(diagnostics,r['action_diagnostics'])
    trace=json.loads((directory/'policy_trace.json').read_bytes());by_key={(t.sym,t.tf,t.entry_ts):t for t in model_trades}
    seen=set();deferred={};actual_rows=0
    with policy('baseline'):
        for row in trace['decisions']:
            key=tuple(row['key'])
            if key in seen or key not in by_key:raise ValueError('duplicated/unmatched soft decision')
            seen.add(key);state=state_at(by_key[key],cache,row['at'],progress)
            d=cache[state.sym,state.tf][0];idx=rb._find_last_closed_candle_index(d['t'],row['at'],rb.BAR_MS[state.tf])
            x=(position_features(state,cache[state.sym,'15m'][0],row['at'],state.exit_price)
               if valid_origin(state,d,idx,row['at'],state.exit_price) else None)
            if x is None:
                if row['features'] is not None:raise ValueError('unknown state published numeric features')
            else:np.testing.assert_allclose(row['features'],x,rtol=1e-11,atol=1e-10)
            score=None;model_id=None
            for k,f in enumerate(folds):
                if f['fit_at']<=row['at']<f['stop'] and k in models:
                    model_id=k;score=float(models[k].predict(x[None,:])[0]) if x is not None else None;break
            assert_nested(score,row['prediction'])
            if row['model_id']!=model_id or row['defer']!=(score is not None and score>THRESHOLD):raise ValueError('runtime decision mismatch')
            if row['defer']:deferred[key]=row
            actual_rows+=1
    if (actual_rows,len(deferred))!=(r['policy_counts']['actual_soft_decisions'],r['policy_counts']['actual_deferrals']):
        raise ValueError('actual policy denominator mismatch')
    for outcome in trace['outcomes']:
        key=tuple(outcome['key'])
        if key not in deferred:raise ValueError('outcome without actual deferral')
        if outcome['kind']=='TIMEOUT':
            t=by_key[key];origin=deferred[key]
            if t.exit_reason!=origin['origin_reason']+' [action one-bar deadline]' or t.exit_ts!=origin['deadline']:
                raise ValueError('timeout/cooldown reason mismatch')
            if rb._cooldown_bars_after_exit(t.mode,t.exit_reason)!=rb._cooldown_bars_after_exit(t.mode,origin['origin_reason']):raise ValueError('cooldown altered')
        elif rb._is_weak_exit_reason(outcome['reason']):raise ValueError('hard outcome masked')
    series={s:closed_price_series(cache[s,'15m'][0],bar_ms=BAR,start_ms=r['start_ms'],end_ms=r['end_ms']) for s in m['eligible_symbols']}
    grid=list(range(r['start_ms'],r['end_ms']+1,BAR));checked={}
    assert_nested(_benchmark_result(series['BTCUSDT'],initial_capital=10000,fee_bps=fee,slippage_bps=slip),r['benchmark'])
    for name,trades in (('control',control),('exit_model',model_trades)):
        a=r['accounts'][name];ref=_simulate_account(trades,price_series_by_symbol=series,valuation_timestamps=grid,capacity=10,initial_capital=10000,fee_bps=fee,slippage_bps=slip)
        if ref.violations or ref.fully_valued_points!=len(grid):raise ValueError('canonical violation')
        np.testing.assert_allclose(a['curve'],ref.equity_curve,rtol=1e-10,atol=1e-7)
        np.testing.assert_allclose([a['totals']['fees'],a['totals']['slippage']],[ref.fees_quote,ref.slippage_quote],rtol=1e-10,atol=1e-7)
        ledger=json.loads((directory/f'ledger_{name}.json').read_bytes())
        if len(ledger)!=len(trades) or a['trades']!=len(trades):raise ValueError('trade denominator mismatch')
        for row in ledger:
            if not math.isclose(row['net_pnl'],row['raw_price_pnl']-row['fees']-row['slippage'],abs_tol=1e-8):raise ValueError('cash attribution mismatch')
        for field,total in a['totals'].items():
            if not math.isclose(sum(v[field] for v in ledger),total,abs_tol=1e-7):raise ValueError('cash sum mismatch')
        c=np.array(a['curve']);peak=np.maximum(10000,np.maximum.accumulate(c[:,1]));test=dict(a['curve'])[r['split_boundaries_ms'][1]]
        np.testing.assert_allclose([100*(c[-1,1]/10000-1),float(100*(1-c[:,1]/peak).max()),100*(c[-1,1]/test-1)],
            [a['net_return_pct'],a['max_drawdown_pct'],a['test_return_pct']],rtol=0,atol=1e-8)
        if not math.isclose(a['alpha_pp'],a['net_return_pct']-r['benchmark']['net_return_after_costs_pct'],abs_tol=1e-8):raise ValueError('alpha mismatch')
        if objective(trades,cache,m['eligible_symbols'],r['start_ms'],r['end_ms'])!=r['missions'][name]:raise ValueError('mission mismatch')
        cut=r['split_boundaries_ms'][1]
        if objective([t for t in trades if t.entry_ts>=cut],cache,m['eligible_symbols'],cut,r['end_ms'])!=r['test_missions'][name]:raise ValueError('TEST mission mismatch')
        assert_nested(exit_diagnostics([asdict(t) for t in trades]),r['exit_diagnostics'][name])
        checked[name]={'trades':len(trades),'equity_points':len(grid)}
    ci=single_interval(r['accounts']['control']['curve'],r['accounts']['exit_model']['curve'],r['split_boundaries_ms'][1],r['end_ms'])
    verdict=gate(r['accounts']['control'],r['accounts']['exit_model'],tuple(r['missions'][k] for k in ('control','exit_model')),
        tuple(r['test_missions'][k] for k in ('control','exit_model')),ci['interval'],ci['days'])
    test=diagnostics['test'];verdict['checks']['positive_action_mean']=test['selected_mean_advantage_pp'] is not None and test['selected_mean_advantage_pp']>0
    verdict['checks']['hurt_rate']=test['selected_known_n']>0 and test['selected_hurt_n']/test['selected_known_n']<=.35
    verdict['numerical_gate']='PASS' if all(verdict['checks'].values()) else 'REJECTED' if ci['days']>=30 else 'INCONCLUSIVE'
    assert_nested({**verdict,**ci},r['comparisons']['exit_model'])
    return {'status':'PASS','dataset_soft_exits':len(cases),'native_control_predictions':native_rows,
        'actual_soft_decisions':actual_rows,'actual_deferrals':len(deferred),'models':len(models),'accounts':checked,
        'result_sha256':sha(directory/'result.json'),'verifier_sha256':sha(Path(__file__)),
        'limitations':['indicator checkpoints hash/producer-bound; no independent full indicator recomputation',
                       'numeric/state/policy audit, no live/PIT/exchange-fill certification']}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True)
    a=p.parse_args();v=verify(a.run);freeze(a.run/'independent_verification.json',json.dumps(v,indent=2).encode());print(json.dumps(v,indent=2))
