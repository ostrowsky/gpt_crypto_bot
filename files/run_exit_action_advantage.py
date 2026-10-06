"""Registered causal concrete SELL-action model and maximum-period portfolio replay."""
import argparse
import asyncio
from dataclasses import asdict
import json
from pathlib import Path
from unittest.mock import patch
import numpy as np
from catboost import CatBoostRegressor
import config
import replay_backtest as rb
from capacity_catboost import split_boundaries
from compare_price_volatility_bot import reuse_features,reuse_candidate_snapshot,replacement_options
from historical_signal_evaluation import sha,freeze
from impulse_entry_catboost import BAR,DAY,gate
from exit_action_advantage import (MODEL_PARAMETERS,THRESHOLD,position_features,label_action,
                                  valid_origin,deadline,cohorts,ExitPolicy)
from run_joint_direction_amplitude import reuse_parent,SOURCES as PARENT_SOURCES
from run_impulse_entry_catboost import single_interval
from run_turnover_economics import publish,validate_data,objective
from research_rocket_capture import policy
from audit_negative_day_rebound import finalize_at_boundary
from portfolio_alpha import _simulate_account,_benchmark_result,closed_price_series
from turnover_economics import cash_ledger,grouped_attribution
from render_turnover_economics import exit_diagnostics

ROOT=Path(__file__).resolve().parent.parent
SOURCES=('run_exit_action_advantage.py','exit_action_advantage.py','render_turnover_economics.py')+PARENT_SOURCES


def dataset(trades,cache,end,fee,slip,progress):
    cases=[];xs=[];ys=[];clock=[];available=[]
    for i,t in enumerate(trades):
        if not rb._is_weak_exit_reason(t.exit_reason):continue
        d,f=cache[t.sym,t.tf];idx=rb._find_last_closed_candle_index(d['t'],t.exit_ts,rb.BAR_MS[t.tf])
        x=(position_features(t,cache[t.sym,'15m'][0],t.exit_ts,t.exit_price)
           if idx is not None and valid_origin(t,d,idx,t.exit_ts,t.exit_price) else None)
        target=label_action(t,cache,end,fee,slip,progress) if x is not None else None
        until=deadline(d,idx,t.tf) if idx is not None else t.exit_ts+rb.BAR_MS[t.tf]
        cases.append({'trade_id':i,'key':[t.sym,t.tf,t.entry_ts],'at':t.exit_ts,
                      'target':target,'feature_known':x is not None,'available_at':until})
        xs.append(np.full(22,np.nan) if x is None else x);ys.append(np.nan if target is None else target['advantage'])
        clock.append(t.exit_ts);available.append(until)
    return cases,dict(clock=np.asarray(clock,dtype=np.int64),available=np.asarray(available,dtype=np.int64),
                      x=np.asarray(xs),y=np.asarray(ys))


def fit_blocks(z,start,end,output):
    clock=z['clock'];valid_x=np.isfinite(z['x']).all(axis=1);valid=valid_x&np.isfinite(z['y'])
    z['prediction']=np.full(len(clock),np.nan);z['fold']=np.full(len(clock),-1,dtype=np.int64)
    folds=[];models={}
    for at in range(start+30*DAY,end,30*DAY):
        stop=min(end,at+30*DAY);train,val,cut=cohorts(clock,z['available'],valid,at,start)
        issued=(clock>=at)&(clock<stop);score=issued&valid_x
        record={'fit_at':at,'stop':stop,'validation_boundary':cut,'train_n':int(train.sum()),
                'validation_n':int(val.sum()),'issued_n':int(issued.sum())}
        if train.sum()<300 or val.sum()<50:
            record['state']='UNTRAINED_IMMEDIATE_SELL';folds.append(record);continue
        model=CatBoostRegressor(**MODEL_PARAMETERS)
        model.fit(z['x'][train],z['y'][train],eval_set=(z['x'][val],z['y'][val]),early_stopping_rounds=40,use_best_model=True)
        k=len(folds);name=f'model_{k}.cbm';model.save_model(str(output/name));models[k]=model
        z['prediction'][score]=model.predict(z['x'][score]);z['fold'][issued]=k
        record.update(state='FROZEN',model=name,model_sha256=sha(output/name),trees=model.tree_count_,
            train_max_label_at=int(z['available'][train].max()),validation_max_label_at=int(z['available'][val].max()),
            selected_n=int((z['prediction'][issued]>THRESHOLD).sum()))
        folds.append(record);print(json.dumps(record),flush=True)
    publish(output/'folds.json',folds);np.savez_compressed(output/'dataset.npz',**z)
    def predictor(at,x):
        for k,f in enumerate(folds):
            if f['fit_at']<=at<f['stop'] and k in models:
                return (float(models[k].predict(np.asarray(x)[None,:])[0]),k) if x is not None else (None,k)
        return None,None
    return predictor,folds


def action_diagnostics(z,cut):
    result={}
    for name,mask in (('all_oos',z['fold']>=0),('test',(z['fold']>=0)&(z['clock']>=cut))):
        known=mask&np.isfinite(z['y'])&np.isfinite(z['prediction']);p=z['prediction'][known];y=z['y'][known]
        selected=p>THRESHOLD;target=y[selected]
        result[name]={'issued_n':int(mask.sum()),'known_n':int(known.sum()),'selected_known_n':int(selected.sum()),
            'always_hold_mean_advantage_pp':float(y.mean()) if len(y) else None,
            'selected_mean_advantage_pp':float(target.mean()) if len(target) else None,
            'selected_median_advantage_pp':float(np.median(target)) if len(target) else None,
            'selected_hurt_n':int((target<0).sum()),
            'selected_p10_pp':float(np.quantile(target,.1)) if len(target) else None,
            'mae_pp':float(np.mean(abs(p-y))) if len(y) else None,'immediate_sell_mae_pp':float(np.mean(abs(y))) if len(y) else None,
            'scope':'control-entry first-WEAK action diagnostic, not policy account PnL'}
    return result


async def run(parent,output):
    if not parent.resolve().is_relative_to(ROOT/'.runtime'):raise ValueError('trusted parent only')
    old=json.loads((parent/'registration.json').read_bytes());market=Path(old['market']);features=Path(old['features']);candidates=Path(old['candidates'])
    for p in (market,features,candidates):
        if not p.resolve().is_relative_to(ROOT/'.runtime'):raise ValueError('trusted runtime only')
    output.mkdir(parents=True,exist_ok=False);(output/'source_snapshot').mkdir();sources={}
    for name in SOURCES:
        path=Path(__file__).with_name(name);freeze(output/'source_snapshot'/name,path.read_bytes());sources[name]=sha(path)
    freeze(output/'registered_spec.md',(ROOT/'docs/specs/exit-action-advantage.md').read_bytes())
    m=json.loads((market/'manifest.json').read_bytes());start,end=m['start_ms'],m['end_ms'];cuts=split_boundaries(start,end)
    fee=max(7.5,float(config.PAPER_FEE_BPS));slip=5.
    reg={'contract':'exit-action-advantage-v1','parent':str(parent.resolve()),'market':str(market.resolve()),
        'sources':sources,'start_ms':start,'end_ms':end,'market_sha256':sha(market/'manifest.json'),
        'spec_sha256':sha(output/'registered_spec.md'),'parameters':MODEL_PARAMETERS,'threshold_pp':THRESHOLD,
        'fee_bps':fee,'slippage_bps':slip,'split_boundaries_ms':cuts,'runtime_eligible':False}
    publish(output/'registration.json',reg)
    def verify():
        if sha(market/'manifest.json')!=reg['market_sha256']:raise ValueError('manifest drift')
        for name,h in sources.items():
            if sha(Path(__file__).with_name(name))!=h:raise ValueError('source drift: '+name)
        for name,h in m['input_hashes'].items():
            if Path(name).name!=name or sha(market/'market'/name)!=h:raise ValueError('market drift')
    verify();reuse_parent(parent,market,output);cache,frames,hashes=reuse_features(features,m,output);del frames;validate_data(cache,m)
    control=[rb.ReplayTrade(**t) for t in json.loads((output/'trades_control.json').read_bytes())]
    symbols=m['eligible_symbols'];c15={s:cache[s,'15m'] for s in symbols};c4={s:cache[s,'4h'] for s in symbols}
    context=rb._build_bull_day_context(cache['BTCUSDT','1h'][0]);progress=rb._update_trade_progress
    with policy('baseline'),patch.object(rb,'_load_temporal_scout_events',return_value=({},{})):
        cases,z=dataset(control,cache,end,fee,slip,progress);publish(output/'cases.json',cases)
        print(json.dumps({'phase':'FIT','weak_exits':len(cases),'known_labels':int(np.isfinite(z['y']).sum())}),flush=True)
        predictor,folds=fit_blocks(z,start,end,output)
        snapshot=reuse_candidate_snapshot(candidates,m,output);enabled,delta=replacement_options(config)
        wrapper=ExitPolicy(progress,cache,predictor)
        print(json.dumps({'phase':'SIMULATE','arm':'exit_model','candidates':snapshot[2]}),flush=True)
        with patch.object(rb,'_update_trade_progress',side_effect=wrapper):
            trades,stat=await rb.simulate_portfolio(symbols,['15m','1h'],cache,c15,c4,context,max_open_positions=10,
                enable_replacement=enabled,replace_min_delta=delta,variant='score_replace_cluster',
                top_gainer_score_min=float(config.TOP_GAINER_SCORE_GATE_MIN_SCORE),candidate_snapshot=snapshot)
        finalize_at_boundary(trades,cache,end);publish(output/'trades_exit_model.json',[asdict(t) for t in trades])
        publish(output/'policy_trace.json',{'decisions':wrapper.decisions,'outcomes':wrapper.outcomes,
            'remaining_deferrals':len(wrapper.active),'scope':'remaining states may correspond to replacement or boundary liquidation'})
    series={s:closed_price_series(c15[s][0],bar_ms=BAR,start_ms=start,end_ms=end) for s in symbols};grid=list(range(start,end+1,BAR))
    benchmark=_benchmark_result(series['BTCUSDT'],initial_capital=10000,fee_bps=fee,slippage_bps=slip)
    accounts={};missions={};test_missions={};attribution={};stress={};exits={}
    prior=json.loads((output/'parent_result.json').read_bytes())
    for name,rows in (('control',control),('exit_model',trades)):
        a=cash_ledger(rows,series,grid,fee,slip)
        ref=_simulate_account(rows,price_series_by_symbol=series,valuation_timestamps=grid,capacity=10,initial_capital=10000,fee_bps=fee,slippage_bps=slip)
        if ref.violations or ref.fully_valued_points!=len(grid):raise ValueError('cash/grid violation')
        np.testing.assert_allclose(a['curve'],ref.equity_curve,rtol=1e-10,atol=1e-7)
        np.testing.assert_allclose([a['totals']['fees'],a['totals']['slippage']],[ref.fees_quote,ref.slippage_quote],rtol=1e-10,atol=1e-7)
        ledger=a.pop('ledger');publish(output/f'ledger_{name}.json',ledger);curve=dict(a['curve'])
        if name=='control':np.testing.assert_allclose(a['curve'],prior['accounts']['control']['curve'],rtol=0,atol=1e-9)
        accounts[name]={**a,'test_return_pct':100*(curve[end]/curve[cuts[1]]-1),'trades':len(rows),
            'alpha_pp':a['net_return_pct']-benchmark['net_return_after_costs_pct'],
            'replacements':stat.replacements_total if name=='exit_model' else prior['accounts']['control']['replacements']}
        missions[name]=objective(rows,cache,symbols,start,end)
        test_missions[name]=objective([t for t in rows if t.entry_ts>=cuts[1]],cache,symbols,cuts[1],end)
        attribution[name]={'mode':grouped_attribution(ledger,'mode')};exits[name]=exit_diagnostics([asdict(t) for t in rows])
        stress[name]={k:v for k,v in cash_ledger(rows,series,grid,2*fee,2*slip).items() if k not in ('curve','ledger')}
    ci=single_interval(accounts['control']['curve'],accounts['exit_model']['curve'],cuts[1],end)
    verdict=gate(accounts['control'],accounts['exit_model'],(missions['control'],missions['exit_model']),
        (test_missions['control'],test_missions['exit_model']),ci['interval'],ci['days'])
    diagnostics=action_diagnostics(z,cuts[1]);test=diagnostics['test']
    verdict['checks']['positive_action_mean']=test['selected_mean_advantage_pp'] is not None and test['selected_mean_advantage_pp']>0
    verdict['checks']['hurt_rate']=test['selected_known_n']>0 and test['selected_hurt_n']/test['selected_known_n']<=.35
    verdict['numerical_gate']='PASS' if all(verdict['checks'].values()) else 'REJECTED' if ci['days']>=30 else 'INCONCLUSIVE'
    verify()
    for sym,h in hashes.items():
        if sha(features/f'{sym}.npz')!=h:raise ValueError('feature drift')
    result={'status':'COMPLETED_RETROSPECTIVE_PREQUENTIAL','runtime_eligible':False,'achievement_claimed':False,
        'start_ms':start,'end_ms':end,'days':(end-start)/DAY,'split_boundaries_ms':cuts,
        'population_complete':len(symbols),'population_requested':len(m['requested_symbols']),
        'costs':{'fee_bps':fee,'slippage_bps':slip},'benchmark':benchmark,'accounts':accounts,'missions':missions,
        'test_missions':test_missions,'comparisons':{'exit_model':{**verdict,**ci}},'action_diagnostics':diagnostics,
        'exit_diagnostics':exits,'attribution':attribution,'cost_stress':stress,'folds':folds,
        'policy_counts':{'actual_soft_decisions':len(wrapper.decisions),'actual_deferrals':sum(d['defer'] for d in wrapper.decisions)},
        'limitations':prior['limitations']+['training population control soft exits; changed-policy state distribution differs',
            'isolated one-bar action target omits capacity/replacement opportunity cost',
            'fixed deadline is 15m or1h; original hard exits may execute earlier; no intrabar fill guarantee']}
    publish(output/'result.json',result);publish(output/'receipt.json',{p.name:sha(p) for p in output.iterdir() if p.is_file()})
    print(json.dumps({'accounts':{n:{k:a[k] for k in ('net_return_pct','test_return_pct','trades')} for n,a in accounts.items()},'verdict':verdict},indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('parent','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();asyncio.run(run(a.parent,a.output))
