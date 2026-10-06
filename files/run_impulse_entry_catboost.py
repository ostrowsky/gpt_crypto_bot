"""Registered maximum-history causal impulse-entry experiment, no live adoption."""
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
from audit_negative_day_rebound import finalize_at_boundary
from capacity_catboost import split_boundaries
from compare_price_volatility_bot import reuse_features, reuse_candidate_snapshot, replacement_options
from historical_signal_evaluation import sha
from impulse_entry_catboost import (BAR, DAY, HORIZON, PARAMETERS, FEATURE_NAMES,
                                    features, net_target, fit_cohorts, admit, screen, gate)
from portfolio_alpha import _simulate_account, _benchmark_result, closed_price_series
from research_rocket_capture import policy
from run_turnover_economics import publish, validate_data, objective
from turnover_economics import cash_ledger, grouped_attribution

ROOT = Path(__file__).resolve().parent.parent
SOURCES = ('run_impulse_entry_catboost.py','impulse_entry_catboost.py',
           'run_turnover_economics.py','turnover_economics.py','capacity_catboost.py',
           'compare_price_volatility_bot.py','replay_backtest.py','portfolio_alpha.py',
           'config.py','indicators.py','strategy.py','monitor.py',
           'research_rocket_capture.py','audit_negative_day_rebound.py')


def make_dataset(snapshot, cache, fee, slip):
    rows=[]; xs=[]; ys=[]
    for at in sorted(snapshot[0]):
        for j,c in enumerate(snapshot[0][at]):
            if c.mode != 'impulse_speed': continue
            d=cache[c.sym,'15m'][0]
            x=features(d,at,c.tf); y=net_target(d,at,fee,slip)
            rows.append((at,j,c.sym,c.tf)); xs.append(np.full(len(FEATURE_NAMES),np.nan) if x is None else x)
            ys.append(np.nan if y is None else y)
    return rows,np.asarray(xs),np.asarray(ys)


def fit_predict(rows,x,y,start,end,output):
    clocks=np.array([r[0] for r in rows],dtype=np.int64)
    valid_x=np.isfinite(x).all(axis=1); valid_y=np.isfinite(y)
    predictions=np.full(len(rows),np.nan); fold_id=np.full(len(rows),-1,dtype=np.int64)
    folds=[]; decisions={}
    for at in range(start+30*DAY,end,30*DAY):
        stop=min(end,at+30*DAY)
        train,val,cut=fit_cohorts(clocks,valid_x & valid_y,at,start)
        issued=(clocks>=at)&(clocks<stop); score=issued & valid_x
        record={'fit_at':at,'stop':stop,'validation_boundary':cut,
                'train_n':int(train.sum()),'validation_n':int(val.sum()),
                'issued_n':int(issued.sum()),'unknown_feature_n':int((issued & ~valid_x).sum())}
        if train.sum()<500 or val.sum()<100:
            record['state']='UNTRAINED_PASSTHROUGH';folds.append(record);continue
        model=CatBoostRegressor(**PARAMETERS)
        model.fit(x[train],y[train],eval_set=(x[val],y[val]),early_stopping_rounds=40,use_best_model=True)
        path=output/f'model_{len(folds)}.cbm';model.save_model(str(path))
        predictions[score]=model.predict(x[score]);fold_id[issued]=len(folds)
        for i in np.flatnonzero(issued): decisions[tuple(rows[i][:2])]=admit(predictions[i])
        record.update(state='FROZEN',model=path.name,model_sha256=sha(path),trees=model.tree_count_,
                      train_max_label_at=int((clocks[train]+HORIZON).max()),
                      validation_max_label_at=int((clocks[val]+HORIZON).max()),
                      accepted_n=int(sum(admit(p) for p in predictions[issued])))
        folds.append(record);print(json.dumps(record),flush=True)
    publish(output/'folds.json',folds)
    np.savez_compressed(output/'dataset.npz',clock=clocks,index=np.array([r[1] for r in rows]),
        symbol=np.array([r[2] for r in rows]),tf=np.array([r[3] for r in rows]),
        x=x,y=y,prediction=predictions,fold=fold_id)
    return decisions,folds,predictions,fold_id


async def run(market,features_dir,candidates,output):
    for p in (market,features_dir,candidates):
        if not p.resolve().is_relative_to(ROOT/'.runtime'): raise ValueError('trusted runtime only')
    m=json.loads((market/'manifest.json').read_bytes());start,end=m['start_ms'],m['end_ms']
    cuts=split_boundaries(start,end);fee=max(7.5,float(config.PAPER_FEE_BPS));slip=5.
    output.mkdir(parents=True,exist_ok=False);(output/'source_snapshot').mkdir()
    sources={}
    for name in SOURCES:
        p=Path(__file__).with_name(name);(output/'source_snapshot'/name).write_bytes(p.read_bytes());sources[name]=sha(p)
    reg={'contract':'impulse-entry-net-v1','parameters':PARAMETERS,'feature_names':FEATURE_NAMES,
         'start_ms':start,'end_ms':end,'market':str(market.resolve()),'features':str(features_dir.resolve()),
         'candidates':str(candidates.resolve()),'sources':sources,'market_sha256':sha(market/'manifest.json'),
         'fee_bps':fee,'slippage_bps':slip,'threshold':0,'initial_passthrough_days':30,
         'refit_days':30,'split_boundaries_ms':cuts,'runtime_eligible':False}
    publish(output/'registration.json',reg)
    def verify():
        if sha(market/'manifest.json')!=reg['market_sha256']: raise ValueError('manifest drift')
        for n,h in m['input_hashes'].items():
            if Path(n).name!=n or sha(market/'market'/n)!=h: raise ValueError('market drift')
        for n,h in sources.items():
            if sha(Path(__file__).with_name(n))!=h: raise ValueError('source drift '+n)
    verify();cache,frames,hashes=reuse_features(features_dir,m,output);del frames
    validate_data(cache,m);symbols=m['eligible_symbols']
    c15={s:cache[s,'15m'] for s in symbols};c4={s:cache[s,'4h'] for s in symbols}
    context=rb._build_bull_day_context(cache['BTCUSDT','1h'][0])
    with policy('baseline'),patch.object(rb,'_load_temporal_scout_events',return_value=({},{})):
        snapshot=reuse_candidate_snapshot(candidates,m,output)
        spot=[]
        for sym,tf in (('BTCUSDT','15m'),('ETHUSDT','1h')):
            d,f=cache[sym,tf]
            actual=await rb._build_candidates_for_symbol(sym,tf,d,f,c15,c4,context,variant='score_replace_cluster',include_trend_start=False)
            actual=[asdict(c) for c in actual if start<=c.ts_ms<end]
            expected=[asdict(c) for at in sorted(snapshot[0]) for c in snapshot[0][at] if c.sym==sym and c.tf==tf]
            if actual!=expected: raise ValueError('candidate kernel differs')
            spot.append({'symbol':sym,'tf':tf,'n':len(actual)})
        publish(output/'candidate_spot_verification.json',{'status':'PASS','pairs':spot})
        rows,x,y=make_dataset(snapshot,cache,fee,slip)
        print(json.dumps({'phase':'FIT','impulse_candidates':len(rows)}),flush=True)
        decisions,folds,pred,fold_ids=fit_predict(rows,x,y,start,end,output)
        filtered=screen(snapshot,decisions);enabled,delta=replacement_options(config)
        trades_by_arm={};stats={}
        for name,stream in (('control',snapshot),('catboost',filtered)):
            print(json.dumps({'phase':'SIMULATE','arm':name,'candidates':stream[2]}),flush=True)
            trades,stat=await rb.simulate_portfolio(symbols,['15m','1h'],cache,c15,c4,context,
                max_open_positions=10,enable_replacement=enabled,replace_min_delta=delta,
                variant='score_replace_cluster',top_gainer_score_min=float(config.TOP_GAINER_SCORE_GATE_MIN_SCORE),
                candidate_snapshot=stream)
            finalize_at_boundary(trades,cache,end);trades_by_arm[name]=trades;stats[name]=asdict(stat)
            publish(output/f'trades_{name}.json',[asdict(t) for t in trades])
    series={s:closed_price_series(c15[s][0],bar_ms=BAR,start_ms=start,end_ms=end) for s in symbols}
    grid=list(range(start,end+1,BAR));accounts={};missions={};test_missions={};attribution={};stress={}
    benchmark=_benchmark_result(series['BTCUSDT'],initial_capital=10000,fee_bps=fee,slippage_bps=slip)
    for name,trades in trades_by_arm.items():
        a=cash_ledger(trades,series,grid,fee,slip)
        ref=_simulate_account(trades,price_series_by_symbol=series,valuation_timestamps=grid,capacity=10,
                              initial_capital=10000,fee_bps=fee,slippage_bps=slip)
        if ref.violations or ref.fully_valued_points!=len(grid): raise ValueError('cash/grid violation')
        np.testing.assert_allclose(a['curve'],ref.equity_curve,rtol=1e-10,atol=1e-7)
        np.testing.assert_allclose([a['totals']['fees'],a['totals']['slippage']],
                                   [ref.fees_quote,ref.slippage_quote],rtol=1e-10,atol=1e-7)
        ledger=a.pop('ledger');publish(output/f'ledger_{name}.json',ledger);curve=dict(a['curve'])
        accounts[name]={**a,'test_return_pct':100*(curve[end]/curve[cuts[1]]-1),
                        'alpha_pp':a['net_return_pct']-benchmark['net_return_after_costs_pct'],
                        'trades':len(trades),'replacements':stats[name]['replacements_total']}
        missions[name]=objective(trades,cache,symbols,start,end)
        test_missions[name]=objective([t for t in trades if t.entry_ts>=cuts[1]],cache,symbols,cuts[1],end)
        attribution[name]={'mode':grouped_attribution(ledger,'mode')}
        stress[name]={k:v for k,v in cash_ledger(trades,series,grid,2*fee,2*slip).items() if k not in ('curve','ledger')}
    ci=single_interval(accounts['control']['curve'],accounts['catboost']['curve'],cuts[1],end)
    verdict=gate(accounts['control'],accounts['catboost'],(missions['control'],missions['catboost']),
                 (test_missions['control'],test_missions['catboost']),ci['interval'],ci['days'])
    forecast={}
    for name,mask in (('all_oos',fold_ids>=0),('test',np.array([r[0]>=cuts[1] for r in rows]) & (fold_ids>=0))):
        valid=mask & np.isfinite(pred) & np.isfinite(y);p=pred[valid];t=y[valid]
        forecast[name]={'issued_n':int(mask.sum()),'label_known_n':int(valid.sum()),
                        'missing_target_or_feature_n':int((mask & ~valid).sum()),
                        'mae_pct':float(np.mean(abs(p-t))),'zero_mae_pct':float(np.mean(abs(t))),
                        'rmse_pct':float(np.sqrt(np.mean((p-t)**2))),
                        'accepted_n':int((p>0).sum()),'accepted_mean_net_proxy_pct':float(t[p>0].mean()) if (p>0).any() else None,
                        'base_positive_n':int((t>0).sum()),'direction_correct_n':int(((p>0)==(t>0)).sum())}
    verify()
    for s,h in hashes.items():
        if sha(features_dir/f'{s}.npz')!=h: raise ValueError('feature drift')
    result={'status':'COMPLETED_RETROSPECTIVE_PREQUENTIAL','runtime_eligible':False,'achievement_claimed':False,
            'start_ms':start,'end_ms':end,'days':(end-start)/DAY,'split_boundaries_ms':cuts,
            'population_complete':len(symbols),'population_requested':len(m['requested_symbols']),
            'costs':{'fee_bps':fee,'slippage_bps':slip},'benchmark':benchmark,'accounts':accounts,
            'missions':missions,'test_missions':test_missions,'attribution':attribution,'cost_stress':stress,
            'comparisons':{'catboost':{**verdict,**ci}},'forecast_metrics':forecast,'folds':folds,
            'filter_audit':{'all_candidates':snapshot[2],'impulse_candidates':len(rows),
                            'scored_impulse_candidates':int((fold_ids>=0).sum()),'retained_all_candidates':filtered[2]},
            'limitations':['previously exposed retrospective history, prequential OOS not sealed',
                '93/105 complete symbols; historical PIT/live-agent/receive-time parity uncertified',
                '75m net target is a proxy; actual portfolio uses unchanged original exits',
                'closed-bar idealized fills and static costs, not observed executions',
                'first30 days untrained passthrough; TEST inherits earlier holdings and equity']}
    publish(output/'result.json',result)
    publish(output/'receipt.json',{p.name:sha(p) for p in output.iterdir() if p.is_file()})
    print(json.dumps({'accounts':{n:{k:a[k] for k in ('net_return_pct','test_return_pct','trades')} for n,a in accounts.items()},'verdict':verdict},indent=2),flush=True)


def single_interval(base,target,start,end):
    # Reuse exact calendar grid construction but independently regenerate quantiles.
    from datetime import datetime,time,timedelta,timezone
    from capacity_catboost import TZ
    b=dict(base);t=dict(target);day=datetime.fromtimestamp(start/1000,timezone.utc).astimezone(TZ).date()
    last=datetime.fromtimestamp(end/1000,timezone.utc).astimezone(TZ).date();delta=[]
    while day<last:
        a=int(datetime.combine(day,time(),tzinfo=TZ).timestamp()*1000)
        z=int(datetime.combine(day+timedelta(days=1),time(),tzinfo=TZ).timestamp()*1000)
        if start<=a and z<=end and all(v in b and v in t for v in (a,z)):
            delta.append(np.log(t[z]/t[a])-np.log(b[z]/b[a]))
        day+=timedelta(days=1)
    if len(delta)<3:return {'days':len(delta),'interval':None}
    rng=np.random.default_rng(42);starts=rng.integers(0,len(delta)-2,size=(5000,(len(delta)+2)//3))
    indices=(starts[:,:,None]+np.arange(3)).reshape(5000,-1)[:,:len(delta)]
    values=np.asarray(delta)[indices].mean(axis=1)*10000
    return {'days':len(delta),'interval':np.quantile(values,[.025,.975]).tolist(),
            'mean_daily_log_uplift_bp':float(np.mean(delta)*10000),'confidence':.95,'block_days':3,'draws':5000}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('market','features','candidates','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();asyncio.run(run(a.market,a.features,a.candidates,a.output))
