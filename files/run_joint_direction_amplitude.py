"""Frozen direction/amplitude experiment with verified same-profile references."""
import argparse
import asyncio
from dataclasses import asdict
import json
from pathlib import Path
from unittest.mock import patch
import numpy as np
import config
import replay_backtest as rb
from capacity_catboost import split_boundaries,forward_return,HORIZON
from compare_price_volatility_bot import reuse_features,reuse_candidate_snapshot,replacement_options
from historical_signal_evaluation import sha,freeze
from impulse_entry_catboost import BAR,DAY,FEATURE_NAMES,PARAMETERS,admit,screen,gate
from joint_direction_amplitude import cohorts,enough,fit_models,predict,net_from_gross,diagnostics
from run_impulse_entry_catboost import SOURCES as PARENT_SOURCES,single_interval
from run_turnover_economics import publish,validate_data,objective
from research_rocket_capture import policy
from audit_negative_day_rebound import finalize_at_boundary
from portfolio_alpha import closed_price_series,_simulate_account,_benchmark_result
from turnover_economics import cash_ledger,grouped_attribution

ROOT=Path(__file__).resolve().parent.parent
SOURCES=('run_joint_direction_amplitude.py','joint_direction_amplitude.py')+PARENT_SOURCES


def reuse_parent(parent,market,output):
    if not parent.resolve().is_relative_to(ROOT/'.runtime'):raise ValueError('trusted parent only')
    reg=json.loads((parent/'registration.json').read_bytes())
    receipt=json.loads((parent/'receipt.json').read_bytes())
    for name,h in receipt.items():
        if Path(name).name!=name or sha(parent/name)!=h:raise ValueError('parent receipt drift')
    audit=json.loads((parent/'independent_verification.json').read_bytes())
    if audit['status']!='PASS' or audit['result_sha256']!=sha(parent/'result.json'):raise ValueError('parent not audited')
    if sha(market/'manifest.json')!=reg['market_sha256']:raise ValueError('parent market differs')
    for name,h in reg['sources'].items():
        if sha(Path(__file__).with_name(name))!=h:raise ValueError('parent kernel differs: '+name)
    inherited={}
    for target,name in [('parent_dataset.npz','dataset.npz'),('parent_result.json','result.json'),
                         ('parent_registration.json','registration.json'),('parent_audit.json','independent_verification.json'),
                         ('candidate_spot_verification.json','candidate_spot_verification.json'),
                         ('trades_control.json','trades_control.json'),('trades_regressor.json','trades_catboost.json')]:
        freeze(output/target,(parent/name).read_bytes())
        inherited[target]={'source':str((parent/name).resolve()),'sha256':sha(parent/name)}
    publish(output/'inheritance_receipt.json',{'parent':str(parent.resolve()),'files':inherited,
        'parent_receipt_sha256':sha(parent/'receipt.json'),'parent_audit_sha256':sha(parent/'independent_verification.json')})
    return reg


def fit_blocks(z,start,end,fee,slip,output):
    clocks=z['clock'];x=z['x'];g=z['gross']
    valid_x=np.isfinite(x).all(axis=1);valid=valid_x&np.isfinite(g)
    for name in ('raw_p','p','up','down','prediction','climatology','train_mean_net'):
        z[name]=np.full(len(clocks),np.nan)
    z['fold']=np.full(len(clocks),-1,dtype=np.int64);folds=[];decisions={}
    for at in range(start+30*DAY,end,30*DAY):
        stop=min(end,at+30*DAY);train,val,cal,cuts=cohorts(clocks,valid,at,start)
        issued=(clocks>=at)&(clocks<stop);scored=issued&valid_x
        record={'fit_at':at,'stop':stop,'boundaries':cuts,'train_n':int(train.sum()),
                'validation_n':int(val.sum()),'calibration_n':int(cal.sum()),'issued_n':int(issued.sum())}
        if not enough(g,train,val,cal):
            record['state']='UNTRAINED_PASSTHROUGH';folds.append(record);continue
        models,coef,intercept=fit_models(x,g,train,val,cal)
        checksums={};trees={}
        for name,model in models.items():
            filename=f'model_{len(folds)}_{name}.cbm';model.save_model(str(output/filename))
            checksums[filename]=sha(output/filename);trees[name]=model.tree_count_
        raw,p,up,down,net=predict(models,x[scored],coef,intercept,fee,slip)
        for name,value in zip(('raw_p','p','up','down','prediction'),(raw,p,up,down,net)):z[name][scored]=value
        z['climatology'][scored]=np.mean(g[train]>0)
        z['train_mean_net'][scored]=net_from_gross(np.mean(g[train]),fee,slip)
        z['fold'][issued]=len(folds)
        for i in np.flatnonzero(issued):decisions[int(clocks[i]),int(z['index'][i])]=admit(z['prediction'][i])
        record.update(state='FROZEN',models=checksums,trees=trees,platt_coef=coef,platt_intercept=intercept,
            train_max_label_at=int((clocks[train]+HORIZON).max()),validation_max_label_at=int((clocks[val]+HORIZON).max()),
            calibration_max_label_at=int((clocks[cal]+HORIZON).max()),
            accepted_n=int((net>0).sum()),train_up_n=int((g[train]>0).sum()),
            validation_up_n=int((g[val]>0).sum()),calibration_up_n=int((g[cal]>0).sum()))
        folds.append(record);print(json.dumps(record),flush=True)
    publish(output/'folds.json',folds);np.savez_compressed(output/'dataset.npz',**z)
    return decisions,folds


async def run(parent,output):
    if not parent.resolve().is_relative_to(ROOT/'.runtime'):raise ValueError('trusted runtime only')
    previous=json.loads((parent/'registration.json').read_bytes())
    market=Path(previous['market']);features=Path(previous['features']);candidates=Path(previous['candidates'])
    for p in (market,features,candidates):
        if not p.resolve().is_relative_to(ROOT/'.runtime'):raise ValueError('trusted runtime only')
    output.mkdir(parents=True,exist_ok=False);(output/'source_snapshot').mkdir();sources={}
    for name in SOURCES:
        path=Path(__file__).with_name(name);freeze(output/'source_snapshot'/name,path.read_bytes());sources[name]=sha(path)
    spec=ROOT/'docs/specs/joint-direction-amplitude.md';freeze(output/'registered_spec.md',spec.read_bytes())
    m=json.loads((market/'manifest.json').read_bytes());start,end=m['start_ms'],m['end_ms'];cuts=split_boundaries(start,end)
    fee=max(7.5,float(config.PAPER_FEE_BPS));slip=5.
    reg={'contract':'joint-direction-amplitude-v1','parent':str(parent.resolve()),'market':str(market.resolve()),
        'sources':sources,'start_ms':start,'end_ms':end,'market_sha256':sha(market/'manifest.json'),
        'registered_spec_sha256':sha(output/'registered_spec.md'),'parameters':PARAMETERS,'features':FEATURE_NAMES,
        'classifier_loss':'Logloss','platt':{'C':1.,'solver':'lbfgs','max_iter':1000},
        'fee_bps':fee,'slippage_bps':slip,'split_boundaries_ms':cuts,'runtime_eligible':False}
    publish(output/'registration.json',reg)
    def verify():
        if sha(market/'manifest.json')!=reg['market_sha256']:raise ValueError('manifest drift')
        for name,h in sources.items():
            if sha(Path(__file__).with_name(name))!=h:raise ValueError('source drift: '+name)
        for name,h in m['input_hashes'].items():
            if Path(name).name!=name or sha(market/'market'/name)!=h:raise ValueError('raw drift')
    verify();old=reuse_parent(parent,market,output)
    if (old['start_ms'],old['end_ms'],old['fee_bps'],old['slippage_bps'])!=(start,end,fee,slip):raise ValueError('parent profile differs')
    cache,frames,hashes=reuse_features(features,m,output);del frames;validate_data(cache,m)
    with np.load(output/'parent_dataset.npz',allow_pickle=False) as p:
        z={name:p[name] for name in ('clock','index','symbol','tf','x','y')}
    gross=[]
    for at,sym in zip(z['clock'],z['symbol']):
        value=forward_return(cache[str(sym),'15m'][0],int(at));gross.append(np.nan if value is None else value)
    z['gross']=np.asarray(gross)
    np.testing.assert_allclose(net_from_gross(z['gross'],fee,slip),z['y'],rtol=1e-11,atol=1e-10,equal_nan=True)
    decisions,folds=fit_blocks(z,start,end,fee,slip,output)
    symbols=m['eligible_symbols'];c15={s:cache[s,'15m'] for s in symbols};c4={s:cache[s,'4h'] for s in symbols}
    context=rb._build_bull_day_context(cache['BTCUSDT','1h'][0])
    with policy('baseline'),patch.object(rb,'_load_temporal_scout_events',return_value=({},{})):
        snapshot=reuse_candidate_snapshot(candidates,m,output)
        expected=[(at,j,c.sym,c.tf) for at in sorted(snapshot[0]) for j,c in enumerate(snapshot[0][at]) if c.mode=='impulse_speed']
        if expected!=list(zip(z['clock'].tolist(),z['index'].tolist(),z['symbol'].tolist(),z['tf'].tolist())):raise ValueError('dataset population differs')
        stream=screen(snapshot,decisions);enabled,delta=replacement_options(config)
        print(json.dumps({'phase':'SIMULATE','arm':'joint','candidates':stream[2]}),flush=True)
        trades,stat=await rb.simulate_portfolio(symbols,['15m','1h'],cache,c15,c4,context,max_open_positions=10,
            enable_replacement=enabled,replace_min_delta=delta,variant='score_replace_cluster',
            top_gainer_score_min=float(config.TOP_GAINER_SCORE_GATE_MIN_SCORE),candidate_snapshot=stream)
        finalize_at_boundary(trades,cache,end);publish(output/'trades_joint.json',[asdict(t) for t in trades])
    trades_by_arm={'joint':trades}
    for arm in ('control','regressor'):trades_by_arm[arm]=[rb.ReplayTrade(**t) for t in json.loads((output/f'trades_{arm}.json').read_bytes())]
    series={s:closed_price_series(c15[s][0],bar_ms=BAR,start_ms=start,end_ms=end) for s in symbols};grid=list(range(start,end+1,BAR))
    benchmark=_benchmark_result(series['BTCUSDT'],initial_capital=10000,fee_bps=fee,slippage_bps=slip)
    accounts={};missions={};test_missions={};attribution={};stress={}
    previous_result=json.loads((output/'parent_result.json').read_bytes())
    for arm in ('control','regressor','joint'):
        trades=trades_by_arm[arm];a=cash_ledger(trades,series,grid,fee,slip)
        ref=_simulate_account(trades,price_series_by_symbol=series,valuation_timestamps=grid,capacity=10,
            initial_capital=10000,fee_bps=fee,slippage_bps=slip)
        if ref.violations or ref.fully_valued_points!=len(grid):raise ValueError('cash/grid violation')
        np.testing.assert_allclose(a['curve'],ref.equity_curve,rtol=1e-10,atol=1e-7)
        np.testing.assert_allclose([a['totals']['fees'],a['totals']['slippage']],[ref.fees_quote,ref.slippage_quote],rtol=1e-10,atol=1e-7)
        ledger=a.pop('ledger');publish(output/f'ledger_{arm}.json',ledger);curve=dict(a['curve'])
        if arm!='joint':np.testing.assert_allclose(a['curve'],previous_result['accounts']['control' if arm=='control' else 'catboost']['curve'],rtol=0,atol=1e-9)
        replacements=stat.replacements_total if arm=='joint' else previous_result['accounts']['control' if arm=='control' else 'catboost']['replacements']
        accounts[arm]={**a,'test_return_pct':100*(curve[end]/curve[cuts[1]]-1),'trades':len(trades),
            'alpha_pp':a['net_return_pct']-benchmark['net_return_after_costs_pct'],'replacements':replacements}
        missions[arm]=objective(trades,cache,symbols,start,end)
        test_missions[arm]=objective([t for t in trades if t.entry_ts>=cuts[1]],cache,symbols,cuts[1],end)
        attribution[arm]={'mode':grouped_attribution(ledger,'mode')}
        stress[arm]={k:v for k,v in cash_ledger(trades,series,grid,2*fee,2*slip).items() if k not in ('curve','ledger')}
    ci=single_interval(accounts['control']['curve'],accounts['joint']['curve'],cuts[1],end)
    verdict=gate(accounts['control'],accounts['joint'],(missions['control'],missions['joint']),
        (test_missions['control'],test_missions['joint']),ci['interval'],ci['days'])
    verify()
    for sym,h in hashes.items():
        if sha(features/f'{sym}.npz')!=h:raise ValueError('feature drift')
    result={'status':'COMPLETED_RETROSPECTIVE_PREQUENTIAL','runtime_eligible':False,'achievement_claimed':False,
        'start_ms':start,'end_ms':end,'days':(end-start)/DAY,'split_boundaries_ms':cuts,
        'population_complete':len(symbols),'population_requested':len(m['requested_symbols']),
        'costs':{'fee_bps':fee,'slippage_bps':slip},'accounts':accounts,'benchmark':benchmark,'missions':missions,
        'test_missions':test_missions,'attribution':attribution,'cost_stress':stress,'folds':folds,
        'comparisons':{'joint':{**verdict,**ci}},'forecast_metrics':diagnostics(z,cuts[1],fee,slip),
        'filter_audit':{'input':snapshot[2],'impulse_candidates':len(z['clock']),'scored':int((z['fold']>=0).sum()),'retained':stream[2]},
        'limitations':previous_result['limitations']+['regressor is unchanged previously rejected reference, not another new policy test',
            'conditional magnitudes are not an independently predicted volatility process or future price path']}
    publish(output/'result.json',result);publish(output/'receipt.json',{p.name:sha(p) for p in output.iterdir() if p.is_file()})
    print(json.dumps({'accounts':{n:{k:a[k] for k in ('net_return_pct','test_return_pct','trades')} for n,a in accounts.items()},'verdict':verdict},indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('parent','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();asyncio.run(run(a.parent,a.output))
