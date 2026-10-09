"""Maximum-history mission-aligned prequential training and full portfolio replay."""
from __future__ import annotations
import argparse,asyncio,json,pickle,shutil,time
from pathlib import Path
from dataclasses import asdict,replace
from unittest.mock import patch
import numpy as np
from catboost import CatBoostClassifier
import config,replay_backtest as rb
from mission_contract import candidate_target,local_day,window,CONTRACT
from mission_learning import *
from compare_price_volatility_bot import build_features,parallel_candidates,replacement_options,read_feature_pack
from closed_grid_policy_replay import event_clock
from mission_rule_policy import rules_only
from audit_negative_day_rebound import finalize_at_boundary
from historical_signal_evaluation import sha
from capacity_catboost import split_boundaries
from run_leader_mission_reaudit import load_market
import leader_mission_metrics as lm
from leader_trend_continuation import paired_accompaniment

ROOT=Path(__file__).resolve().parent.parent
SOURCES=('run_mission_learning.py','mission_learning.py','mission_contract.py','mission_rule_policy.py','compare_price_volatility_bot.py','closed_grid_policy_replay.py',
    'replay_backtest.py','config.py','strategy.py','monitor.py','indicators.py','research_rocket_capture.py','capacity_catboost.py',
    'impulse_entry_catboost.py','audit_negative_day_rebound.py','leader_mission_metrics.py','leader_trend_continuation.py','run_leader_mission_reaudit.py')

def write(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False),encoding='utf-8')
def key(c):return c.ts_ms,c.sym,c.tf,c.mode,c.price


def fit_heads(dataset,start,end,output,names):
    """Refit using strictly mature labels. Validation never chooses acceptance thresholds."""
    clocks=dataset['clock'];available=dataset['available'];x=dataset['x'];ys=dataset['y']
    predictions=np.full(ys.shape,np.nan);folds=[]
    for at in range(start+30*DAY,end,30*DAY):
        stop=min(at+30*DAY,end);issued=(clocks>=at)&(clocks<stop);record=dict(fit_at=at,stop=stop,heads={})
        for h,name in enumerate(names):
            valid=np.isfinite(x).all(axis=1)&np.isfinite(ys[:,h]);train,val,cut=fit_cohorts(clocks,available[:,h],valid,at,start)
            r=dict(train_n=int(train.sum()),validation_n=int(val.sum()),train_positive_n=int(ys[train,h].sum()),validation_positive_n=int(ys[val,h].sum()),validation_boundary=cut)
            if train.sum()<500 or val.sum()<100 or len(np.unique(ys[train,h]))<2 or len(np.unique(ys[val,h]))<2:r['state']='UNTRAINED';record['heads'][name]=r;continue
            model=CatBoostClassifier(**PARAMETERS);model.fit(x[train],ys[train,h],eval_set=(x[val],ys[val,h]),early_stopping_rounds=40,use_best_model=True)
            scoring=issued&np.isfinite(x).all(axis=1);predictions[scoring,h]=model.predict_proba(x[scoring])[:,1]
            p=output/f'{name}_{at}.cbm';model.save_model(str(p));r.update(state='FROZEN',model=p.name,model_sha256=sha(p),trees=model.tree_count_,train_max_available=int(available[train,h].max()),validation_max_available=int(available[val,h].max()),scored_n=int(scoring.sum()))
            record['heads'][name]=r
        folds.append(record);print('FIT '+json.dumps(record),flush=True)
    write(output/('folds_'+names[0]+'.json'),folds);np.savez_compressed(output/('dataset_'+names[0]+'.npz'),**dataset,prediction=predictions)
    diagnostics={}
    for h,name in enumerate(names):
        known=np.isfinite(predictions[:,h])&np.isfinite(ys[:,h]);p=predictions[known,h];y=ys[known,h]
        diagnostics[name]=dict(issued_n=int(np.isfinite(predictions[:,h]).sum()),known_n=int(known.sum()),positive_n=int(y.sum()),base_rate=float(y.mean()) if len(y) else None,brier=float(np.mean((p-y)**2)) if len(y) else None,scope='OOS proxy, not portfolio achievement')
    write(output/('diagnostics_'+names[0]+'.json'),diagnostics);return predictions,folds


async def run(market,labels_path,output,workers,prepared=None):
    for p in (market,labels_path):
        if not p.resolve().is_relative_to(ROOT/'.runtime'):raise ValueError('trusted workspace runtime only')
    manifest=json.loads((market/'manifest.json').read_bytes());labels=json.loads(labels_path.read_bytes());start,end=manifest['start_ms'],manifest['end_ms'];boundary=split_boundaries(start,end)[1]
    output.mkdir(parents=True,exist_ok=False);(output/'source_snapshot').mkdir();sources={}
    for n in SOURCES:p=Path(__file__).with_name(n);shutil.copy2(p,output/'source_snapshot'/n);sources[n]=sha(p)
    spec=ROOT/'docs/specs/mission-aligned-learning-cycle.md';shutil.copy2(spec,output/'registered_spec.md')
    write(output/'registration.json',dict(contract=CONTRACT,parameters=PARAMETERS,sources=sources,spec_sha256=sha(spec),market=str(market.resolve()),market_manifest_sha256=sha(market/'manifest.json'),labels_sha256=sha(labels_path),labels=str(labels_path.resolve()),registered_at=time.time(),start_ms=start,end_ms=end,test_boundary=boundary,runtime_eligible=False,exposed_retrospective=True,ranking_formula='original score + 8*(early_probability-.5); untrained unchanged'))
    for n,h in manifest['input_hashes'].items():
        if sha(market/'market'/n)!=h:raise ValueError('market hash drift')
    if prepared:
        if not prepared.resolve().is_relative_to(ROOT/'.runtime'):raise ValueError('trusted prepared runtime only')
        old= json.loads((prepared/'registration.json').read_bytes())
        if (old['start_ms'],old['end_ms'],old['market_manifest_sha256'],old['labels_sha256'])!=(start,end,sha(market/'manifest.json'),sha(labels_path)):raise ValueError('prepared inputs differ')
        for n,h in old['sources'].items():
            if n!='run_mission_learning.py' and sha(Path(__file__).with_name(n))!=h:raise ValueError('prepared feature/kernel source differs '+n)
        cache={};fh=json.loads((prepared/'features/hashes.json').read_bytes())
        if set(fh)!=set(manifest['eligible_symbols']):raise ValueError('prepared population differs')
        for s,h in fh.items():
            p=prepared/'features'/(s+'.npz')
            if sha(p)!=h:raise ValueError('prepared features drift')
            for tf,pack in read_feature_pack(p).items():cache[s,tf]=pack
        with (prepared/'candidate_snapshot.pkl').open('rb') as f:snapshot=pickle.load(f)
        if snapshot[2]!=sum(map(len,snapshot[0].values())) or any(c.ts_ms!=at for at,rows in snapshot[0].items() for c in rows):raise ValueError('prepared candidate identity mismatch')
        receipt=dict(parent=str(prepared.resolve()),registration_sha256=sha(prepared/'registration.json'),abort_sha256=sha(prepared/'ABORTED_RESOURCE_GROWTH.json'),feature_hashes=fh,candidate_sha256=sha(prepared/'candidate_snapshot.pkl'),reused_files={})
        for p in prepared.iterdir():
            if p.is_file() and (p.suffix in ('.cbm','.npz') or p.name.startswith(('folds_','diagnostics_'))):
                shutil.copy2(p,output/p.name);receipt['reused_files'][p.name]=sha(p)
        write(output/'prepared_parent_receipt.json',receipt);print('REUSED IMMUTABLE PREPARED INPUTS '+str(snapshot[2]),flush=True)
    else:
        print('BUILD FEATURES',len(manifest['eligible_symbols']),flush=True)
        cache,_=build_features(manifest['eligible_symbols'],market/'market',output/'features',workers)
    symbols=manifest['eligible_symbols'];c15={s:cache[s,'15m'] for s in symbols};c4={s:cache[s,'4h'] for s in symbols};context=rb._build_bull_day_context(cache['BTCUSDT','1h'][0])
    if not prepared:
        raw,times,_=parallel_candidates(symbols,cache,output/'features',context,output,workers);snapshot=event_clock(raw,times,start,end)
    with (output/'candidate_snapshot.pkl').open('wb') as f:pickle.dump(snapshot,f)
    print('ENTRY DATA',snapshot[2],flush=True)
    keys=[];xs=[];ys=[];available=[];clocks=[]
    for at in sorted(snapshot[0]):
        for c in snapshot[0][at]:
            if prepared:
                keys.append(key(c));continue
            x=entry_features(cache[c.sym,'15m'][0],at,c.tf,c.top_gainer_score,c.mode)
            label=labels.get(local_day(at));t=candidate_target(c.sym,at,c.price,label) if label and not label['coverage']['missing'] else None
            keys.append(key(c));clocks.append(at);xs.append(np.full(18,np.nan) if x is None else x)
            ys.append([t['leader'],t['early_leader']] if t else [np.nan,np.nan]);available.append([t['available_at']]*2 if t else [end+DAY]*2)
    if prepared:
        with np.load(output/'dataset_discovery.npz',allow_pickle=False) as saved:
            dataset={k:saved[k] for k in ('clock','available','x','y')};predictions=saved['prediction']
        if not np.array_equal(dataset['clock'],np.array([k[0] for k in keys])) or len(keys)!=len(predictions):raise ValueError('prepared prediction alignment differs')
        for i in np.linspace(0,len(keys)-1,33,dtype=int):
            at,s,tf,mode,p=keys[i];c=next(c for c in snapshot[0][at] if key(c)==keys[i]);x=entry_features(cache[s,'15m'][0],at,tf,c.top_gainer_score,mode)
            np.testing.assert_allclose(dataset['x'][i],np.full(18,np.nan) if x is None else x,rtol=0,atol=0,equal_nan=True)
        folds=json.loads((output/'folds_discovery.json').read_bytes())
    else:
        dataset=dict(clock=np.asarray(clocks,np.int64),available=np.asarray(available,np.int64),x=np.asarray(xs),y=np.asarray(ys))
        predictions,folds=fit_heads(dataset,start,end,output,('discovery','early'))
    scores={k:p[1] for k,p in zip(keys,predictions)}
    del xs,ys,available,dataset
    progress=rb._update_trade_progress;skip=rb._record_cooldown_skip;states={'continuation':[],'reentry':[]};seen=set();trades_by_arm={};stats={}
    def observe(trade,data,feat,idx,**kwargs):
        clock=kwargs.get('ts_ms');ident=(trade.sym,trade.tf,trade.entry_ts,clock)
        if clock is not None and start<=clock<end and ident not in seen:
            seen.add(ident);x=state_features(cache[trade.sym,'15m'][0],clock,trade.tf,trade.top_gainer_score,trade.mode,trade.entry_ts,trade.entry_price,max(trade.entry_price,trade.max_price_since_entry))
            if x is not None:states['continuation'].append((clock,trade.sym,trade.tf,trade.mode,x))
        return progress(trade,data,feat,idx,**kwargs)
    def observe_skip(trade,candidate,cache_arg,stat):
        if trade:
            clock=candidate.ts_ms;x=state_features(cache[candidate.sym,'15m'][0],clock,candidate.tf,candidate.top_gainer_score,candidate.mode,trade.entry_ts,trade.entry_price,max(trade.entry_price,trade.max_price_since_entry))
            if x is not None:states['reentry'].append((clock,candidate.sym,candidate.tf,candidate.mode,x))
        return skip(trade,candidate,cache_arg,stat)
    enabled,delta=replacement_options(config)
    with rules_only():
        for name in ('control','early_ranker'):
            stream={}
            for at,rows in snapshot[0].items():
                stream[at]=[replace(c,top_gainer_score=c.top_gainer_score+8*(scores[key(c)]-.5)) if name=='early_ranker' and np.isfinite(scores.get(key(c),np.nan)) else replace(c) for c in rows]
            print('SIMULATE '+name,flush=True)
            with patch.object(rb,'_update_trade_progress',new=observe if name=='control' else progress),patch.object(rb,'_record_cooldown_skip',new=observe_skip if name=='control' else skip):
                trades,stat=await rb.simulate_portfolio(symbols,['15m','1h'],cache,c15,c4,context,max_open_positions=10,enable_replacement=enabled,replace_min_delta=delta,variant='score_replace_cluster',top_gainer_score_min=float(config.TOP_GAINER_SCORE_GATE_MIN_SCORE),candidate_snapshot=(stream,snapshot[1],snapshot[2]))
            finalize_at_boundary(trades,cache,end);rows=[asdict(t) for t in trades];write(output/f'trades_{name}.json',rows);trades_by_arm[name]=rows;stats[name]=asdict(stat)
            print('TRADES '+name+' '+str(len(rows)),flush=True)
    for name,horizon in (('continuation',3600000),('reentry',14400000)):
        entries=states[name];x=np.asarray([r[4] for r in entries]);ys=[];available=[]
        for clock,s,*_ in entries:
            y=continuation_target(cache[s,'15m'][0],clock,horizon);ys.append([y['target']] if y else [np.nan]);available.append([y['available_at']] if y else [end+horizon])
        if entries:
            fit_heads(dict(clock=np.array([r[0] for r in entries],np.int64),available=np.asarray(available,np.int64),x=x,y=np.asarray(ys)),start,end,output,(name,))
        write(output/('states_'+name+'.json'),[dict(clock=r[0],symbol=r[1],tf=r[2],mode=r[3]) for r in entries])
    market_data=load_market(market,manifest);ind={s:lm.indicators(d) for s,d in market_data.items()};arms={};daily={};episodes={}
    for name,rows in trades_by_arm.items():
        full,fd,entries=mission_metrics(rows,labels);test,td,_=mission_metrics(rows,labels,boundary);by_sym={s:[t for t in rows if t['sym']==s] for s in symbols}
        ep=[lm.episode(r,market_data[r['symbol']],ind[r['symbol']],by_sym[r['symbol']]) for r in entries];episodes[name]=ep
        arms[name]=dict(full=full,test=test,exits=lm.exit_summary([r for r in ep if window(r['day'])[0]>=boundary]));daily[name]=td
        write(output/f'episodes_{name}.json',ep)
    ci=paired_entry_interval(daily['control'],daily['early_ranker']);base,target=(arms[n]['test'] for n in ('control','early_ranker'))
    exit_cmp=paired_accompaniment(*([e for e in episodes[n] if window(e['day'])[0]>=boundary] for n in ('control','early_ranker')))
    checks=dict(early_count_gain=target['early']>base['early'],early_ci_positive=ci is not None and ci['early_count_delta95'][0]>0,coverage_not_worse=target['captured']>=base['captured'],precision_not_worse=target['unique_precision_n']*base['unique_precision_N']>=base['unique_precision_n']*target['unique_precision_N'] if min(base['unique_precision_N'],target['unique_precision_N'])>0 else False,
        companion_exit_pass=False,population_complete=len(symbols)==len(manifest['requested_symbols']),historical_PIT_certified=False)
    readiness=release_readiness(dict(retrospective_mission_pass=all(checks.values()),rollback_available=True,source_model_hashes=True))
    for n,h in sources.items():
        if sha(Path(__file__).with_name(n))!=h:raise ValueError('source drift '+n)
    if sha(labels_path)!=json.loads((output/'registration.json').read_bytes())['labels_sha256']:raise ValueError('labels drift')
    write(output/'result.json',dict(status='COMPLETED_EXPOSED_RETROSPECTIVE',runtime_eligible=False,achievement_claimed=False,days=(end-start)/DAY,population_complete=len(symbols),population_requested=len(manifest['requested_symbols']),arms=arms,paired_entry_interval=ci,paired_exit_comparison=exit_cmp,checks=checks,readiness=readiness,stats=stats,head_state_counts={n:len(r) for n,r in states.items()},limitations=['Current-selected universe; historical PIT uncertified','Previously exposed history, prequential predictions not sealed TEST','Ideal closed-bar fills, actual acceptance unknown','Continuation/reentry proxies are action-neutral; no exit gain claimed','Companion exit gate not yet independently certified']))
    write(output/'daily_metrics.json',daily);write(output/'receipt.json',{p.name:sha(p) for p in output.iterdir() if p.is_file()});print('COMPLETE '+json.dumps(dict(arms=arms,checks=checks,readiness=readiness)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--market',type=Path,required=True);p.add_argument('--labels',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--workers',type=int,default=4);p.add_argument('--prepared-parent',type=Path);a=p.parse_args();asyncio.run(run(a.market,a.labels,a.output,a.workers,a.prepared_parent))
