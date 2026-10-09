"""Fixed-model, public-data forward observations. Never places orders or enables models."""
from __future__ import annotations
import argparse,asyncio,json,shutil,time,hashlib
from pathlib import Path
from datetime import datetime,timezone
from concurrent.futures import ThreadPoolExecutor,as_completed
import numpy as np
from catboost import CatBoostClassifier
import replay_backtest as rb
from mission_learning import entry_features,release_readiness,BAR
from mission_contract import CONTRACT,local_day,window,daily_label,target_symbol
from fetch_mission_history import request,digest,daily_request_range
from mission_rule_policy import rules_only

ROOT=Path(__file__).resolve().parent.parent


def validate_issue(feature_clock,received_ms,issued_ms,model_fit_at):
    if not model_fit_at<feature_clock<=received_ms<=issued_ms:raise ValueError('invalid model/feature/receipt/issue clock order')
    if issued_ms-feature_clock>5*60000:raise ValueError('stale forecast origin')


def hourly_day_row(day,rows):
    """UTC hourly aggregation handles23/25-hour local days without a fixed offset."""
    lo,hi=window(day)
    if [r[0] for r in rows]!=list(range(lo,hi,3600000)) or any(r[6]!=r[0]+3599999 for r in rows):raise ValueError('incomplete hourly day')
    if not all(np.isfinite([float(r[i]) for i in (1,2,3,4,5,7)]).all() for r in rows):raise ValueError('invalid hourly day')
    return [lo,float(rows[0][1]),max(float(r[2]) for r in rows),min(float(r[3]) for r in rows),float(rows[-1][4]),sum(float(r[5]) for r in rows),hi-1,sum(float(r[7]) for r in rows)]


def initialize(experiment,output):
    result=json.loads((experiment/'result.json').read_bytes());reg=json.loads((experiment/'registration.json').read_bytes())
    if result['status']!='COMPLETED_EXPOSED_RETROSPECTIVE' or result['runtime_eligible']:raise ValueError('invalid research artifact')
    receipt=json.loads((experiment/'receipt.json').read_bytes())
    for n,h in receipt.items():
        if Path(n).name!=n or digest(experiment/n)!=h:raise ValueError('research artifact drift')
    for n,h in reg['sources'].items():
        if digest(ROOT/'files'/n)!=h:raise ValueError('kernel source changed '+n)
    output.mkdir(parents=True,exist_ok=False);(output/'models').mkdir();(output/'cycles').mkdir()
    models={}
    for name in ('discovery','early','continuation','reentry'):
        foldpath=experiment/('folds_'+('discovery' if name=='early' else name)+'.json')
        if not foldpath.exists():models[name]=dict(state='UNTRAINED');continue
        choices=[(f['fit_at'],f['heads'][name]) for f in json.loads(foldpath.read_bytes()) if f['heads'][name]['state']=='FROZEN']
        if not choices:models[name]=dict(state='UNTRAINED');continue
        at,r=choices[-1];p=experiment/r['model']
        if digest(p)!=r['model_sha256']:raise ValueError('model drift')
        shutil.copy2(p,output/'models'/p.name);models[name]=dict(state='FIXED',fit_at=at,file=p.name,sha256=r['model_sha256'])
    if any(models[n]['state']!='FIXED' for n in ('discovery','early')):raise ValueError('discovery and early models required')
    with np.load(experiment/'dataset_discovery.npz',allow_pickle=False) as z:
        fit=min(models[n]['fit_at'] for n in ('discovery','early'));valid=(z['clock']<fit)&(z['available'][:,0]<fit)&np.isfinite(z['x']).all(axis=1)
        bounds=np.quantile(z['x'][valid],[.01,.99],axis=0).tolist()
    source_hashes={**reg['sources'],**{n:digest(Path(__file__).with_name(n)) for n in ('mission_forward_shadow.py','fetch_mission_history.py')}}
    write(output/'registration.json',dict(contract=CONTRACT,started_at=datetime.now(timezone.utc).isoformat(),experiment=str(experiment.resolve()),experiment_result_sha256=digest(experiment/'result.json'),source_hashes=source_hashes,models=models,runtime_eligible=False,training_feature_bounds_01_99=bounds,
        scope='observational actual rule candidates; forecasts cannot change BUY/SELL; model and parameters frozen',fresh_start_ms=int(time.time()*1000),acceptance_clocks_verified=False))


def write(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False),encoding='utf-8')


async def collect(output):
    reg=json.loads((output/'registration.json').read_bytes())
    for n,h in reg['source_hashes'].items():
        if digest(ROOT/'files'/n)!=h:raise ValueError('frozen source drift '+n)
    cycle=output/'cycles'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ');cycle.mkdir();(cycle/'raw').mkdir()
    raw,meta=request('time');(cycle/'server_time.json').write_bytes(raw);server=json.loads(raw)['serverTime'];origin=server//BAR*BAR
    raw,universe_meta=request('exchangeInfo',dict(permissions='SPOT'));(cycle/'exchangeInfo.json').write_bytes(raw)
    watchraw=(ROOT/'files/watchlist.json').read_bytes();(cycle/'watchlist.json').write_bytes(watchraw);watch=json.loads(watchraw)
    tradable={r['symbol'] for r in json.loads(raw)['symbols'] if r['status']=='TRADING' and r['quoteAsset']=='USDT'}
    requested=sorted(set(watch)&tradable);excluded=sorted(set(watch)-tradable);cache={};receipts={}
    def one(s,tf):
        try:
            if not s.isalnum():raise ValueError('invalid market symbol')
            raw,received=request('klines',dict(symbol=s,interval=tf,endTime=server,limit=256));rows=json.loads(raw);p=cycle/'raw'/(s+'_'+tf+'.json');p.write_bytes(raw)
            step=BAR if tf=='15m' else 4*BAR
            if any(len(r)!=12 or r[6]!=r[0]+step-1 for r in rows):raise ValueError('unexpected schema')
            # Builder expects one final incomplete sentinel; request includes current
            # bar but feature extraction only reads exact closed prefixes.
            data=np.array([(r[0],*[float(r[i]) for i in range(1,6)]) for r in rows],dtype=rb._KLINE_DTYPE)
            return (s,tf),dict(state='RECEIVED',sha256=digest(p),**received),data
        except Exception as e:return (s,tf),dict(state='FAILED',error=repr(e)),None
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures=[pool.submit(one,s,tf) for s in requested for tf in ('15m','1h')]
        for f in as_completed(futures):
            identity,r,d=f.result();receipts['_'.join(identity)]=r
            if d is not None:cache[identity]=(d,rb.compute_features(d['o'],d['h'],d['l'],d['c'],d['v']))
    models={}
    for n in ('discovery','early'):
        r=reg['models'][n]
        if r['state']=='FIXED':
            p=output/'models'/r['file']
            if digest(p)!=r['sha256']:raise ValueError('fixed model drift')
            model=CatBoostClassifier();model.load_model(str(p));models[n]=model
    complete=[s for s in requested if (s,'15m') in cache and (s,'1h') in cache]
    c15={s:cache[s,'15m'] for s in complete};c4={s:(rb._aggregate_1h_to_4h(cache[s,'1h'][0]),{}) for s in complete}
    for s,(d,_) in c4.items():c4[s]=(d,rb.compute_features(d['o'],d['h'],d['l'],d['c'],d['v']))
    context=rb._build_bull_day_context(cache['BTCUSDT','1h'][0]) if ('BTCUSDT','1h') in cache else None
    forecasts=[];late=[]
    with rules_only():
        for s in complete:
            for tf in ('15m','1h'):
                d,feat=cache[s,tf]
                candidates=await rb._build_candidates_for_symbol(s,tf,d,feat,c15,c4,context,variant='score_replace_cluster',include_trend_start=False)
                for c in candidates:
                    if c.ts_ms!=origin:continue
                    x=entry_features(c15[s][0],origin,tf,c.top_gainer_score,c.mode)
                    if x is None:continue
                    issued=int(time.time()*1000);received=max(int(datetime.fromisoformat(receipts[s+'_'+t]['received_utc']).timestamp()*1000) for t in ('15m','1h'))
                    try:
                        for n in models:validate_issue(origin,received,issued,reg['models'][n]['fit_at'])
                    except ValueError as e:late.append(dict(symbol=s,tf=tf,reason=str(e)));continue
                    bounds=np.asarray(reg['training_feature_bounds_01_99']);outside=(x<bounds[0])|(x>bounds[1])
                    forecasts.append(dict(symbol=s,tf=tf,mode=c.mode,feature_clock=origin,received_ms=received,issued_ms=issued,price=c.price,score=c.top_gainer_score,feature_outside_training_n=int(outside.sum()),feature_count=len(x),feature_drift_warning=int(outside.sum())>=4,
                        probabilities={n:float(model.predict_proba(x.reshape(1,-1))[0,1]) for n,model in models.items()},actual_acceptance=None,model_hashes={n:reg['models'][n]['sha256'] for n in models}))
    result=dict(state='OBSERVED_SHADOW' if forecasts else 'NO_FRESH_CANDIDATES',origin=origin,day=local_day(origin),requested=len(requested),complete=len(complete),excluded_nontradable=excluded,failed_pairs=[n for n,r in receipts.items() if r['state']=='FAILED'],stale_candidates=late,forecast_n=len(forecasts),universe_received=universe_meta,
        asof_watchlist_sha256=digest(cycle/'watchlist.json'),asof_universe_sha256=digest(cycle/'exchangeInfo.json'),runtime_eligible=False,actual_acceptance_verified=False,full_day_eligible=reg['fresh_start_ms']<=window(local_day(origin))[0])
    write(cycle/'receipts.json',receipts);write(cycle/'forecasts.json',forecasts);write(cycle/'result.json',result)
    write(output/'status.json',dict(updated_at=datetime.now(timezone.utc).isoformat(),last_cycle=str(cycle.resolve()),collection=result,release=release_readiness(dict(source_model_hashes=True,rollback_available=True))))
    print(json.dumps(result),flush=True)


def mature_days(output):
    """Score future outcomes separately; classifier forecasts are NOT accepted BUYs."""
    from datetime import timedelta
    now=int(time.time()*1000);groups={};maturity=output/'matured';maturity.mkdir(exist_ok=True)
    for cycle in sorted((output/'cycles').iterdir()):
        p=cycle/'result.json'
        if not p.exists():continue
        r=json.loads(p.read_bytes());day=r['day']
        if window(day)[1]>=now or not r['full_day_eligible']:continue
        groups.setdefault(day,[]).append(cycle)
    for day,cycles in sorted(groups.items()):
        if (maturity/day).exists():continue
        first=cycles[0];out=maturity/day;out.mkdir();(out/'raw').mkdir()
        results=[json.loads((c/'result.json').read_bytes()) for c in cycles];lo,hi=window(day)
        full_grid=set(range(lo,hi,BAR));seen={r['origin'] for r in results}
        complete_grid=full_grid<=seen and len({r['asof_watchlist_sha256'] for r in results})==1 and all(r['complete']==r['requested'] and not r['failed_pairs'] and not r['stale_candidates'] for r in results)
        universe=json.loads((first/'exchangeInfo.json').read_bytes());watch=json.loads((first/'watchlist.json').read_bytes())
        final_raw,final_meta=request('exchangeInfo',dict(permissions='SPOT'));(out/'label_exchangeInfo.json').write_bytes(final_raw)
        all_registries=[json.loads((c/'exchangeInfo.json').read_bytes()) for c in cycles]+[json.loads(final_raw)]
        symbols=sorted({r['symbol'] for u in all_registries for r in u['symbols'] if r.get('quoteAsset')=='USDT' and target_symbol(r['symbol'])});bars={};receipts={}
        active={r['symbol'] for u in all_registries[:-1] for r in u['symbols'] if r['status']=='TRADING'}
        def one(s):
            try:
                if not s.isalnum():raise ValueError('invalid label symbol')
                parameters=daily_request_range(lo,hi) if hi-lo==86400000 else dict(interval='1h',startTime=lo,endTime=hi-1,limit=30)
                raw,meta=request('klines',dict(symbol=s,**parameters));p=out/'raw'/(s+'.json');p.write_bytes(raw)
                rows=json.loads(raw)
                if not rows:return s,None,dict(state='NO_BARS',sha256=digest(p),**meta)
                if parameters['interval']=='1d':
                    if len(rows)!=1 or rows[0][0]!=lo or rows[0][6]!=hi-1:raise ValueError('partial native label day')
                    row=rows[0]
                else:row=hourly_day_row(day,rows)
                return s,row,dict(state='COMPLETE',sha256=digest(p),**meta)
            except Exception as e:return s,None,dict(state='UNKNOWN',error=repr(e))
        with ThreadPoolExecutor(max_workers=4) as pool:
            for s,row,r in pool.map(one,symbols):
                receipts[s]=r
                if row is not None:bars[s]=row
        missing=[s for s,r in receipts.items() if r['state']=='UNKNOWN' or (r['state']=='NO_BARS' and s in active)];label=daily_label(day,bars,watch,now,dict(missing=missing,historical_PIT_certified=True,scope='native local-day targets, union of asof and label-only final registries',final_registry_sha256=digest(out/'label_exchangeInfo.json'),final_registry_received=final_meta))
        forecasts=[r for c in cycles for r in json.loads((c/'forecasts.json').read_bytes()) if digest(c/'watchlist.json')==digest(first/'watchlist.json')];metrics={}
        from mission_contract import candidate_target
        for head,target in (('discovery','leader'),('early','early_leader')):
            pairs=[]
            for f in forecasts:
                y=candidate_target(f['symbol'],f['feature_clock'],f['price'],label)
                if y is not None and head in f['probabilities']:pairs.append((f['probabilities'][head],y[target]))
            metrics[head]=dict(known_n=len(pairs),positive_n=sum(y for _,y in pairs),brier=float(np.mean([(p-y)**2 for p,y in pairs])) if pairs else None,scope='forecast proxy only; actual BUY effect unverified')
        write(out/'receipts.json',receipts);write(out/'label.json',label)
        write(out/'result.json',dict(state='COMPLETE_FORECAST_COHORT' if complete_grid and not missing else 'PARTIAL_OR_UNKNOWN',day=day,forecast_n=len(forecasts),complete_origin_grid=complete_grid,missing_origin_n=len(full_grid-seen),missing_symbols=missing,leader_pairs=len(label['leaders']),metrics=metrics,actual_acceptance_verified=False,mission_improvement_claimed=False,runtime_eligible=False))


async def run(args):
    if not 1<=args.max_days<=35:raise ValueError('shadow duration must be1..35days')
    if args.initialize:initialize(args.experiment,args.output)
    while True:
        if (args.output/'STOP').exists():return
        registered=json.loads((args.output/'registration.json').read_bytes())
        if int(time.time()*1000)-registered['fresh_start_ms']>=args.max_days*86400000:
            write(args.output/'status.json',dict(state='BOUNDED_SHADOW_ENDED',runtime_eligible=False));return
        try:await collect(args.output)
        except Exception as e:
            write(args.output/'status.json',dict(state='COLLECTION_FAILED',error=repr(e),updated_at=datetime.now(timezone.utc).isoformat(),runtime_eligible=False));print('FAILED '+repr(e),flush=True)
            if not args.loop:raise
        if args.score_mature:mature_days(args.output)
        if not args.loop:return
        await asyncio.sleep(max(1,(BAR-int(time.time()*1000)%BAR)/1000+2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--experiment',type=Path);p.add_argument('--output',type=Path,required=True);p.add_argument('--initialize',action='store_true');p.add_argument('--loop',action='store_true');p.add_argument('--score-mature',action='store_true');p.add_argument('--max-days',type=int,default=35);a=p.parse_args();asyncio.run(run(a))
