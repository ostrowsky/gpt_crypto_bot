"""Retrospective causal overlays on maximum closed-archive rule-only replay.

No production imports of this module, no model/config/position writes.
"""
from __future__ import annotations
import argparse, asyncio, hashlib, json, math, pickle, re, shutil, time, urllib.request, urllib.parse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

STEP=900_000
DAY=86_400_000
HORIZON=4
ARMS=('baseline','volatility','direction','combined','ewma_control')
VAR_FLOOR=1e-12
TARGET_SIGMA=.01

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def market_frame(data):
    """Origin is candle CLOSE; every feature ends at that origin."""
    d=pd.DataFrame({k:data[k] for k in ('t','o','h','l','c','v')})
    if not np.array_equal(np.diff(d.t),np.full(len(d)-1,STEP)):
        raise ValueError('Noncontiguous market grid')
    log=np.log(d.c);r=log.diff();sq=r*r
    result=pd.DataFrame({'origin':d.t+STEP,'price':d.c})
    for k in (1,4,16,96):result['r'+str(k)]=log.diff(k)
    for k in (4,16,96):result['rv'+str(k)]=sq.rolling(k).mean()*HORIZON
    for k in (4,16,96):result['v'+str(k)]=np.log1p(d.v)-np.log1p(d.v.rolling(k).mean())
    result['range']=(d.h-d.l)/d.c
    result['body']=(d.c-d.o)/d.o
    angle=2*np.pi*((result.origin//60_000)%1440)/1440
    result['sin']=np.sin(angle);result['cos']=np.cos(angle)
    result['ewma']=sq.ewm(alpha=.06,adjust=False).mean()*HORIZON
    result['return_target']=log.shift(-HORIZON)-log
    result['variance_target']=sum(sq.shift(-k) for k in range(1,HORIZON+1))
    result['label_close']=result.origin+HORIZON*STEP
    return result

DIR_FEATURES=['r1','r4','r16','r96','rv4','rv16','rv96','v4','v16','v96','range','body','sin','cos']
VOL_FEATURES=['rv4','rv16','rv96']

def fit_pair(train):
    if train.empty:raise ValueError('No prior training labels')
    x=train[DIR_FEATURES].to_numpy();scaler=StandardScaler().fit(x)
    direction=Ridge(alpha=10).fit(scaler.transform(x),train.return_target)
    v=np.log(np.maximum(train[VOL_FEATURES].to_numpy(),VAR_FLOOR))
    vs=StandardScaler().fit(v)
    target=np.log(np.maximum(train.variance_target.to_numpy(),VAR_FLOOR))
    har=Ridge(alpha=10).fit(vs.transform(v),target)
    # TRAIN-only retransformation to the variance mean, not volatility median.
    smearing=float(np.exp(target-har.predict(vs.transform(v))).mean())
    return scaler,direction,vs,har,smearing

def predict_pair(pair,frame):
    scaler,direction,vs,har,smear=pair
    ret=direction.predict(scaler.transform(frame[DIR_FEATURES].to_numpy()))
    var=np.exp(har.predict(vs.transform(np.log(np.maximum(frame[VOL_FEATURES].to_numpy(),VAR_FLOOR)))))*smear
    if not np.isfinite(ret).all() or not np.isfinite(var).all() or (var<=0).any():
        raise ValueError('Invalid model output')
    return ret,var

def walk_forward(frames,start,end):
    predictions={s:np.full((len(f),2),np.nan) for s,f in frames.items()}
    folds=[]
    for boundary in range(start,end,30*DAY):
        stop=min(end,boundary+30*DAY)
        pieces=[]
        for s,f in frames.items():
            finite=np.isfinite(f[DIR_FEATURES+['return_target','variance_target']]).all(axis=1)
            mask=finite&(f.origin>=boundary-60*DAY)&(f.label_close<boundary)&((f.origin//STEP)%4==0)
            pieces.append(f.loc[mask])
        train=pd.concat(pieces,ignore_index=True)
        assert train.label_close.max()<boundary
        pair=fit_pair(train)
        for s,f in frames.items():
            mask=(f.origin>=boundary)&(f.origin<stop)&np.isfinite(f[DIR_FEATURES]).all(axis=1)
            ret,var=predict_pair(pair,f.loc[mask])
            predictions[s][mask]=np.column_stack([ret,var])
        fold=dict(start=boundary,end=stop,training_rows=len(train),last_training_label=int(train.label_close.max()),
                  direction_coef=pair[1].coef_.tolist(),har_coef=pair[3].coef_.tolist(),smearing=pair[4])
        folds.append(fold);print('Walk-forward fold',boundary,stop,len(train),flush=True)
    return predictions,folds

def break_even(fee_bps,slippage_bps):
    f,s=fee_bps/10000,slippage_bps/10000
    if not 0<f<1 or not 0<s<1:raise ValueError('Require positive realistic costs')
    return math.log((1+s)*(1+f)/((1-s)*(1-f)))

def sizing(variance):
    if not math.isfinite(variance) or variance<=0:raise ValueError('Missing forecast variance')
    return min(1.0,TARGET_SIGMA/math.sqrt(variance))

def lookup(frames,predictions,symbol,at):
    f=frames[symbol]
    idx=int(np.searchsorted(f.origin.to_numpy(),at,side='right')-1)
    if idx<0 or int(f.origin.iloc[idx])!=at:raise ValueError('No same-close causal forecast')
    ret,var=predictions[symbol][idx]
    if not np.isfinite([ret,var]).all():raise ValueError('Unavailable causal forecast')
    return float(ret),float(var),float(f.ewma.iloc[idx])

def weighted_account(trades,series,grid,weights,fee_bps,slippage_bps):
    """Canonical liquidation-equity accounting with entry-frozen size multipliers."""
    from portfolio_alpha import _normalized_series,_price_at_or_before
    fee,slip=fee_bps/10000,slippage_bps/10000
    if not 0<=fee<1 or not 0<=slip<1:raise ValueError('Invalid costs')
    if len(weights)!=len(trades) or not np.isfinite(weights).all() or min(weights,default=1)<=0 or max(weights,default=1)>1:
        raise ValueError('Invalid multipliers')
    normalized={s:_normalized_series(v) for s,v in series.items()}
    events=defaultdict(list)
    for i,t in enumerate(trades):
        if not grid[0]<=t.entry_ts<=t.exit_ts<=grid[-1] or min(t.entry_price,t.exit_price)<=0:
            raise ValueError('Invalid trade timing/price')
        events[t.entry_ts].append((2,i,'entry',t.entry_price))
        if t.partial_exit_taken:
            if not (t.entry_ts<=t.partial_exit_ts<=t.exit_ts and 0<t.partial_exit_fraction<1 and t.partial_exit_price>0):
                raise ValueError('Invalid partial exit')
            events[t.partial_exit_ts].append((0,i,'partial',t.partial_exit_price))
        events[t.exit_ts].append((3 if t.exit_ts==t.entry_ts else 1,i,'exit',t.exit_price))
    cash=1.0;positions={};symbols=set();curve=[];gross=[];cost=0.;max_slots=0
    def mark(at):
        equity=cash;exposure=0.
        for sym,qty in positions.values():
            price=_price_at_or_before(normalized[sym],at)
            if price is None:raise ValueError('Missing/stale holding mark')
            exposure+=qty*price;equity+=qty*price*(1-slip)*(1-fee)
        if not math.isfinite(equity) or equity<=0:raise ValueError('Invalid equity')
        return equity,exposure
    grid_set=set(grid)
    for at in sorted(grid_set|set(events)):
        for _,i,kind,price in sorted(events.get(at,[])):
            t=trades[i]
            if kind=='entry':
                if t.sym in symbols or len(positions)>=10:raise ValueError('Duplicate/capacity violation')
                equity,_=mark(at);budget=min(cash,equity/10*weights[i])
                if budget<=0:raise ValueError('No cash')
                notional=budget/(1+fee);qty=notional/(price*(1+slip))
                positions[i]=(t.sym,qty);symbols.add(t.sym);cash-=budget
                cost+=notional*fee+qty*price*slip;max_slots=max(max_slots,len(positions))
            else:
                if i not in positions:raise ValueError('Unmatched exit')
                sym,qty=positions[i];sold=qty*(t.partial_exit_fraction if kind=='partial' else 1)
                proceeds=sold*price*(1-slip);cash+=proceeds*(1-fee)
                cost+=proceeds*fee+sold*price*slip
                if kind=='exit':del positions[i];symbols.remove(sym)
                else:positions[i]=(sym,qty-sold)
            if cash < -1e-10:raise ValueError('Borrowing')
        if at in grid_set:
            eq,exposure=mark(at);curve.append((at,eq));gross.append(exposure/eq)
    if positions:raise ValueError('Unliquidated position')
    values=np.array([v for _,v in curve]);peak=np.maximum(1.0,np.maximum.accumulate(values))
    return dict(net_return_pct=100*(cash-1),max_drawdown_pct=100*np.max(1-values/peak),
        average_gross_exposure_pct=100*np.mean(gross),costs_initial_capital=cost,max_positions=max_slots,
        trades=len(trades),valuation_points=len(curve),curve=curve)

def forecast_metrics(frames,predictions,start,end):
    buckets=defaultdict(list)
    for sym,f in frames.items():
        mask=(f.origin>=start)&(f.label_close<=end)&((f.origin//STEP)%4==0)
        mask&=np.isfinite(f.return_target)&np.isfinite(f.variance_target)&np.isfinite(predictions[sym]).all(axis=1)
        actual=f.return_target.to_numpy()[mask];rv=f.variance_target.to_numpy()[mask]
        ret,var=predictions[sym][mask].T
        buckets['return_abs'].extend(np.abs(actual-ret));buckets['control_abs'].extend(np.abs(actual))
        buckets['direction'].extend(np.sign(actual)==np.sign(ret))
        buckets['up'].extend(actual>0);buckets['zero_rv'].extend(rv<VAR_FLOOR)
        y=np.maximum(rv,VAR_FLOOR)
        for name,p in [('har',var),('past_rv',f.rv4.to_numpy()[mask]),('ewma',f.ewma.to_numpy()[mask])]:
            ratio=y/np.maximum(p,VAR_FLOOR)
            buckets[name].extend(ratio-np.log(ratio)-1)
    n=len(buckets['direction'])
    return dict(n=n,return_MAE=float(np.mean(buckets['return_abs'])),control_MAE=float(np.mean(buckets['control_abs'])),
        direction_correct=int(sum(buckets['direction'])),always_up_correct=int(sum(buckets['up'])),
        variance_floor_count=int(sum(buckets['zero_rv'])),
        qlike={k:float(np.mean(buckets[k])) for k in ('har','past_rv','ewma')})

def paired_intervals(accounts):
    series={name:pd.Series(dict(row['curve'])) for name,row in accounts.items()}
    first,last=series['baseline'].index[[0,-1]]
    bounds=pd.date_range(pd.Timestamp(first,unit='ms',tz='UTC').tz_convert('Europe/Budapest'),
                         pd.Timestamp(last,unit='ms',tz='UTC').tz_convert('Europe/Budapest'),freq='D')
    stamps=(bounds.tz_convert('UTC').asi8//1_000_000).tolist()
    daily={}
    for name,s in series.items():
        values=s.reindex(stamps).to_numpy().copy()
        # First curve point follows same-origin fees; the full first day must
        # start from actual initial capital and include these opening costs.
        values[0]=1.0
        daily[name]=np.diff(np.log(values))
    if not all(np.isfinite(v).all() for v in daily.values()):raise ValueError('Incomplete daily boundary curve')
    output={}
    for name in ('volatility','direction','combined'):
        diff=daily[name]-daily['baseline'];results={}
        for block in (1,3):
            rng=np.random.default_rng(42);n=len(diff)
            if n<30:raise ValueError('Insufficient complete portfolio days')
            starts=rng.integers(0,n,size=(5000,math.ceil(n/block)))
            indices=(starts[:,:,None]+np.arange(block))%n
            samples=diff[indices.reshape(5000,-1)[:,:n]].mean(axis=1)*10000
            lo,hi=np.quantile(samples,[.05/(2*3),1-.05/(2*3)])
            results[str(block)]=dict(mean_daily_log_gain_bps=float(diff.mean()*10000),familywise95=[float(lo),float(hi)])
        output[name]=dict(n_days=len(diff),blocks=results)
    return output

def fetch_tail(symbol,tf,start,end,directory):
    interval=STEP if tf=='15m' else 4*STEP
    cursor=start;rows=[];receipts=[]
    while cursor<end:
        params=dict(symbol=symbol,interval=tf,startTime=cursor,endTime=end-1,limit=1000)
        url='https://data-api.binance.vision/api/v3/klines?'+urllib.parse.urlencode(params)
        for attempt in range(3):
            try:
                with urllib.request.urlopen(url,timeout=30) as response:raw=response.read()
                batch=json.loads(raw)
                if not isinstance(batch,list) or not batch:raise ValueError('Missing recent candles')
                break
            except Exception:
                if attempt==2:raise
                time.sleep(.5*(attempt+1))
        name=f'{symbol}_{tf}_{cursor}.json';(directory/name).write_bytes(raw)
        receipts.append(dict(file=name,sha256=hashlib.sha256(raw).hexdigest(),source=url,retrieved_at=pd.Timestamp.now(tz='UTC').isoformat()))
        for r in batch:
            if len(r)!=12 or int(r[6])!=int(r[0])+interval-1:raise ValueError('Changed/unclosed kline schema')
            rows.append(dict(t=int(r[0]),o=float(r[1]),h=float(r[2]),l=float(r[3]),c=float(r[4]),v=float(r[5])))
        next_cursor=rows[-1]['t']+interval
        if next_cursor<=cursor:raise ValueError('Pagination stalled')
        cursor=next_cursor
    return rows,receipts

def prepare_market(archive,cache_dir,output):
    from closed_grid_policy_replay import validate_series
    manifest=json.loads((archive/'manifest.json').read_bytes());original_hash=digest(archive/'manifest.json')
    for name,checksum in manifest['input_hashes'].items():
        if Path(name).name!=name or digest(archive/'market'/name)!=checksum:raise ValueError('Archive input hash mismatch')
    pattern=re.compile(r'.+_(15m|1h)_(\d+)_(\d+)\.json$')
    ends=[int(m.group(3)) for p in cache_dir.glob('*.json') if (m:=pattern.fullmatch(p.name))]
    end=max([manifest['end_ms']]+ends)//(4*STEP)*(4*STEP)
    if end>int(pd.Timestamp.now(tz='UTC').value//1_000_000):raise ValueError('Cache end is in future')
    market=output/'market';raw_dir=output/'raw';market.mkdir();raw_dir.mkdir()
    results=[]
    def one(identity):
        symbol,tf=identity;step=STEP if tf=='15m' else 4*STEP
        old=archive/'market'/f"{symbol}_{tf}_{manifest['archive_start_ms']}_{manifest['end_ms']}.json"
        rows=json.loads(old.read_bytes());receipt=[]
        if end>manifest['end_ms']:
            more,receipt=fetch_tail(symbol,tf,manifest['end_ms'],end,raw_dir);rows+=more
        validate_series(rows,step,manifest['archive_start_ms'],end)
        file=market/f'{symbol}_{tf}.json';file.write_text(json.dumps(rows,separators=(',',':')),encoding='utf-8')
        print('Market complete',symbol,tf,len(rows),flush=True)
        return dict(symbol=symbol,tf=tf,sha256=digest(file),rows=len(rows),receipts=receipt)
    with ThreadPoolExecutor(max_workers=4) as pool:
        results=list(pool.map(one,[(s,tf) for s in manifest['eligible_symbols'] for tf in ('15m','1h')]))
    manifest.update(end_ms=end,archive_end_ms=end,input_hashes={f"{r['symbol']}_{r['tf']}.json":r['sha256'] for r in results},
        original_manifest_sha256=original_hash,recent_receipts=results,maximum_cache_end_ms=max(ends,default=end))
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    return manifest

def frozen_market(source,output):
    """Reuse the registered inputs; never move the boundary after seeing outcomes."""
    manifest=json.loads((source/'manifest.json').read_bytes())
    for name,checksum in manifest['input_hashes'].items():
        if Path(name).name!=name or digest(source/'market'/name)!=checksum:
            raise ValueError('Frozen input hash mismatch')
    shutil.copytree(source/'market',output/'market')
    if (source/'raw').exists():shutil.copytree(source/'raw',output/'raw')
    manifest.update(frozen_market_source=str(source.resolve()),frozen_manifest_sha256=digest(source/'manifest.json'))
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    return manifest

def feature_worker(request):
    """Identical indicator function in separate processes, with native-array checkpoints."""
    import replay_backtest as rb
    symbol,market,directory=request;payload={}
    for tf in ('15m','1h'):
        rows=json.loads((Path(market)/f'{symbol}_{tf}.json').read_bytes())
        data=np.array([tuple(r[k] for k in ('t','o','h','l','c','v')) for r in rows],dtype=rb._KLINE_DTYPE)
        payload[tf+'__data']=data
    payload['4h__data']=rb._aggregate_1h_to_4h(payload['1h__data'])
    for tf in ('15m','1h','4h'):
        data=payload[tf+'__data']
        feat=rb.compute_features(data['o'],data['h'],data['l'],data['c'],data['v'])
        for key,value in feat.items():
            value=np.asarray(value)
            if value.dtype.kind=='O':
                if not all(isinstance(x,str) for x in value):raise ValueError('Unsafe feature dtype')
                value=value.astype(str)
            payload[tf+'__'+key]=value
    path=Path(directory)/f'{symbol}.npz';np.savez_compressed(path,**payload)
    return symbol,str(path),digest(path)

def build_features(symbols,market,directory,workers=4):
    directory.mkdir();cache={};frames={};checksums={}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        jobs=[pool.submit(feature_worker,(s,str(market),str(directory))) for s in symbols]
        for job in as_completed(jobs):
            sym,path,checksum=job.result();checksums[sym]=checksum
            with np.load(path,allow_pickle=False) as payload:
                for tf in ('15m','1h','4h'):
                    prefix=tf+'__'
                    cache[sym,tf]=(payload[prefix+'data'],{k[len(prefix):]:payload[k] for k in payload.files if k.startswith(prefix) and k!=prefix+'data'})
            frames[sym]=market_frame(cache[sym,'15m'][0])
            print('Features complete',len(frames),len(symbols),sym,flush=True)
    (directory/'hashes.json').write_text(json.dumps(checksums,indent=2),encoding='utf-8')
    # Stable symbol order is required for bitwise deterministic pooled training.
    return cache,{s:frames[s] for s in symbols}

def read_feature_pack(path):
    with np.load(path,allow_pickle=False) as payload:
        return {tf:(payload[tf+'__data'],{k[len(tf)+2:]:payload[k] for k in payload.files if k.startswith(tf+'__') and k!=tf+'__data'}) for tf in ('15m','1h','4h')}

def reuse_features(directory,manifest,output):
    """Reuse only identical market/config/indicator checkpoints, in stable order."""
    parent=directory.parent
    previous=json.loads((parent/'manifest.json').read_bytes())
    registered=json.loads((parent/'registration.json').read_bytes())
    if previous['input_hashes']!=manifest['input_hashes']:raise ValueError('Feature market differs')
    for name in ('config.py','indicators.py','replay_backtest.py'):
        if digest(Path(__file__).with_name(name))!=registered['source_hashes'][name]:raise ValueError('Feature code differs')
    hashes=json.loads((directory/'hashes.json').read_bytes());cache={};frames={}
    if set(hashes)!=set(manifest['eligible_symbols']):raise ValueError('Incomplete feature cohort')
    for sym in manifest['eligible_symbols']:
        if digest(directory/f'{sym}.npz')!=hashes[sym]:raise ValueError('Feature hash mismatch')
        packs=read_feature_pack(directory/f'{sym}.npz')
        for tf,pack in packs.items():cache[sym,tf]=pack
        frames[sym]=market_frame(packs['15m'][0])
    (output/'reused_features.json').write_text(json.dumps(dict(source=str(directory.resolve()),hashes=hashes),indent=2),encoding='utf-8')
    print('Verified reused features',len(frames),flush=True)
    return cache,frames,hashes

def candidate_worker(request):
    import replay_backtest as rb
    from research_rocket_capture import policy
    sym,tf,features,context_file,directory=request
    packs=read_feature_pack(Path(features)/f'{sym}.npz')
    with np.load(context_file,allow_pickle=False) as saved:context=tuple(saved[k] for k in ('t','bull','vs'))
    data,feat=packs[tf]
    with policy('baseline'):
        rows=asyncio.run(rb._build_candidates_for_symbol(sym,tf,data,feat,{sym:packs['15m']},{sym:packs['4h']},context,
                        variant='score_replace_cluster',include_trend_start=False))
    path=Path(directory)/f'{sym}_{tf}.json'
    path.write_text(json.dumps([asdict(r) for r in rows],allow_nan=False),encoding='utf-8')
    return sym,tf,str(path),digest(path),len(rows)

def parallel_candidates(symbols,cache,features,context,output,workers):
    """Independent symbol/timeframe generators; merge in original serial order."""
    import replay_backtest as rb
    directory=output/'candidates';directory.mkdir()
    context_file=output/'market_context.npz';np.savez_compressed(context_file,t=context[0],bull=context[1],vs=context[2])
    identities=[(s,tf) for s in symbols for tf in ('15m','1h')];completed={}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        jobs=[pool.submit(candidate_worker,(s,tf,str(features),str(context_file),str(directory))) for s,tf in identities]
        for job in as_completed(jobs):
            sym,tf,path,checksum,n=job.result();completed[sym,tf]=(path,checksum)
            print('Candidates complete',len(completed),len(identities),sym,tf,n,flush=True)
    raw={};times=set();total=0
    for sym,tf in identities:
        path,checksum=completed[sym,tf]
        if digest(path)!=checksum:raise ValueError('Candidate checkpoint drift')
        rows=[rb.ReplayCandidate(**r) for r in json.loads(Path(path).read_bytes())]
        total+=len(rows);times.update(int(x) for x in cache[sym,tf][0]['t'][25:])
        for row in rows:raw.setdefault(row.ts_ms,[]).append(row)
    (directory/'hashes.json').write_text(json.dumps({Path(p).name:h for p,h in completed.values()},indent=2),encoding='utf-8')
    return raw,times,total

def replacement_options(config):
    """Match the established closed-grid/rule-only evaluator defaults."""
    return bool(getattr(config,'PORTFOLIO_REPLACE_ENABLED',True)),float(getattr(config,'PORTFOLIO_REPLACE_MIN_DELTA',8.0))

def reuse_candidate_snapshot(source,manifest,output):
    """Read only the trusted local snapshot with frozen source/market receipt."""
    receipt=json.loads((source/'candidate_checkpoint_receipt.json').read_bytes())
    if receipt['market_hashes']!=manifest['input_hashes'] or (receipt['start_ms'],receipt['end_ms'])!=(manifest['start_ms'],manifest['end_ms']):
        raise ValueError('Candidate market/boundary differs')
    for name in ('replay_backtest.py','monitor.py','strategy.py','indicators.py','config.py','research_rocket_capture.py'):
        if digest(Path(__file__).with_name(name))!=receipt['source_hashes'][name]:raise ValueError('Candidate rule source differs')
    path=source/'candidate_snapshot.pkl'
    if digest(path)!=receipt['snapshot_sha256']:raise ValueError('Candidate snapshot hash mismatch')
    with path.open('rb') as file:snapshot=pickle.load(file)
    raw,times,n=snapshot
    if n!=sum(map(len,raw.values())) or any(not manifest['start_ms']<=at<manifest['end_ms'] for at in raw):
        raise ValueError('Invalid candidate count/clock')
    if any(row.ts_ms!=at for at,rows in raw.items() for row in rows):raise ValueError('Candidate close differs')
    (output/'reused_candidate_checkpoint.json').write_text(json.dumps(dict(source=str(source.resolve()),receipt=receipt),indent=2),encoding='utf-8')
    print('Verified reused candidate snapshot',n,flush=True)
    return snapshot

async def run(args):
    import replay_backtest as rb, config
    from closed_grid_policy_replay import event_clock
    from audit_negative_day_rebound import finalize_at_boundary
    from research_rocket_capture import policy
    from portfolio_alpha import closed_price_series,_benchmark_result,_simulate_account
    args.output.mkdir(parents=True,exist_ok=False)
    sources={name:digest(Path(__file__).with_name(name)) for name in ('compare_price_volatility_bot.py','replay_backtest.py','monitor.py','strategy.py','indicators.py','config.py','portfolio_alpha.py','research_rocket_capture.py')}
    (args.output/'registration.json').write_text(json.dumps(dict(arms=ARMS,source_hashes=sources,registered_at=pd.Timestamp.now(tz='UTC').isoformat(),runtime_eligible=False)),encoding='utf-8')
    manifest=frozen_market(args.frozen_market,args.output) if args.frozen_market else prepare_market(args.archive,args.cache_dir,args.output)
    symbols=manifest['eligible_symbols'];start,end=manifest['start_ms'],manifest['end_ms']
    if args.frozen_features:
        cache,frames,feature_hashes=reuse_features(args.frozen_features,manifest,args.output)
        feature_directory=args.frozen_features
    else:
        feature_directory=args.output/'features'
        cache,frames=build_features(symbols,args.output/'market',feature_directory,args.workers)
        feature_hashes=json.loads((feature_directory/'hashes.json').read_bytes())
    predictions,folds=walk_forward(frames,start,end)
    (args.output/'folds.json').write_text(json.dumps(folds,indent=2),encoding='utf-8')
    np.savez_compressed(args.output/'predictions.npz',**predictions)
    c15={s:cache[s,'15m'] for s in symbols};c4={s:cache[s,'4h'] for s in symbols}
    context=rb._build_bull_day_context(cache['BTCUSDT','1h'][0])
    with policy('baseline'),patch.object(rb,'_load_temporal_scout_events',return_value=({},{})):
        if args.frozen_candidates:
            snapshot=reuse_candidate_snapshot(args.frozen_candidates,manifest,args.output)
        else:
            raw,times,_=parallel_candidates(symbols,cache,feature_directory,context,args.output,args.workers)
            snapshot=event_clock(raw,times,start,end)
        # Trusted local runtime checkpoint, never distributed or committed.
        with (args.output/'candidate_snapshot.pkl').open('wb') as f:pickle.dump(snapshot,f)
        fee=max(7.5,float(config.PAPER_FEE_BPS));slip=5.;threshold=break_even(fee,slip)
        filtered={at:[c for c in rows if lookup(frames,predictions,c.sym,c.ts_ms)[0]>threshold] for at,rows in snapshot[0].items()}
        streams={'baseline':snapshot,'direction':(filtered,snapshot[1],sum(map(len,filtered.values())))}
        trades_by_arm={};stats={}
        for name,stream in streams.items():
            print('Simulating admission/exits',name,stream[2],flush=True)
            replace_enabled,replace_delta=replacement_options(config)
            trades,stat=await rb.simulate_portfolio(symbols,['15m','1h'],cache,c15,c4,context,max_open_positions=10,
                enable_replacement=replace_enabled,replace_min_delta=replace_delta,
                variant='score_replace_cluster',top_gainer_score_min=float(config.TOP_GAINER_SCORE_GATE_MIN_SCORE),candidate_snapshot=stream)
            finalize_at_boundary(trades,cache,end);trades_by_arm[name]=trades;stats[name]=asdict(stat)
            (args.output/f'trades_{name}.json').write_text(json.dumps([asdict(t) for t in trades]),encoding='utf-8')
    series={s:closed_price_series(cache[s,'15m'][0],bar_ms=STEP,start_ms=start,end_ms=end) for s in symbols}
    grid=list(range(start,end+1,STEP));accounts={};stress={}
    for name in ARMS:
        trades=trades_by_arm['direction' if name in ('direction','combined') else 'baseline']
        weights=[sizing(lookup(frames,predictions,t.sym,t.entry_ts)[2 if name=='ewma_control' else 1]) if name in ('volatility','combined','ewma_control') else 1. for t in trades]
        accounts[name]=weighted_account(trades,series,grid,weights,fee,slip)
        stress[name]=weighted_account(trades,series,grid,weights,2*fee,2*slip)
        if name=='baseline':
            reference=_simulate_account(trades,price_series_by_symbol=series,valuation_timestamps=grid,capacity=10,initial_capital=1,fee_bps=fee,slippage_bps=slip)
            if reference.violations:raise ValueError('Canonical account violations: '+str(reference.violations))
            np.testing.assert_allclose(accounts[name]['curve'],reference.equity_curve,rtol=1e-10,atol=1e-12)
    btc=_benchmark_result(series['BTCUSDT'],initial_capital=1,fee_bps=fee,slippage_bps=slip)
    # Existing daily leader labels are evaluator-only, never inputs to models.
    objective=rb._daily_top_objective(symbols,cache,start_ms=start,end_ms=end,top_n=10)
    captures={}
    for name in ARMS:
        trades=trades_by_arm['direction' if name in ('direction','combined') else 'baseline']
        from datetime import datetime,timezone
        report=rb._make_report(label=name,start=datetime.fromtimestamp(start/1000,timezone.utc),end=datetime.fromtimestamp(end/1000,timezone.utc),
            symbols=symbols,timeframes=['15m','1h'],trades=trades,run_stats=rb.ReplayRunStats(),daily_objective=objective)
        captures[name]=report['objective']
    for name,checksum in sources.items():
        if digest(Path(__file__).with_name(name))!=checksum:raise ValueError('Source drift')
    for name,checksum in manifest['input_hashes'].items():
        if digest(args.output/'market'/name)!=checksum:raise ValueError('Frozen market drift')
    for symbol,checksum in feature_hashes.items():
        if digest(feature_directory/f'{symbol}.npz')!=checksum:raise ValueError('Frozen feature drift')
    results=dict(status='COMPLETED_RETROSPECTIVE_DIAGNOSTIC',runtime_eligible=False,achievement_claimed=False,
        source_hashes=sources,start_ms=start,end_ms=end,days=(end-start)/DAY,population_complete=len(symbols),population_requested=len(manifest['requested_symbols']),
        costs=dict(fee_bps=fee,slippage_bps=slip),benchmark=btc,accounts=accounts,cost_stress=stress,
        paired_intervals=paired_intervals(accounts),captures=captures,forecast_metrics=forecast_metrics(frames,predictions,start,end),
        candidate_counts={k:v[2] for k,v in streams.items()},folds=folds,
        limitations=['current rule-only replay, not historical live-model or agent-only parity','fixed complete symbols; unknown historical watchlist and survivorship exposure',
            'coarse four-15m-return variance proxy','static fill costs and idealized closed-bar fills','already-seen retrospective history, not sealed holdout',
            'size overlay does not feed back into admission','full Truth Harness FAIL TH-11 remains separate'])
    (args.output/'result.json').write_text(json.dumps(results,indent=2,allow_nan=False),encoding='utf-8')
    print('COMPARISON COMPLETE',json.dumps({k:{n:v for n,v in a.items() if n!='curve'} for k,a in accounts.items()}),flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive',type=Path,default=Path('.runtime/closed_grid_policy_replay/20261001_max_archive_v1'))
    p.add_argument('--cache-dir',type=Path,default=Path('.runtime/signal_quality_cache'))
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--frozen-market',type=Path,help='Reuse verified registered candles without extending/selecting the period')
    p.add_argument('--frozen-features',type=Path,help='Reuse verified market/config/indicator-equivalent feature checkpoints')
    p.add_argument('--frozen-candidates',type=Path,help='Reuse trusted local candidate snapshot bound to exact rule/market hashes')
    p.add_argument('--workers',type=int,choices=range(1,5),default=4)
    asyncio.run(run(p.parse_args()))

if __name__=='__main__':main()
