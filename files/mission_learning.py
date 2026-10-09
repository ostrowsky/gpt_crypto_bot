"""Causal, separately supervised mission heads and fail-closed release checks."""
from datetime import datetime,timedelta,timezone
import numpy as np
from impulse_entry_catboost import features as market_features
from mission_contract import window,local_day,candidate_target,remaining,TZ

BAR=900000;DAY=86400000
MODES=('trend','retest','impulse_speed','breakout','strong_trend','impulse','alignment')
PARAMETERS=dict(iterations=300,depth=4,learning_rate=.03,l2_leaf_reg=10,random_seed=42,
    thread_count=2,allow_writing_files=False,verbose=False,loss_function='Logloss')


def entry_features(data,clock,tf,score,mode):
    x=market_features(data,clock,tf)
    if x is None or not np.isfinite(score):return None
    return np.r_[x,score,[float(mode==m) for m in MODES]]


def state_features(data,clock,tf,score,mode,entry_clock,entry_price,peak_price):
    x=entry_features(data,clock,tf,score,mode)
    if x is None or entry_clock>clock or min(entry_price,peak_price)<=0:return None
    i=np.searchsorted(data['t']+BAR,clock,side='right')-1;p=float(data['c'][i])
    return np.r_[x,(clock-entry_clock)/3600000,100*(p/entry_price-1),100*(peak_price/p-1)]


def continuation_target(data,clock,horizon):
    """Explicit proxy: future close rises without >2 origin ATR adverse closes."""
    closes=data['t']+BAR;i=int(np.searchsorted(closes,clock));n=horizon//BAR
    if i<14 or i+n>=len(data) or not np.array_equal(closes[i:i+n+1],clock+np.arange(n+1)*BAR):return None
    before=data[i-13:i+1];prev=data['c'][i-14:i]
    atr=float(np.maximum(before['h']-before['l'],np.maximum(abs(before['h']-prev),abs(before['l']-prev))).mean())
    p=float(data['c'][i]);future=data['c'][i+1:i+n+1]
    if not np.isfinite(future).all() or not np.isfinite(atr) or atr<=0:return None
    return dict(target=int(future[-1]>p and future.min()>=p-2*atr),available_at=clock+horizon)


def fit_cohorts(clocks,available,valid,fit_at,start):
    first=datetime.fromtimestamp(start/1000,timezone.utc).astimezone(TZ).date()
    last=datetime.fromtimestamp(fit_at/1000,timezone.utc).astimezone(TZ).date()
    cut=window((first+timedelta(days=int((last-first).days*.8))).isoformat())[0]
    train=valid&(clocks<cut)&(available<cut)
    val=valid&(clocks>=cut)&(clocks<fit_at)&(available<fit_at)
    return train,val,cut


def mission_metrics(trades,labels,boundary=None):
    """Unique day/symbol first BUY, global exchange top filtered by watchlist."""
    used={d:r for d,r in labels.items() if (boundary is None or window(d)[0]>=boundary) and not r['coverage']['missing'] and r['state']!='PENDING'}
    first={};unknown=0
    for t in sorted(trades,key=lambda r:r['entry_ts']):
        day=local_day(t['entry_ts'])
        if day not in used:continue
        if t['sym'] not in used[day]['values']:unknown+=1;continue
        first.setdefault((day,t['sym']),t)
    records=[];daily=[]
    for (d,s),t in sorted(first.items()):
        label=used[d];v=label['values'][s];r=remaining(v['open'],v['close'],t['entry_price']);leader=s in label['leaders']
        records.append(dict(day=d,symbol=s,leader=leader,early=bool(leader and r is not None and r>=.35),capture=r,trade=t,entry_ts=t['entry_ts'],entry_price=t['entry_price']))
    for d,r in sorted(used.items()):
        picks=[v for v in records if v['day']==d];daily.append(dict(day=d,leaders=len(r['leaders']),early=sum(v['early'] for v in picks),captured=sum(v['leader'] for v in picks),precision_n=sum(v['leader'] for v in picks),precision_N=len(picks)))
    summary=dict(days=len(used),excluded_partial_days=len(labels)-len(used) if boundary is None else sum(window(d)[0]>=boundary and d not in used for d in labels),
        leader_pairs=sum(r['leaders'] for r in daily),early=sum(r['early'] for r in daily),captured=sum(r['captured'] for r in daily),
        unique_precision_n=sum(r['precision_n'] for r in daily),unique_precision_N=sum(r['precision_N'] for r in daily),unknown_entry_events=unknown,
        historical_PIT_certified=False)
    return summary,daily,[r for r in records if r['leader']]


def paired_entry_interval(a,b):
    if [r['day'] for r in a]!=[r['day'] for r in b]:raise ValueError('different windows')
    n=len(a)
    if n<3:return None
    x=np.array([[r['early'],r['captured'],r['precision_n'],r['precision_N']] for r in a],float)
    y=np.array([[r['early'],r['captured'],r['precision_n'],r['precision_N']] for r in b],float)
    rng=np.random.default_rng(42);starts=rng.integers(0,n,size=(5000,int(np.ceil(n/3))))
    idx=((starts[:,:,None]+np.arange(3))%n).reshape(5000,-1)[:,:n]
    sx=x[idx].sum(axis=1);sy=y[idx].sum(axis=1)
    delta=sy[:,0]-sx[:,0]
    return dict(days=n,block_days=3,draws=5000,early_count_delta95=np.quantile(delta,[.025,.975]).tolist())


def release_readiness(evidence):
    checks={k:bool(evidence.get(k,False)) for k in ('retrospective_mission_pass','fresh_mission_pass','pit_universe','asof_watchlist','receive_clock_verified','acceptance_clock_verified','complete_coverage','source_model_hashes','companion_exit_pass','rollback_available')}
    checks['fresh_sample']=evidence.get('fresh_completed_days',0)>=30 and evidence.get('fresh_leader_pairs',0)>=100
    return dict(state='READY_FOR_CANARY_REVIEW' if all(checks.values()) else 'SHADOW_ONLY',runtime_eligible=False,checks=checks,blockers=[k for k,v in checks.items() if not v])
