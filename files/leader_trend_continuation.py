"""Offline causal trend-confirmed soft SELL; no live config imports or future labels."""
from __future__ import annotations
import numpy as np
import leader_mission_metrics as metrics
import replay_backtest as rb

BAR=metrics.BAR
MAX_DELAY=4*BAR


def confirmation(raw,feat,entry,price,at):
    """Features use closed15m prefix, entry price, never final trade annotations."""
    i=int(np.searchsorted(raw['time'],entry));j=int(np.searchsorted(raw['time'],at))
    if (price<=0 or i<13 or j<=i or j>=len(raw['time']) or
        not np.array_equal(raw['time'][i:j+1],np.arange(entry,at+1,BAR))):return None
    c=float(raw['close'][j]);ema=float(feat['ema'][j]);previous=float(feat['ema'][j-1])
    atr=float(feat['atr'][j]);entry_atr=float(feat['atr'][i])
    peak=max(price,float(raw['close'][i:j+1].max()));tol=price*1e-12
    if not all(np.isfinite(v) for v in (c,ema,previous,atr,entry_atr,peak)) or min(atr,entry_atr)<=0:return None
    valid=bool(c>ema+tol and ema>previous+tol and peak>=price+entry_atr-tol and c>=peak-atr-tol)
    return dict(close=c,ema=ema,previous_ema=previous,atr=atr,entry_atr=entry_atr,peak=peak,confirmed=valid)


class TrendPolicy:
    def __init__(self,progress,market,features):
        self.progress=progress;self.market=market;self.features=features
        self.active={};self.considered=set();self.decisions=[];self.ticks=[]

    def __call__(self,trade,data,feat,idx,**kwargs):
        at=int(kwargs['ts_ms']);key=(trade.sym,trade.tf,trade.entry_ts)
        reason=self.progress(trade,data,feat,idx,**kwargs)
        state=self.active.get(key)
        if state is not None:
            x=confirmation(self.market[trade.sym],self.features[trade.sym],trade.entry_ts,trade.entry_price,at)
            kind=('HARD' if reason and not rb._is_weak_exit_reason(reason) else
                  'TIMEOUT' if at>=state['deadline'] else
                  'CONFIRMATION_LOST' if x is None or not x['confirmed'] else 'HOLD')
            self.ticks.append(dict(key=list(key),at=at,deadline=state['deadline'],kind=kind,features=x,reason=reason))
            if kind=='HARD':del self.active[key];return reason
            if kind=='HOLD':trade.exit_ts=0;trade.exit_price=0.;return None
            del self.active[key]
            # Explicit latest CLOSED15m price for1h positions, never stale1h fill.
            raw=self.market[trade.sym];j=int(np.searchsorted(raw['time'],at))
            if j>=len(raw['time']) or raw['time'][j]!=at:raise ValueError('missing forced exit price')
            trade.exit_ts=at;trade.exit_price=float(raw['close'][j])
            return state['origin_reason']+' [trend continuation '+kind+']'
        if not rb._is_weak_exit_reason(reason) or key in self.considered:return reason
        self.considered.add(key)
        exact=(int(data['t'][idx])+rb.BAR_MS[trade.tf]==at)
        x=confirmation(self.market[trade.sym],self.features[trade.sym],trade.entry_ts,trade.entry_price,at) if exact else None
        price=float(data['c'][idx])
        if x is not None and not np.isclose(x['close'],price,rtol=1e-10,atol=1e-12):x=None
        defer=bool(x is not None and x['confirmed'])
        self.decisions.append(dict(key=list(key),at=at,entry_price=trade.entry_price,origin_price=price,
            reason=reason,features=x,defer=defer,deadline=at+MAX_DELAY))
        if defer:
            self.active[key]=dict(deadline=at+MAX_DELAY,origin_reason=reason)
            trade.exit_ts=0;trade.exit_price=0.;return None
        return reason


def category(reason):
    if 'ATR trail' in reason:return 'ATR_TRAIL'
    if rb._is_weak_exit_reason(reason):return 'WEAK'
    if 'time (' in reason:return 'TIME_LIMIT'
    if 'replace' in reason.lower():return 'REPLACEMENT'
    if 'open_at_end' in reason:return 'BOUNDARY'
    return 'OTHER_HARD'


def diagnostics(episodes,skips,boundary=None):
    selected=[e for e in episodes if boundary is None or metrics.day_bounds(e['day'])[0]>=boundary]
    valid=[e for e in selected if e['state']=='CONFIRMED' and not e['terminal_forced']]
    groups={}
    for kind in sorted({category(e['exit_reason']) for e in valid}):
        rows=[e for e in valid if category(e['exit_reason'])==kind]
        groups[kind]=dict(n=len(rows),N=len(valid),before=sum(e['first_exit_before_marker'] for e in rows),
            rebound_n=sum(e.get('new_high_after_first_exit') is True for e in rows),
            rebound_N=sum(e.get('new_high_after_first_exit') is not None for e in rows))
    exact={}
    for e in valid:
        key=e['exit_reason'];r=exact.setdefault(key,dict(n=0,before=0));r['n']+=1;r['before']+=int(e['first_exit_before_marker'])
    associated=[]
    by_symbol={}
    for e in valid:by_symbol.setdefault(e['symbol'],[]).append(e)
    for s in skips:
        if boundary is not None and s['at']<boundary:continue
        related=[e for e in by_symbol.get(s['symbol'],[]) if e['entry_ts']<=s['at']<e['marker_ts']]
        if related:associated.append(dict(**s,leader_episodes=[e['day'] for e in related]))
    return dict(confirmed_n=len(valid),categories=groups,exact_reasons=exact,
        cooldown_events_in_confirmed_leader_episodes=len(associated),
        cooldown_unique_episode_pairs=len({(e['symbol'],day) for e in associated for day in e['leader_episodes']}),
        cooldown_episode_events=associated,scope='actual blocked candidates, not successful counterfactual BUYs')


def paired_accompaniment(base,target):
    a={(r['day'],r['symbol']):r for r in base};b={(r['day'],r['symbol']):r for r in target};rows=[]
    for key in sorted(a.keys()&b.keys()):
        x,y=a[key],b[key]
        if (x['entry_ts'],x['entry_price'])!=(y['entry_ts'],y['entry_price']):continue
        if any(r['state']!='CONFIRMED' or r['terminal_forced'] for r in (x,y)):continue
        if x['marker_ts']!=y['marker_ts']:raise ValueError('paired marker drift')
        rows.append(dict(day=key[0],symbol=key[1],delta=y['accompaniment_fraction']-x['accompaniment_fraction']))
    days=sorted({r['day'] for r in rows});ci=None
    if len(days)>=3:
        sums=np.array([sum(r['delta'] for r in rows if r['day']==d) for d in days]);counts=np.array([sum(r['day']==d for r in rows) for d in days])
        rng=np.random.default_rng(42);st=rng.integers(0,len(days),(5000,int(np.ceil(len(days)/3))))
        idx=((st[:,:,None]+np.arange(3))%len(days)).reshape(5000,-1)[:,:len(days)]
        ci=np.quantile(sums[idx].sum(axis=1)/counts[idx].sum(axis=1),[.05/48,1-.05/48]).tolist()
    return dict(n=len(rows),days=len(days),mean=float(np.mean([r['delta'] for r in rows])) if rows else None,
        interval=ci,confidence_familywise=.95,comparisons=24)


def verdict(arms,comparison):
    checks={}
    for cohort in ('full','test'):
        a,b=(arms[k][cohort] for k in ('control','trend'))
        checks[cohort+'_early']=b['early']>=a['early'];checks[cohort+'_coverage']=b['captured']>=a['captured']
        checks[cohort+'_precision']=bool(a['precision_N'] and b['precision_N'] and b['precision_n']/b['precision_N']>=a['precision_n']/a['precision_N'])
    p=comparison['accompaniment'];e=comparison['exits']
    checks['held_time_gain']=p['mean'] is not None and p['mean']>0
    checks['held_time_confirmed']=p['days']>=30 and p['interval'] is not None and p['interval'][0]>0
    checks['retention_not_worse']=e['retention_delta_mean'] is not None and e['retention_delta_mean']>=0
    checks['no_more_early_exits']=e['first_early_delta_n']<=0
    return dict(checks=checks,status='RETROSPECTIVE_GAIN_REQUIRES_FORWARD' if all(checks.values()) else
        'MISSION_TRADEOFF_OR_WORSE' if not all(v for k,v in checks.items() if k.startswith(('full_','test_'))) else
        'NO_CONFIRMED_MISSION_GAIN',runtime_eligible=False)
