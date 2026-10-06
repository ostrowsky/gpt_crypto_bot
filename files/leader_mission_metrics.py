"""Raw closed-candle leader metrics and causal weakening evaluation labels."""
from __future__ import annotations
from datetime import datetime,timedelta,time,timezone
from zoneinfo import ZoneInfo
import numpy as np

BAR=900000
TZ=ZoneInfo('Europe/Budapest')


def day_bounds(day):
    date=datetime.fromisoformat(day).date()
    return tuple(int(datetime.combine(date,t,tzinfo=TZ).timestamp()*1000) for t in (time(),time(22)))


def day_key(clock):return datetime.fromtimestamp(clock/1000,timezone.utc).astimezone(TZ).date().isoformat()


def daily_labels(market,start,end):
    labels={};days=[];day=datetime.fromtimestamp(start/1000,timezone.utc).astimezone(TZ).date()
    last=datetime.fromtimestamp(end/1000,timezone.utc).astimezone(TZ).date()
    while day<=last:
        key=day.isoformat();lo,hi=day_bounds(key)
        if start<=lo and hi<=end:
            values={}
            for symbol,d in market.items():
                i=np.searchsorted(d['time'],lo+BAR);j=np.searchsorted(d['time'],hi)
                if i>=len(d['time']) or j>=len(d['time']):continue
                if not np.array_equal(d['time'][i:j+1],np.arange(lo+BAR,hi+1,BAR)):continue
                opening=float(d['open'][i]);closing=float(d['close'][j])
                values[symbol]=dict(open=opening,close=closing,cutoff=hi,return_pct=100*(closing/opening-1))
            if len(values)>=15:
                leaders=sorted(values,key=lambda s:(values[s]['return_pct'],s),reverse=True)[:15]
                labels[key]=dict(values=values,leaders=leaders);days.append(key)
        day+=timedelta(days=1)
    return labels,days


def mission(trades,labels,boundary=None):
    used={d:r for d,r in labels.items() if boundary is None or day_bounds(d)[0]>=boundary}
    eligible=[];first={}
    for t in sorted(trades,key=lambda t:t['entry_ts']):
        key=day_key(t['entry_ts'])
        if key not in used or t['entry_ts']>day_bounds(key)[1] or t['sym'] not in used[key]['values']:continue
        eligible.append(t);first.setdefault((key,t['sym']),t)
    records=[]
    for (day,symbol),t in sorted(first.items()):
        row=used[day]['values'][symbol];move=row['close']-row['open']
        capture=float(np.clip((row['close']-t['entry_price'])/move,0,1.5)) if move>0 else None
        records.append(dict(day=day,symbol=symbol,leader=symbol in used[day]['leaders'],
            entry_ts=t['entry_ts'],entry_price=t['entry_price'],capture=capture,early=capture is not None and capture>=.35,
            lead_minutes=(row['cutoff']-t['entry_ts'])/60000,stored_capture=t.get('capture_ratio_at_entry'),
            stored_early=t.get('capture_ratio_at_entry') is not None and t['capture_ratio_at_entry']>=.35,
            trade=t))
    leaders=[r for r in records if r['leader']];objective=sum(t['sym'] in used[day_key(t['entry_ts'])]['leaders'] for t in eligible)
    dayrows=[]
    for day in sorted(used):
        picks=[r for r in leaders if r['day']==day];b=[t for t in eligible if day_key(t['entry_ts'])==day]
        dayrows.append(dict(day=day,early=sum(r['early'] for r in picks),captured=len(picks),leaders=15,
            precision_n=sum(t['sym'] in used[day]['leaders'] for t in b),precision_N=len(b)))
    summary=dict(days=len(used),leader_pairs=len(used)*15,early=sum(r['early'] for r in leaders),captured=len(leaders),
        precision_n=objective,precision_N=len(eligible),unique_precision_n=len(leaders),unique_precision_N=len(records),
        stored_early=sum(r['stored_early'] for r in leaders),
        corrected_early_disagreements=sum(r['early']!=r['stored_early'] for r in leaders),
        capture_known=sum(r['capture'] is not None for r in leaders),
        capture_mean=float(np.mean([r['capture'] for r in leaders if r['capture'] is not None])) if any(r['capture'] is not None for r in leaders) else None,
        lead_minutes_mean=float(np.mean([r['lead_minutes'] for r in leaders])) if leaders else None)
    return summary,dayrows,leaders


def indicators(data):
    c=data['close'];h=data['high'];l=data['low'];n=len(c)
    ema=c.copy()
    for i in range(1,n):ema[i]=(2/21)*c[i]+(19/21)*ema[i-1]
    tr=np.maximum(h-l,np.maximum(np.abs(h-np.r_[c[0],c[:-1]]),np.abs(l-np.r_[c[0],c[:-1]])))
    atr=np.full(n,np.nan)
    if n>=14:atr[13:]=np.convolve(tr,np.ones(14)/14,mode='valid')
    return dict(ema=ema,atr=atr)


def episode(record,data,feat,related_trades=None):
    t=record['trade'];entry=int(t['entry_ts']);price=float(t['entry_price']);exit_=int(t['exit_ts'])
    out={k:record[k] for k in ('day','symbol','entry_ts','entry_price')};out.update(exit_ts=exit_,exit_reason=t['exit_reason'],terminal_forced='open_at_end' in t['exit_reason'])
    i=int(np.searchsorted(data['time'],entry));j=i+96
    if i>=len(data['time']) or j>=len(data['time']) or not np.array_equal(data['time'][i:j+1],entry+np.arange(97)*BAR):
        return dict(out,state='UNKNOWN_FOLLOWUP')
    if not np.isfinite(feat['atr'][i]) or feat['atr'][i]<=0:return dict(out,state='UNKNOWN_ENTRY_ATR')
    closes=data['close'][i:j+1];peak=np.maximum.accumulate(np.r_[price,closes])[1:]
    tolerance=1e-12*price
    established=peak-price>=feat['atr'][i]-tolerance
    downward=np.r_[False,np.diff(feat['ema'][i:j+1]) < -tolerance]
    condition=established&(closes<=peak-2*feat['atr'][i:j+1]+tolerance)&downward
    confirmed=np.flatnonzero(condition[1:]&condition[:-1])+1
    state='CONFIRMED' if len(confirmed) else ('UPSWING_UNCONFIRMED' if established.any() else 'NO_ESTABLISHED_UPSWING')
    out.update(state=state,terminal_forced='open_at_end' in t['exit_reason'])
    # First reduction and later new-close-high: tail is separate from no breakout.
    first_exit=int(t['partial_exit_ts']) if t['partial_exit_taken'] else exit_
    k=int(np.searchsorted(data['time'],first_exit));later=k+16
    out['first_exit_ts']=first_exit
    if (k>=i and later<len(data['time']) and data['time'][k]==first_exit
        and np.array_equal(data['time'][k:later+1],first_exit+np.arange(17)*BAR)):
        past_peak=max(price,float(data['close'][i:k+1].max()))
        out['new_high_after_first_exit']=bool(data['close'][k+1:later+1].max()>past_peak)
    else:out['new_high_after_first_exit']=None
    if len(confirmed):
        at=i+int(confirmed[0]);marker=int(data['time'][at]);top=float(peak[int(confirmed[0])]);rise=top-price
        fraction=float(t['partial_exit_fraction']) if t['partial_exit_taken'] else 0.
        effective=(fraction*float(t['partial_exit_price'])+(1-fraction)*float(t['exit_price'])) if fraction else float(t['exit_price'])
        remaining=0. if exit_<=marker else (1-fraction if fraction and int(t['partial_exit_ts'])<=marker else 1.)
        out.update(marker_ts=marker,peak_close=top,retention=(effective-price)/rise,
            giveback_pp=100*(top-effective)/price,exit_before_marker=exit_<marker,
            first_exit_before_marker=first_exit<marker,delay_bars=(exit_-marker)/BAR,remaining_fraction_at_marker=remaining)
        out.update(accompaniment(related_trades or [t],entry,marker))
    return out


def accompaniment(trades,start,marker):
    seconds=0.;remaining=0.;intervals=[]
    for t in trades:
        a=max(start,t['entry_ts']);b=min(marker,t['exit_ts'])
        partial=t['partial_exit_taken'];p=t['partial_exit_ts'] if partial else t['exit_ts'];fraction=t['partial_exit_fraction'] if partial else 0.
        if b>a:
            seconds+=max(0,min(b,p)-a)+max(0,b-max(a,p))*(1-fraction)
            intervals.append((a,b))
        if t['entry_ts']<=marker<t['exit_ts']:remaining+=1-fraction if partial and p<=marker else 1.
    intervals.sort()
    if any(b[0]<a[1] for a,b in zip(intervals,intervals[1:])) or remaining>1+1e-12:raise ValueError('overlapping symbol accompaniment')
    return dict(accompaniment_fraction=seconds/(marker-start) if marker>start else None,
        any_remaining_at_marker=remaining>0,all_positions_remaining_fraction=remaining,accompanied_positions=len(intervals))


def exit_summary(records):
    valid=[r for r in records if r['state']=='CONFIRMED' and not r['terminal_forced']]
    rebound=[r for r in records if r.get('new_high_after_first_exit') is not None and not r.get('terminal_forced',False)]
    summary=dict(captured_pairs=len(records),states={s:sum(r['state']==s for r in records) for s in sorted({r['state'] for r in records})},
        confirmed_n=len(valid),forced_n=sum(r.get('terminal_forced',False) for r in records),
        early_exit_n=sum(r['exit_before_marker'] for r in valid),first_early_exit_n=sum(r['first_exit_before_marker'] for r in valid),
        retention_mean=float(np.mean([r['retention'] for r in valid])) if valid else None,
        giveback_mean_pp=float(np.mean([r['giveback_pp'] for r in valid])) if valid else None,
        delay_bars_median=float(np.median([r['delay_bars'] for r in valid])) if valid else None,
        remaining_mean=float(np.mean([r['remaining_fraction_at_marker'] for r in valid])) if valid else None,
        accompaniment_mean=float(np.mean([r['accompaniment_fraction'] for r in valid])) if valid else None,
        any_remaining_n=sum(r['any_remaining_at_marker'] for r in valid),
        rebound_n=sum(r['new_high_after_first_exit'] for r in rebound),rebound_N=len(rebound))
    return summary


def paired_intervals(control,candidate):
    by0={r['day']:r for r in control};by1={r['day']:r for r in candidate}
    if by0.keys()!=by1.keys():raise ValueError('different mission days')
    days=sorted(by0);n=len(days)
    def values(rows):
        return np.array([[r['early']/r['leaders'],r['captured']/r['leaders'],
            r['precision_n'],r['precision_N']] for r in rows],float)
    a=values([by0[d] for d in days]);b=values([by1[d] for d in days]);out={}
    if n<3:return dict(days=n,intervals=None)
    rng=np.random.default_rng(42);starts=rng.integers(0,n,(5000,int(np.ceil(n/3))))
    idx=((starts[:,:,None]+np.arange(3))%n).reshape(5000,-1)[:,:n]
    aa=a[idx];bb=b[idx]
    delta=bb[:,:,:2].mean(axis=1)-aa[:,:,:2].mean(axis=1)
    den0=aa[:,:,3].sum(axis=1);den1=bb[:,:,3].sum(axis=1)
    precision=np.divide(bb[:,:,2].sum(axis=1),den1,out=np.full(5000,np.nan),where=den1>0)-np.divide(aa[:,:,2].sum(axis=1),den0,out=np.full(5000,np.nan),where=den0>0)
    delta=np.column_stack([delta,precision])*100
    for j,k in enumerate(('early_pp','capture_pp','precision_pp')):
        v=delta[:,j];out[k]=np.quantile(v[np.isfinite(v)],[.05/48,1-.05/48]).tolist() if np.isfinite(v).any() else None
    return dict(days=n,intervals=out,confidence_familywise=.95,comparisons=24)


def compare_exits(base,target):
    a={(r['day'],r['symbol']):r for r in base};b={(r['day'],r['symbol']):r for r in target}
    common=sorted(a.keys()&b.keys());matched=[];different=0
    for pair in common:
        x,y=a[pair],b[pair]
        if x['entry_ts']!=y['entry_ts'] or x['entry_price']!=y['entry_price']:different+=1;continue
        if x['state']!='CONFIRMED' or y['state']!='CONFIRMED' or x['terminal_forced'] or y['terminal_forced']:continue
        if x['marker_ts']!=y['marker_ts']:raise ValueError('same-entry marker differs')
        matched.append(dict(day=pair[0],symbol=pair[1],retention_delta=y['retention']-x['retention'],
            first_early_delta=int(y['first_exit_before_marker'])-int(x['first_exit_before_marker'])))
    ci=None
    days=sorted({r['day'] for r in matched})
    if len(days)>=3:
        sums=np.array([sum(r['retention_delta'] for r in matched if r['day']==d) for d in days]);counts=np.array([sum(r['day']==d for r in matched) for d in days])
        rng=np.random.default_rng(42);starts=rng.integers(0,len(days),(5000,int(np.ceil(len(days)/3))))
        idx=((starts[:,:,None]+np.arange(3))%len(days)).reshape(5000,-1)[:,:len(days)]
        means=sums[idx].sum(axis=1)/counts[idx].sum(axis=1);ci=np.quantile(means,[.05/48,1-.05/48]).tolist()
    return dict(common_pairs=len(common),different_entry_pairs=different,confirmed_same_entry_n=len(matched),
        retention_delta_mean=float(np.mean([r['retention_delta'] for r in matched])) if matched else None,
        first_early_delta_n=sum(r['first_early_delta'] for r in matched),retention_interval=ci,days=len(days))


def verdict(base,target,comparison,exits):
    precision=lambda r:r['precision_n']/r['precision_N'] if r['precision_N'] else None
    p0,p1=precision(base),precision(target)
    if p0 is None or p1 is None:return 'INSUFFICIENT_SELECTION_DENOMINATOR'
    noninferior=target['early']>=base['early'] and target['captured']>=base['captured'] and p1>=p0
    if not noninferior:return 'MISSION_TRADEOFF_OR_WORSE'
    changed=target['early']>base['early'] or target['captured']>base['captured'] or p1>p0
    intervals=comparison['intervals']
    entry_robust=changed and comparison['days']>=30 and intervals is not None and any(v is not None and v[0]>0 for v in intervals.values())
    exit_robust=(exits['retention_delta_mean'] is not None and exits['retention_delta_mean']>0 and exits['first_early_delta_n']<=0
        and exits['days']>=30 and exits['retention_interval'] is not None and exits['retention_interval'][0]>0)
    if entry_robust or exit_robust:return 'RETROSPECTIVE_MISSION_GAIN_REQUIRES_FORWARD'
    if changed or exits['retention_delta_mean'] not in (None,0):return 'DESCRIPTIVE_CHANGE_NOT_CONFIRMED'
    return 'NO_MISSION_GAIN'
