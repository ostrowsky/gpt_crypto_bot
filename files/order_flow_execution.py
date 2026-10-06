"""Research-only bounded execution; no live dependencies or order submission."""
from __future__ import annotations
import numpy as np

HORIZONS=(5,10,30)
FEE=.00075
THRESHOLD_BP=1.


def decide(predicted_return,side,horizon):
    if side not in (1,-1) or horizon not in HORIZONS:
        raise ValueError('invalid execution side/horizon')
    p=np.asarray(predicted_return,float)
    if not np.isfinite(p).all():raise ValueError('unavailable forecast')
    return np.where(side*p*10000 < -THRESHOLD_BP,horizon,0)


def walk(book,quantity,side):
    """Full fixed-base-quantity five-rank VWAP; inadequate depth remains NaN."""
    q=np.asarray(book,float);amount=np.broadcast_to(np.asarray(quantity,float),q.shape[:-1])
    if q.shape[-1]!=20 or side not in (1,-1):raise ValueError('bad book/side')
    prices=q[...,2::4] if side==1 else q[...,0::4]
    sizes=q[...,3::4] if side==1 else q[...,1::4]
    prior=np.cumsum(sizes,axis=-1)-sizes
    taken=np.minimum(np.maximum(amount[...,None]-prior,0),sizes)
    good=np.isfinite(q).all(axis=-1)&(q>0).all(axis=-1)&np.isfinite(amount)&(amount>0)
    good&=(np.diff(q[...,0::4],axis=-1)<0).all(axis=-1)&(np.diff(q[...,2::4],axis=-1)>0).all(axis=-1)&(q[...,0]<q[...,2])
    good&=sizes.sum(axis=-1)>=amount*(1-1e-12)
    with np.errstate(invalid='ignore',divide='ignore'):
        price=(prices*taken).sum(axis=-1)/amount
    return np.where(good,price*(1+side*FEE),np.nan)


def arrival(data,clocks,quantity,side,delay):
    t=np.asarray(clocks,np.int64);target=t+delay*1000
    origin=np.searchsorted(data['time'],t);idx=np.searchsorted(data['time'],target)
    safe=np.minimum(idx,len(data['time'])-1);start=np.minimum(origin,len(data['time'])-1)
    seg=data['segment'];changes=np.r_[0,np.cumsum(seg[1:]!=seg[:-1])]
    good=(origin<len(seg))&(idx<len(seg))&(data['time'][start]==t)&(data['time'][safe]==target)
    good&=(seg[start]>=0)&(seg[safe]>=0)&(changes[safe]==changes[start])&(data['age'][safe]<=250)
    result=walk(data['book'][safe],quantity,side)
    return np.where(good,result,np.nan)


def metrics(time,mid,immediate,wait,selected,side_signs=(1,-1)):
    """Matrices have BUY then SELL rows; equal weights are diagnostic only."""
    mid=np.asarray(mid,float)[:,None];base=np.asarray(immediate,float);future=np.asarray(wait,float)
    choose=np.asarray(selected,bool);side=np.array(side_signs)[None,:]
    if base.shape!=future.shape or base.shape!=choose.shape or base.shape!=(len(time),len(side_signs)):raise ValueError('shape')
    action=np.where(choose,future,base);known=np.isfinite(action)
    matched=np.isfinite(base)&np.isfinite(future)
    gain=side*(base-action)/mid*10000;shortfall=side*(action/mid-1)*10000
    values=gain[matched];loss=shortfall[matched];chosen=matched&choose
    days=np.asarray(time)//86400000
    interior=[int(d) for d in np.unique(days) if time.min()<d*86400000 and time.max()>=(d+1)*86400000]
    daily=[]
    for d in interior:
        take=matched&(days==d)[:,None]
        if take.any():daily.append({'day':d,'gain_bp':float(gain[take].mean()),'n':int(take.sum())})
    return dict(issued=int(choose.size),deferred=int(choose.sum()),action_known=int(known.sum()),
        action_unknown=int((~known).sum()),matched=int(matched.sum()),matched_deferred=int(chosen.sum()),
        harmed=int((matched&(gain < -1e-10)).sum()),benefited=int((matched&(gain > 1e-10)).sum()),
        mean_gain_bp=float(values.mean()) if len(values) else None,
        selected_mean_gain_bp=float(gain[chosen].mean()) if chosen.any() else None,
        mean_shortfall_bp=float(loss.mean()) if len(loss) else None,
        p95_shortfall_bp=float(np.quantile(loss,.95)) if len(loss) else None,
        p99_shortfall_bp=float(np.quantile(loss,.99)) if len(loss) else None,
        daily=daily,calendar_interior_days=len(interior),fully_observed_candidate_days=0,
        scope='matched synthetic orders, source gaps excluded; not fills or portfolio alpha')


def interval(daily):
    a=np.array(daily,float);n=len(a)
    if n<3:return None
    rng=np.random.default_rng(42);starts=rng.integers(0,n,(5000,int(np.ceil(n/3))))
    idx=((starts[:,:,None]+np.arange(3))%n).reshape(5000,-1)[:,:n]
    return np.quantile(a[idx].mean(axis=1),[.05/18,1-.05/18]).tolist()


def extract_orders(trades,ledger):
    if len(trades)!=len(ledger):raise ValueError('allocation count')
    out=[]
    for i,(t,l) in enumerate(zip(trades,ledger)):
        if l['trade_id']!=i or l['symbol']!=t['sym'] or l['entry_ts']!=t['entry_ts'] or l['exit_ts']!=t['exit_ts']:
            raise ValueError('allocation provenance')
        qty=l['budget']/((1+FEE)*t['entry_price']*1.0005)
        if not np.isfinite(qty) or qty<=0:raise ValueError('allocation quantity')
        actions=[('BUY',t['entry_ts'],qty,False)]
        partial=t.get('partial_exit_taken',False)
        if partial:
            fraction=t['partial_exit_fraction']
            if not 0<fraction<1:raise ValueError('partial')
            actions.append(('SELL_PARTIAL',t['partial_exit_ts'],qty*fraction,True));qty*=1-fraction
        actions.append(('SELL',t['exit_ts'],qty,True))
        for kind,clock,size,sell in actions:
            out.append(dict(trade_id=i,symbol=t['sym'],time=int(clock),quantity=float(size),kind=kind,
                side=-1 if sell else 1,protected=sell and (kind=='SELL_PARTIAL' or 'WEAK:' not in t['exit_reason'])))
    return out
