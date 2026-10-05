"""Causal stationary book/OFI features, second-horizon targets and error metrics."""
from __future__ import annotations
import numpy as np

HORIZONS=(5,10,15,20,25,30)
REGISTERED=(5,10,30)
DIRECTION_EPS=1e-12  # Numerical zero, far below the instruments' price ticks.

def prefix_sum(value):return np.concatenate([np.zeros((1,)+value.shape[1:]),np.cumsum(value,axis=0)])
def sums(prefix,indices,width):return prefix[indices+1]-prefix[np.maximum(0,indices-width+1)]

def features(data,symbol_index):
    t,q,seg=data['time'],data['book'],data['segment'];n=len(t)
    indices=np.flatnonzero(t%5000==0);mid=(q[:,0]+q[:,2])/2
    bids=q[:,0::4];bqty=q[:,1::4];asks=q[:,2::4];aqty=q[:,3::4]
    total=np.nansum(bqty+aqty,axis=1);depth=prefix_sum(total)
    log=np.log(mid);r=np.r_[0,np.diff(log)]
    # Invalid rows are retained in the clock; only eligible prefixes are used.
    square=prefix_sum(np.nan_to_num(r*r));counts=prefix_sum(data['events'].astype(float))
    signed=prefix_sum(np.nan_to_num(data['flow']));absolute=prefix_sum(np.nan_to_num(np.abs(data['flow'])))
    selected=q[indices];p=mid[indices];current_depth=total[indices]
    with np.errstate(divide='ignore',invalid='ignore'):
        book=np.empty_like(selected)
        book[:,0::4]=10000*(selected[:,0::4]/p[:,None]-1)
        book[:,2::4]=10000*(selected[:,2::4]/p[:,None]-1)
        book[:,1::4]=selected[:,1::4]/current_depth[:,None]
        book[:,3::4]=selected[:,3::4]/current_depth[:,None]
        blocks=[book]
        for k in (1,3,5):
            b=bqty[indices,:k].sum(axis=1);a=aqty[indices,:k].sum(axis=1)
            blocks.append(((b-a)/(b+a))[:,None])
        micro=(q[indices,2]*q[indices,1]+q[indices,0]*q[indices,3])/(q[indices,1]+q[indices,3])
        blocks.append(np.column_stack([10000*np.log(micro/p),10000*(q[indices,2]-q[indices,0])/p,data['age'][indices]/1000]))
        for width in (1,5,15,60):
            ret=log[indices]-log[np.maximum(0,indices-width)]
            blocks.append(np.column_stack([ret*10000,np.sqrt(sums(square,indices,width))*10000,sums(counts,indices,width)/width]))
        hot=np.zeros((len(indices),3));hot[:,symbol_index]=1;blocks.append(hot)
        state_width=sum(b.shape[1] for b in blocks)
        for width in (1,5,15,60):
            normalizer=np.maximum(sums(depth,indices,width)/width,1e-12)
            blocks.append(np.column_stack([sums(signed,indices,width)/normalizer[:,None],
                sums(absolute,indices,width)/normalizer[:,None]]))
    x=np.concatenate(blocks,axis=1).astype(np.float32)
    bad=prefix_sum((seg<0).astype(int));changes=np.r_[0,np.cumsum(seg[1:]!=seg[:-1])]
    starts=np.maximum(0,indices-60)
    good=(indices>=60)&(seg[indices]>=0)&(bad[indices+1]-bad[starts]==0)&(changes[indices]==changes[starts])&np.isfinite(x).all(axis=1)
    return dict(indices=indices,time=t[indices],x=x,good=good,mid=mid[indices],micro_return=np.log(micro/p),state_width=state_width)

def targets(mid,segment):
    out=np.full((len(mid),len(HORIZONS)),np.nan);changes=np.r_[0,np.cumsum(segment[1:]!=segment[:-1])]
    for j,h in enumerate(HORIZONS):
        row=np.arange(max(0,len(mid)-h));future=row+h
        good=(segment[row]>=0)&(changes[row]==changes[future])&np.isfinite(mid[row])&np.isfinite(mid[future])&(mid[row]>0)&(mid[future]>0)
        out[row[good],j]=np.log(mid[future[good]]/mid[row[good]])
    return out

def masks(time,good,returns,cuts):
    mature=np.isfinite(returns).all(axis=1);purge=31_000;out={}
    for name,lo,hi in [('train',cuts['start'],cuts['validation']),('validation',cuts['validation'],cuts['calibration']),
                       ('calibration',cuts['calibration'],cuts['test']),('test',cuts['test'],cuts['end'])]:
        out[name]=good&mature&(time>=lo)&(time+purge<hi)
        if name=='train':out[name]&=time%30_000==0
    out['inference']=good&(time>=cuts['test'])&(time+purge<cuts['end'])
    return out

def quantiles(actual,prediction):
    residual=np.abs(actual-prediction)
    if residual.shape!=prediction.shape or not len(residual) or not np.isfinite(residual).all():raise ValueError('Invalid calibration residuals')
    rank=min(len(residual),int(np.ceil((len(residual)+1)*.9)))-1
    return np.partition(residual,rank,axis=0)[rank]

def metrics(actual,prediction,widths,prices=None):
    if actual.shape!=prediction.shape or not len(actual) or not np.isfinite(actual).all() or not np.isfinite(prediction).all():raise ValueError('Invalid matched cohort')
    rows=[]
    for j,h in enumerate(HORIZONS):
        a=actual[:,j];p=prediction[:,j];err=p-a
        sa=np.where(np.abs(a)<=DIRECTION_EPS,0,np.sign(a));sp=np.where(np.abs(p)<=DIRECTION_EPS,0,np.sign(p));nz=sa!=0
        signs=[int((sp<0).sum()),int((sp==0).sum()),int((sp>0).sum())]
        row=dict(horizon_sec=h,n=len(a),mae_bp=float(np.abs(err).mean()*10000),rmse_bp=float(np.sqrt(np.mean(err**2))*10000),
            direction_correct=int((sp==sa)[nz].sum()),direction_n=int(nz.sum()),sign_correct=int((sp==sa).sum()),sign_n=len(a),
            observed_majority_correct=int(max((sa[nz]<0).sum(),(sa[nz]>0).sum())),
            actual_sign_counts=[int((sa<0).sum()),int((sa==0).sum()),int((sa>0).sum())],predicted_sign_counts=signs,
            interval_covered=int((np.abs(err)<=widths[j]).sum()),interval_n=len(a),interval_width_bp=float(widths[j]*20000))
        if prices is not None:
            delta=prices*(np.exp(p)-np.exp(a));row.update(mae_USDT=float(np.abs(delta).mean()),rmse_USDT=float(np.sqrt(np.mean(delta**2))))
        rows.append(row)
    return rows

def paired_days(time,actual,predictions):
    day=time//86400000;report={}
    full=[d for d in np.unique(day) if time.min()<d*86400000 and time.max()>=(d+1)*86400000]
    for h in REGISTERED:
        j=HORIZONS.index(h);base=np.abs(predictions['CatBoost_OFI'][:,j]-actual[:,j])
        for control in ('CatBoost_Book','Ridge_OFI','Zero'):
            losses=np.abs(predictions[control][:,j]-actual[:,j])-base
            daily=np.array([losses[day==d].mean()*10000 for d in full])
            item=dict(n_days=len(daily),mean_mae_gain_bp=float(daily.mean()) if len(daily) else None,
                daily=[dict(day=int(d),gain_bp=float(v)) for d,v in zip(full,daily)],robust_claim_allowed=False)
            if len(daily)>=3:
                rng=np.random.default_rng(42);n=len(daily);starts=rng.integers(0,n,(5000,int(np.ceil(n/3))))
                idx=((starts[:,:,None]+np.arange(3))%n).reshape(5000,-1)[:,:n]
                lo,hi=np.quantile(daily[idx].mean(axis=1),[.05/18,1-.05/18]);item['familywise95_bp']=[float(lo),float(hi)]
                item['robust_claim_allowed']=bool(n>=30 and lo>0)
            report[f'{h}s OFI vs {control}']=item
    return report
