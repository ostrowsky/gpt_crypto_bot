"""Compact DeepLOB topology, causal inputs and separated probability calibration."""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.metrics import confusion_matrix,f1_score,balanced_accuracy_score

HORIZONS=(1,3,5)
SEQUENCE=100
MINUTE=60_000
DEADBAND=.0002

def softmax(logits):
    z=np.asarray(logits,dtype=float);z=z-z.max(axis=-1,keepdims=True)
    p=np.exp(z);return p/p.sum(axis=-1,keepdims=True)

def calibrate(probabilities,labels):
    p=np.asarray(probabilities);y=np.asarray(labels,dtype=int)
    if len(y)==0 or p.shape!=(len(y),3):raise ValueError('Invalid calibration cohort')
    z=np.log(np.maximum(p,1e-12))
    objective=lambda t:float(-np.log(np.maximum(softmax(z/t)[np.arange(len(y)),y],1e-12)).mean())
    result=minimize_scalar(objective,bounds=(.25,4),method='bounded')
    return float(result.x)

def temperature(p,t):return softmax(np.log(np.maximum(p,1e-12))/t)

def normalize_sequence(book):
    a=np.array(book,dtype=np.float32,copy=True)
    # Compute price ratios in float64 first to retain small BTC spreads.
    mid=(book[-1,0]+book[-1,2])/2
    if not np.isfinite(book).all() or mid<=0:raise ValueError('Invalid causal sequence')
    a[:,0::4]=10000*(book[:,0::4]/mid-1);a[:,2::4]=10000*(book[:,2::4]/mid-1)
    q=np.concatenate([book[:,1::4],book[:,3::4]],axis=1);scale=float(q.mean())
    if scale<=0:raise ValueError('Empty sequence depth')
    a[:,1::4]=np.log1p(book[:,1::4]/scale);a[:,3::4]=np.log1p(book[:,3::4]/scale)
    return a

def causal_features(time,book,flow,segment,trade_gap,symbol_index):
    """No future targets in this function; identical to prefix-only inference."""
    n=len(time);mid=(book[:,0]+book[:,2])/2
    bq=book[:,3::4];aq=book[:,1::4];total=bq.sum(axis=1)+aq.sum(axis=1)
    normalized=np.empty_like(book)
    normalized[:,0::4]=10000*(book[:,0::4]/mid[:,None]-1)
    normalized[:,2::4]=10000*(book[:,2::4]/mid[:,None]-1)
    normalized[:,1::4]=aq/total[:,None];normalized[:,3::4]=bq/total[:,None]
    features=[normalized]
    for k in (1,5,10):features.append(((bq[:,:k].sum(axis=1)-aq[:,:k].sum(axis=1))/(bq[:,:k].sum(axis=1)+aq[:,:k].sum(axis=1)))[:,None])
    micro=(book[:,0]*book[:,3]+book[:,2]*book[:,1])/(book[:,1]+book[:,3])
    features.append(np.column_stack([10000*(micro/mid-1),10000*(book[:,0]-book[:,2])/mid]))
    log=pd.Series(np.log(mid));one=log.diff()
    for k in (6,18,30,60,90):features.append((10000*log.diff(k)).to_numpy()[:,None])
    for k in (6,18,30,90):
        rv=(one*one).rolling(k,min_periods=k).sum().to_numpy()
        f=pd.DataFrame(flow).rolling(k,min_periods=k).sum().to_numpy()
        features.append(np.column_stack([np.sqrt(rv)*10000,np.log1p(f[:,0]),np.log1p(f[:,1]),
            np.divide(f[:,2],f[:,1],out=np.zeros(n),where=f[:,1]>0),
            np.divide(f[:,4],f[:,3],out=np.zeros(n),where=f[:,3]>0)]))
    onehot=np.zeros((n,3));onehot[:,symbol_index]=1;features.append(onehot)
    x=np.concatenate(features,axis=1).astype(np.float32)
    # A whole past sequence must be in the same valid session, with no trade gap.
    good=(segment>=0)&np.isfinite(x).all(axis=1)
    start=np.maximum(0,np.arange(n)-SEQUENCE+1)
    changes=np.r_[0,np.cumsum(segment[1:]!=segment[:-1])]
    gaps=np.r_[0,np.cumsum(trade_gap)]
    good&=(np.arange(n)>=SEQUENCE-1)&(changes==changes[start])&(gaps[1:]-gaps[start]==0)
    good&=time%MINUTE==0
    return x,good,mid

def targets(mid,segment):
    n=len(mid);returns=np.full((n,3),np.nan);labels=np.full((n,3),-1,dtype=np.int64)
    changes=np.r_[0,np.cumsum(segment[1:]!=segment[:-1])]
    for j,h in enumerate(HORIZONS):
        lag=h*6;row=np.arange(n-lag);future=row+lag
        ok=(segment[row]>=0)&(changes[row]==changes[future])&np.isfinite(mid[row])&np.isfinite(mid[future])
        r=np.log(mid[future[ok]]/mid[row[ok]]);returns[row[ok],j]=r
        labels[row[ok],j]=np.where(r>DEADBAND,2,np.where(r<-DEADBAND,0,1))
    return returns,labels

def boundaries(start,end):
    span=end-start
    cuts=[start]+[int((start+span*f)//MINUTE*MINUTE) for f in (.5,.65,.75)]+[end]
    if any(b<=a for a,b in zip(cuts,cuts[1:])):raise ValueError('Invalid clock span')
    return dict(zip(('start','validation','calibration','test','end'),cuts))

def split_masks(time,good,labels,cuts):
    eligible=good&(labels>=0).all(axis=1);purge=5*MINUTE+10_000
    masks={}
    for name,lo,hi in [('train',cuts['start'],cuts['validation']),('validation',cuts['validation'],cuts['calibration']),
                       ('calibration',cuts['calibration'],cuts['test']),('test',cuts['test'],cuts['end'])]:
        masks[name]=eligible&(time>=lo)&(time+purge<hi)
        if name=='train':masks[name]&=time%(5*MINUTE)==0
    masks['inference']=good&(time>=cuts['test'])&(time+purge<cuts['end'])
    return masks

def metrics(y,p,actual):
    y=np.asarray(y,dtype=int);p=np.asarray(p,dtype=float);actual=np.asarray(actual)
    if len(y)==0 or p.shape!=(len(y),3) or not np.isfinite(p).all() or (p<0).any() or not np.allclose(p.sum(axis=1),1):raise ValueError('Invalid score cohort')
    prediction=p.argmax(axis=1);counts=np.bincount(y,minlength=3)
    confidence=p.max(axis=1);correct=prediction==y;ece=0;bins=[]
    for i in range(10):
        mask=(confidence>=i/10)&(confidence<(i+1)/10 if i<9 else confidence<=1)
        n=int(mask.sum())
        if n:
            mean=float(confidence[mask].mean());accuracy=float(correct[mask].mean());ece+=n/len(y)*abs(mean-accuracy)
            bins.append(dict(bin=i,n=n,correct=int(correct[mask].sum()),mean_confidence=mean))
    nonneutral=y!=1;direction=(p[:,2]>p[:,0])==(actual>0)
    raw_nonzero=actual!=0
    return dict(n=len(y),correct=int(correct.sum()),accuracy=float(correct.mean()),
        class_counts=counts.tolist(),base_majority_correct=int(counts.max()),base_majority_accuracy=float(counts.max()/len(y)),
        confusion=confusion_matrix(y,prediction,labels=[0,1,2]).tolist(),
        macro_f1=float(f1_score(y,prediction,labels=[0,1,2],average='macro',zero_division=0)),
        balanced_accuracy=float(balanced_accuracy_score(y,prediction)),
        log_loss=float(-np.log(np.maximum(p[np.arange(len(y)),y],1e-12)).mean()),
        brier=float(np.mean(np.sum((p-np.eye(3)[y])**2,axis=1))),ece=float(ece),calibration_bins=bins,
        nonneutral_n=int(nonneutral.sum()),nonneutral_correct=int(direction[nonneutral].sum()),
        nonneutral_majority_correct=int(max((y==0).sum(),(y==2).sum())),
        raw_direction_n=int(raw_nonzero.sum()),raw_direction_correct=int(direction[raw_nonzero].sum()),
        raw_direction_majority_correct=int(max((actual<0).sum(),(actual>0).sum())))

def loss_intervals(time,y,probabilities):
    days=time//86_400_000;output={}
    losses={name:-np.log(np.maximum(p[np.arange(len(y)),y],1e-12)) for name,p in probabilities.items()}
    for a,b in [('CatBoost','Logistic'),('DeepLOB','CatBoost'),('DeepLOB','Logistic')]:
        if a not in losses or b not in losses:continue
        # First/last partial UTC days are descriptive only.
        complete=[day for day in np.unique(days) if time.min()<day*86_400_000 and time.max()>=(day+1)*86_400_000]
        diff=np.array([np.mean((losses[b]-losses[a])[days==day]) for day in complete])
        result=dict(n_days=len(diff),mean_loss_gain=float(diff.mean()) if len(diff) else None)
        if len(diff)>=3:
            rng=np.random.default_rng(42);n=len(diff);starts=rng.integers(0,n,(5000,int(np.ceil(n/3))))
            indices=((starts[:,:,None]+np.arange(3))%n).reshape(5000,-1)[:,:n]
            samples=diff[indices].mean(axis=1)
            lo,hi=np.quantile(samples,[.05/18,1-.05/18]);result['familywise95']=[float(lo),float(hi)]
        result['robust_claim_allowed']=bool(len(diff)>=30 and result.get('familywise95',[0])[0]>0)
        output[a+' vs '+b]=result
    return output

def build_deeplob():
    import torch
    from torch import nn
    class DeepLOB(nn.Module):
        def __init__(self):
            super().__init__();c=16
            def temporal():return [nn.Conv2d(c,c,(3,1),padding=(1,0)),nn.LeakyReLU(.01),nn.Conv2d(c,c,(3,1),padding=(1,0)),nn.LeakyReLU(.01)]
            self.spatial=nn.Sequential(nn.Conv2d(1,c,(1,2),stride=(1,2)),nn.LeakyReLU(.01),*temporal(),
                nn.Conv2d(c,c,(1,2),stride=(1,2)),nn.LeakyReLU(.01),*temporal(),
                nn.Conv2d(c,c,(1,10)),nn.LeakyReLU(.01),*temporal())
            self.branch3=nn.Sequential(nn.Conv2d(c,16,1),nn.LeakyReLU(.01),nn.Conv2d(16,16,(3,1),padding=(1,0)),nn.LeakyReLU(.01))
            self.branch5=nn.Sequential(nn.Conv2d(c,16,1),nn.LeakyReLU(.01),nn.Conv2d(16,16,(5,1),padding=(2,0)),nn.LeakyReLU(.01))
            self.branchpool=nn.Sequential(nn.MaxPool2d((3,1),stride=1,padding=(1,0)),nn.Conv2d(c,16,1),nn.LeakyReLU(.01))
            self.lstm=nn.LSTM(48,32,batch_first=True)
            self.heads=nn.ModuleList([nn.Linear(32,3) for _ in HORIZONS])
        def forward(self,x):
            z=self.spatial(x[:,None]);z=torch.cat([self.branch3(z),self.branch5(z),self.branchpool(z)],dim=1)
            z=z.squeeze(-1).transpose(1,2);z,_=self.lstm(z)
            return torch.stack([head(z[:,-1]) for head in self.heads],dim=1)
    return DeepLOB()
