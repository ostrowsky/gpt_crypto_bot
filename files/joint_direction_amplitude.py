"""Causal probability/conditional-magnitude mixture, research only."""
from datetime import datetime,time,timedelta,timezone
import numpy as np
from catboost import CatBoostClassifier,CatBoostRegressor
from sklearn.linear_model import LogisticRegression
from capacity_catboost import TZ,BAR,HORIZON
from impulse_entry_catboost import PARAMETERS,DAY


def net_from_gross(gross,fee,slip):
    if not 0<=fee<10000 or not 0<=slip<10000:raise ValueError('invalid costs')
    f,s=fee/10000,slip/10000
    return 100*((1+np.asarray(gross)/100)*(1-f)*(1-s)/((1+f)*(1+s))-1)


def cohorts(clocks,valid,at,start):
    first=datetime.fromtimestamp(start/1000,timezone.utc).astimezone(TZ).date()
    last=datetime.fromtimestamp(at/1000,timezone.utc).astimezone(TZ).date()
    cuts=[int(datetime.combine(first+timedelta(days=int((last-first).days*q)),time(),tzinfo=TZ).timestamp()*1000) for q in (.6,.8)]
    available=clocks+HORIZON
    train=valid&(clocks<cuts[0])&(available<cuts[0])
    val=valid&(clocks>=cuts[0])&(clocks<cuts[1])&(available<cuts[1])
    cal=valid&(clocks>=cuts[1])&(available<at)
    return train,val,cal,cuts


def enough(gross,train,val,cal):
    up=gross>0
    return bool(train.sum()>=500 and val.sum()>=100 and cal.sum()>=100 and
                all((train&(up==side)).sum()>=200 and (val&(up==side)).sum()>=50 for side in (False,True)) and
                len(np.unique(up[cal]))==2)


def logits(p):
    p=np.clip(np.asarray(p),1e-6,1-1e-6)
    return np.log(p/(1-p))


def calibrated(p,coef,intercept):
    from scipy.special import expit
    return expit(float(coef)*logits(p)+float(intercept))


def mixture(p,up,down,fee,slip):
    p=np.asarray(p);up=np.maximum(0,np.asarray(up));down=np.maximum(0,np.asarray(down))
    if np.any((p<0)|(p>1)):raise ValueError('invalid probability')
    return net_from_gross(p*up-(1-p)*down,fee,slip)


def fit_models(x,g,train,val,cal):
    direction=CatBoostClassifier(**dict(PARAMETERS,loss_function='Logloss'))
    direction.fit(x[train],(g[train]>0).astype(int),eval_set=(x[val],(g[val]>0).astype(int)),
                  early_stopping_rounds=40,use_best_model=True)
    models={'direction':direction}
    for name,side in (('up',True),('down',False)):
        t=train&((g>0)==side);v=val&((g>0)==side)
        model=CatBoostRegressor(**PARAMETERS)
        model.fit(x[t],abs(g[t]),eval_set=(x[v],abs(g[v])),early_stopping_rounds=40,use_best_model=True)
        models[name]=model
    raw=direction.predict_proba(x[cal])[:,1]
    platt=LogisticRegression(C=1.,solver='lbfgs',max_iter=1000)
    platt.fit(logits(raw)[:,None],(g[cal]>0).astype(int))
    return models,float(platt.coef_[0,0]),float(platt.intercept_[0])


def predict(models,x,coef,intercept,fee,slip):
    raw=models['direction'].predict_proba(x)[:,1]
    p=calibrated(raw,coef,intercept)
    up=np.maximum(0,models['up'].predict(x));down=np.maximum(0,models['down'].predict(x))
    return raw,p,up,down,mixture(p,up,down,fee,slip)


def probability_metrics(prob,truth):
    prob=np.asarray(prob,float);truth=np.asarray(truth,bool)
    if not len(prob) or len(prob)!=len(truth) or not np.isfinite(prob).all() or np.any((prob<0)|(prob>1)):
        raise ValueError('finite known probability cohort required')
    bins=[];ece=0.;index=np.minimum((prob*10).astype(int),9)
    for i in range(10):
        mask=index==i;n=int(mask.sum());positives=int(truth[mask].sum())
        mean=float(prob[mask].mean()) if n else None;rate=positives/n if n else None
        bins.append({'bin':i,'n':n,'up_n':positives,'mean_probability':mean,'observed_up_rate':rate})
        if n:ece+=n/len(prob)*abs(mean-rate)
    clipped=np.clip(prob,1e-6,1-1e-6)
    return {'n':len(prob),'up_n':int(truth.sum()),'direction_correct_n':int(((prob>.5)==truth).sum()),
            'brier':float(np.mean((prob-truth)**2)),
            'logloss':float(np.mean(np.where(truth,-np.log(clipped),-np.log1p(-clipped)))),
            'ece10':float(ece),'bins':bins}


def diagnostics(z,cut,fee,slip):
    results={}
    for name,mask in (('all_oos',z['fold']>=0),('test',(z['fold']>=0)&(z['clock']>=cut))):
        known=mask & np.isfinite(z['gross']) & np.isfinite(z['prediction'])
        y=z['y'][known];g=z['gross'][known];p=z['prediction'][known];accepted=p>0
        results[name]={'issued_n':int(mask.sum()),'known_n':int(known.sum()),'unknown_future_or_feature_n':int((mask&~known).sum()),
            'raw_probability':probability_metrics(z['raw_p'][known],g>0),
            'calibrated_probability':probability_metrics(z['p'][known],g>0),
            'climatology':probability_metrics(z['climatology'][known],g>0),
            'net_mae':float(np.mean(abs(p-y))),'net_rmse':float(np.sqrt(np.mean((p-y)**2))),
            'zero_net_mae':float(np.mean(abs(y))),
            'flat_price_net_mae':float(np.mean(abs(net_from_gross(0,fee,slip)-y))),
            'train_mean_net_mae':float(np.mean(abs(z['train_mean_net'][known]-y))),
            'accepted_known_n':int(accepted.sum()),
            'accepted_realized_mean_net_pct':float(y[accepted].mean()) if accepted.any() else None}
    return results
