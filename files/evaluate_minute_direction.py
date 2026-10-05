"""Registered CatBoost then compact DeepLOB on genuine public crypto L2 data."""
from __future__ import annotations
import argparse,copy,json,random,shutil,time,warnings
from pathlib import Path
import numpy as np
from minute_direction_data import sha,SYMBOLS,STEP
import minute_direction_models as m

def load_frames(folder):
    coverage=json.loads((folder/'coverage.json').read_bytes());frames={}
    for si,symbol in enumerate(SYMBOLS):
        path=folder/(symbol+'.npz')
        if sha(path)!=coverage['assets'][symbol]['sha256']:raise ValueError('Prepared book drift')
        with np.load(path,allow_pickle=False) as f:a={k:f[k] for k in f.files}
        if not np.array_equal(np.diff(a['time']),np.full(len(a['time'])-1,STEP)):raise ValueError('Irregular prepared clock')
        x,good,mid=m.causal_features(a['time'],a['book'],a['flow'],a['segment'],a['trade_gap'],si)
        returns,labels=m.targets(mid,a['segment']);a.update(x=x,good=good,mid=mid,returns=returns,labels=labels)
        frames[symbol]=a
    start=max(a['time'][0] for a in frames.values());end=min(a['time'][-1] for a in frames.values())
    cuts=m.boundaries(int(start),int(end))
    for a in frames.values():a['masks']=m.split_masks(a['time'],a['good'],a['labels'],cuts)
    return frames,cuts,coverage

def cohort(frames,split):
    rows=[(symbol,i) for symbol,a in frames.items() for i in np.flatnonzero(a['masks'][split])]
    rows.sort(key=lambda r:(frames[r[0]]['time'][r[1]],r[0]))
    if not rows:raise ValueError('No eligible '+split+' rows')
    return rows

def matrix(frames,rows,key):return np.array([frames[s][key][i] for s,i in rows])

def prior_probabilities(y,n):
    counts=np.bincount(y,minlength=3)+1
    return np.tile(counts/counts.sum(),(n,1))

def momentum_probabilities(xtrain,y,x):
    # First return lag follows normalized 40 columns + three imbalances + micro/spread.
    direction=np.sign(xtrain[:,45]);future=np.sign(x[:,45]);p=np.empty((len(x),3))
    for sign in (-1,0,1):p[future==sign]=prior_probabilities(y[direction==sign],int((future==sign).sum()))
    return p

def fit_logistic(x,y):
    from sklearn.linear_model import LogisticRegression
    from sklearn.exceptions import ConvergenceWarning
    with warnings.catch_warnings():
        warnings.simplefilter('error',ConvergenceWarning)
        return LogisticRegression(C=1,max_iter=5000,random_state=42).fit(x,y)

def fit_catboost(frames,cohorts,out):
    from catboost import CatBoostClassifier
    from sklearn.preprocessing import StandardScaler
    from joblib import dump
    x={k:matrix(frames,r,'x') for k,r in cohorts.items()};y={k:matrix(frames,r,'labels') for k,r in cohorts.items()}
    predictions={name:np.zeros((len(cohorts['inference']),3,3)) for name in ('CatBoost','Logistic','Prior','Momentum')}
    scaler=StandardScaler().fit(x['train']);xt=scaler.transform(x['train'])
    report=[]
    for j,h in enumerate(m.HORIZONS):
        if set(y['train'][:,j])!={0,1,2}:raise ValueError('Training misses a label class')
        cat=CatBoostClassifier(iterations=600,depth=6,learning_rate=.05,l2_leaf_reg=10,loss_function='MultiClass',
            random_seed=42,thread_count=2,allow_writing_files=False,verbose=False)
        cat.fit(x['train'],y['train'][:,j],eval_set=(x['validation'],y['validation'][:,j]),early_stopping_rounds=60,use_best_model=True)
        cat.save_model(str(out/f'catboost_h{h}.cbm'))
        raw_cal=cat.predict_proba(x['calibration']);t=m.calibrate(raw_cal,y['calibration'][:,j])
        predictions['CatBoost'][:,j]=m.temperature(cat.predict_proba(x['inference']),t)
        logit=fit_logistic(xt,y['train'][:,j])
        lt=m.calibrate(logit.predict_proba(scaler.transform(x['calibration'])),y['calibration'][:,j])
        predictions['Logistic'][:,j]=m.temperature(logit.predict_proba(scaler.transform(x['inference'])),lt)
        predictions['Prior'][:,j]=prior_probabilities(y['train'][:,j],len(cohorts['inference']))
        predictions['Momentum'][:,j]=momentum_probabilities(x['train'],y['train'][:,j],x['inference'])
        dump(dict(scaler=scaler,model=logit,temperature=lt),out/f'logistic_h{h}.joblib')
        report.append(dict(horizon=h,iterations=cat.tree_count_,temperature=t,logistic_temperature=lt,logistic_iterations=int(logit.n_iter_.max()),
            validation_logloss=float(cat.best_score_['validation']['MultiClass']),train_n=len(y['train']),
            validation_n=len(y['validation']),calibration_n=len(y['calibration'])))
        print('CatBoost frozen',report[-1],flush=True)
    return predictions,report

def deep_probabilities(model,frames,rows,batch_size=128):
    import torch
    output=[];model.eval()
    with torch.no_grad():
        for offset in range(0,len(rows),batch_size):
            block=rows[offset:offset+batch_size]
            x=np.stack([m.normalize_sequence(frames[s]['book'][i-m.SEQUENCE+1:i+1]) for s,i in block])
            output.append(torch.softmax(model(torch.from_numpy(x)),dim=-1).numpy())
    return np.concatenate(output)

def fit_deeplob(frames,cohorts,out):
    import torch
    from torch.utils.data import Dataset,DataLoader
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.manual_seed(42);random.seed(42);np.random.seed(42)
    torch.use_deterministic_algorithms(True)
    class SequenceDataset(Dataset):
        def __init__(self,rows):self.rows=rows
        def __len__(self):return len(self.rows)
        def __getitem__(self,k):
            s,i=self.rows[k];a=frames[s]
            return torch.from_numpy(m.normalize_sequence(a['book'][i-m.SEQUENCE+1:i+1])),torch.from_numpy(a['labels'][i])
    loader=DataLoader(SequenceDataset(cohorts['train']),batch_size=128,shuffle=True,num_workers=0,
                      generator=torch.Generator().manual_seed(42))
    model=m.build_deeplob();optimizer=torch.optim.Adam(model.parameters(),lr=.001,weight_decay=.0001)
    best=float('inf');best_state=None;patience=0;history=[];best_epoch=None
    yval=matrix(frames,cohorts['validation'],'labels')
    for epoch in range(1,9):
        model.train();total=0.;n=0;started=time.monotonic()
        for x,y in loader:
            optimizer.zero_grad();logits=model(x)
            loss=torch.nn.functional.cross_entropy(logits.reshape(-1,3),y.reshape(-1))
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);optimizer.step()
            total+=float(loss.detach())*len(x);n+=len(x)
        p=deep_probabilities(model,frames,cohorts['validation'])
        loss=float(-np.log(np.maximum(np.take_along_axis(p,yval[:,:,None],axis=2),1e-12)).mean())
        history.append(dict(epoch=epoch,train_loss=total/n,validation_loss=loss,seconds=time.monotonic()-started))
        print('DeepLOB epoch',history[-1],flush=True)
        if loss<best:
            best=loss;best_state=copy.deepcopy(model.state_dict());best_epoch=epoch;patience=0
        else:patience+=1
        if patience>=2:break
    model.load_state_dict(best_state);model.eval()
    from safetensors.torch import save_file
    save_file({k:v.contiguous() for k,v in best_state.items()},str(out/'deeplob.safetensors'))
    ycal=matrix(frames,cohorts['calibration'],'labels');pcal=deep_probabilities(model,frames,cohorts['calibration'])
    raw=deep_probabilities(model,frames,cohorts['inference']);temps=[]
    for j,h in enumerate(m.HORIZONS):
        t=m.calibrate(pcal[:,j],ycal[:,j]);raw[:,j]=m.temperature(raw[:,j],t);temps.append(t)
    return raw,dict(best_epoch=best_epoch,history=history,temperature=temps,train_n=len(cohorts['train']),
        validation_n=len(cohorts['validation']),calibration_n=len(cohorts['calibration']),
        architecture='DeepLOB compact: spatial CNN + temporal CNN + 3 Inception branches + LSTM + 3 heads')

def account(frames,rows,probabilities,horizon,cuts,fee_bps=7.5,slip_bps=5):
    """Long-only portfolio diagnostic, all inference origins including future gaps."""
    fee,slip=fee_bps/10000,slip_bps/10000;cash=1.;positions={};last_bid={};curve=[];exposure=[]
    signals={};trades=[];costs=0.;funding_paid=0.;missing=0;opportunities=0;max_positions=0;delayed_exits=0;unfilled_signals=0
    settlements={}
    for s,a in frames.items():
        for stamp,rate,mark in a.get('funding',[]):
            at=int(np.ceil(stamp/STEP))*STEP
            if cuts['test']<=at<=cuts['end']:settlements.setdefault(at,[]).append((s,float(rate),float(mark)))
    for (s,i),p in zip(rows,probabilities):
        if p[2]>=.5 and p[2]>p[0]:signals.setdefault(int(frames[s]['time'][i])+STEP,[]).append((s,int(frames[s]['time'][i])))
    indices={s:int(np.searchsorted(a['time'],cuts['test'])) for s,a in frames.items()}
    def quote(s,at):
        a=frames[s];i=indices[s]
        if i>=len(a['time']) or int(a['time'][i])!=at or a['segment'][i]<0:return None
        bid,ask=float(a['book'][i,2]),float(a['book'][i,0])
        if not np.isfinite([bid,ask]).all() or bid>=ask:return None
        return bid,ask
    def equity():return cash+sum(v['qty']*last_bid[s]*(1-slip)*(1-fee)-v['funding'] for s,v in positions.items())
    for at in range(cuts['test'],cuts['end']+1,STEP):
        quotes={s:quote(s,at) for s in frames}
        for s,q in quotes.items():
            if q is not None:last_bid[s]=q[0]
        for s,rate,mark in settlements.get(at,[]):
            if s in positions:
                charge=positions[s]['qty']*mark*rate
                positions[s]['funding']+=charge;funding_paid+=charge
        for s in list(positions):
            v=positions[s]
            if at>=v['due']:
                if quotes[s] is None:delayed_exits+=1;continue
                bid=quotes[s][0];gross=v['qty']*bid*(1-slip);proceeds=gross*(1-fee)-v['funding'];cash+=proceeds
                costs+=gross*fee+v['qty']*bid*slip
                trades.append(dict(symbol=s,origin=v['origin'],entry=v['entry'],exit=at,due=v['due'],
                    budget=v['budget'],proceeds=proceeds,funding=v['funding'],return_pct=100*(proceeds/v['budget']-1)))
                del positions[s]
        for s,origin in signals.get(at,[]):
            if s in positions:continue
            if quotes[s] is None:unfilled_signals+=1;continue
            budget=min(cash,equity()/3)
            if budget<=0:raise ValueError('Borrowing/cash exhausted')
            ask=quotes[s][1];notional=budget/(1+fee);qty=notional/(ask*(1+slip));cash-=budget
            costs+=notional*fee+qty*ask*slip
            positions[s]=dict(qty=qty,due=origin+horizon*m.MINUTE+STEP,origin=origin,entry=at,budget=budget,funding=0.)
            max_positions=max(max_positions,len(positions))
        for s in positions:
            opportunities+=1;missing+=int(quotes[s] is None)
        value=equity()
        if not np.isfinite(value) or value<=0 or cash<-1e-10 or len(positions)>3:raise ValueError('Invalid unified account')
        curve.append([at,value]);exposure.append(sum(v['qty']*last_bid[s] for s,v in positions.items())/value)
        for s in indices:indices[s]+=1
    if positions:raise ValueError('No observed final liquidation quote')
    a=np.array(curve);peak=np.maximum(1,np.maximum.accumulate(a[:,1]))
    return dict(return_pct=100*(cash-1),max_drawdown_pct=100*np.max(1-a[:,1]/peak),
        mean_exposure_pct=100*float(np.mean(exposure)),trades=len(trades),costs_initial_capital=costs,funding_paid_initial_capital=funding_paid,
        max_positions=max_positions,missing_holding_marks=missing,holding_mark_opportunities=opportunities,
        drawdown_fully_observed=missing==0,delayed_exit_grid_points=delayed_exits,unfilled_signals=unfilled_signals,
        curve=a,trades_detail=trades)

def btc_benchmark(frames,cuts):
    a=frames['BTCUSDT'];good=(a['time']>=cuts['test']+STEP)&(a['time']<=cuts['end'])&(a['segment']>=0)
    idx=np.flatnonzero(good)
    if not len(idx):raise ValueError('Missing BTC benchmark')
    start,end=idx[0],idx[-1];fee=.00075;slip=.0005
    value=a['book'][end,2]*(1-slip)*(1-fee)/(a['book'][start,0]*(1+slip)*(1+fee))
    funding=a.get('funding',np.empty((0,3)));mask=(funding[:,0]>a['time'][start])&(funding[:,0]<=a['time'][end])
    funding_fraction=float(np.sum(funding[mask,1]*funding[mask,2])/(a['book'][start,0]*(1+slip)*(1+fee)))
    value-=funding_fraction
    return dict(return_pct=float(100*(value-1)),start=int(a['time'][start]),end=int(a['time'][end]),
                funding_pct=100*funding_fraction,complete=bool(a['time'][end]==cuts['end']))

def evaluate(frames,cohorts,predictions,cuts,out,metadata,coverage):
    inference=cohorts['inference'];test=set(cohorts['test']);select=np.array([r in test for r in inference])
    y=matrix(frames,inference,'labels');actual=matrix(frames,inference,'returns');clock=np.array([frames[s]['time'][i] for s,i in inference])
    symbols=np.array([s for s,i in inference]);result=dict(status='COMPLETED_RETROSPECTIVE_L2_DIAGNOSTIC',runtime_eligible=False,
        cuts=cuts,coverage=coverage,metadata=metadata,cohort_counts={k:len(v) for k,v in cohorts.items()},
        benchmark=btc_benchmark(frames,cuts),metrics={},paired_losses={},accounts={},limitations=[
            'March/April public BTC/ETH/SOL perpetual top-20 states only; not current spot bot candidate population',
            'archive lacks independent receipt times; event-time availability and idealized quoted fills',
            'retrospective registered holdout; not independent forward/shadow proof',
            'neutral band 2bp is not a fee break-even threshold; economic results include positive costs',
            'drawdown may include stale marks, explicitly counted; missing intragap risk is unobserved',
            'full Truth Harness FAIL TH-11 remains separate'])
    np.savez_compressed(out/'test_predictions.npz',time=clock,symbol=symbols,labels=y,returns=actual,
        scored=select,**{name:p for name,p in predictions.items()})
    for j,h in enumerate(m.HORIZONS):
        result['metrics'][str(h)]={}
        for name,p in predictions.items():
            by_asset={s:m.metrics(y[(symbols==s)&select,j],p[(symbols==s)&select,j],actual[(symbols==s)&select,j]) for s in frames}
            result['metrics'][str(h)][name]=dict(pooled=m.metrics(y[select,j],p[select,j],actual[select,j]),assets=by_asset)
            row=account(frames,inference,p[:,j],h,cuts);stress=account(frames,inference,p[:,j],h,cuts,15,10)
            np.savez_compressed(out/f'curve_{name}_{h}.npz',curve=row.pop('curve'),stress_curve=stress.pop('curve'))
            (out/f'trades_{name}_{h}.json').write_text(json.dumps(row.pop('trades_detail')),encoding='utf-8');stress.pop('trades_detail')
            result['accounts'].setdefault(str(h),{})[name]=dict(**row,double_cost_return_pct=stress['return_pct'])
        result['paired_losses'][str(h)]=m.loss_intervals(clock[select],y[select,j],{name:p[select,j] for name,p in predictions.items()})
    return result

def run(args):
    args.output.mkdir(parents=True,exist_ok=False);snapshot=args.output/'source_snapshot';snapshot.mkdir()
    sources={}
    for name in ('minute_direction_data.py','minute_direction_partial.py','minute_direction_models.py','evaluate_minute_direction.py'):
        path=Path(__file__).with_name(name);sources[name]=sha(path);shutil.copy2(path,snapshot/name)
    frames,cuts,coverage=load_frames(args.books);cohorts={name:cohort(frames,name) for name in ('train','validation','calibration','test','inference')}
    import importlib.metadata
    versions={name:importlib.metadata.version(name) for name in ('catboost','torch','numpy','pandas','scipy','scikit-learn','pyarrow')}
    registration=dict(source_hashes=sources,cuts=cuts,coverage_sha256=sha(args.books/'coverage.json'),versions=versions,
        registered_at=time.time(),seed=42,cohort_counts={k:len(v) for k,v in cohorts.items()},parameters=dict(
            catboost=dict(depth=6,lr=.05,l2=10,max_iterations=600,early_stopping=60),
            logistic=dict(C=1,max_iterations=5000,convergence_warning='fail'),
            deeplob=dict(channels=16,inception=16,lstm=32,sequence=100,epochs=8,patience=2,batch=128,lr=.001)))
    (args.output/'registration.json').write_text(json.dumps(registration,indent=2),encoding='utf-8')
    print('Registered',registration,flush=True)
    predictions,cat=fit_catboost(frames,cohorts,args.output)
    # Preserve the frozen CatBoost phase before starting the registered DL phase.
    np.savez_compressed(args.output/'catboost_phase.npz',**predictions)
    (args.output/'catboost_metadata.json').write_text(json.dumps(cat,indent=2),encoding='utf-8')
    for j,h in enumerate(m.HORIZONS):
        inf=cohorts['inference'];test=set(cohorts['test']);mask=np.array([r in test for r in inf]);y=matrix(frames,inf,'labels');r=matrix(frames,inf,'returns')
        print('CatBoost TEST',h,m.metrics(y[mask,j],predictions['CatBoost'][mask,j],r[mask,j]),flush=True)
    deep,dl=fit_deeplob(frames,cohorts,args.output);predictions['DeepLOB']=deep
    result=evaluate(frames,cohorts,predictions,cuts,args.output,dict(catboost=cat,deeplob=dl,registration=registration),coverage)
    for name,checksum in sources.items():
        if sha(Path(__file__).with_name(name))!=checksum:raise ValueError('Experiment code changed during run')
    for symbol in frames:
        if sha(args.books/(symbol+'.npz'))!=coverage['assets'][symbol]['sha256']:raise ValueError('Book data changed during run')
    (args.output/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    print('COMPLETE',args.output,flush=True)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--books',type=Path,default=Path('.runtime/minute_direction_partial_books'))
    parser.add_argument('--output',type=Path,required=True);run(parser.parse_args())

if __name__=='__main__':main()
