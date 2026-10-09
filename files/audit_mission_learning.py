"""Independent raw-price, fold-clock and frozen-control audit of mission learning."""
import argparse,json,pickle,hashlib
from pathlib import Path
from datetime import datetime,timezone
from zoneinfo import ZoneInfo
import numpy as np

TZ=ZoneInfo('Europe/Budapest')
ENTRY_FIELDS=('sym','tf','mode','entry_ts','entry_price','entry_i','trail_k','max_hold_bars','entry_score','ranker_final_score','ranker_top_gainer_prob','top_gainer_score','allocation_score','entry_rsi','entry_daily_range','entry_intraday_change_pct')

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p):return json.loads(p.read_bytes())

def verify_targets(experiment,reg,labels):
    manifest=load(Path(reg['market'])/'manifest.json');market={};count=0
    for s in manifest['eligible_symbols']:
        rows=load(Path(reg['market'])/'market'/(s+'_15m.json'))
        market[s]={k:np.array([r[k] for r in rows]) for k in ('t','h','l','c')};market[s]['time']=market[s]['t']+900000
    with (experiment/'candidate_snapshot.pkl').open('rb') as f:snapshot=pickle.load(f)
    candidates=[c for at in sorted(snapshot[0]) for c in snapshot[0][at]]
    with np.load(experiment/'dataset_discovery.npz',allow_pickle=False) as z:
        clocks,ys,available=z['clock'],z['y'],z['available']
        if len(candidates)!=len(clocks):raise ValueError('entry alignment mismatch')
        for i,c in enumerate(candidates):
            at=c.ts_ms;d=datetime.fromtimestamp(at/1000,timezone.utc).astimezone(TZ).date().isoformat();label=labels.get(d);data=market[c.sym];j=int(np.searchsorted(data['time'],at))
            if j>=len(data['time']) or data['time'][j]!=at or not np.isclose(c.price,data['c'][j],rtol=1e-12,atol=0):raise ValueError('candidate origin price is not raw closed price')
            known=label and not label['coverage']['missing'] and c.sym in label['values']
            if not known:
                if np.isfinite(ys[i]).any():raise ValueError('missing label became known training target')
                continue
            v=label['values'][c.sym];leader=c.sym in label['leaders'];early=leader and v['close']>v['open'] and (v['close']-c.price)/(v['close']-v['open'])>=.35
            if tuple(ys[i])!=(int(leader),int(early)) or not np.all(available[i]==label['available_at']):raise ValueError('entry raw future label mismatch')
            count+=1
    state_counts={}
    for family,horizon in (('continuation',4),('reentry',16)):
        meta=load(experiment/('states_'+family+'.json'));checked=0
        with np.load(experiment/('dataset_'+family+'.npz'),allow_pickle=False) as z:
            clocks,ys,available=z['clock'],z['y'],z['available']
            if len(meta)!=len(clocks):raise ValueError('state target alignment mismatch')
            for i,r in enumerate(meta):
                d=market[r['symbol']];at=r['clock'];j=int(np.searchsorted(d['time'],at))
                if clocks[i]!=at or j<14 or j>=len(d['time']) or d['time'][j]!=at:raise ValueError('state source clock mismatch')
                if j+horizon>=len(d['time']):
                    if np.isfinite(ys[i,0]):raise ValueError('unobserved future state labeled')
                    continue
                h=d['h'][j-13:j+1];l=d['l'][j-13:j+1];previous=d['c'][j-14:j]
                atr=np.maximum(h-l,np.maximum(abs(h-previous),abs(l-previous))).mean();p=d['c'][j];future=d['c'][j+1:j+horizon+1]
                expected=int(future[-1]>p and future.min()>=p-2*atr)
                if ys[i,0]!=expected or available[i,0]!=at+horizon*900000:raise ValueError('state future target mismatch')
                checked+=1
        state_counts[family]=checked
    return dict(state='PASS',entry_known_targets=count,state_targets=state_counts,scope='all known targets reconstructed from raw prices; unknown futures remain absent')

def audit(experiment,daily,parent):
    receipt=load(experiment/'receipt.json');reg=load(experiment/'registration.json');result=load(experiment/'result.json');checks={};count=0
    for n,h in receipt.items():
        if Path(n).name!=n or digest(experiment/n)!=h:raise ValueError('artifact drift '+n)
    for n,h in reg['sources'].items():
        if digest(experiment/'source_snapshot'/n)!=h:raise ValueError('source snapshot drift')
    labels=load(Path(reg['labels']));native={};raw_receipt=load(daily/'daily_receipt.json')
    daily_reg=load(daily/'registration.json');watch=set(load(daily/'watchlist.json'))
    if digest(daily/'watchlist.json')!=daily_reg['watchlist_sha256']:raise ValueError('watchlist binding mismatch')
    for s,r in raw_receipt.items():
        if r['state']!='FETCHED':raise ValueError('unresolved raw daily coverage')
        p=daily/'daily_raw'/(s+'.json')
        if digest(p)!=r['sha256']:raise ValueError('raw day hash drift')
        native[s]={datetime.fromtimestamp(r[0]/1000,timezone.utc).astimezone(TZ).date().isoformat():r for r in load(p)}
    for d,label in labels.items():
        for s,v in label['values'].items():
            raw=native[s][d]
            if raw[6]!=label['available_at']-1 or (float(raw[1]),float(raw[4]),float(raw[7]))!=(v['open'],v['close'],v['quote_volume']):raise ValueError('daily raw boundary/value mismatch')
            count+=1
        eligible=[s for s,v in label['values'].items() if v['quote_volume']>=1000000]
        expected=sorted(eligible,key=lambda s:(label['values'][s]['close']/label['values'][s]['open']-1,s),reverse=True)[:20]
        if expected!=label['exchange_top']:raise ValueError('exchange rank mismatch')
        if label['leaders']!=[s for s in expected if s in watch]:raise ValueError('global top/watchlist intersection mismatch')
    checks['raw_day_values']=dict(state='PASS',checked=count)
    checks['raw_training_targets']=verify_targets(experiment,reg,labels)
    folds_count=0
    for family in ('discovery','continuation','reentry'):
        p=experiment/('folds_'+family+'.json')
        if not p.exists():continue
        with np.load(experiment/('dataset_'+family+'.npz'),allow_pickle=False) as z:
            for fold in load(p):
                for h,(name,r) in enumerate(fold['heads'].items()):
                    if r['state']!='FROZEN':continue
                    clock=z['clock'];available=z['available'][:,h];valid=np.isfinite(z['x']).all(axis=1)&np.isfinite(z['y'][:,h])
                    train=valid&(clock<r['validation_boundary'])&(available<r['validation_boundary']);val=valid&(clock>=r['validation_boundary'])&(clock<fold['fit_at'])&(available<fold['fit_at'])
                    if (int(train.sum()),int(val.sum()))!=(r['train_n'],r['validation_n']):raise ValueError('fold denominator mismatch')
                    if available[train].max()!=r['train_max_available'] or available[val].max()!=r['validation_max_available']:raise ValueError('fold maturity mismatch')
                    if digest(experiment/r['model'])!=r['model_sha256']:raise ValueError('model binding mismatch')
                    folds_count+=1
    checks['mature_fold_clocks']=dict(state='PASS',heads_checked=folds_count)
    old=load(parent/'trades_control.json');old_reg=load(parent/'registration.json');old_end=load(Path(old_reg['market'])/'manifest.json')['end_ms']
    base=load(experiment/'trades_control.json');eligible_old=[r for r in old if r['exit_ts']<old_end and 'open_at_end' not in r['exit_reason']]
    eligible_new=[r for r in base if r['exit_ts']<old_end and 'open_at_end' not in r['exit_reason']]
    cols=ENTRY_FIELDS+('exit_ts','exit_price','exit_reason','partial_exit_taken','partial_exit_fraction','partial_exit_ts','partial_exit_price')
    a=[tuple(r[k] for k in cols) for r in eligible_old];b=[tuple(r[k] for k in cols) for r in eligible_new]
    if a!=b:raise ValueError('frozen historical control prefix differs')
    checks['control_prefix']=dict(state='PASS',trades=len(a),boundary=old_end,fields=list(cols))
    verified={}
    for name in ('control','early_ranker'):
        rows=load(experiment/('trades_'+name+'.json'));known={d:r for d,r in labels.items() if not r['coverage']['missing'] and r['available_at']<=reg['end_ms'] and datetime.fromisoformat(d).replace(tzinfo=TZ).timestamp()*1000>=reg['test_boundary']};first={};unknown=0
        for t in sorted(rows,key=lambda r:r['entry_ts']):
            d=datetime.fromtimestamp(t['entry_ts']/1000,timezone.utc).astimezone(TZ).date().isoformat()
            if d not in known:continue
            if t['sym'] not in known[d]['values']:unknown+=1;continue
            first.setdefault((d,t['sym']),t)
        captured=early=0
        for (d,s),t in first.items():
            label=known[d]
            if s not in label['leaders']:continue
            captured+=1;v=label['values'][s];move=v['close']-v['open']
            if move>0 and (v['close']-t['entry_price'])/move>=.35:early+=1
        expected=dict(early=early,captured=captured,leader_pairs=sum(len(r['leaders']) for r in known.values()),unique_precision_n=captured,unique_precision_N=len(first),unknown_entry_events=unknown)
        actual=result['arms'][name]['test']
        if any(actual[k]!=v for k,v in expected.items()):raise ValueError('mission metric mismatch '+name)
        verified[name]=expected
    checks['mission_denominators']=dict(state='PASS',test=verified)
    result_audit=dict(status='PASS',result_sha256=digest(experiment/'result.json'),checks=checks,limits='Retrospective internal consistency, not PIT/actual fills/fresh deployment approval')
    (experiment/'independent_verification.json').write_text(json.dumps(result_audit,indent=2),encoding='utf-8');print(json.dumps(result_audit),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--experiment',type=Path,required=True);p.add_argument('--daily',type=Path,required=True);p.add_argument('--parent',type=Path,required=True);a=p.parse_args();audit(a.experiment,a.daily,a.parent)
