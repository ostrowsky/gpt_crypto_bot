"""Independent native prediction, execution arithmetic and source-bound audit."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from collections import Counter
from catboost import CatBoostRegressor
from minute_direction_data import sha,SYMBOLS
from evaluate_second_order_flow import load_asset
from second_order_flow_models import features,HORIZONS


def verify_spot(spot):
    report=json.loads((spot/'result.json').read_bytes());counts=Counter();last=0;updates={};regress=0;rows=0
    if sha(spot/'messages.jsonl')!=report['messages_sha256']:raise ValueError('capture hash')
    with (spot/'messages.jsonl').open(encoding='utf-8') as file:
        for line in file:
            x=json.loads(line)
            if x['receive_monotonic_ns']<last:raise ValueError('receive clock regression')
            last=x['receive_monotonic_ns'];p=json.loads(x['raw']);s=p['stream'].split('@')[0].upper()
            kind='depth' if '@depth' in p['stream'] else 'trade';counts[s+':'+kind]+=1;rows+=1
            if kind=='depth':
                u=p['data']['lastUpdateId'];regress+=u<updates.get(s,0);updates[s]=u
                b=np.array(p['data']['bids'],float);a=np.array(p['data']['asks'],float)
                if b.shape!=(20,2) or a.shape!=(20,2) or not np.isfinite(b).all() or not np.isfinite(a).all():raise ValueError('depth shape')
                if (b<=0).any() or (a<=0).any() or b[0,0]>=a[0,0] or (np.diff(b[:,0])>=0).any() or (np.diff(a[:,0])<=0).any():raise ValueError('depth validity')
            elif float(p['data']['p'])<=0 or float(p['data']['q'])<=0:raise ValueError('trade validity')
    if dict(counts)!=report['counts'] or regress or report['invalid']:raise ValueError('capture counts/update IDs')
    out=dict(status='PASS',messages=rows,counts=dict(counts),depth_update_regressions=regress,
        messages_sha256=report['messages_sha256'],verifier_sha256=sha(__file__),
        limits=['no fills or bot decisions','wall clock not synchronized; not transport latency certification'])
    with (spot/'independent_verification.json').open('x',encoding='utf-8') as file:json.dump(out,file,indent=2)
    return out


def scalar_price(data,at,quantity,side,delay):
    i=int(np.searchsorted(data['time'],at));j=int(np.searchsorted(data['time'],at+delay*1000))
    if i>=len(data['time']) or j>=len(data['time']) or data['time'][i]!=at or data['time'][j]!=at+delay*1000:return None
    if data['segment'][i]<0 or not np.all(data['segment'][i:j+1]==data['segment'][i]) or data['age'][j]>250:return None
    q=data['book'][j];remaining=quantity;value=0
    if not np.isfinite(q).all() or (q<=0).any() or q[0]>=q[2]:return None
    for rank in range(5):
        price=q[4*rank+(2 if side==1 else 0)];size=q[4*rank+(3 if side==1 else 1)]
        fill=min(remaining,size);value+=price*fill;remaining-=fill
    if remaining>quantity*1e-12:return None
    return value/quantity*(1+side*.00075)


def audit_metrics(a,scores):
    total=0
    for j,h in enumerate((5,10,30)):
        for method,report in scores[str(h)]['methods'].items():
            choices=a['selected_'+method][:,:,j];base=a['immediate'];wait=a['future'][:,:,j]
            action=np.where(choices,wait,base);matched=np.isfinite(base)&np.isfinite(wait)
            g=(base-action)*np.array([1,-1])[None,:]/a['mid'][:,None]*10000
            sf=(action/a['mid'][:,None]-1)*np.array([1,-1])[None,:]*10000
            def check(r,take):
                mask=matched&take;known=np.isfinite(action)&take;n=int(mask.sum())
                counts=dict(issued=int(take.sum()),deferred=int((choices&take).sum()),matched=n,
                    matched_deferred=int((mask&choices).sum()),action_known=int(known.sum()),action_unknown=int((~np.isfinite(action)&take).sum()),
                    harmed=int((mask&(g < -1e-10)).sum()),benefited=int((mask&(g > 1e-10)).sum()))
                for k,v in counts.items():
                    if r[k]!=v:raise ValueError('count mismatch '+k)
                vals={'mean_gain_bp':float(g[mask].mean()) if n else None,'mean_shortfall_bp':float(sf[mask].mean()) if n else None,
                    'p95_shortfall_bp':float(np.percentile(sf[mask],95)) if n else None,'p99_shortfall_bp':float(np.percentile(sf[mask],99)) if n else None}
                selected=mask&choices;vals['selected_mean_gain_bp']=float(g[selected].mean()) if selected.any() else None
                for k,v in vals.items():
                    if v is None:
                        if r[k] is not None:raise ValueError('undefined metric')
                    elif not np.isclose(r[k],v,rtol=1e-10,atol=1e-10):raise ValueError('metric mismatch '+k)
            check(report,np.ones_like(matched,bool));total+=1
            for s in SYMBOLS:check(report['assets'][s],np.broadcast_to((a['symbol']==s)[:,None],matched.shape));total+=1
            for k,side in enumerate(('BUY','SELL')):check(report['sides'][side],np.broadcast_to((np.arange(2)==k)[None,:],matched.shape));total+=1
            day=a['time']//86400000
            for row in report['daily']:
                mask=matched&(day==row['day'])[:,None]
                if row['n']!=int(mask.sum()) or not np.isclose(row['gain_bp'],g[mask].mean(),atol=1e-10):raise ValueError('daily mismatch')
    for h in ('5','10','30'):
        matched=np.isfinite(a['immediate'])&np.isfinite(a['future'][:,:,(5,10,30).index(int(h))])
        days=a['time']//86400000
        full=sum(int((days==d).sum())==17280*3 and int(matched[days==d].sum())==17280*3*2 for d in np.unique(days))
        if scores[h]['methods']['OFI']['fully_observed_candidate_days']!=full:raise ValueError('complete day mismatch')
        for control,r in scores[h]['paired'].items():
            left={x['day']:x['gain_bp'] for x in scores[h]['methods']['OFI']['daily']}
            right={x['day']:x['gain_bp'] for x in scores[h]['methods'][control]['daily']}
            if left.keys()!=right.keys():raise ValueError('paired day mismatch')
            daily=np.array([left[d]-right[d] for d in sorted(left)]);np.testing.assert_allclose(daily,r['daily_gain_bp'])
            if len(daily)>=3:
                rng=np.random.default_rng(42);starts=rng.integers(0,len(daily),(5000,int(np.ceil(len(daily)/3))))
                idx=((starts[:,:,None]+np.arange(3))%len(daily)).reshape(5000,-1)[:,:len(daily)]
                np.testing.assert_allclose(np.quantile(daily[idx].mean(axis=1),[.05/18,1-.05/18]),r['familywise95_bp'])
    return total


def verify(output,spot=None):
    receipt=json.loads((output/'receipt.json').read_bytes())
    for name,digest in receipt.items():
        if sha(output/name)!=digest:raise ValueError('output drift')
    reg=json.loads((output/'registration.json').read_bytes());parent=Path(reg['parent']);books=Path(reg['books']);orders=Path(reg['orders'])
    for name,digest in reg['source_hashes'].items():
        if sha(Path(__file__).with_name(name))!=digest or sha(output/'source_snapshot'/name)!=digest:raise ValueError('source drift')
    for path,key in [(parent/'result.json','parent_result_sha256'),(parent/'predictions.npz','prediction_sha256'),
                     (orders/'trades_control.json','orders_sha256'),(orders/'ledger_control.json','ledger_sha256'),(books/'coverage.json','coverage_sha256')]:
        if sha(path)!=reg[key]:raise ValueError('input drift')
    old=json.loads((parent/'result.json').read_bytes());coverage=json.loads((books/'coverage.json').read_bytes())
    with np.load(output/'outcomes.npz',allow_pickle=False) as z:a={k:z[k] for k in z.files}
    with np.load(parent/'predictions.npz',allow_pickle=False) as z:p={k:z[k] for k in z.files}
    result=json.loads((output/'result.json').read_bytes());actual=json.loads((output/'actual_orders.json').read_bytes())
    trades=json.loads((orders/'trades_control.json').read_bytes());ledger=json.loads((orders/'ledger_control.json').read_bytes())
    expected=[]
    for i,(t,l) in enumerate(zip(trades,ledger)):
        size=l['budget']/(1.00075*1.0005*t['entry_price']);expected.append((i,t['sym'],'BUY',t['entry_ts'],size,False))
        if t['partial_exit_taken']:
            expected.append((i,t['sym'],'SELL_PARTIAL',t['partial_exit_ts'],size*t['partial_exit_fraction'],True));size*=1-t['partial_exit_fraction']
        expected.append((i,t['sym'],'SELL',t['exit_ts'],size,'WEAK:' not in t['exit_reason']))
    if len(expected)!=len(actual):raise ValueError('actual population mismatch')
    for e,o in zip(expected,actual):
        if e[:4]!=(o['trade_id'],o['symbol'],o['kind'],o['time']) or e[5]!=o['protected'] or not np.isclose(e[4],o['quantity']):raise ValueError('order reconstruction')
    samples=0;native_n=0;joined=0
    for si,s in enumerate(SYMBOLS):
        data=load_asset(books,s,coverage);frame=features(data,si);take=a['symbol']==s;pt=p['symbol']==s
        clock=a['time'][take];np.testing.assert_array_equal(clock,p['time'][pt]);np.testing.assert_allclose(a['mid'][take],p['origin_price'][pt])
        row=np.searchsorted(frame['time'],clock);np.testing.assert_array_equal(frame['time'][row],clock)
        if not frame['good'][row].all():raise ValueError('ineligible past features')
        if np.any(clock<reg['cuts']['test']) or np.any(clock+31000>=reg['cuts']['end']):raise ValueError('issuance cut')
        for method,name in [('OFI','CatBoost_OFI'),('Book','CatBoost_Book')]:
            model=CatBoostRegressor();model.load_model(str(parent/(name+'.cbm')))
            width=old['models'][name]['input_width'];prediction=model.predict(frame['x'][row,:width])*np.array(old['train_target_scale'])+np.array(old['train_target_mean'])
            np.testing.assert_allclose(prediction,p[name][pt],rtol=1e-7,atol=1e-10);native_n+=len(row)
        np.testing.assert_allclose(frame['micro_return'][row],p['Microprice'][pt,0],rtol=1e-8,atol=1e-12)
        for method,name in [('OFI','CatBoost_OFI'),('Book','CatBoost_Book'),('Microprice','Microprice')]:
            for j,h in enumerate((5,10,30)):
                pp=p[name][pt,HORIZONS.index(h)]
                for k,side in enumerate((1,-1)):
                    np.testing.assert_array_equal(a['selected_'+method][take,k,j],side*pp*10000 < -1)
        if a['selected_Immediate'][take].any() or not a['selected_AlwaysWait'][take].all():raise ValueError('controls')
        for ri in np.unique(np.linspace(0,len(clock)-1,128,dtype=int)):
            for k,side in enumerate((1,-1)):
                for col,delay in enumerate((1,6,11,31)):
                    value=scalar_price(data,int(clock[ri]),1000/a['mid'][take][ri],side,delay)
                    expected_value=a['immediate'][take][ri,k] if col==0 else a['future'][take][ri,k,col-1]
                    if value is None:
                        if np.isfinite(expected_value):raise ValueError('fabricated fill')
                    elif not np.isclose(value,expected_value,atol=1e-10):raise ValueError('scalar fill mismatch')
                    samples+=1
        for o in actual:
            if o['symbol']!=s:continue
            ri=np.searchsorted(clock,o['time']);has=ri<len(clock) and clock[ri]==o['time']
            if not has:
                if o['state']!='NO_PAST_FORECAST':raise ValueError('future selected order')
                continue
            joined+=1;side=o['side'];base=scalar_price(data,o['time'],o['quantity'],side,1)
            for j,h in enumerate((5,10,30)):
                delay=h if side*p['CatBoost_OFI'][pt,HORIZONS.index(h)][ri]*10000 < -1 and not o['protected'] else 0
                if o['delays'][j]!=delay:raise ValueError('unsafe actual delay')
                price=scalar_price(data,o['time'],o['quantity'],side,delay+1)
                if price is None:
                    if o['net_unit_prices'][j] is not None:raise ValueError('unknown actual fill')
                elif not np.isclose(price,o['net_unit_prices'][j]):raise ValueError('actual price')
                g=side*(base-price)/o['origin_mid']*10000 if base is not None and price is not None else None
                if g is None:
                    if o['gain_bp'][j] is not None:raise ValueError('unknown actual gain')
                elif not np.isclose(g,o['gain_bp'][j]):raise ValueError('actual gain')
        del data,frame
        print('AUDITED',s,flush=True)
    metrics_n=audit_metrics(a,result['scores'])
    if result['runtime_eligible'] or result['verdict']!='NOT_APPROVED':raise ValueError('unsupported promotion')
    audit=dict(status='PASS',result_sha256=sha(output/'result.json'),native_prediction_rows=native_n,
        scalar_book_walk_checks=samples,metric_groups=metrics_n,actual_orders_reconstructed=len(actual),
        transferred_orders_joined=joined,prepared_books='all part hashes; producer/source bound, not all raw events independently decoded',
        limitations=['scalar book-walk audit is sampled, not full independent arrival recomputation',
            'same-venue actual execution and maximum-period portfolio effect remain unproven'])
    audit['verifier_sha256']=sha(__file__)
    if spot is not None:audit['spot_capture']=verify_spot(spot)
    with (output/'independent_verification.json').open('x',encoding='utf-8') as f:json.dump(audit,f,indent=2)
    print(json.dumps(audit),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--spot',type=Path)
    a=p.parse_args();verify(a.output,a.spot)
