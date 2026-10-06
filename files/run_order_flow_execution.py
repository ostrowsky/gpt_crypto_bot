"""Maximum available event-time book execution diagnostic, immutable outputs."""
from __future__ import annotations
import argparse,json,shutil,time
from pathlib import Path
import numpy as np
from minute_direction_data import sha,SYMBOLS
from evaluate_second_order_flow import load_asset,validate_source
import order_flow_execution as m

METHODS=('OFI','Book','Microprice','AlwaysWait','Immediate')


def publish(path,obj):
    with path.open('x',encoding='utf-8') as f:json.dump(obj,f,indent=2,allow_nan=False)


def summarize(a):
    result={}
    for j,h in enumerate(m.HORIZONS):
        scores={}
        for method in METHODS:
            choice=a['selected_'+method][:,:,j]
            scores[method]=m.metrics(a['time'],a['mid'],a['immediate'],a['future'][:,:,j],choice)
            scores[method]['assets']={s:m.metrics(a['time'][a['symbol']==s],a['mid'][a['symbol']==s],a['immediate'][a['symbol']==s],
                a['future'][a['symbol']==s,:,j],choice[a['symbol']==s]) for s in SYMBOLS}
            scores[method]['sides']={name:m.metrics(a['time'],a['mid'],a['immediate'][:,k:k+1],a['future'][:,k:k+1,j],choice[:,k:k+1],(side,))
                for k,(name,side) in enumerate((('BUY',1),('SELL',-1)))}
        intervals={}
        for control in ('Immediate','Book','Microprice'):
            left={d['day']:d['gain_bp'] for d in scores['OFI']['daily']};right={d['day']:d['gain_bp'] for d in scores[control]['daily']}
            if set(left)!=set(right):raise ValueError('unequal paired days')
            dif=[left[d]-right[d] for d in sorted(left)]
            intervals[control]=dict(n_days=len(dif),daily_gain_bp=dif,mean_daily_gain_bp=float(np.mean(dif)) if dif else None,
                familywise95_bp=m.interval(dif),robust_claim_allowed=False)
        result[str(h)]=dict(methods=scores,paired=intervals)
    return result


def run(raw,books,parent,orders,output):
    output.mkdir(parents=True,exist_ok=False)
    coverage=json.loads((books/'coverage.json').read_bytes());old=json.loads((parent/'result.json').read_bytes())
    audit=json.loads((parent/'verification.json').read_bytes())
    if audit['status']!='PASS':raise ValueError('parent audit')
    if sha(parent/'result.json')!=audit['result_sha256'] or sha(parent/'predictions.npz')!=audit['predictions_sha256']:raise ValueError('parent drift')
    for name,digest in audit['native_sha256'].items():
        if sha(parent/name)!=digest:raise ValueError('native drift')
    for name,digest in old['registration']['source_hashes'].items():
        if sha(Path(__file__).with_name(name))!=digest:raise ValueError('producer drift')
    if sha(books/'coverage.json')!=old['registration']['coverage_sha256']:raise ValueError('coverage drift')
    receipt=json.loads((orders/'receipt.json').read_bytes());order_audit=json.loads((orders/'independent_verification.json').read_bytes())
    if order_audit['status']!='PASS':raise ValueError('order audit')
    for name in ('result.json','trades_control.json','ledger_control.json','registration.json'):
        if sha(orders/name)!=receipt[name]:raise ValueError('order provenance')
    trades=json.loads((orders/'trades_control.json').read_bytes());ledger=json.loads((orders/'ledger_control.json').read_bytes())
    actual=m.extract_orders(trades,ledger);order_reg=json.loads((orders/'registration.json').read_bytes())
    spec=Path('docs/specs/order-flow-execution.md');shutil.copy2(spec,output/'registered_spec.md')
    names=('order_flow_execution.py','run_order_flow_execution.py','second_order_flow_models.py','second_order_flow_data.py','evaluate_second_order_flow.py')
    source=output/'source_snapshot';source.mkdir()
    hashes={}
    for name in names:
        p=Path(__file__).with_name(name);hashes[name]=sha(p);shutil.copy2(p,source/name)
    registration=dict(registered_at=time.time(),spec_sha256=sha(spec),source_hashes=hashes,
        parent=str(parent.resolve()),orders=str(orders.resolve()),books=str(books.resolve()),raw=str(raw.resolve()),
        parent_result_sha256=sha(parent/'result.json'),prediction_sha256=sha(parent/'predictions.npz'),
        orders_result_sha256=sha(orders/'result.json'),orders_sha256=sha(orders/'trades_control.json'),
        ledger_sha256=sha(orders/'ledger_control.json'),coverage_sha256=sha(books/'coverage.json'),
        cuts=old['registration']['cuts'],horizons=list(m.HORIZONS),notional_USDT=1000,arrival_seconds=1,
        threshold_bp=m.THRESHOLD_BP,fee_bps=m.FEE*10000,disclosed_test=True,
        order_start_ms=order_reg['start_ms'],order_end_ms=order_reg['end_ms'],runtime_eligible=False)
    publish(output/'registration.json',registration);print('REGISTERED execution',flush=True)
    validate_source(raw,books,coverage);print('All raw LFS hashes verified',flush=True)
    with np.load(parent/'predictions.npz',allow_pickle=False) as z:pred={k:z[k] for k in z.files}
    blocks=[];views=[]
    for s in SYMBOLS:
        data=load_asset(books,s,coverage);take=pred['symbol']==s;clock=pred['time'][take];mid=pred['origin_price'][take];qty=1000/mid
        imm=np.column_stack([m.arrival(data,clock,qty,side,1) for side in (1,-1)])
        future=np.stack([np.column_stack([m.arrival(data,clock,qty,side,h+1) for side in (1,-1)]) for h in m.HORIZONS],axis=2)
        indices=[old['registration']['horizon_seconds'].index(h) for h in m.HORIZONS]
        forecasts={k:pred[name][take][:,indices] for k,name in [('OFI','CatBoost_OFI'),('Book','CatBoost_Book'),('Microprice','Microprice')]}
        a=dict(time=clock,mid=mid,symbol=np.full(len(clock),s),immediate=imm,future=future)
        for name in METHODS:
            if name in forecasts:
                a['selected_'+name]=np.stack([np.column_stack([
                    m.decide(forecasts[name][:,j],side,h)>0 for j,h in enumerate(m.HORIZONS)]) for side in (1,-1)],axis=1)
            else:a['selected_'+name]=np.full(future.shape,name=='AlwaysWait',bool)
        blocks.append(a)
        for order in actual:
            if order['symbol']!=s:continue
            at=order['time'];row=int(np.searchsorted(clock,at))
            if row>=len(clock) or clock[row]!=at:order['state']='NO_PAST_FORECAST';continue
            side=order['side'];k=0 if side==1 else 1;gain=[];delay=[];prices=[]
            p0=float(mid[row]);base=m.arrival(data,np.array([at]),order['quantity'],side,1)[0]
            for j,h in enumerate(m.HORIZONS):
                wait=bool(a['selected_OFI'][row,k,j]) and not order['protected'];delay.append(h if wait else 0)
                price=m.arrival(data,np.array([at]),order['quantity'],side,(h if wait else 0)+1)[0]
                prices.append(float(price) if np.isfinite(price) else None)
                gain.append(float(side*(base-price)/p0*10000) if np.isfinite(base) and np.isfinite(price) else None)
            order.update(state='JOINED_PERPETUAL_PROXY',forecast=forecasts['OFI'][row].tolist(),origin_mid=p0,
                immediate_net_unit=float(base) if np.isfinite(base) else None,delays=delay,net_unit_prices=prices,gain_bp=gain)
            ti=np.searchsorted(data['time'],at);slice_=slice(max(0,ti-60),min(len(data['time']),ti+32))
            history=(data['book'][slice_,0]+data['book'][slice_,2])/2
            views.append(dict(order_id=len(views),trade_id=order['trade_id'],symbol=s,kind=order['kind'],time=at,
                clock=data['time'][slice_].tolist(),mid=[float(p) if np.isfinite(p) else None for p in history],
                forecast_prices=(p0*np.exp(forecasts['OFI'][row])).tolist(),delays=delay,net_prices=prices))
        print(s,'issued',len(clock),'paired5',int(np.isfinite(imm).all(axis=1).sum()),flush=True)
        del data
    a={k:np.concatenate([b[k] for b in blocks]) for k in blocks[0]}
    order_idx=np.lexsort((a['symbol'],a['time']));a={k:v[order_idx] for k,v in a.items()}
    for o in actual:o.setdefault('state','NO_SYMBOL_BOOK_ARCHIVE')
    scores=summarize(a)
    states={state:sum(o['state']==state for o in actual) for state in sorted({o['state'] for o in actual})}
    joined=[o for o in actual if o['state']=='JOINED_PERPETUAL_PROXY']
    actual_scores={str(h):dict(issued=len(actual),joined=len(joined),deferred=sum(o['delays'][j]>0 for o in joined),
        known=sum(o['gain_bp'][j] is not None for o in joined),gains_bp=[o['gain_bp'][j] for o in joined],
        scope='transferred spot replay clocks on perpetual quotes; not actual fills') for j,h in enumerate(m.HORIZONS)}
    result=dict(status='COMPLETED_DIAGNOSTIC',verdict='NOT_APPROVED',runtime_eligible=False,
        scores=scores,actual_orders=dict(trades=len(trades),issued=len(actual),states=states,scores=actual_scores),
        limitations=['perpetual books vs spot bot','event time only, no receive times or actual fills',
            'synthetic BUY/SELL tasks are not candidate population','short exposed TEST with source holes',
            'book walking assumes no impact and fixed 1-second arrival; depth failure remains unknown',
            'maximum 186-day portfolio execution validation cannot be certified with available books'])
    np.savez_compressed(output/'outcomes.npz',**a)
    publish(output/'actual_orders.json',actual);publish(output/'views.json',views);publish(output/'result.json',result)
    for name,digest in hashes.items():
        if sha(Path(__file__).with_name(name))!=digest:raise ValueError('run source drift')
    for path,digest in [(parent/'result.json',registration['parent_result_sha256']),(parent/'predictions.npz',registration['prediction_sha256']),
                        (orders/'trades_control.json',registration['orders_sha256']),(orders/'ledger_control.json',registration['ledger_sha256'])]:
        if sha(path)!=digest:raise ValueError('run input drift')
    publish(output/'receipt.json',{p.name:sha(p) for p in output.iterdir() if p.is_file()})
    print(json.dumps(dict(actual=result['actual_orders'],summary={h:{k:v['mean_gain_bp'] for k,v in s['methods'].items()} for h,s in scores.items()})),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('raw','books','parent','orders','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();run(a.raw,a.books,a.parent,a.orders,a.output)
