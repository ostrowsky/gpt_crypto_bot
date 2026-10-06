"""Independent raw daily labels, first-entry counts and scalar episode audit."""
from __future__ import annotations
import argparse,json,math
from bisect import bisect_left
from datetime import datetime,timedelta,time,timezone
from zoneinfo import ZoneInfo
from pathlib import Path
import numpy as np
from historical_signal_evaluation import sha

TZ=ZoneInfo('Europe/Budapest');BAR=900000


def identity(clock):return datetime.fromtimestamp(clock/1000,timezone.utc).astimezone(TZ).date().isoformat()


def bounds(day):
    date=datetime.fromisoformat(day).date()
    return tuple(int(datetime.combine(date,t,tzinfo=TZ).timestamp()*1000) for t in (time(),time(22)))


def scalar_episode(entry,raw,ema,atr,related=None,clock_list=None):
    tr=entry['trade'];at=tr['entry_ts'];price=tr['entry_price'];times=clock_list if clock_list is not None else [r['t']+BAR for r in raw]
    i=bisect_left(times,at)
    if i+96>=len(times) or times[i:i+97]!=list(range(at,at+97*BAR,BAR)):return {'state':'UNKNOWN_FOLLOWUP'}
    if not math.isfinite(atr[i]) or atr[i]<=0:return {'state':'UNKNOWN_ENTRY_ATR'}
    peak=price;established=False;previous=False;marker=None;top=None
    for k in range(i,i+97):
        c=raw[k]['c'];peak=max(peak,c);tolerance=price*1e-12;established|=peak-price>=atr[i]-tolerance
        declining=k>i and ema[k]<ema[k-1]-tolerance
        condition=peak-price>=atr[i]-tolerance and c<=peak-2*atr[k]+tolerance and declining
        if condition and previous:marker=times[k];top=peak;break
        previous=condition
    state='CONFIRMED' if marker is not None else ('UPSWING_UNCONFIRMED' if established else 'NO_ESTABLISHED_UPSWING')
    out={'state':state};final=tr['exit_ts'];partial=tr['partial_exit_taken'];first=tr['partial_exit_ts'] if partial else final
    k=bisect_left(times,first)
    if k>=i and k+16<len(times) and times[k:k+17]==list(range(first,first+17*BAR,BAR)):
        high=max(price,max(r['c'] for r in raw[i:k+1]));out['new_high_after_first_exit']=max(r['c'] for r in raw[k+1:k+17])>high
    else:out['new_high_after_first_exit']=None
    if marker is not None:
        fraction=tr['partial_exit_fraction'] if partial else 0
        realized=tr['partial_exit_price']*fraction+tr['exit_price']*(1-fraction) if partial else tr['exit_price']
        remain=0 if final<=marker else (1-fraction if partial and tr['partial_exit_ts']<=marker else 1)
        out.update(marker_ts=marker,peak_close=top,retention=(realized-price)/(top-price),
            giveback_pp=100*(top-realized)/price,exit_before_marker=final<marker,first_exit_before_marker=first<marker,
            delay_bars=(final-marker)/BAR,remaining_fraction_at_marker=remain)
        held=0;active=0;positions=0
        for position in related or [tr]:
            begin=max(at,position['entry_ts']);finish=min(marker,position['exit_ts'])
            if finish>begin:
                positions+=1
                if position['partial_exit_taken']:
                    split=position['partial_exit_ts'];frac=position['partial_exit_fraction']
                    held+=max(0,min(finish,split)-begin)+(1-frac)*max(0,finish-max(begin,split))
                else:held+=finish-begin
            if position['entry_ts']<=marker<position['exit_ts']:
                active+=1-position['partial_exit_fraction'] if position['partial_exit_taken'] and position['partial_exit_ts']<=marker else 1
        out.update(accompaniment_fraction=held/(marker-at),any_remaining_at_marker=active>0,
            all_positions_remaining_fraction=active,accompanied_positions=positions)
    return out


def assert_values(expected,actual):
    for k,v in expected.items():
        a=actual[k]
        if v is None or isinstance(v,(str,bool,int)):
            if a!=v:raise ValueError('scalar mismatch '+k)
        elif not np.isclose(v,a,rtol=1e-9,atol=1e-9):raise ValueError('numeric mismatch '+k)


def verify(output):
    receipt=json.loads((output/'receipt.json').read_bytes())
    for name,digest in receipt.items():
        if sha(output/name)!=digest:raise ValueError('receipt drift')
    reg=json.loads((output/'registration.json').read_bytes());result=json.loads((output/'result.json').read_bytes())
    for name,digest in reg['source_hashes'].items():
        if sha(Path(__file__).with_name(name))!=digest or sha(output/'source_snapshot'/name)!=digest:raise ValueError('source drift')
    market=Path(reg['market']);manifest=json.loads((market/'manifest.json').read_bytes());raw={};times={};ema={};atr={}
    if sha(market/'manifest.json')!=reg['market_manifest_sha256']:raise ValueError('manifest drift')
    for s in manifest['eligible_symbols']:
        name=s+'_15m.json'
        if sha(market/'market'/name)!=manifest['input_hashes'][name]:raise ValueError('raw price drift')
        raw[s]=json.loads((market/'market'/name).read_bytes());times[s]=[r['t']+BAR for r in raw[s]]
        values=[];true=[]
        for i,r in enumerate(raw[s]):
            values.append(r['c'] if i==0 else values[-1]+(2/21)*(r['c']-values[-1]))
            prior=raw[s][i-1]['c'] if i else r['c'];true.append(max(r['h']-r['l'],abs(r['h']-prior),abs(r['l']-prior)))
        ema[s]=values;atr[s]=[float('nan')]*13+[sum(true[i-13:i+1])/14 for i in range(13,len(true))]
    labels=json.loads((output/'daily_labels.json').read_bytes());expected_labels={};date=datetime.fromtimestamp(reg['start_ms']/1000,timezone.utc).astimezone(TZ).date()
    end_date=datetime.fromtimestamp(reg['end_ms']/1000,timezone.utc).astimezone(TZ).date()
    while date<=end_date:
        day=date.isoformat();lo,hi=bounds(day)
        if reg['start_ms']<=lo and hi<=reg['end_ms']:
            ranks=[];vals={}
            for s in raw:
                i=bisect_left(times[s],lo+BAR);j=bisect_left(times[s],hi)
                if i>=len(times[s]) or j>=len(times[s]) or times[s][i:j+1]!=list(range(lo+BAR,hi+1,BAR)):continue
                o=raw[s][i]['o'];c=raw[s][j]['c'];value=(c/o-1)*100;ranks.append((value,s));vals[s]=(o,c,hi,value)
            if len(ranks)>=15:
                leaders=[s for value,s in sorted(ranks,reverse=True)[:15]];expected_labels[day]=(leaders,vals)
                if labels[day]['leaders']!=leaders:raise ValueError('leader rank drift')
                for s,(o,c,cut,v) in vals.items():assert_values(dict(open=o,close=c,cutoff=cut,return_pct=v),labels[day]['values'][s])
        date+=timedelta(days=1)
    if set(expected_labels)!=set(labels):raise ValueError('day population')
    binding={name:Path(parent['path']) for name,parent in reg['parents'].items()}
    for name,parent in reg['parents'].items():
        for file,key in [('result.json','result'),('receipt.json','receipt'),('independent_verification.json','audit')]:
            if sha(binding[name]/file)!=parent[key]:raise ValueError('parent drift')
    maps={'control':('turnover','control'),'amplitude':('turnover','amplitude_cost'),'no_replacement':('turnover','no_replacement'),
        'combined':('turnover','combined'),'impulse_catboost':('impulse','catboost'),'joint':('joint','joint'),'exit_model':('exit','exit_model')}
    audited_entries=0;episodes_n=0;audited_daily={};all_episodes={}
    for arm,(parent,name) in maps.items():
        trades=json.loads((binding[parent]/('trades_'+name+'.json')).read_bytes());recorded=json.loads((output/('leader_entries_'+arm+'.json')).read_bytes())
        first={};eligible=[];eligible_by_day={}
        for t in sorted(trades,key=lambda t:t['entry_ts']):
            day=identity(t['entry_ts'])
            if day in expected_labels and t['entry_ts']<=bounds(day)[1] and t['sym'] in expected_labels[day][1]:
                eligible.append(t);eligible_by_day.setdefault(day,[]).append(t);first.setdefault((day,t['sym']),t)
        leaders={p:t for p,t in first.items() if p[1] in expected_labels[p[0]][0]}
        if len(leaders)!=len(recorded):raise ValueError('first leader count')
        for r in recorded:
            pair=(r['day'],r['symbol']);t=leaders[pair]
            if r['trade']!=t:raise ValueError('first trade selection')
            o,c,cut,_=expected_labels[pair[0]][1][pair[1]]
            capture=max(0,min(1.5,(c-t['entry_price'])/(c-o))) if c>o else None
            assert_values(dict(capture=capture,early=capture is not None and capture>=.35,lead_minutes=(cut-t['entry_ts'])/60000),r)
            audited_entries+=1
        for cohort in ('full','test'):
            days=[d for d in expected_labels if cohort=='full' or bounds(d)[0]>=reg['test_boundary_ms']]
            records=[r for r in recorded if r['day'] in days];chosen=[t for t in eligible if identity(t['entry_ts']) in days]
            picks=[p for p in first if p[0] in days]
            expected=dict(days=len(days),leader_pairs=len(days)*15,early=sum(r['early'] for r in records),captured=len(records),
                precision_n=sum(t['sym'] in expected_labels[identity(t['entry_ts'])][0] for t in chosen),precision_N=len(chosen),
                unique_precision_n=len(records),unique_precision_N=len(picks),stored_early=sum(r['stored_early'] for r in records),
                corrected_early_disagreements=sum(r['early']!=r['stored_early'] for r in records))
            assert_values(expected,result['arms'][arm][cohort])
        daily=[]
        for day in sorted(expected_labels):
            if bounds(day)[0]<reg['test_boundary_ms']:continue
            chosen=eligible_by_day.get(day,[]);entry_rows=[r for r in recorded if r['day']==day]
            daily.append(dict(day=day,early=sum(r['early'] for r in entry_rows),captured=len(entry_rows),leaders=15,
                precision_n=sum(t['sym'] in expected_labels[day][0] for t in chosen),precision_N=len(chosen)))
        audited_daily[arm]=daily
        ep=json.loads((output/('episodes_'+arm+'.json')).read_bytes())
        if len(ep)!=len(recorded):raise ValueError('episode count')
        by_symbol={s:[t for t in trades if t['sym']==s] for s in raw}
        for r,e in zip(recorded,ep):
            # ALL episodes independently reconstructed; no frozen peak/stop fields.
            independent=scalar_episode(r,raw[r['symbol']],ema[r['symbol']],atr[r['symbol']],by_symbol[r['symbol']],times[r['symbol']])
            try:assert_values(independent,e)
            except ValueError as exc:raise ValueError(f"{arm} {r['day']} {r['symbol']} {r['entry_ts']} expected={independent} actual={e}") from exc
            episodes_n+=1
        all_episodes[arm]=[e for e in ep if bounds(e['day'])[0]>=reg['test_boundary_ms']]
        for cohort in ('full','test'):
            used=[e for e in ep if cohort=='full' or bounds(e['day'])[0]>=reg['test_boundary_ms']]
            valid=[e for e in used if e['state']=='CONFIRMED' and not e['terminal_forced']]
            rebound=[e for e in used if e.get('new_high_after_first_exit') is not None and not e['terminal_forced']]
            expected=dict(captured_pairs=len(used),confirmed_n=len(valid),early_exit_n=sum(e['exit_before_marker'] for e in valid),
                first_early_exit_n=sum(e['first_exit_before_marker'] for e in valid),rebound_n=sum(e['new_high_after_first_exit'] for e in rebound),rebound_N=len(rebound),
                retention_mean=sum(e['retention'] for e in valid)/len(valid) if valid else None,
                giveback_mean_pp=sum(e['giveback_pp'] for e in valid)/len(valid) if valid else None,
                delay_bars_median=float(np.median([e['delay_bars'] for e in valid])) if valid else None)
            expected.update(accompaniment_mean=sum(e['accompaniment_fraction'] for e in valid)/len(valid) if valid else None,
                any_remaining_n=sum(e['any_remaining_at_marker'] for e in valid))
            assert_values(expected,result['arms'][arm]['exits_'+cohort])
        print('AUDITED leader mission',arm,flush=True)
    # Independent paired bootstrap and common-entry exit comparisons.
    for arm,comparison in result['comparisons'].items():
        base=audited_daily['control'];target=audited_daily[arm];n=len(base)
        if [r['day'] for r in base]!=[r['day'] for r in target]:raise ValueError('cohort day mismatch')
        a=np.array([[r['early']/15,r['captured']/15,r['precision_n'],r['precision_N']] for r in base]);b=np.array([[r['early']/15,r['captured']/15,r['precision_n'],r['precision_N']] for r in target])
        rng=np.random.default_rng(42);starts=rng.integers(0,n,(5000,int(np.ceil(n/3))))
        idx=((starts[:,:,None]+np.arange(3))%n).reshape(5000,-1)[:,:n];aa=a[idx];bb=b[idx]
        metric=np.column_stack([bb[:,:,:2].mean(axis=1)-aa[:,:,:2].mean(axis=1),
            bb[:,:,2].sum(axis=1)/bb[:,:,3].sum(axis=1)-aa[:,:,2].sum(axis=1)/aa[:,:,3].sum(axis=1)])*100
        for j,k in enumerate(('early_pp','capture_pp','precision_pp')):
            np.testing.assert_allclose(np.quantile(metric[:,j],[.05/48,1-.05/48]),comparison['entry']['intervals'][k],atol=1e-10)
        e0={(e['day'],e['symbol']):e for e in all_episodes['control']};e1={(e['day'],e['symbol']):e for e in all_episodes[arm]}
        deltas=[];early_delta=[];different=0
        for p in sorted(e0.keys()&e1.keys()):
            x,y=e0[p],e1[p]
            if (x['entry_ts'],x['entry_price'])!=(y['entry_ts'],y['entry_price']):different+=1;continue
            if x['state']!='CONFIRMED' or y['state']!='CONFIRMED' or x['terminal_forced'] or y['terminal_forced']:continue
            deltas.append((p[0],y['retention']-x['retention']));early_delta.append(int(y['first_exit_before_marker'])-int(x['first_exit_before_marker']))
        ex=comparison['exits'];assert_values(dict(common_pairs=len(e0.keys()&e1.keys()),different_entry_pairs=different,
            confirmed_same_entry_n=len(deltas),retention_delta_mean=float(np.mean([v for d,v in deltas])) if deltas else None,first_early_delta_n=sum(early_delta)),ex)
        days=sorted({d for d,v in deltas})
        if len(days)>=3:
            sums=np.array([sum(v for d,v in deltas if d==day) for day in days]);count=np.array([sum(d==day for d,v in deltas) for day in days])
            rng=np.random.default_rng(42);st=rng.integers(0,len(days),(5000,int(np.ceil(len(days)/3))))
            ii=((st[:,:,None]+np.arange(3))%len(days)).reshape(5000,-1)[:,:len(days)]
            np.testing.assert_allclose(np.quantile(sums[ii].sum(axis=1)/count[ii].sum(axis=1),[.05/48,1-.05/48]),ex['retention_interval'])
    capacity=binding['capacity'];cases=[r for r in json.loads((capacity/'dataset.json').read_bytes()) if r['cohort']=='test']
    records=json.loads((output/'capacity_records.json').read_bytes());n_known=0;kept_leader=0;model_leader=0;kept_early=0;model_early=0
    with np.load(capacity/'test_predictions.npz',allow_pickle=False) as z:scores=z['scores']
    for i,(case,r) in enumerate(zip(cases,records)):
        choice=0 if scores[i,0]>scores[i,1] else 1
        if r['choice']!=choice:raise ValueError('capacity action')
        for k,s in enumerate(case['symbols']):
            if r['options'][k] is None:continue
            day=identity(case['clock']);o,c,cut,v=expected_labels[day][1][s];at=bisect_left(times[s],case['clock'])
            cap=max(0,min(1.5,(c-raw[s][at]['c'])/(c-o))) if c>o else None
            assert_values(dict(leader=s in expected_labels[day][0],capture=cap),r['options'][k])
        if r['known']:
            n_known+=1;keep=r['options'][1];chosen=r['options'][choice]
            kept_leader+=keep['leader'];model_leader+=chosen['leader']
            kept_early+=bool(keep['leader'] and keep['capture'] is not None and keep['capture']>=.35)
            model_early+=bool(chosen['leader'] and chosen['capture'] is not None and chosen['capture']>=.35)
    assert_values(dict(known=n_known,issued=len(cases)),result['capacity'])
    assert_values(dict(leader_n=kept_leader,early_n=kept_early),result['capacity']['keep'])
    assert_values(dict(leader_n=model_leader,early_n=model_early),result['capacity']['model'])
    joined_leader=0;control_trades=json.loads((binding['turnover']/'trades_control.json').read_bytes())
    for order in json.loads((binding['execution']/'actual_orders.json').read_bytes()):
        if order['state']!='JOINED_PERPETUAL_PROXY':continue
        t=control_trades[order['trade_id']];day=identity(t['entry_ts'])
        joined_leader+=day in expected_labels and t['sym'] in expected_labels[day][0]
    if result['runtime_eligible']:raise ValueError('unsupported promotion')
    audit=dict(status='PASS',result_sha256=sha(output/'result.json'),verifier_sha256=sha(__file__),
        raw_symbols=len(raw),daily_labels=len(labels)*15,first_leader_entries=audited_entries,
        scalar_episode_checks=episodes_n,limitations=['no live/PIT certification','causal marker operational, not ultimate peak'])
    audit.update(capacity_cases=len(cases),capacity_mission_known=n_known,joined_execution_leader_actions=int(joined_leader),paired_comparisons=6)
    with (output/'independent_verification.json').open('x',encoding='utf-8') as file:json.dump(audit,file,indent=2)
    print(json.dumps(audit),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);verify(p.parse_args().output)
