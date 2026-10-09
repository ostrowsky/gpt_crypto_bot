"""Extend immutable historical candles for the mission experiment, public data only."""
import argparse,json,shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
from datetime import datetime,timedelta,timezone
from fetch_mission_history import request,digest,completed_rows
from mission_contract import window,TZ,daily_label,CONTRACT


def verify_grid(rows,step,start,end):
    if [r['t'] for r in rows]!=list(range(start,end,step)):raise ValueError('incomplete chronological candle grid')
    import math
    for r in rows:
        if not all(math.isfinite(r[k]) for k in ('o','h','l','c','v')) or min(r[k] for k in ('o','h','l','c'))<=0 or r['v']<0:raise ValueError('invalid candle')
        if r['h']<max(r['o'],r['c']) or r['l']>min(r['o'],r['c']):raise ValueError('invalid OHLC')


def relabel(daily,output):
    reg=json.loads((daily/'registration.json').read_bytes());receipt=json.loads((daily/'daily_receipt.json').read_bytes())
    watch=set(json.loads((daily/'watchlist.json').read_bytes()));by_day={}
    for s,r in receipt.items():
        if r['state']!='FETCHED':continue
        p=daily/'daily_raw'/(s+'.json')
        if digest(p)!=r['sha256']:raise ValueError('raw day drift')
        for row in completed_rows(json.loads(p.read_bytes()),reg['start_ms'],reg['end_ms'])[0]:
            key=datetime.fromtimestamp(row[0]/1000,timezone.utc).astimezone(TZ).date().isoformat();by_day.setdefault(key,{})[s]=row
    old=json.loads((daily/'labels.json').read_bytes())
    labels={d:daily_label(d,by_day.get(d,{}),watch,reg['end_ms'],r['coverage']) for d,r in old.items()}
    output.mkdir(parents=True,exist_ok=False)
    for n in ('mission_contract.py','prepare_mission_market.py'):shutil.copy2(Path(__file__).with_name(n),output/n)
    (output/'registration.json').write_text(json.dumps(dict(contract=CONTRACT,parent=str(daily.resolve()),raw_receipt_sha256=digest(daily/'daily_receipt.json'),old_labels_sha256=digest(daily/'labels.json'),source_hashes={n:digest(output/n) for n in ('mission_contract.py','prepare_mission_market.py')},correction='known below-volume candidates retained as negatives; ranking unchanged'),indent=2),encoding='utf-8')
    (output/'labels.json').write_text(json.dumps(labels,indent=2),encoding='utf-8')
    if any(labels[d]['leaders']!=old[d]['leaders'] for d in labels):raise ValueError('rank changed during denominator repair')
    print(json.dumps(dict(days=len(labels),leader_pairs=sum(len(r['leaders']) for r in labels.values()),labels_sha256=digest(output/'labels.json'))),flush=True)


def extend(parent,daily,output):
    old=json.loads((parent/'manifest.json').read_bytes());reg=json.loads((daily/'registration.json').read_bytes())
    end=reg['end_ms'];start=old['archive_start_ms'];watch=json.loads((daily/'watchlist.json').read_bytes())
    symbols=sorted(set(old['requested_symbols'])|set(watch));output.mkdir(parents=True,exist_ok=False)
    (output/'market').mkdir();(output/'raw').mkdir();(output/'source_snapshot').mkdir()
    for n in ('prepare_mission_market.py','fetch_mission_history.py','mission_contract.py'):shutil.copy2(Path(__file__).with_name(n),output/'source_snapshot'/n)
    (output/'registration.json').write_text(json.dumps(dict(parent=str(parent.resolve()),parent_manifest_sha256=digest(parent/'manifest.json'),daily_registration_sha256=digest(daily/'registration.json'),start_ms=start,end_ms=end,requested_symbols=symbols,runtime_eligible=False,source_hashes={p.name:digest(p) for p in (output/'source_snapshot').iterdir()}),indent=2),encoding='utf-8')
    def one(s,tf):
        receipts=[];step={'15m':900000,'1h':3600000}[tf];name=s+'_'+tf+'.json';p=parent/'market'/name
        try:
            rows=[];cursor=start
            if name in old['input_hashes']:
                if digest(p)!=old['input_hashes'][name]:raise ValueError('parent drift')
                rows=json.loads(p.read_bytes());verify_grid(rows,step,start,old['end_ms']);cursor=old['end_ms']
            while cursor<end:
                raw,meta=request('klines',dict(symbol=s,interval=tf,startTime=cursor,endTime=end-1,limit=1000));batch=json.loads(raw)
                rp=output/'raw'/f'{s}_{tf}_{cursor}.json';rp.write_bytes(raw);receipts.append(dict(file=rp.name,sha256=digest(rp),**meta))
                if not isinstance(batch,list) or not batch:raise ValueError('missing tail / delisted history')
                for r in batch:
                    if len(r)!=12 or r[6]!=r[0]+step-1:raise ValueError('unclosed schema')
                    rows.append(dict(t=int(r[0]),**{k:float(r[i]) for k,i in zip(('o','h','l','c','v'),range(1,6))}))
                next_at=int(batch[-1][0])+step
                if next_at<=cursor:raise ValueError('pagination stalled')
                cursor=next_at
            verify_grid(rows,step,start,end);dest=output/'market'/name;dest.write_text(json.dumps(rows,separators=(',',':')),encoding='utf-8')
            return name,dict(state='COMPLETE',sha256=digest(dest),rows=len(rows),receipts=receipts,parent_prefix_sha256=old['input_hashes'].get(name))
        except Exception as e:return name,dict(state='UNAVAILABLE_OR_GAPPED',error=repr(e),receipts=receipts)
    results={}
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures=[pool.submit(one,s,tf) for s in symbols for tf in ('15m','1h')]
        for f in as_completed(futures):
            n,r=f.result();results[n]=r;print('INTRADAY '+str(len(results))+'/'+str(len(futures))+' '+n+' '+r['state'],flush=True)
    eligible=[s for s in symbols if all(results[s+'_'+tf+'.json']['state']=='COMPLETE' for tf in ('15m','1h'))]
    manifest={**old,'end_ms':end,'archive_end_ms':end,'requested_symbols':symbols,'eligible_symbols':eligible,
        'input_hashes':{s+'_'+tf+'.json':results[s+'_'+tf+'.json']['sha256'] for s in eligible for tf in ('15m','1h')},'historical_PIT_certified':False,'collection_receipts':results}
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps(dict(status='COLLECTED',days=(end-old['start_ms'])/86400000,complete=len(eligible),requested=len(symbols),missing=sorted(set(symbols)-set(eligible)))),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('operation',choices=['relabel','extend']);p.add_argument('--daily',type=Path,required=True);p.add_argument('--parent',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.operation=='relabel':relabel(a.daily,a.output)
    else:extend(a.parent,a.daily,a.output)
