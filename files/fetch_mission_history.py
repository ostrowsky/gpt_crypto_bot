"""Public closed-candle collection for a versioned mission; never accesses accounts."""
from __future__ import annotations
import argparse,json,hashlib,time,urllib.request,urllib.parse,urllib.error,shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
from datetime import datetime,timedelta,timezone
from mission_contract import TZ,window,target_symbol,daily_label,CONTRACT

API='https://api.binance.com/api/v3/'


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def request(endpoint,parameters=None):
    if endpoint not in ('time','exchangeInfo','klines'):raise ValueError('public market endpoints only')
    url=API+endpoint+('?' + urllib.parse.urlencode(parameters) if parameters else '')
    for attempt in range(3):
        try:
            with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'MissionLearningAudit/1.0'}),timeout=25) as response:
                raw=response.read();return raw,dict(url=url,received_utc=datetime.now(timezone.utc).isoformat(),status=response.status)
        except urllib.error.HTTPError as e:
            if e.code not in (429,500,502,503,504) or attempt==2:raise
            time.sleep(min(60,max(2,float(e.headers.get('Retry-After',2)))))
        except (TimeoutError,urllib.error.URLError):
            if attempt==2:raise
            time.sleep(2)
    raise AssertionError('unreachable')


def daily_request_range(start,end):
    first=datetime.fromtimestamp(start/1000,timezone.utc).astimezone(TZ)
    last=datetime.fromtimestamp((end-1)/1000,timezone.utc).astimezone(TZ)
    if first.utcoffset()!=last.utcoffset() or end-start>1000*86400000:
        raise ValueError('daily bulk range must have constant timezone offset; DST transition needs hourly aggregation')
    seconds=int(first.utcoffset().total_seconds());sign='-' if seconds<0 else ''
    return dict(interval='1d',timeZone=f'{sign}{abs(seconds)//3600}:{abs(seconds)%3600//60:02}',
        startTime=start,endTime=end-1,limit=1000)


def completed_rows(rows,start,end):
    valid=[];partial=[]
    for r in rows:
        if r[0]<start or r[6]>=end:raise ValueError('requested range mismatch')
        key=datetime.fromtimestamp(r[0]/1000,timezone.utc).astimezone(TZ).date().isoformat()
        lo,hi=window(key)
        if r[0]!=lo:raise ValueError('daily open timezone mismatch')
        if r[6]!=hi-1:partial.append(dict(day=key,open_time=r[0],close_time=r[6]));continue
        valid.append(r)
    return valid,partial


def collect(output,parent):
    output.mkdir(parents=True,exist_ok=False);(output/'daily_raw').mkdir();(output/'source_snapshot').mkdir()
    for name in ('mission_contract.py',Path(__file__).name):shutil.copy2(Path(__file__).with_name(name),output/'source_snapshot'/name)
    raw,received=request('time');(output/'server_time.json').write_bytes(raw);server=json.loads(raw)['serverTime']
    raw,exchange_received=request('exchangeInfo',dict(permissions='SPOT'));(output/'exchangeInfo.json').write_bytes(raw)
    watchpath=Path('files/watchlist.json');shutil.copy2(watchpath,output/'watchlist.json');watch=set(json.loads(watchpath.read_bytes()))
    manifest=json.loads((parent/'manifest.json').read_bytes());start=manifest['start_ms']
    day=datetime.fromtimestamp(server/1000,timezone.utc).astimezone(TZ).date();end=window(day.isoformat())[0]
    if end<=start:raise ValueError('no completed history')
    symbols={r['symbol'] for r in json.loads(raw)['symbols'] if r.get('quoteAsset')=='USDT' and target_symbol(r['symbol'])}
    symbols|={s for s in watch if target_symbol(s)}
    # Supplement the present symbol registry with recorded historical exchange-top names.
    for path in Path('.runtime/reports').glob('top_gainer_critic_????-??-??_final.json'):
        old=json.loads(path.read_bytes());symbols.update(r['symbol'] for r in old.get('exchange_top_gainers',[]) if target_symbol(r['symbol']))
    parameters=daily_request_range(start,end)
    reg=dict(contract=CONTRACT,registered_utc=datetime.now(timezone.utc).isoformat(),start_ms=start,end_ms=end,
        server_received=received,exchange_received=exchange_received,watchlist_sha256=digest(output/'watchlist.json'),
        exchange_sha256=digest(output/'exchangeInfo.json'),parent_market=str(parent.resolve()),parent_manifest_sha256=digest(parent/'manifest.json'),
        requested_symbols=sorted(symbols),historical_PIT_certified=False,
        universe_scope='current public exchange registry plus recorded historical top names; not certified historical PIT',
        source_hashes={p.name:digest(p) for p in (output/'source_snapshot').iterdir()})
    (output/'registration.json').write_text(json.dumps(reg,indent=2),encoding='utf-8');print('REGISTERED '+str(len(symbols))+' public symbols',flush=True)
    def one(symbol):
        try:
            if not symbol.isalnum():raise ValueError('invalid filename symbol')
            raw,meta=request('klines',dict(parameters,symbol=symbol));rows=json.loads(raw)
            if not isinstance(rows,list):raise ValueError('unexpected response')
            path=output/'daily_raw'/(symbol+'.json');path.write_bytes(raw)
            if len({r[0] for r in rows})!=len(rows):raise ValueError('duplicate native day')
            valid,partial=completed_rows(rows,start,end)
            return symbol,dict(state='FETCHED',rows=len(rows),completed_rows=len(valid),partial_rows=partial,sha256=digest(path),**meta)
        except Exception as e:return symbol,dict(state='UNKNOWN_FETCH',error=repr(e))
    receipts={}
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures=[pool.submit(one,s) for s in sorted(symbols)]
        for future in as_completed(futures):
            s,v=future.result();receipts[s]=v
            if len(receipts)%50==0:print('DAILY '+str(len(receipts))+'/'+str(len(symbols)),flush=True)
    (output/'daily_receipt.json').write_text(json.dumps(receipts,indent=2),encoding='utf-8')
    missing=[s for s,r in receipts.items() if r['state']!='FETCHED'];by_day={}
    for s,r in receipts.items():
        if r['state']!='FETCHED':continue
        for row in completed_rows(json.loads((output/'daily_raw'/(s+'.json')).read_bytes()),start,end)[0]:
            key=datetime.fromtimestamp(row[0]/1000,timezone.utc).astimezone(TZ).date().isoformat();by_day.setdefault(key,{})[s]=row
    labels={};date=datetime.fromtimestamp(start/1000,timezone.utc).astimezone(TZ).date()
    while window(date.isoformat())[1]<=end:
        key=date.isoformat();partial=[s for s,r in receipts.items() if any(x['day']==key for x in r.get('partial_rows',[]))]
        labels[key]=daily_label(key,by_day.get(key,{}),watch,end,
            dict(requested=len(symbols),fetched=len(symbols)-len(missing),missing=sorted(set(missing+partial)),partial_native_symbols=partial,historical_PIT_certified=False))
        date+=timedelta(days=1)
    (output/'labels.json').write_text(json.dumps(labels,indent=2),encoding='utf-8')
    result=dict(status='COLLECTED_OBSERVED_UNIVERSE',days=len(labels),symbols_requested=len(symbols),symbols_fetched=len(symbols)-len(missing),
        missing=missing,leader_pairs=sum(len(r['leaders']) for r in labels.values()),historical_PIT_certified=False,
        labels_sha256=digest(output/'labels.json'),runtime_eligible=False)
    (output/'result.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--parent',type=Path,required=True)
    a=p.parse_args();collect(a.output,a.parent)
