"""Verified standing Binance UM top-20 states; never replay them as depth diffs."""
from __future__ import annotations
import argparse,csv,io,json,time,urllib.request,urllib.parse,zipfile
from concurrent.futures import ThreadPoolExecutor,ProcessPoolExecutor,as_completed
from pathlib import Path
import numpy as np
from minute_direction_data import sha,fetch,SYMBOLS,STEP

DATASET='predict-quant/binance-future-orderbook'
REVISION='b8590b83452d7a32fbb274ff7741b6db000b3984'

def download(root):
    root.mkdir(parents=True,exist_ok=True)
    url=f'https://huggingface.co/api/datasets/{DATASET}/tree/{REVISION}?recursive=true&expand=false&limit=1000'
    raw=urllib.request.urlopen(url,timeout=45).read();(root/'tree.json').write_bytes(raw)
    rows=[r for r in json.loads(raw) if r['type']=='file' and r['path'].endswith('.parquet') and r['path'].split('/')[0] in SYMBOLS]
    if len(rows)!=90:raise ValueError('Pinned top-20 archive changed')
    manifest=dict(dataset=DATASET,revision=REVISION,files=rows,retrieved_at=time.time())
    (root/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    def one(row):
        path=root/row['path'];path.parent.mkdir(exist_ok=True);expected=row['lfs']['oid']
        if path.exists() and path.stat().st_size==row['size'] and sha(path)==expected:return
        url=f"https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}/{row['path']}?download=true"
        for attempt in range(4):
            try:
                partial=path.with_suffix('.partial');fetch(url,partial)
                if partial.stat().st_size!=row['size'] or sha(partial)!=expected:raise ValueError('Source size/hash mismatch')
                partial.replace(path);print('Verified state file',row['path'],flush=True);return
            except Exception:
                if attempt==3:raise
                time.sleep(2)
    with ThreadPoolExecutor(max_workers=3) as pool:list(pool.map(one,rows))
    receipts=[]
    for symbol in SYMBOLS:
        for month in ('2026-03','2026-04'):
            path=root/f'{symbol}-1m-{month}.zip'
            url=f'https://data.binance.vision/data/futures/um/monthly/klines/{symbol}/1m/{path.name}'
            check=urllib.request.urlopen(url+'.CHECKSUM',timeout=45).read().decode().split()[0]
            if not path.exists() or sha(path)!=check:fetch(url,path)
            if sha(path)!=check:raise ValueError('Official kline checksum differs')
            receipts.append(dict(file=path.name,url=url,sha256=check))
        # Entire funding interval, all available records, no secret API/key.
        params=dict(symbol=symbol,startTime=1772668800000,endTime=1775779200000,limit=1000)
        url='https://fapi.binance.com/fapi/v1/fundingRate?'+urllib.parse.urlencode(params)
        raw=urllib.request.urlopen(url,timeout=45).read();funding=json.loads(raw)
        if not isinstance(funding,list) or not funding or len(funding)==1000:raise ValueError('Incomplete funding response')
        path=root/(symbol+'_funding.json');path.write_bytes(raw)
        receipts.append(dict(file=path.name,url=url,sha256=sha(path),n=len(funding)))
        print('Official volume/funding complete',symbol,flush=True)
    manifest['official_receipts']=receipts
    (root/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')

def validate_state(bids,asks):
    b=np.array(bids,dtype=float);a=np.array(asks,dtype=float)
    if b.shape!=(20,2) or a.shape!=(20,2):raise ValueError('Not a full top-20 state')
    if not np.isfinite(b).all() or not np.isfinite(a).all() or (b<=0).any() or (a<=0).any():raise ValueError('Invalid standing levels')
    if (np.diff(b[:,0])>=0).any() or (np.diff(a[:,0])<=0).any() or b[0,0]>=a[0,0]:raise ValueError('Unordered/crossed standing book')
    return np.column_stack([a[:10,0],a[:10,1],b[:10,0],b[:10,1]]).ravel()

def sample_states(paths):
    import pyarrow.parquet as pq
    import pyarrow as pa
    pa.set_cpu_count(1)
    clock=None;previous=None;last_time=None;session=0;times=[];books=[];segments=[]
    audit=dict(raw_rows=0,clock_gaps=0,invalid_states=0,stale_samples=0,backward_rows=0)
    for path in paths:
        for batch in pq.ParquetFile(path).iter_batches(batch_size=65536,columns=['E','bids','asks']):
            t=batch.column(0).to_numpy();b=batch.column(1);a=batch.column(2)
            if not np.isfinite(t).all() or (t<=0).any():raise ValueError('Missing/invalid event timestamp')
            # Late full states cannot overwrite a newer state or be sorted back
            # into the past. Keep recorded order, quarantine their older clocks.
            floor=np.maximum.accumulate(t)
            if last_time is not None:floor=np.maximum(floor,last_time)
            mapping=np.flatnonzero(t>=floor);audit['raw_rows']+=len(t);audit['backward_rows']+=len(t)-len(mapping)
            if not len(mapping):continue
            t=t[mapping]
            delta=np.r_[STEP+1 if last_time is None else t[0]-last_time,np.diff(t)]
            gaps=delta>STEP;ids=session+np.cumsum(gaps);session=int(ids[-1]);audit['clock_gaps']+=int(gaps.sum())
            if clock is None:clock=(int(t[0])//STEP+1)*STEP
            while clock<int(t[-1]):
                i=int(np.searchsorted(t,clock,side='right')-1)
                candidate=(int(t[i]),b[int(mapping[i])].as_py(),a[int(mapping[i])].as_py(),int(ids[i])) if i>=0 else previous
                if candidate is None or not 0<=clock-candidate[0]<=STEP:
                    row=np.full(40,np.nan);segment=-1;audit['stale_samples']+=1
                else:
                    try:row=validate_state(json.loads(candidate[1]),json.loads(candidate[2]));segment=candidate[3]
                    except ValueError:row=np.full(40,np.nan);segment=-1;audit['invalid_states']+=1
                times.append(clock);books.append(row);segments.append(segment);clock+=STEP
            previous=(int(t[-1]),b[int(mapping[-1])].as_py(),a[int(mapping[-1])].as_py(),int(ids[-1]));last_time=int(t[-1])
        print('Sampled',path.name,'grid',len(times),flush=True)
    if previous is not None and 0<=clock-previous[0]<=STEP:
        try:row=validate_state(json.loads(previous[1]),json.loads(previous[2]));segment=previous[3]
        except ValueError:row=np.full(40,np.nan);segment=-1;audit['invalid_states']+=1
        times.append(clock);books.append(row);segments.append(segment)
    return np.array(times,dtype=np.int64),np.array(books),np.array(segments,dtype=np.int64),audit

def closed_volume(root,symbol,times):
    flow=np.zeros((len(times),5));bad=np.zeros(len(times),dtype=bool);seen=set()
    for path in sorted(root.glob(symbol+'-1m-*.zip')):
        with zipfile.ZipFile(path) as z:
            if len(z.namelist())!=1:raise ValueError('Unexpected official archive members')
            with z.open(z.namelist()[0]) as f:
                reader=csv.reader(io.TextIOWrapper(f,encoding='utf-8'))
                for row in reader:
                    if row[0] in ('open_time','Open time'):continue
                    if len(row)!=12:raise ValueError('Official kline schema changed')
                    opened=int(row[0]);closed=int(row[6])
                    if closed!=opened+59999 or opened in seen:raise ValueError('Invalid/duplicate official candle')
                    seen.add(opened);available=opened+60000+STEP
                    if not times[0]<=available<=times[-1]:continue
                    i=(available-int(times[0]))//STEP
                    qty=float(row[5]);buy=float(row[9]);notional=float(row[7]);buyvalue=float(row[10])
                    if min(qty,buy,notional,buyvalue)<0 or buy>qty+1e-8:raise ValueError('Invalid official volume')
                    flow[i]=[float(row[8]),qty,buy,notional,2*buyvalue-notional]
    for opened in range((int(times[0])//60000-1)*60000,int(times[-1])+1,60000):
        available=opened+60000+STEP
        if times[0]<=available<=times[-1] and opened not in seen:bad[(available-int(times[0]))//STEP]=True
    return flow,bad

def prepare_symbol(request):
    root,output,symbol=request;root=Path(root);output=Path(output)
    times,books,segments,audit=sample_states(sorted((root/symbol).glob('*.parquet')))
    flow,bad=closed_volume(root,symbol,times)
    funding=json.loads((root/(symbol+'_funding.json')).read_bytes())
    funding=np.array([[int(r['fundingTime']),float(r['fundingRate']),float(r['markPrice'])] for r in funding],dtype=float)
    if not np.isfinite(funding).all() or (np.diff(funding[:,0])<=0).any() or (funding[:,2]<=0).any():raise ValueError('Invalid funding')
    np.savez_compressed(output/(symbol+'.npz'),time=times,book=books,segment=segments,flow=flow,trade_gap=bad,funding=funding)
    valid=segments>=0
    result=dict(start=int(times[0]),end=int(times[-1]),samples=len(times),valid=int(valid.sum()),invalid=int((~valid).sum()),
                trade_gap_bins=int(bad.sum()),state_audit=audit,sha256=sha(output/(symbol+'.npz')))
    print('Prepared independent states',symbol,result,flush=True);return symbol,result

def prepare(root,output):
    output.mkdir(parents=True,exist_ok=False);source=sha(__file__)
    manifest=json.loads((root/'manifest.json').read_bytes())
    if manifest['dataset']!=DATASET or manifest['revision']!=REVISION:raise ValueError('Source revision differs')
    for row in manifest['files']:
        if sha(root/row['path'])!=row['lfs']['oid']:raise ValueError('State data drift')
    for row in manifest['official_receipts']:
        if sha(root/row['file'])!=row['sha256']:raise ValueError('Official data drift')
    coverage={}
    with ProcessPoolExecutor(max_workers=3) as pool:
        jobs=[pool.submit(prepare_symbol,(str(root),str(output),s)) for s in SYMBOLS]
        for future in as_completed(jobs):symbol,report=future.result();coverage[symbol]=report
    if sha(__file__)!=source:raise ValueError('Preparation source changed')
    (output/'coverage.json').write_text(json.dumps(dict(source_manifest_sha256=sha(root/'manifest.json'),source_hash=source,
        dataset=DATASET,revision=REVISION,market='Binance USD-M perpetual',assets=coverage),indent=2),encoding='utf-8')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['download','prepare'])
    p.add_argument('--data',type=Path,default=Path('.runtime/minute_direction_partial_data'))
    p.add_argument('--output',type=Path,default=Path('.runtime/minute_direction_partial_books'))
    a=p.parse_args();download(a.data) if a.action=='download' else prepare(a.data,a.output)
