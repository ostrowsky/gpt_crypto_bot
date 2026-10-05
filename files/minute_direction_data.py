"""Pinned public L2/trade archive and causal, gap-aware ten-second books."""
from __future__ import annotations
import argparse, hashlib, json, time, urllib.request
from concurrent.futures import ThreadPoolExecutor,ProcessPoolExecutor,as_completed
from pathlib import Path
import numpy as np

DATASET='MaximumLeverage/crypto-lob-stream'
REVISION='873f31e729ae23b1c309cd5dcb33feed27c407de'
SYMBOLS=('BTCUSDT','ETHUSDT','SOLUSDT')
STEP=10_000

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(2**20),b''):h.update(block)
    return h.hexdigest()

def fetch(url,path):
    with urllib.request.urlopen(url,timeout=120) as response,Path(path).open('wb') as f:
        for block in iter(lambda:response.read(2**20),b''):f.write(block)

def download(root):
    root.mkdir(parents=True,exist_ok=True)
    url=f'https://huggingface.co/api/datasets/{DATASET}/tree/{REVISION}?recursive=true&expand=false&limit=1000'
    raw=urllib.request.urlopen(url,timeout=40).read();(root/'tree.json').write_bytes(raw)
    entries=[r for r in json.loads(raw) if r['type']=='file' and r['path'].endswith('.parquet')]
    if len(entries)!=18:raise ValueError('Pinned archive coverage changed')
    manifest=dict(dataset=DATASET,revision=REVISION,files=entries,retrieved_at=time.time())
    (root/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    def one(row):
        path=root/row['path'];path.parent.mkdir(parents=True,exist_ok=True)
        expected=row['lfs']['oid']
        if path.exists() and path.stat().st_size==row['size'] and sha(path)==expected:return
        url=f"https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}/{row['path']}?download=true"
        partial=path.with_suffix('.partial')
        for attempt in range(4):
            try:
                fetch(url,partial)
                if partial.stat().st_size!=row['size'] or sha(partial)!=expected:raise ValueError('Source size/hash mismatch')
                partial.replace(path);print('Verified',row['path'],flush=True);return
            except Exception:
                if attempt==3:raise
                time.sleep(2)
    with ThreadPoolExecutor(max_workers=3) as pool:list(pool.map(one,sorted(entries,key=lambda r:r['size'])))
    return manifest

class Book:
    def __init__(self):
        from sortedcontainers import SortedDict
        self.bids=SortedDict();self.asks=SortedDict();self.last_id=None
        self.last_time=None;self.valid=False;self.segment=0
        self.stats=dict(anchors=0,gaps=0,stale_resets=0,crossed=0,updates=0,covered_updates=0)
    def anchor(self,t,uid,side,price,qty):
        self.bids.clear();self.asks.clear()
        if not np.isfinite(price).all() or not np.isfinite(qty).all() or (price<=0).any() or (qty<=0).any():
            raise ValueError('Invalid standing snapshot')
        for b,p,q in zip(side,price,qty):(self.bids if b else self.asks)[float(p)]=float(q)
        self.prune();self.last_id=int(uid);self.last_time=int(t);self.valid=True
        self.segment+=1;self.stats['anchors']+=1
        self.check()
    def prune(self):
        while len(self.bids)>1000:self.bids.popitem(index=0)
        while len(self.asks)>1000:self.asks.popitem(index=-1)
    def check(self):
        if not self.bids or not self.asks or self.bids.peekitem(-1)[0]>=self.asks.peekitem(0)[0]:
            self.valid=False;self.segment+=1;self.stats['crossed']+=1
    def update(self,t,first,last,side,price,qty,validated=False):
        if self.last_id is None or last<=self.last_id:
            self.stats['covered_updates']+=1;return
        if t<self.last_time:raise ValueError('Backward event clock')
        if first>self.last_id+1:
            if self.valid:self.segment+=1;self.stats['gaps']+=1
            self.valid=False
        if t-self.last_time>STEP:
            if self.valid:self.segment+=1;self.stats['stale_resets']+=1
            self.valid=False
        if not validated and (not np.isfinite(price).all() or not np.isfinite(qty).all() or (price<=0).any() or (qty<0).any()):
            raise ValueError('Invalid depth update')
        if self.valid:
            for b,p,q in zip(side,price,qty):
                levels=self.bids if b else self.asks;p=float(p)
                if q==0:levels.pop(p,None)
                else:levels[p]=float(q)
            self.prune();self.check()
        self.last_id=int(last);self.last_time=int(t);self.stats['updates']+=1
    def sample(self,at):
        if not self.valid or self.last_time is None or not 0<=at-self.last_time<=STEP or min(len(self.bids),len(self.asks))<10:
            return np.full(40,np.nan),-1
        bids=list(self.bids.items()[-10:])[::-1];asks=list(self.asks.items()[:10])
        return np.array([[ap,aq,bp,bq] for (bp,bq),(ap,aq) in zip(bids,asks)]).ravel(),self.segment

def grouped_depth(paths):
    """Do not split an atomic update when a Parquet batch/file ends."""
    import pyarrow.parquet as pq
    import pyarrow.compute as pc
    carry=None
    cols=['timestamp_ms','first_update_id','last_update_id','side','price','quantity']
    for path in paths:
        for batch in pq.ParquetFile(path).iter_batches(batch_size=262144,columns=cols):
            arrays=[batch.column(i).to_numpy(zero_copy_only=False) for i in (0,1,2)]
            arrays+=[pc.equal(batch.column(3),'bid').to_numpy(zero_copy_only=False)]
            arrays+=[batch.column(i).to_numpy(zero_copy_only=False) for i in (4,5)]
            if carry is not None:arrays=[np.concatenate([a,b]) for a,b in zip(carry,arrays)]
            if not np.isfinite(arrays[4]).all() or not np.isfinite(arrays[5]).all() or (arrays[4]<=0).any() or (arrays[5]<0).any():raise ValueError('Invalid raw depth prices/size')
            same=arrays[2][1:]==arrays[2][:-1]
            if np.any(same&((arrays[0][1:]!=arrays[0][:-1])|(arrays[1][1:]!=arrays[1][:-1]))):raise ValueError('Atomic update identity differs')
            ends=np.flatnonzero(arrays[2][1:]!=arrays[2][:-1])+1
            a=0
            for b in ends:
                yield int(arrays[0][a]),int(arrays[1][a]),int(arrays[2][a]),arrays[3][a:b],arrays[4][a:b],arrays[5][a:b]
                a=b
            carry=[v[a:] for v in arrays]
    if carry is not None and len(carry[0]):
        if np.any(carry[0]!=carry[0][0]) or np.any(carry[1]!=carry[1][0]):raise ValueError('Final atomic identity differs')
        yield int(carry[0][0]),int(carry[1][0]),int(carry[2][0]),carry[3],carry[4],carry[5]

def snapshots(paths):
    import pyarrow.parquet as pq
    import pyarrow.compute as pc
    rows=[]
    for path in paths:
        a=pq.read_table(path)
        t=a['timestamp_ms'].to_numpy();u=a['last_update_id'].to_numpy()
        side=pc.equal(a['side'],'bid').to_numpy();p=a['price'].to_numpy();q=a['quantity'].to_numpy()
        # Each anchor identity is time/id; repeated prices in an anchor fail.
        for at,uid in dict.fromkeys(zip(t,u)):
            mask=(t==at)&(u==uid)
            if len(set(zip(side[mask],p[mask])))!=int(mask.sum()):raise ValueError('Duplicate snapshot level')
            rows.append((int(at),int(uid),side[mask],p[mask],q[mask]))
    return sorted(rows,key=lambda x:x[0])

def reconstruct(depth,anchors,validated=False):
    book=Book();anchor_i=0;clock=None;times=[];values=[];segments=[]
    previous_event=None
    for t,first,last,side,price,qty in depth:
        if previous_event is not None and t<previous_event:raise ValueError('Unsorted depth timestamps')
        previous_event=t
        if clock is None:clock=(t//STEP+1)*STEP
        while clock<t:
            # Apply standing anchors available by this grid, before emitting it.
            while anchor_i<len(anchors) and anchors[anchor_i][0]<=clock:
                book.anchor(*anchors[anchor_i]);anchor_i+=1
            row,segment=book.sample(clock);times.append(clock);values.append(row);segments.append(segment);clock+=STEP
        while anchor_i<len(anchors) and anchors[anchor_i][0]<=t:
            book.anchor(*anchors[anchor_i]);anchor_i+=1
        book.update(t,first,last,side,price,qty,validated=validated)
        if book.stats['updates'] and book.stats['updates']%500000==0:print('Depth updates',book.stats['updates'],'grid',len(times),'valid',book.valid,flush=True)
    # Last grid after the last event is a causal sample, not new observation.
    if clock is not None and previous_event is not None and clock-previous_event<=STEP:
        row,segment=book.sample(clock);times.append(clock);values.append(row);segments.append(segment)
    return np.array(times,dtype=np.int64),np.asarray(values),np.array(segments,dtype=np.int64),book.stats

def aggregate_trades(paths,times):
    import pyarrow.parquet as pq
    import pyarrow.compute as pc
    sums=np.zeros((len(times),5));bad=np.zeros(len(times),dtype=bool);last_id=None;last_t=None
    for path in paths:
        for batch in pq.ParquetFile(path).iter_batches(batch_size=262144,columns=['timestamp_ms','trade_id','price','quantity','buyer_maker']):
            t,uid,p,q,m=[batch.column(i).to_numpy(zero_copy_only=False) for i in range(5)]
            if (np.diff(t)<0).any() or (last_t is not None and t[0]<last_t):raise ValueError('Backward trade clock')
            gap=np.r_[last_id is not None and uid[0]!=last_id+1,np.diff(uid)!=1]
            if not np.isfinite(p).all() or not np.isfinite(q).all() or (p<=0).any() or (q<=0).any():raise ValueError('Invalid trades')
            index=((t//STEP+1)*STEP-times[0])//STEP
            valid=(index>=0)&(index<len(times));idx=index[valid].astype(int)
            for j,v in enumerate((np.ones(len(t)),q,np.where(m,0,q),p*q,np.where(m,-p*q,p*q))):
                sums[:,j]+=np.bincount(idx,weights=v[valid],minlength=len(times))
            bad[index[gap&valid].astype(int)]=True;last_id=int(uid[-1]);last_t=int(t[-1])
    return sums,bad

def prepare_symbol(request):
    root,output,symbol=request
    root=Path(root);output=Path(output)
    files=lambda kind:sorted((root/kind/'binance'/symbol).glob('*.parquet'))
    anchors=snapshots(files('snapshots'))
    print('Reconstructing',symbol,'anchors',len(anchors),flush=True)
    ticks,books,segments,stats=reconstruct(grouped_depth(files('depth')),anchors,validated=True)
    flow,trade_gaps=aggregate_trades(files('trades'),ticks)
    np.savez_compressed(output/(symbol+'.npz'),time=ticks,book=books,segment=segments,flow=flow,trade_gap=trade_gaps)
    valid=segments>=0
    result=dict(start=int(ticks[0]),end=int(ticks[-1]),samples=len(ticks),valid=int(valid.sum()),
                invalid=int((~valid).sum()),trade_gap_bins=int(trade_gaps.sum()),book_audit=stats,
                sha256=sha(output/(symbol+'.npz')))
    print('Prepared',symbol,result,flush=True)
    return symbol,result

def prepare(root,output):
    output.mkdir(parents=True,exist_ok=False)
    manifest=json.loads((root/'manifest.json').read_bytes())
    if manifest['dataset']!=DATASET or manifest['revision']!=REVISION:raise ValueError('Dataset revision differs')
    for row in manifest['files']:
        if sha(root/row['path'])!=row['lfs']['oid']:raise ValueError('Raw file drift')
    coverage={}
    with ProcessPoolExecutor(max_workers=2) as pool:
        jobs=[pool.submit(prepare_symbol,(str(root),str(output),symbol)) for symbol in SYMBOLS]
        for future in as_completed(jobs):symbol,result=future.result();coverage[symbol]=result
    (output/'coverage.json').write_text(json.dumps(dict(source_manifest_sha256=sha(root/'manifest.json'),
        source_hash=sha(__file__),dataset=DATASET,revision=REVISION,assets=coverage),indent=2),encoding='utf-8')

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=['download','prepare'])
    parser.add_argument('--data',type=Path,default=Path('.runtime/minute_direction_data'))
    parser.add_argument('--output',type=Path,default=Path('.runtime/minute_direction_books'))
    args=parser.parse_args();download(args.data) if args.action=='download' else prepare(args.data,args.output)
