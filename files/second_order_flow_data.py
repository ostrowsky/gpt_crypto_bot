"""Causal one-second books and OFI from every pinned ~100ms standing state."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from minute_direction_data import sha,SYMBOLS

STEP=1000
MAX_AGE=250
MAX_GAP=500

def quote_pattern():
    number=r'[0-9]+(?:\.[0-9]+)?'
    level=lambda k:rf'\[\s*"(?P<p{k}>{number})"\s*,\s*"(?P<q{k}>{number})"\s*\]'
    return r'^\[\s*'+r'\s*,\s*'.join(level(k) for k in range(5))

def parse_side(strings):
    import pyarrow as pa
    import pyarrow.compute as pc
    found=pc.extract_regex(strings,quote_pattern());cols=[]
    missing=pc.is_null(found)
    for k in range(5):
        for key in (f'p{k}',f'q{k}'):
            field=pc.if_else(missing,pa.scalar(None,type=pa.string()),found.field(key))
            cols.append(pc.cast(field,pa.float64()).to_numpy(zero_copy_only=False))
    a=np.column_stack(cols)
    # Pinned strings contain twenty rank pairs; unexpected formatting is rejected.
    count=pc.count_substring(strings,pattern='], [').to_numpy(zero_copy_only=False)
    a[count!=19]=np.nan
    return a

def validate_quotes(bid,ask):
    good=np.isfinite(bid).all(axis=1)&np.isfinite(ask).all(axis=1)&(bid>0).all(axis=1)&(ask>0).all(axis=1)
    good&=(np.diff(bid[:,0::2],axis=1)<0).all(axis=1)&(np.diff(ask[:,0::2],axis=1)>0).all(axis=1)
    good&=bid[:,0]<ask[:,0]
    return good

def order_flow(previous,current):
    """Each rank: positive buy-side supply/demand pressure, negative sell-side."""
    bp,bq,ap,aq=previous[:,0::4],previous[:,1::4],previous[:,2::4],previous[:,3::4]
    nbp,nbq,nap,naq=current[:,0::4],current[:,1::4],current[:,2::4],current[:,3::4]
    return (nbp>=bp)*nbq-(nbp<=bp)*bq-(nap<=ap)*naq+(nap>=ap)*aq

class Sampler:
    def __init__(self):
        self.clock=None;self.last_time=None;self.previous=None;self.previous_valid=False;self.session=0
        self.carry=np.zeros(5);self.carry_count=0;self.carry_reset=0
        self.audit=dict(raw_rows=0,backward_rows=0,invalid_states=0,source_gaps=0,stale_seconds=0,reset_seconds=0)
    def push(self,time,bid,ask):
        time=np.asarray(time,dtype=np.int64);self.audit['raw_rows']+=len(time)
        floor=np.maximum.accumulate(time)
        if self.last_time is not None:floor=np.maximum(floor,self.last_time)
        keep=time>=floor;self.audit['backward_rows']+=int((~keep).sum())
        time=time[keep];bid=bid[keep];ask=ask[keep]
        if not len(time):return None
        valid=validate_quotes(bid,ask);self.audit['invalid_states']+=int((~valid).sum())
        quote=np.column_stack([bid[:,0::2],bid[:,1::2],ask[:,0::2],ask[:,1::2]]).reshape(-1,4,5).transpose(0,2,1).reshape(-1,20)
        prior=np.vstack([quote[0] if self.previous is None else self.previous,quote[:-1]])
        delta=np.r_[MAX_GAP+1 if self.last_time is None else time[0]-self.last_time,np.diff(time)]
        prev_valid=np.r_[self.previous_valid,valid[:-1]]
        resets=(delta>MAX_GAP)|(~valid)|(~prev_valid)
        ids=self.session+np.cumsum(resets);self.audit['source_gaps']+=int((delta>MAX_GAP).sum())
        ofi=order_flow(prior,quote);ofi[resets]=0
        if self.clock is None:self.clock=int((time[0]+STEP-1)//STEP*STEP)
        clocks=np.arange(self.clock,int(time[-1]),STEP,dtype=np.int64)
        output=None
        if len(clocks):
            index=np.searchsorted(time,clocks,side='right')-1
            q=quote[np.maximum(index,0)].copy();sessions=ids[np.maximum(index,0)].copy()
            stamp=time[np.maximum(index,0)].copy();okay=valid[np.maximum(index,0)].copy()
            before=index<0
            if before.any():
                q[before]=np.nan if self.previous is None else self.previous
                stamp[before]=time[0] if self.last_time is None else self.last_time
                sessions[before]=self.session;okay[before]=self.previous_valid
            age=clocks-stamp;okay&=(age>=0)&(age<=MAX_AGE)
            prefix=np.vstack([np.zeros(5),np.cumsum(ofi,axis=0)])
            lo=np.searchsorted(time,clocks-STEP,side='right');hi=index+1
            flow=prefix[hi]-prefix[lo];counts=hi-lo
            reset_prefix=np.r_[0,np.cumsum(resets)]
            bin_resets=reset_prefix[hi]-reset_prefix[lo];bin_resets[0]+=self.carry_reset
            self.audit['stale_seconds']+=int((~okay).sum());self.audit['reset_seconds']+=int((bin_resets>0).sum())
            okay&=bin_resets==0;q[~okay]=np.nan;sessions[~okay]=-1
            flow[0]+=self.carry;counts[0]+=self.carry_count
            flow[bin_resets>0]=0
            self.carry=prefix[-1]-prefix[hi[-1]];self.carry_count=len(time)-int(hi[-1])
            self.carry_reset=int(reset_prefix[-1]-reset_prefix[hi[-1]])
            output=dict(time=clocks,book=q,segment=sessions,flow=flow,events=counts,age=age.astype(np.float32))
            self.clock=int(clocks[-1]+STEP)
        else:
            self.carry+=ofi.sum(axis=0);self.carry_count+=len(time);self.carry_reset+=int(resets.sum())
        self.previous=quote[-1].copy();self.previous_valid=bool(valid[-1]);self.last_time=int(time[-1]);self.session=int(ids[-1])
        return output

def prepare(root,output):
    import pyarrow.parquet as pq
    import pyarrow as pa
    pa.set_cpu_count(1)
    output.mkdir(parents=True,exist_ok=False)
    manifest=json.loads((root/'manifest.json').read_bytes())
    if manifest['dataset']!='predict-quant/binance-future-orderbook' or manifest['revision']!='b8590b83452d7a32fbb274ff7741b6db000b3984':
        raise ValueError('Wrong source archive')
    if len(manifest['files'])!=90:raise ValueError('Maximum archive coverage changed')
    for row in manifest['files']:
        if sha(root/row['path'])!=row['lfs']['oid']:raise ValueError('Original source drift')
    coverage={};source=sha(__file__)
    for symbol in SYMBOLS:
        sampler=Sampler();parts=[];directory=output/symbol;directory.mkdir()
        for part,path in enumerate(sorted((root/symbol).glob('*.parquet'))):
            blocks=[]
            for batch in pq.ParquetFile(path).iter_batches(batch_size=65536,columns=['E','bids','asks']):
                time=batch.column(0).to_numpy()
                if not np.isfinite(time).all() or (time<=0).any():raise ValueError('Invalid source event clocks')
                block=sampler.push(time,parse_side(batch.column(1)),parse_side(batch.column(2)))
                if block is not None:blocks.append(block)
            if blocks:
                arrays={k:np.concatenate([b[k] for b in blocks]) for k in blocks[0]}
                dest=directory/f'{part:03d}.npz';np.savez_compressed(dest,**arrays)
                parts.append(dict(file=dest.name,sha256=sha(dest),n=len(arrays['time']),start=int(arrays['time'][0]),end=int(arrays['time'][-1]),
                    valid=int((arrays['segment']>=0).sum())))
            print('One-second OFI',symbol,path.name,'audit',sampler.audit,flush=True)
        coverage[symbol]=dict(parts=parts,audit=sampler.audit)
    if sha(__file__)!=source:raise ValueError('Preparation code changed during run')
    report=dict(status='PREPARED_CAUSAL_BOOK_OFI',source_manifest_sha256=sha(root/'manifest.json'),source_hash=source,
        step_ms=STEP,max_quote_age_ms=MAX_AGE,max_source_gap_ms=MAX_GAP,assets=coverage)
    (output/'coverage.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('COMPLETE one-second OFI',output,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();prepare(a.data,a.output)
