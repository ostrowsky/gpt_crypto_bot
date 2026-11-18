"""RFD-1 causal receipt-clock features. Offline diagnostics, no trading API."""
from collections import deque
from datetime import datetime, timedelta
import hashlib
import json
import math
from pathlib import Path
from zoneinfo import ZoneInfo

HORIZONS=(5,10,30)


def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.next')
    temporary.write_text(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False),encoding='utf-8')
    temporary.replace(path)


def file_sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block:=stream.read(65536):digest.update(block)
    return digest.hexdigest()


def freeze(path,value):
    path=Path(path)
    if path.exists():
        if read_frozen(path)!=value:raise ValueError('REGISTRATION_CHANGED')
        return
    write_json(path,value)
    path.with_suffix(path.suffix+'.sha256').write_text(file_sha(path),encoding='ascii')


def read_frozen(path):
    path=Path(path)
    if file_sha(path)!=path.with_suffix(path.suffix+'.sha256').read_text(encoding='ascii'):
        raise ValueError('ARTIFACT_HASH')
    return json.loads(path.read_text(encoding='utf-8'))


def book_ofi(old,new):
    b,q,a,v=old;bn,qn,an,vn=new
    return (qn*(bn>=b)-q*(bn<=b))-(vn*(an<=a)-v*(an>=a))


class ReceiptGrid:
    """Strict-before snapshot grid, bounded ten-second per-symbol event context."""
    def __init__(self,symbols,*,start_ms,end_ms):
        self.symbols=tuple(symbols);self.next_ms=start_ms;self.end_ms=end_ms
        self.last_ms=None;self.active=False
        self.states={s:self._new() for s in symbols}

    @staticmethod
    def _new(epoch=0):
        return dict(epoch=epoch,book=None,trade_ms=None,trades=deque(),ofi=deque(),
                    history=deque(maxlen=11),previous_valid=False)

    def _reset(self,symbol):
        self.states[symbol]=self._new(self.states[symbol]['epoch']+1)

    def _snapshot(self,symbol,t):
        s=self.states[symbol];book=s['book']
        valid=(self.active and book is not None and s['trade_ms'] is not None and
               0<t-book['at_ms']<=1000 and 0<t-s['trade_ms']<=5000)
        if not valid and s['previous_valid']:
            s['epoch']+=1;s['history'].clear()
        s['previous_valid']=bool(valid)
        for name in ('trades','ofi'):
            while s[name] and s[name][0][0]<t-10000:s[name].popleft()
        mid=(book['bid']+book['ask'])/2 if valid else None
        s['history'].append((t,mid,s['epoch']))
        row=dict(symbol=symbol,at_ms=t,feature_end_ms=t-1,epoch=s['epoch'],
                 valid=bool(valid),mid=mid,features=None)
        history=s['history']
        if (not valid or len(history)!=11 or
                any(m is None or e!=s['epoch'] or at!=t-(10-i)*1000
                    for i,(at,m,e) in enumerate(history))):return row
        prices=[x[1] for x in history]
        returns=[math.log(b/a)*10000 for a,b in zip(prices,prices[1:])]
        p={f'return_{w}':math.log(mid/prices[-1-w])*10000 for w in (1,5,10)}
        p['rv_10']=math.sqrt(sum(v*v for v in returns))
        b=dict(p)
        b['spread_bps']=(book['ask']-book['bid'])/mid*10000
        b['micro_bps']=((book['ask']*book['bq']+book['bid']*book['aq'])/
                         (book['bq']+book['aq'])/mid-1)*10000
        for n in (1,5,20):
            qb=sum(q for _,q in book['bids'][:n]);qa=sum(q for _,q in book['asks'][:n])
            b[f'depth_imbalance_{n}']=(qb-qa)/(qb+qa)
        depth=sum(q for _,q in book['bids']+book['asks'])
        b['log_depth_20']=math.log1p(depth);b['quote_age_ms']=t-book['at_ms']
        f=dict(b)
        for w in (1,5,10):
            trades=[v for at,v in s['trades'] if at>=t-w*1000]
            qty=sum(abs(v) for v in trades)
            f[f'trade_imbalance_{w}']=sum(trades)/qty if qty else 0.
            f[f'log_trade_qty_{w}']=math.log1p(qty)
            f[f'trade_count_{w}']=len(trades)
            f[f'book_ofi_{w}']=sum(v for at,v in s['ofi'] if at>=t-w*1000)/(depth or 1.)
        row['features']={'P':p,'B':b,'F':f}
        return row

    def _advance(self,until):
        while self.next_ms<=until and self.next_ms<self.end_ms:
            t=self.next_ms
            self.next_ms+=1000
            for symbol in self.symbols:yield self._snapshot(symbol,t)

    def feed(self,record,*,parsed=None):
        wall=record['wall_ms']
        if self.last_ms is not None and wall<self.last_ms:raise ValueError('RECEIPT_REGRESSION')
        self.last_ms=wall
        yield from self._advance(wall)
        kind=record['kind']
        if kind in ('SESSION_START','SESSION_END') or record.get('stream')=='!serverShutdown':
            for symbol in self.symbols:self._reset(symbol)
            self.active=kind=='SESSION_START'
            return
        if kind!='MARKET':raise ValueError('UNKNOWN_RECORD')
        v=record['validation']
        if not v['valid']:raise ValueError('INVALID_FRAME')
        topic=record['stream'];symbol=topic.split('@')[0].upper()
        if symbol not in self.states:raise ValueError('UNEXPECTED_SYMBOL')
        if v['continuity']=='DUPLICATE':return
        if v['continuity']=='GAP':self._reset(symbol)
        s=self.states[symbol]
        data=parsed if parsed is not None else json.loads(record['raw'])['data']
        if topic.endswith('@aggTrade'):
            s['trade_ms']=wall
            s['trades'].append((wall,float(data['q'])*(-1 if data['m'] else 1)))
        else:
            bids=[(float(p),float(q)) for p,q in data['bids']]
            asks=[(float(p),float(q)) for p,q in data['asks']]
            new=dict(at_ms=wall,bid=bids[0][0],bq=bids[0][1],ask=asks[0][0],aq=asks[0][1],bids=bids,asks=asks)
            old=s['book']
            if old is not None:
                s['ofi'].append((wall,book_ofi(tuple(old[k] for k in ('bid','bq','ask','aq')),
                                                tuple(new[k] for k in ('bid','bq','ask','aq')))))
            s['book']=new

    def finish(self):
        return self._advance(self.end_ms-1)


def revalidate(validator,record):
    """Replay producer validation, never trust saved validity as training input."""
    kind=record['kind'];wall=record['wall_ms'];mono=record['mono_ns'];sid=record['session_id']
    if (validator.last_wall is not None and wall<validator.last_wall or
            validator.last_mono is not None and mono<validator.last_mono):raise ValueError('RECEIPT_REGRESSION')
    if kind=='SESSION_START':
        validator.begin_session(sid,wall_ms=wall,mono_ns=mono);return None
    if sid!=validator.session_id:raise ValueError('SESSION_OWNERSHIP')
    if kind=='SESSION_END':
        validator.last_wall,validator.last_mono=wall,mono
        return None
    if kind!='MARKET':raise ValueError('UNKNOWN_RECORD')
    result=validator.ingest(record['raw'],wall_ms=wall,mono_ns=mono,session_id=sid)
    if result['validation']!=record['validation'] or result['stream']!=record['stream']:
        raise ValueError('RAW_VALIDATION_MISMATCH')
    if not result['validation']['valid']:raise ValueError('INVALID_FRAME')
    return json.loads(record['raw'])['data']


def calendar_splits(registration):
    zone=ZoneInfo(registration['timezone'])
    first=datetime.fromtimestamp(registration['full_days_start_ms']/1000,zone)
    bounds=[int((first+timedelta(days=d)).timestamp()*1000) for d in (0,3,4,5,7)]
    if bounds[-1]!=registration['planned_end_ms']:raise ValueError('CALENDAR_SCHEDULE')
    return dict(zip(('train','tune','cal','test'),zip(bounds,bounds[1:])))


def matched_examples(rows,start,end):
    by_symbol={}
    for row in rows:by_symbol.setdefault(row['symbol'],{})[row['at_ms']]=row
    issued=[]
    for symbol,grid in by_symbol.items():
        for t,row in sorted(grid.items()):
            if not start<=t<end or t%5000 or row['features'] is None:continue
            targets=None
            if t+30000<end:
                future=[grid.get(t+d*1000) for d in range(1,31)]
                if all(r is not None and r['valid'] and r['epoch']==row['epoch'] for r in future):
                    targets={h:math.log(grid[t+h*1000]['mid']/row['mid'])*10000 for h in HORIZONS}
            issued.append(dict(row,targets=targets))
    return sorted(issued,key=lambda r:(r['at_ms'],r['symbol']))


def majority_direction(targets):
    return 1 if sum(y>0 for y in targets)>=sum(y<0 for y in targets) else -1


def metrics(targets,predictions,*,majority_sign):
    if len(targets)!=len(predictions):raise ValueError('METRIC_LENGTH')
    if any(not math.isfinite(v) for v in list(targets)+list(predictions)):raise ValueError('NONFINITE_METRIC')
    n=len(targets);direction_n=sum(y!=0 for y in targets)
    sign=lambda x:1 if x>0 else -1 if x<0 else 0
    correct=sum(y!=0 and sign(y)==sign(p) for y,p in zip(targets,predictions))
    return dict(N=n,mae_bps=sum(abs(y-p) for y,p in zip(targets,predictions))/n if n else None,
        rmse_bps=math.sqrt(sum((y-p)**2 for y,p in zip(targets,predictions))/n) if n else None,
        zero_mae_bps=sum(abs(y) for y in targets)/n if n else None,
        zero_rmse_bps=math.sqrt(sum(y*y for y in targets)/n) if n else None,
        direction_correct_n=correct,direction_N=direction_n,direction_accuracy=correct/direction_n if direction_n else None,
        three_sign_correct_n=sum(sign(y)==sign(p) for y,p in zip(targets,predictions)),three_sign_N=n,
        positive_n=sum(y>0 for y in targets),negative_n=sum(y<0 for y in targets),unchanged_n=sum(y==0 for y in targets),
        majority_sign=majority_sign,majority_correct_n=sum(y!=0 and sign(y)==majority_sign for y in targets))


def conformal_radius(residuals,coverage=.9):
    if not residuals or not 0<coverage<1 or any(not math.isfinite(v) or v<0 for v in residuals):
        raise ValueError('CALIBRATION_INPUT')
    rank=math.ceil((len(residuals)+1)*coverage)
    return sorted(residuals)[rank-1] if rank<=len(residuals) else math.inf


def chart_origins(issued,symbols,thresholds):
    result=[]
    for symbol in symbols:
        times=sorted(r['at_ms'] for r in issued if r['symbol']==symbol and r['features'] is not None)
        for threshold in thresholds:
            chosen=next((t for t in times if t>=threshold),None)
            if chosen is not None:result.append((symbol,chosen))
    return result
