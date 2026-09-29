"""Separate, resumable closed-candle cache; never mutate original history."""
import argparse
import asyncio
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import time

import aiohttp

ROOT = Path(__file__).resolve().parent.parent
STEPS = {'15m': 900000, '1h': 3600000}


def valid(row, step):
    try:
        t = int(row['t'])
        o,h,l,c,v = (float(row[k]) for k in ('o','h','l','c','v'))
        return t % step == 0 and all(math.isfinite(x) for x in (o,h,l,c,v)) and min(o,h,l,c)>0 and v>=0 and h>=max(o,c,l) and l<=min(o,c,h)
    except (KeyError,TypeError,ValueError,OverflowError):
        return False


def merge_interior(fragments, start, end, step):
    rows, conflicts, errors, sources = {}, set(), 0, []
    for path in fragments:
        raw = path.read_bytes()
        sources.append({'file':path.name,'sha256':hashlib.sha256(raw).hexdigest()})
        try:
            payload = json.loads(raw)
            if not isinstance(payload,list): raise ValueError('not list')
            last = max((int(r['t']) for r in payload), default=-1)
            for r in payload:
                t = int(r['t'])
                # A successor, not filename horizon, proves this is not the active tail.
                if not start <= t < end or t >= last: continue
                if not valid(r,step):
                    conflicts.add(t)
                    continue
                r = {k: int(r[k]) if k=='t' else float(r[k]) for k in ('t','o','h','l','c','v')}
                if t in rows and rows[t] != r: conflicts.add(t)
                rows[t] = r
        except (ValueError,KeyError,TypeError): errors += 1
    for t in conflicts: rows.pop(t,None)
    return rows, conflicts, errors, sources


def exchange_rows(payload, step, now_ms):
    out = []
    if not isinstance(payload,list): raise ValueError('exchange response not list')
    for r in payload:
        row = dict(zip(('t','o','h','l','c','v'),[int(r[0]),*(float(x) for x in r[1:6])]))
        if int(r[6]) != row['t']+step-1 or row['t']+step > now_ms: continue
        if not valid(row,step): raise ValueError('invalid exchange candle')
        out.append(row)
    return out


def atomic_json(path, obj):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj,separators=(',',':')),encoding='utf-8')
    tmp.replace(path)


async def recover_one(session, sem, sym, tf, fragments, start, end, output):
    step = STEPS[tf]
    start = (start+step-1)//step*step
    end = end//step*step
    name = f'{sym}_{tf}_{start}_{end}'
    dest, meta = output/(name+'.json'), output/(name+'.manifest.json')
    if dest.exists() and meta.exists():
        old=json.loads(meta.read_bytes())
        if old.get('status')=='complete' and hashlib.sha256(dest.read_bytes()).hexdigest()==old.get('sha256'):
            return old
    async with sem:
        rows,conflicts,errors,sources = await asyncio.to_thread(merge_interior,fragments,start,end,step)
        expected = set(range(start,end,step))
        missing = sorted(expected-rows.keys())
        local_count=len(rows); fetched=0; failure=None
        while missing:
            a=missing[0]
            # Bound each request to a contiguous missing run, at most 1000 bars.
            b=a+step
            while b<end and b not in rows and b<a+1000*step: b+=step
            params=dict(symbol=sym,interval=tf,startTime=a,endTime=b-1,limit=1000)
            payload=None
            for attempt in range(3):
                try:
                    async with session.get('https://api.binance.com/api/v3/klines',params=params) as response:
                        if response.status in (418,429):
                            failure=f'rate_limit_{response.status}'
                            break
                        response.raise_for_status()
                        payload=await response.json()
                    break
                except (aiohttp.ClientError,asyncio.TimeoutError) as exc:
                    failure=type(exc).__name__
                    await asyncio.sleep(attempt+1)
            if payload is None: break
            accepted=exchange_rows(payload,step,int(time.time()*1000))
            additions={r['t']:r for r in accepted if a<=r['t']<b}
            if a not in additions:
                failure='exchange_missing_requested_start'
                break
            rows.update(additions); fetched+=len(additions)
            missing=sorted(expected-rows.keys())
            failure=None
            await asyncio.sleep(.15)
        missing=sorted(expected-rows.keys())
        result=dict(symbol=sym,tf=tf,start=start,end=end,status='complete' if not missing else 'unknown',
                    local_validated_rows=local_count,exchange_refetched_rows=fetched,
                    conflicts=len(conflicts),read_errors=errors,missing_count=len(missing),
                    first_missing=missing[:5],failure=failure,sources=sources,
                    retrieved_at=datetime.now(timezone.utc).isoformat(),
                    provenance='consistent_local_interior_plus_exchange_gap_conflict_recovery')
        if not missing:
            atomic_json(dest,[rows[t] for t in sorted(expected)])
            result['sha256']=hashlib.sha256(dest.read_bytes()).hexdigest()
        atomic_json(meta,result)
        print(f'{sym} {tf}: {result["status"]} local={local_count} fetched={fetched} missing={len(missing)}',flush=True)
        return result


async def run(args):
    import replay_backtest as rb
    import config
    from research_rocket_capture import load_rockets,bounds,DAY
    rockets,_=load_rockets(ROOT/'.runtime/reports')
    dates=sorted(r['day'] for r in rockets)
    start,end=bounds(dates[0])[0]-10*DAY,bounds(dates[-1])[1]
    index=rb._build_market_cache_index(args.source)
    args.output.mkdir(parents=True,exist_ok=True)
    symbols=sorted(set(config.load_watchlist())|{'BTCUSDT'})
    if args.symbols: symbols=args.symbols.split(',')
    sem=asyncio.Semaphore(3)
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=40)) as session:
        results=await asyncio.gather(*(recover_one(session,sem,s,tf,
            [p for a,b,p in index.get((s,tf),[]) if b>=start and a<end],start,end,args.output)
            for s in symbols for tf in STEPS))
    atomic_json(args.output/'recovery_summary.json',dict(start=start,end=end,series=results,
                complete=sum(r['status']=='complete' for r in results),total=len(results)))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',type=Path,default=ROOT/'.runtime/signal_quality_cache')
    parser.add_argument('--output',type=Path,default=ROOT/'.runtime/rocket_closed_cache')
    parser.add_argument('--symbols',default='')
    asyncio.run(run(parser.parse_args()))
