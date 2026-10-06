"""Finite public spot market capture; never accesses keys or places orders."""
from __future__ import annotations
import argparse,asyncio,json,time
from pathlib import Path
from collections import Counter
from minute_direction_data import sha,SYMBOLS

BASE='wss://stream.binance.com:9443/stream?streams='
DOC='https://github.com/binance/binance-spot-api-docs/blob/master/web-socket-streams.md'


def validate_message(message):
    stream=message['stream'];d=message['data'];symbol=stream.split('@')[0].upper()
    if symbol not in SYMBOLS:raise ValueError('unexpected symbol')
    if '@depth20@100ms' in stream:
        b=[[float(p),float(q)] for p,q in d['bids']];a=[[float(p),float(q)] for p,q in d['asks']]
        if not b or not a or len(b)>20 or len(a)>20 or b[0][0]>=a[0][0]:raise ValueError('invalid depth')
        if any(p<=0 or q<=0 for p,q in b+a):raise ValueError('nonpositive depth')
        if any(b[i][0]<=b[i+1][0] for i in range(len(b)-1)) or any(a[i][0]>=a[i+1][0] for i in range(len(a)-1)):
            raise ValueError('unsorted depth')
        int(d['lastUpdateId']);return symbol,'depth'
    if stream.endswith('@trade'):
        if d['s']!=symbol or float(d['p'])<=0 or float(d['q'])<=0:raise ValueError('invalid trade')
        int(d['E']);int(d['T']);int(d['t']);return symbol,'trade'
    raise ValueError('unexpected stream')


async def capture(output,seconds):
    import aiohttp
    if not 1<=seconds<=600:raise ValueError('bounded duration required')
    output.mkdir(parents=True,exist_ok=False);start=time.time_ns();begin=time.monotonic_ns()
    streams=[s.lower()+suffix for s in SYMBOLS for suffix in ('@depth20@100ms','@trade')]
    registration=dict(start_wall_ns=start,source_sha256=sha(__file__),seconds=seconds,streams=streams,
        venue='Binance spot',documentation=DOC,credentials=False,orders=False)
    (output/'registration.json').write_text(json.dumps(registration,indent=2),encoding='utf-8')
    counts=Counter();errors=[];first={};last={};gaps=Counter();invalid=0
    with (output/'messages.jsonl').open('x',encoding='utf-8') as file:
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None,connect=20)) as session:
                async with session.ws_connect(BASE+'/'.join(streams),heartbeat=20) as ws:
                    while (time.monotonic_ns()-begin)/1e9<seconds:
                        remaining=seconds-(time.monotonic_ns()-begin)/1e9
                        try:msg=await asyncio.wait_for(ws.receive(),timeout=min(remaining,10))
                        except asyncio.TimeoutError:continue
                        stamp=time.time_ns();mono=time.monotonic_ns()
                        if msg.type!=aiohttp.WSMsgType.TEXT:
                            if msg.type in (aiohttp.WSMsgType.CLOSED,aiohttp.WSMsgType.ERROR):raise RuntimeError('stream disconnected')
                            continue
                        record=dict(receive_wall_ns=stamp,receive_monotonic_ns=mono,raw=msg.data)
                        try:
                            parsed=json.loads(msg.data);symbol,kind=validate_message(parsed);key=symbol+':'+kind
                            record['valid']=True;counts[key]+=1;first.setdefault(key,mono)
                            if key in last and mono-last[key]>500_000_000:gaps[key]+=1
                            last[key]=mono
                        except (ValueError,KeyError,TypeError):record['valid']=False;invalid+=1
                        file.write(json.dumps(record,ensure_ascii=True)+'\n')
        except (OSError,RuntimeError,asyncio.TimeoutError,aiohttp.ClientError) as exc:
            errors.append(type(exc).__name__+': '+str(exc))
    report=dict(status='CAPTURED_PUBLIC_PILOT' if counts and not errors else 'INCOMPLETE_ACCESS',
        runtime_eligible=False,elapsed_seconds=(time.monotonic_ns()-begin)/1e9,counts=dict(counts),
        invalid=invalid,receive_gaps_over_500ms=dict(gaps),errors=errors,
        first_monotonic_ns=first,last_monotonic_ns=last,messages_sha256=sha(output/'messages.jsonl'),
        limits=['finite connectivity pilot, no bot decisions or fills','depth exchange time absent; receive time retained',
            'wall-clock synchronization unverified','trade clock is not order-fill acknowledgement'])
    (output/'result.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--seconds',type=int,default=120)
    a=p.parse_args();asyncio.run(capture(a.output,a.seconds))
