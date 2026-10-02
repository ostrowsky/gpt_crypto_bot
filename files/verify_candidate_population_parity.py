"""Closed-prefix candidate generation parity, explicitly not live BUY certification."""
import argparse
import asyncio
from dataclasses import asdict
import json
import math
from pathlib import Path

import numpy as np
import replay_backtest as rb
from forward_evidence_service import atomic
from paired_full_policy_replay import frozen_arm
from validated_ranker_rollout import canonical,sha

STEP=900000


def closed_prefix(data,tf,frame):
    n=int(np.searchsorted(data['t'],frame-rb.BAR_MS[tf],side='right'))
    if not n: return None
    closed=data[:n]
    if int(closed[-1]['t'])+rb.BAR_MS[tf]>frame:
        raise ValueError('unclosed input')
    tail=closed[-1:].copy()
    tail['t']+=rb.BAR_MS[tf]
    prefix=np.concatenate((closed,tail))
    return prefix,rb.compute_features(prefix['o'],prefix['h'],prefix['l'],prefix['c'],prefix['v'])


def rows(candidates,frame):
    result=[]
    for candidate in candidates:
        value=asdict(candidate)
        if value['ts_ms']!=frame: raise ValueError('candidate clock mismatch')
        if any(isinstance(v,float) and not math.isfinite(v) for v in value.values()):
            raise ValueError('nonfinite candidate evidence')
        result.append(value)
    return sorted(result,key=lambda row:canonical(row))  # duplicates intentionally retained


def difference(left,right,frame):
    a,b=rows(left,frame),rows(right,frame)
    return {'equal':canonical(a)==canonical(b),'batch_count':len(a),'stream_count':len(b),
            'batch_sha256':sha(canonical(a)),'stream_sha256':sha(canonical(b))}


async def compare(symbols,cache,start,end,progress=None):
    if start%STEP or end%STEP or start>=end: raise ValueError('invalid maximum interval')
    c15,c4=({s:cache[s,tf] for s in symbols} for tf in ('15m','4h'))
    ctx=rb._build_bull_day_context(cache['BTCUSDT','1h'][0])
    batch,_,_=await rb.build_replay_candidate_snapshot(symbols,['15m','1h'],cache,c15,c4,ctx,
                                                       variant='score_replace_cluster')
    warmup=int(cache['BTCUSDT','15m'][0]['t'][0])+STEP
    if warmup>start: raise ValueError('warmup missing')
    state={}
    result={'state':'RUNNING','scope':'replay_batch_vs_closed_prefix_not_monitor_buy_path',
            'runtime_eligible':False,'frames':0,'mismatched_frames':0,'batch_candidates':0,
            'stream_candidates':0,'first_mismatch':None,'bounds':[start,end]}
    for frame in range(warmup,end,STEP):
        prefix={}
        for (s,tf),(data,_) in cache.items():
            pack=closed_prefix(data,tf,frame)
            if pack is not None: prefix[s,tf]=pack
        if ('BTCUSDT','1h') not in prefix: continue  # initial warmup only
        p15,p4=({s:prefix[s,tf] for s in symbols if (s,tf) in prefix} for tf in ('15m','4h'))
        context=rb._build_bull_day_context(prefix['BTCUSDT','1h'][0])
        raw,_,_=await rb.build_replay_candidate_snapshot(symbols,['15m','1h'],prefix,p15,p4,context,
            variant='score_replace_cluster',candidate_stream_state=state,frame_ms=frame)
        state=json.loads(canonical(state))
        if any(t!=frame for t in raw): raise ValueError('stream generated off-clock candidates')
        if frame<start:
            if progress and (frame-warmup)//STEP%96==0:
                progress(dict(result,phase='WARMUP',warmup_frame=frame))
            continue
        verdict=difference(batch.get(frame,[]),raw.get(frame,[]),frame)
        result['frames']+=1
        result['batch_candidates']+=verdict['batch_count']
        result['stream_candidates']+=verdict['stream_count']
        if not verdict['equal']:
            result['mismatched_frames']+=1
            if result['first_mismatch'] is None: result['first_mismatch']=dict(verdict,frame=frame)
        if progress and result['frames']%96==0: progress(dict(result))
    if result['frames']!=(end-start)//STEP: raise ValueError('incomplete candidate clock')
    result['state']='FAIL' if result['mismatched_frames'] else 'PASS'
    result['symbols']=len(symbols)
    return result


async def run(archive,models,output):
    try:
        manifest_raw=(archive/'manifest.json').read_bytes()
        manifest=json.loads(manifest_raw)
        reg_raw=(models/'registration.json').read_bytes()
        registration=json.loads(reg_raw)
        if not isinstance(manifest,dict) or not isinstance(registration,dict):
            raise ValueError('invalid parity manifests')
    except (OSError,ValueError) as exc:
        result={'state':'UNKNOWN','runtime_eligible':False,'closed_loop':False,
                'contract':'maximum-candidate-generation-parity-v1',
                'scope':'replay_batch_vs_closed_prefix_not_monitor_buy_path','reason':str(exc)}
        atomic(output,result)
        return result
    own_sha=sha(Path(__file__).read_bytes())
    def verify():
        if sha((archive/'manifest.json').read_bytes())!=registration['archive_sha256']:
            raise ValueError('archive registration mismatch')
        if registration.get('maximum_available_bounds')!=[manifest['start_ms'],manifest['end_ms']]:
            raise ValueError('maximum archive bounds mismatch')
        if (models/'registration.json').read_bytes()!=reg_raw or sha(Path(__file__).read_bytes())!=own_sha:
            raise ValueError('verification registration/source drift')
        for name,digest in manifest['input_hashes'].items():
            if Path(name).name!=name or sha((archive/'market'/name).read_bytes())!=digest:
                raise ValueError('archive market drift')
        for name,digest in registration['sources'].items():
            if Path(name).name!=name or sha(Path(__file__).with_name(name).read_bytes())!=digest:
                raise ValueError('frozen policy source drift')
        for name,digest in registration['models'].items():
            if Path(name).name!=name or sha((models/(name+'.json')).read_bytes())!=digest:
                raise ValueError('frozen model drift')
    result={'state':'UNKNOWN','runtime_eligible':False,'closed_loop':False,
            'contract':'maximum-candidate-generation-parity-v1','arms':{},
            'scope':'replay_batch_vs_closed_prefix_not_monitor_buy_path',
            'bounds':[manifest.get('start_ms'),manifest.get('end_ms')],
            'archive_sha256':sha(manifest_raw),'registration_sha256':sha(reg_raw),'verifier_sha256':own_sha}
    try:
        verify()
        symbols=manifest['eligible_symbols']
        index=rb._build_market_cache_index(archive/'market')
        cache={}
        for s in symbols:
            for tf in ('15m','1h'):
                data=rb._load_cached_klines(archive/'market',s,tf,manifest['archive_start_ms'],
                                            manifest['end_ms'],cache_index=index)
                if data is None: raise ValueError('missing frozen candle series')
                cache[s,tf]=(data,rb.compute_features(data['o'],data['h'],data['l'],data['c'],data['v']))
            data=rb._aggregate_1h_to_4h(cache[s,'1h'][0])
            cache[s,'4h']=(data,rb.compute_features(data['o'],data['h'],data['l'],data['c'],data['v']))
        for arm in ('champion','candidate'):
            with frozen_arm(models,arm):
                result['arms'][arm]=await compare(symbols,cache,manifest['start_ms'],manifest['end_ms'],
                    lambda data:atomic(output.with_suffix('.progress.json'),dict(data,arm=arm)))
        verify()
        result['state']='PASS' if all(v['state']=='PASS' for v in result['arms'].values()) else 'FAIL'
    except Exception as exc: result.update(state='UNKNOWN',reason=str(exc))
    atomic(output,result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('archive','models','output'): parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    result=asyncio.run(run(args.archive,args.models,args.output))
    print(result['state'])
    raise SystemExit(0 if result['state']=='PASS' else 1)
