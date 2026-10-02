"""Matched real bot paper decisions; never hypothetical canary portfolio proof."""
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time

from forward_evidence_service import atomic
from logical_learning_authority import material
from policy_runtime_receipts import context_path, verify_admission
from validated_ranker_rollout import canonical, sha, seal, unseal

REQUEST = 'actual-canary-exit-request-v1'
EXIT = 'actual-canary-paper-exit-v1'


def identity(body):
    context_path('.',body['sym'],body['tf'],body['bar_ts'])
    return sha(canonical([body['sym'],body['tf'],body['bar_ts'],body['candidate_sha256']]))


def admissions(root,key,champion):
    values={}
    for path in (Path(root)/'admissions').glob('*.json'):
        raw=path.read_bytes()
        body=verify_admission(json.loads(raw),key,champion)
        if body['authorization']['body']['stage']!='CANARY': continue
        name=identity(body)
        if name in values and values[name][1]!=raw:
            raise ValueError('duplicate conflicting canary admission')
        values[name]=(body,raw)
    return values


def immutable(path,value):
    """Runtime observations are append-only; restart cannot replace an event."""
    raw=canonical(value)
    if path.exists():
        if path.read_bytes()!=raw: raise ValueError('canary observation conflict')
    else:
        path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('xb') as handle: handle.write(raw)


def request_exit(root,key,champion,sym,tf,entry_ts,exit_ts,price,reason,now=None):
    now=time.time() if now is None else now
    context_path(root,sym,tf,entry_ts)
    context_path(root,sym,tf,exit_ts)
    if not math.isfinite(price) or price<=0 or exit_ts<entry_ts or exit_ts>now*1000:
        raise ValueError('invalid canary exit timing/price')
    matching=[v for v in admissions(root,key,champion).values()
              if (v[0]['sym'],v[0]['tf'],v[0]['bar_ts'])==(sym,tf,entry_ts)]
    if not matching: return False
    if len(matching)!=1: raise ValueError('ambiguous canary entry')
    body,raw=matching[0]
    path=Path(root)/'actual_canary'/'exit_requests'/(identity(body)+'.json')
    facts={'contract':REQUEST,'admission_sha256':sha(raw),'sym':sym,'tf':tf,
           'entry_ts':entry_ts,'exit_ts':exit_ts,'exit_price':price,'reason':str(reason)}
    if now<body['observed_at']: raise ValueError('exit predates admission observation')
    if path.exists():
        old=unseal(json.loads(path.read_bytes()),key)
        if any(old.get(k)!=v for k,v in facts.items()):
            atomic(Path(root)/'actual_canary'/'conflicts'/(path.stem+'.json'),
                   {'state':'BLOCKED','reason':'exit request changed'})
            raise ValueError('exit request changed')
        return True
    immutable(path,seal(dict(facts,requested_at=now),key))
    return True


def finalize(root,key,champion,positions,fee_bps=7.5,slippage_bps=5,now=None):
    """Called only after successful durable save, not after SELL intent."""
    now=time.time() if now is None else now
    if not math.isfinite(fee_bps) or fee_bps<7.5 or not math.isfinite(slippage_bps) or slippage_bps<5:
        raise ValueError('missing/unsafe cost assumption')
    raw=Path(positions).read_bytes()
    snapshot=json.loads(raw)
    entries=admissions(root,key,champion)
    completed=0
    for path in (Path(root)/'actual_canary'/'exit_requests').glob('*.json'):
        if (Path(root)/'actual_canary'/'conflicts'/(path.stem+'.json')).exists():
            raise ValueError('conflicting exit cannot be finalized')
        request=unseal(json.loads(path.read_bytes()),key)
        if request.get('contract')!=REQUEST or path.stem not in entries:
            raise ValueError('unbound exit request')
        admission,entry_raw=entries[path.stem]
        if request['admission_sha256']!=sha(entry_raw): raise ValueError('canary admission changed')
        current=snapshot.get(request['sym'])
        if current and (current.get('tf'),current.get('entry_ts'))==(request['tf'],request['entry_ts']):
            continue
        target=Path(root)/'actual_canary'/'exits'/(path.stem+'.json')
        if target.exists():
            verify_exit(json.loads(target.read_bytes()),key,champion,entry_raw,now)
            continue
        if not 0<=now-request['requested_at']<=120:
            raise ValueError('save outside causal exit observation budget')
        receipt={'contract':EXIT,'request':json.loads(path.read_bytes()),
                 'admission_sha256':sha(entry_raw),'observed_at':now,
                 'positions_sha256':sha(raw),'persisted_positions':snapshot,
                 'positions_canonical_sha256':sha(canonical(snapshot)),
                 'fee_bps':fee_bps,'slippage_bps':slippage_bps,
                 'scope':'persisted_paper_exit_not_exchange_fill','closed_loop':False}
        immutable(target,seal(receipt,key))
        completed+=1
    return completed


def verify_exit(value,key,champion,entry_raw,now):
    body=unseal(value,key)
    entry=verify_admission(json.loads(entry_raw),key,champion)
    request=unseal(body['request'],key)
    if (body.get('contract')!=EXIT or request.get('contract')!=REQUEST
            or body.get('scope')!='persisted_paper_exit_not_exchange_fill'
            or entry['authorization']['body']['stage']!='CANARY'
            or body['admission_sha256']!=sha(entry_raw) or request['admission_sha256']!=sha(entry_raw)
            or body['positions_canonical_sha256']!=sha(canonical(body['persisted_positions']))
            or (request['sym'],request['tf'],request['entry_ts'])!=(entry['sym'],entry['tf'],entry['bar_ts'])
            or not entry['observed_at']<=request['requested_at']<=body['observed_at']<=now
            or body['observed_at']-request['requested_at']>120
            or not entry['bar_ts']<=request['exit_ts']<=request['requested_at']*1000
            or not math.isfinite(request['exit_price']) or request['exit_price']<=0
            or not math.isfinite(body['fee_bps']) or body['fee_bps']<7.5
            or not math.isfinite(body['slippage_bps']) or body['slippage_bps']<5):
        raise ValueError('invalid actual canary exit')
    current=body['persisted_positions'].get(entry['sym'])
    if current and (current.get('tf'),current.get('entry_ts'))==(entry['tf'],entry['bar_ts']):
        raise ValueError('original canary position still persisted')
    cost=(body['fee_bps']+body['slippage_bps'])/10000
    if cost>=1: raise ValueError('invalid cost fraction')
    net=(request['exit_price']/entry['persisted_position']['entry_price']*(1-cost)**2-1)*100
    return dict(sym=entry['sym'],entry_ts=entry['bar_ts'],exit_ts=request['exit_ts'],net_pct=net)


def tick(deployment,now=None):
    now=time.time() if now is None else now
    result={'state':'UNKNOWN','closed_loop':False,'runtime_eligible':False,
            'run_time':datetime.fromtimestamp(now,timezone.utc).isoformat(),
            'scope':'matched_actual_paper_decisions_not_full_portfolio'}
    try:
        _,key=material(deployment)
        root=Path(deployment['release_root'])
        portfolios=Path(deployment['registry'])/'portfolios'
        cohort=json.loads((portfolios/'cohort.json').read_bytes())
        from learning_cohort_controller import CONTRACT
        if cohort.get('contract')!=CONTRACT: raise ValueError('invalid canary cohort')
        if cohort.get('phase')!='canary' or cohort.get('state')!='COLLECTING':
            result['state']='WAITING_CANARY_COHORT'
            atomic(Path(deployment['registry'])/'canary_outcomes_latest.json',result)
            return result
        champion=(portfolios/'champion.json').read_bytes()
        candidate_sha=sha((portfolios/'candidate.json').read_bytes())
        all_entries=admissions(root,key,champion)
        entries={name:pair for name,pair in all_entries.items()
                 if pair[0]['candidate_sha256']==candidate_sha
                 and cohort['start_ms']<=pair[0]['bar_ts']<=now*1000
                 and cohort['start_ms']/1000<=pair[0]['observed_at']<=now}
        if any((root/'actual_canary'/'conflicts'/(name+'.json')).exists() for name in entries):
            raise ValueError('conflicting canary exit intent')
        values=[]
        for path in (root/'actual_canary'/'exits').glob('*.json'):
            if path.stem not in all_entries: raise ValueError('orphan canary exit')
            if path.stem not in entries: continue
            values.append(verify_exit(json.loads(path.read_bytes()),key,champion,entries[path.stem][1],now))
        result.update(state='MATCHED_DECISIONS_ONLY' if values else 'WAITING_ACTUAL_CANARY_OUTCOMES',
                      admissions=len(entries),closed=len(values),open_or_missing=len(entries)-len(values),
                      cohort_sha256=sha(canonical(cohort)),
                      diagnostic_net_mean_pct=sum(v['net_pct'] for v in values)/len(values) if values else None)
    except Exception as exc: result.update(state='BLOCKED',reason=str(exc))
    atomic(Path(deployment['registry'])/'canary_outcomes_latest.json',result)
    return result


def runtime(operation,*args):
    """Fixed local deployment; observation errors never change trading decisions."""
    import config
    if not getattr(config,'LOCAL_LOGICAL_POLICY_ROLLOUT_ENABLED',False): return
    base=Path(__file__).resolve().parents[1]/'.runtime/learning_roles_local'
    deployment=json.loads((base/'deployment.json').read_bytes())
    _,key=material(deployment)
    from certified_rule_score_policy import champion_bytes
    root=Path(deployment['release_root'])
    if operation=='request': return request_exit(root,key,champion_bytes(),*args)
    if operation=='finalize':
        return finalize(root,key,champion_bytes(),*args,fee_bps=max(7.5,float(config.PAPER_FEE_BPS)))
    raise ValueError('unsupported canary observation')
