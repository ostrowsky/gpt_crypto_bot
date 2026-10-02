"""Durable paper admission and actual fallback observations, not exchange fills."""
import json
import math
from pathlib import Path
import re
import time

from forward_evidence_service import atomic
from validated_ranker_rollout import canonical, sha, seal, unseal, validate_ticket, assigned
from validated_ranker_rollout import CONTRACT as RELEASE_CONTRACT


def context_path(root,sym,tf,bar):
    if not re.fullmatch(r'[A-Z0-9]{3,30}',sym) or not re.fullmatch(r'\d+[mhd]',tf):
        raise ValueError('unsafe receipt identity')
    if not isinstance(bar,int) or isinstance(bar,bool) or bar<=0: raise ValueError('invalid receipt bar')
    return Path(root)/'score_context'/(sym+'_'+tf+'_'+str(bar)+'.json')


def admission(root,key,champion,positions,sym,tf,bar,now=None):
    now=time.time() if now is None else now
    context=json.loads(context_path(root,sym,tf,bar).read_bytes())
    ticket=validate_ticket(context['authorization'],key,champion,now)
    if not assigned(sym,ticket): raise ValueError('symbol not assigned to candidate')
    score=context['score']
    if (score['sym']!=sym or score['tf']!=tf or score['bar_ts']!=bar
            or score['candidate_sha256']!=ticket['candidate_sha256']
            or score['ticket_sha256']!=sha(canonical(ticket))
            or not math.isfinite(score['bonus']) or not -1<=score['bonus']<=1):
        raise ValueError('score receipt mismatch')
    raw=Path(positions).read_bytes()
    position=json.loads(raw)[sym]
    if (position['tf']!=tf or position['entry_ts']!=bar
            or not math.isfinite(position['entry_price']) or position['entry_price']<=0):
        raise ValueError('persisted position does not match scored admission')
    body={'contract':'policy-admission-receipt-v1','observed_at':now,'sym':sym,'tf':tf,'bar_ts':bar,
          'candidate_sha256':ticket['candidate_sha256'],'authorization':context['authorization'],
          'score':score,'positions_sha256':sha(raw),'persisted_position':position,
          'position_sha256':sha(canonical(position)),
          'scope':'persisted_bot_position_not_exchange_fill','closed_loop':False}
    atomic(Path(root)/'admissions'/(sha(canonical(body))+'.json'),seal(body,key))
    return body


def verify_admission(receipt,key,champion):
    body=unseal(receipt,key)
    context_path('.',body['sym'],body['tf'],body['bar_ts'])
    ticket=validate_ticket(body['authorization'],key,champion,body['observed_at'])
    position=body['persisted_position']
    if (body.get('contract')!='policy-admission-receipt-v1'
            or body.get('scope')!='persisted_bot_position_not_exchange_fill'
            or body['candidate_sha256']!=ticket['candidate_sha256']
            or not assigned(body['sym'],ticket)
            or sha(canonical(position))!=body['position_sha256']
            or position['tf']!=body['tf'] or position['entry_ts']!=body['bar_ts']
            or not math.isfinite(position['entry_price']) or position['entry_price']<=0
            or body['score']['ticket_sha256']!=sha(canonical(ticket))
            or body['score']['sym']!=body['sym'] or body['score']['tf']!=body['tf']
            or body['score']['bar_ts']!=body['bar_ts']):
        raise ValueError('admission receipt invalid')
    if (body['score']['candidate_sha256']!=body['candidate_sha256']
            or not math.isfinite(body['score']['bonus']) or not -1<=body['score']['bonus']<=1
            or not re.fullmatch(r'[0-9a-f]{64}',body['positions_sha256'])):
        raise ValueError('admission score invalid')
    return body


def fallback(root,key,sym,tf,bar,now=None):
    root=Path(root)
    context_path(root,sym,tf,bar)
    now=time.time() if now is None else now
    request=json.loads((root/'rollback_request.json').read_bytes())
    if not math.isfinite(now) or now<request['requested_at']: raise ValueError('fallback predates rollback')
    previous=unseal(request['previous_authorization'],key)
    if (previous.get('contract')!=RELEASE_CONTRACT
            or previous.get('stage') not in ('CANARY','PROMOTED') or not assigned(sym,previous)):
        raise ValueError('fallback not assigned to previous candidate')
    pointer=(root/'active.json').read_bytes()
    if sha(pointer)!=request['rollback_pointer_sha256'] or json.loads(pointer).get('state')!='ROLLED_BACK':
        raise ValueError('rollback pointer changed')
    body={'contract':'policy-fallback-receipt-v1','observed_at':now,
          'request_sha256':sha(canonical(request)),'pointer_sha256':sha(pointer),
          'sym':sym,'tf':tf,'bar_ts':bar,'bonus':0.,'closed_loop':False}
    atomic(root/'fallback_receipts'/(sha(canonical(body))+'.json'),seal(body,key))


def verify_rollback(root,key,now=None):
    root=Path(root)
    now=time.time() if now is None else now
    request=json.loads((root/'rollback_request.json').read_bytes())
    pointer=(root/'active.json').read_bytes()
    if json.loads(pointer).get('state')!='ROLLED_BACK' or sha(pointer)!=request['rollback_pointer_sha256']:
        raise ValueError('rollback pointer not applied')
    for path in (root/'fallback_receipts').glob('*.json'):
        try:
            value=unseal(json.loads(path.read_bytes()),key)
            previous=unseal(request['previous_authorization'],key)
            context_path(root,value['sym'],value['tf'],value['bar_ts'])
            if (value.get('contract')=='policy-fallback-receipt-v1' and value['bonus']==0
                    and previous.get('contract')==RELEASE_CONTRACT
                    and previous.get('stage') in ('CANARY','PROMOTED') and assigned(value['sym'],previous)
                    and request['requested_at']<=value['observed_at']<=now
                    and value['pointer_sha256']==sha(pointer)
                    and value['request_sha256']==sha(canonical(request))):
                return {'state':'RUNTIME_FALLBACK_VERIFIED','closed_loop':False,
                        'receipt_sha256':sha(path.read_bytes())}
        except (ValueError,KeyError,TypeError): pass
    return {'state':'POINTER_ONLY_WAITING_RUNTIME','closed_loop':False}


def summarize(root,key,champion,now):
    """Historical admissions are not proof of current authorization or profit."""
    root=Path(root)
    result={'application':'UNKNOWN','rollback':'UNKNOWN','closed_loop':False}
    try:
        active=json.loads((root/'active.json').read_bytes())
        validate_ticket(active,key,champion,now)
        for path in (root/'admissions').glob('*.json'):
            try:
                body=verify_admission(json.loads(path.read_bytes()),key,champion)
                if 0<=now-body['observed_at']<=600 and body['authorization']==active:
                    result['application']='CURRENT_PAPER_ADMISSION_VERIFIED'
                    break
            except (OSError,ValueError,KeyError,TypeError): pass
    except (OSError,ValueError,KeyError,TypeError): pass
    try: result['rollback']=verify_rollback(root,key,now)['state']
    except (OSError,ValueError,KeyError,TypeError): pass
    return result
