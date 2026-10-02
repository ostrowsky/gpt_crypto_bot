"""Immutable disjoint cohorts; validated rejection is not a stage transition."""
import json
from pathlib import Path
import time
from datetime import datetime,timezone

from forward_evidence_service import atomic
from independent_portfolio_gate import evaluate_bundle
from logical_learning_authority import material
from process_lock import process_lock
from validated_ranker_rollout import sha, canonical, validate_ticket

STEP=900000
CONTRACT='independent-learning-cohorts-v1'


def tick(deployment,now=None):
    now=time.time() if now is None else now
    root=Path(deployment['registry'])/'portfolios'
    path=root/'cohort.json'
    result={'state':'BLOCKED','runtime_eligible':False,'closed_loop':False,
            'run_time':datetime.fromtimestamp(now,timezone.utc).isoformat()}
    try:
        authority,key=material(deployment)
        registration=json.loads((root/'registration.json').read_bytes())
        with process_lock(root/'cohort.lock'):
            if not path.exists():
                atomic(path,{'contract':CONTRACT,'phase':'sealed','start_ms':registration['observation_start_ms'],
                    'state':'COLLECTING','previous':None})
            original=path.read_bytes()
            current=json.loads(original)
            if current['contract']!=CONTRACT or current['phase'] not in ('sealed','shadow','canary'):
                raise ValueError('unsupported cohort pointer')
            if current['state']!='COLLECTING':
                result.update(state=current['state'],phase=current['phase'])
                atomic(Path(deployment['registry'])/'cohort_controller_latest.json',result)
                return result
            phase=current['phase']
            result['phase']=phase
            candidate,champion=((root/(arm+'.json')).read_bytes() for arm in ('candidate','champion'))
            def checked(name):
                raw=(root/(name+'_unsigned.json')).read_bytes()
                cert=json.loads((root/(name+'_certification.json')).read_bytes())
                verdict=evaluate_bundle(raw,cert,authority,sha(candidate),sha(champion),now)
                if verdict['phase']!=name: raise ValueError('cohort phase mismatch')
                return verdict
            historical=checked('historical')
            observed=checked(phase)
            if not historical['passed'] or not observed['passed']:
                atomic(path,dict(current,state='REJECTED'))
                result.update(state='REJECTED',reason='certified numerical failure')
            elif observed['start_ms']!=current['start_ms']:
                raise ValueError('cohort start changed')
            else:
                if phase=='canary':
                    result['state']='CANARY_EVIDENCE_READY'
                else:
                    following='shadow' if phase=='sealed' else 'canary'
                    if following=='canary':
                        ticket=validate_ticket(json.loads((Path(deployment['release_root'])/'active.json').read_bytes()),
                                               key,champion,now)
                        if ticket['stage']!='CANARY' or ticket['candidate_sha256']!=sha(candidate):
                            raise ValueError('bound CANARY release missing')
                    start=max(observed['end_ms']+STEP,((int(now*1000)+STEP-1)//STEP)*STEP)
                    new={'contract':CONTRACT,'phase':following,'start_ms':start,'state':'COLLECTING',
                         'previous':sha(original),'previous_end_ms':observed['end_ms']}
                    archive=root/'cohorts'/(phase+'_'+sha(original)+'.json')
                    if not archive.exists(): atomic(archive,current)
                    if path.read_bytes()!=original: raise ValueError('cohort changed concurrently')
                    atomic(path,new)
                    result.update(state='TRANSITIONED',phase=following,start_ms=start)
    except Exception as exc:
        result['reason']=str(exc)
    atomic(Path(deployment['registry'])/'cohort_controller_latest.json',result)
    return result
