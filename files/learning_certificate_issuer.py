"""Issuer validates independent signed facts, never upgrades a replay's scope."""
import json
from pathlib import Path
import time
from datetime import datetime, timezone

from forward_evidence_service import atomic
import independent_portfolio_gate as gate
from logical_learning_authority import CONTRACT, material
from validated_ranker_rollout import canonical, seal, sha, unseal

PROOF = 'independent-full-policy-validation-v1'
CHECKS = ('point_in_time_universe','raw_closed_provenance','candidate_generation_parity',
          'live_execution_parity','maximum_period','training_holdout_excluded',
          'operator_experiment_approved')


def issue(raw, validation, candidate, champion, authority, evaluator_key, now):
    proof = unseal(validation, evaluator_key)
    bundle = json.loads(raw)
    if (proof.get('contract') != PROOF or proof.get('phase') != bundle['phase']
            or proof.get('bundle_sha256') != sha(raw)
            or proof.get('candidate_sha256') != sha(candidate)
            or proof.get('champion_sha256') != sha(champion)
            or proof.get('evaluator_sha256') != gate.source_hash()
            or proof.get('scope') != 'full_candidate_population_and_live_policy'
            or any(proof.get(check) is not True for check in CHECKS)
            or not proof.get('issued_at',float('inf')) <= now < proof.get('expires_at',0)
            or not 0 < proof['expires_at']-proof['issued_at'] <= 86400
            or proof.get('bounds') != [bundle['start_ms'],bundle['end_ms']]):
        raise ValueError('independent full-policy validation missing/mismatched; execution-only parity insufficient')
    if bundle['phase']=='canary' and proof.get('actual_assignment_verified') is not True:
        raise ValueError('real canary assignment not verified')
    # Recompute complete accounts even if the independent proof has passed.
    numeric = gate.compare_accounts(bundle,now)
    body = dict(contract=CONTRACT, isolation_mode='logical_same_user', os_access_isolation=False,
        training_holdout_excluded=True, bundle_sha256=sha(raw), candidate_sha256=sha(candidate),
        champion_sha256=sha(champion), evaluator_sha256=gate.source_hash(), issued_at=now,
        expires_at=min(now+86400,proof['expires_at']), point_in_time_universe=True,
        raw_closed_provenance=True,live_policy_parity=True,operator_experiment_approved=True,
        actual_assignment_verified=proof.get('actual_assignment_verified',False),
        validation_sha256=sha(canonical(validation)))
    return seal(body,authority.key), numeric


def tick(deployment, now=None):
    now=time.time() if now is None else now
    root=Path(deployment['registry'])/'portfolios'
    result={'state':'BLOCKED','runtime_eligible':False,'closed_loop':False,'phases':{},
            'run_time':datetime.fromtimestamp(now,timezone.utc).isoformat()}
    try:
        authority,key=material(deployment)
        candidate,champion=(root/(arm+'.json') for arm in ('candidate','champion'))
        for phase in ('historical','sealed','shadow','canary'):
            try:
                path=root/(phase+'_unsigned.json')
                raw=path.read_bytes()
                validation=json.loads((root/(phase+'_validation.json')).read_bytes())
                cert,numeric=issue(raw,validation,candidate.read_bytes(),champion.read_bytes(),authority,key,now)
                if path.read_bytes()!=raw: raise ValueError('bundle changed during certification')
                atomic(root/(phase+'_certification.json'),cert)
                result['phases'][phase]={'state':'CERTIFIED' if numeric['passed'] else 'CERTIFIED_REJECTED',
                                        'comparison':numeric}
            except Exception as exc:
                result['phases'][phase]={'state':'BLOCKED','reason':str(exc)}
                if (root/(phase+'_unsigned.json')).exists():
                    atomic(root/(phase+'_certification.json'),{'state':'BLOCKED','reason':str(exc)})
        if any(p['state'].startswith('CERTIFIED') for p in result['phases'].values()):
            result['state']='PARTIAL_CERTIFIED'
    except Exception as exc:
        result['reason']=str(exc)
    atomic(Path(deployment['registry'])/'certificate_issuer_latest.json',result)
    return result
