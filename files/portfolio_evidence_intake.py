"""Automatic evidence discovery; never creates population/policy certificates."""
from datetime import datetime, timezone
import json
from pathlib import Path
import time

from forward_evidence_service import atomic
from independent_portfolio_confirmation import CONTRACT, read_bound
from independent_portfolio_gate import evaluate_bundle
from logical_learning_authority import material
from validated_ranker_rollout import sha

REGISTRATION = 'automatic-portfolio-intake-v1'


def discover(root):
    """Fixed producer-owned paths; no trainer-selected arbitrary path intake."""
    registration = {'contract': REGISTRATION, 'evidence': {}}
    for arm in ('candidate', 'champion'):
        path = root/'portfolios'/(arm+'.json')
        registration[arm] = {'path': str(path.relative_to(root)), 'sha256': sha(path.read_bytes())}
    for phase in ('historical', 'sealed', 'shadow', 'canary'):
        paths = [root/'portfolios'/(phase+suffix) for suffix in ('_unsigned.json','_certification.json')]
        if all(p.exists() for p in paths):
            registration['evidence'][phase] = {key: {'path':str(path.relative_to(root)),
                'sha256':sha(path.read_bytes())} for key,path in zip(('bundle','certification'),paths)}
    return registration


def tick(deployment, now=None):
    now = time.time() if now is None else now
    root = Path(deployment['registry'])
    result = {'state': 'BLOCKED', 'runtime_eligible': False, 'closed_loop': False,
              'run_time': datetime.fromtimestamp(now, timezone.utc).isoformat(),
              'phases': {}, 'blockers': []}
    manifest = {'contract': CONTRACT, 'stage': 'CANARY', 'evidence': {}}
    try:
        authority, _ = material(deployment)
        registration_path = root/'portfolio_intake_registration.json'
        if not registration_path.exists():
            registration = discover(root)
            # Discovery is NOT certification. Its descriptor hashes are rechecked
            # by the independent gate before authorization.
            atomic(registration_path, registration)
        original = registration_path.read_bytes()
        registration = json.loads(original)
        if registration.get('contract') != REGISTRATION:
            raise ValueError('unsupported immutable intake registration')
        for arm in ('candidate', 'champion'):
            _, raw = read_bound(root, registration[arm])
            manifest[arm] = registration[arm]
            result[arm+'_sha256'] = sha(raw)
        discovered = discover(root)
        if any(discovered[arm] != registration[arm] for arm in ('candidate','champion')):
            raise ValueError('registered portfolio model changed')
        registration['evidence'].update(discovered['evidence'])
        for phase in ('historical', 'sealed', 'shadow', 'canary'):
            entry = registration.get('evidence', {}).get(phase)
            if entry is None:
                result['phases'][phase] = {'state': 'UNKNOWN', 'reason': 'missing certified phase'}
                continue
            try:
                _, raw = read_bound(root, entry['bundle'])
                _, cert = read_bound(root, entry['certification'])
                checked = evaluate_bundle(raw, json.loads(cert), authority,
                    result['candidate_sha256'], result['champion_sha256'], now)
                if checked['phase'] != phase:
                    raise ValueError('phase identity mismatch')
                result['phases'][phase] = dict(checked,
                    state='PASS' if checked['passed'] else 'REJECTED')
                manifest['evidence'][phase] = entry
            except Exception as exc:
                result['phases'][phase] = {'state': 'UNKNOWN', 'reason': str(exc)}
        if registration_path.read_bytes() != original:
            raise ValueError('intake registration changed during evaluation')
        ready = lambda phase: result['phases'][phase]['state'] == 'PASS'
        # Once canary evidence exists, failure cannot silently downgrade back to
        # pre-canary approval. Controller must reject/rollback instead.
        if 'canary' in registration.get('evidence', {}):
            manifest['stage'] = 'PROMOTED'
        else:
            manifest['evidence'].pop('canary', None)
        required = ('historical', 'sealed', 'shadow')+(
            ('canary',) if manifest['stage'] == 'PROMOTED' else ())
        result['blockers'] = [p+': '+result['phases'][p]['state'] for p in required if not ready(p)]
        result['state'] = ('REJECTED' if any(result['phases'][p]['state'] == 'REJECTED' for p in required)
                           else 'PHASES_READY' if not result['blockers'] else 'BLOCKED')
        # Cross-phase chronology/Harness are still checked by confirm/authorize.
        result['stage'] = manifest['stage']
    except Exception as exc:
        result['blockers'].append(str(exc))
        manifest['evidence'] = {}
    # Publish a blocked/partial intake as well: never leave a stale ready manifest.
    atomic(Path(deployment['portfolio_inputs']), manifest)
    atomic(root/'portfolio_intake_latest.json', result)
    return result
