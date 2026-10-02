"""Read-only loop health: execution, eligibility and improvement are distinct."""
from datetime import datetime, timezone
import json
from pathlib import Path

import policy_provenance as provenance


def read_status(path):
    try:
        value = json.loads(Path(path).read_text(encoding='utf-8-sig'))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def fresh(value, now, budget):
    stamp = provenance.parse_utc(value.get('run_time'))
    return stamp is not None and 0 <= (now-stamp).total_seconds() <= budget


def runtime_root(runtime):
    runtime = Path(runtime)
    local = runtime/'learning_roles_local'
    deployment = read_status(local/'deployment.json')
    if (deployment.get('isolation_mode') == 'logical_same_user' and
            deployment.get('logical_isolation_accepted') is True):
        return local/'evaluator'
    return runtime/'learning_roles'/'evaluator'


def summarize(root, now=None):
    now = now or datetime.now(timezone.utc)
    root = Path(root)
    collector = read_status(root/'status.json')
    exporter = read_status(root/'training_export_latest.json')
    controller = collector.get('controller') or {}
    confirmation = collector.get('portfolio_confirmation') or {}
    local = collector.get('isolation_mode') == 'logical_same_user'
    controller_fresh = True
    if local:
        separate = read_status(root/'controller_latest.json')
        controller_fresh = fresh(separate, now, 600)
        controller = separate.get('controller') or {} if controller_fresh else {}
        confirmation = separate.get('portfolio_confirmation') or {} if controller_fresh else {}
    problems = []
    if local:
        problems.append('logical same-user separation: no OS access isolation')
        if not controller_fresh:
            problems.append('controller: missing/stale/future separate status')
    trainer = read_status(root.parent/'trainer'/'status.json') if local else {}
    collection = collector.get('collection') or {}
    collector_fresh = fresh(collector, now, 240)
    exporter_fresh = fresh(exporter, now, 75*60)
    if not collector_fresh:
        problems.append('collector: missing/stale/future status')
    if collector.get('collection_blocker'):
        problems.append('collector: '+str(collector['collection_blocker']))
    if not collector_fresh or not collection or collector.get('collection_blocker'):
        collection_state = 'UNKNOWN'
    else:
        collection_state = 'PROXY_ONLY'
    if not exporter_fresh or exporter.get('state') != 'EXPORTED_NOT_APPROVED':
        problems.append('training export: missing/stale/blocked')
    if confirmation.get('state') != 'READY':
        problems.extend('portfolio: '+str(v) for v in
                        (confirmation.get('blockers') or ['independent evidence not ready']))
    if controller.get('state') not in ('CANARY', 'PROMOTED') or not controller.get('runtime_eligible'):
        problems.append('controller: '+str(controller.get('reason') or 'not authorized'))
    # Authorization/config switches are not receipts of actual live consumption.
    # No such receipt contract is implemented yet, so never infer application.
    problems.append('live consumption: no verified receipt contract')
    return {'state': 'NOT_CLOSED', 'improvement_verdict': 'UNKNOWN',
            'isolation_mode': 'logical_same_user' if local else 'os_roles_or_unknown',
            'os_access_isolation': False if local else None,
            'trainer_state': trainer.get('state', 'UNKNOWN'),
            'trainer_last_run': trainer.get('run_time'),
            'production_effect': 'UNKNOWN', 'closed_loop': False,
            'collection_state': collection_state, 'collector_fresh': collector_fresh,
            'exporter_fresh': exporter_fresh,
            'observations': collection.get('observations') if collection_state == 'PROXY_ONLY' else None,
            'outcomes': collection.get('outcomes') if collection_state == 'PROXY_ONLY' else None,
            'collector_duration_seconds': collector.get('duration_seconds'),
            'export_duration_seconds': exporter.get('duration_seconds'),
            'activation_authorized': bool(collector_fresh and confirmation.get('state') == 'READY'
                and controller.get('runtime_eligible') and controller.get('state') in ('CANARY','PROMOTED')),
            'blockers': problems}
