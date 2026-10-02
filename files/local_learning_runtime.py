"""User-level role processes; logical separation, never an NTFS isolation claim."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
from forward_evidence_service import atomic, current_sid, run_tick
from process_lock import process_lock

ROOT = Path(__file__).resolve().parents[1]
INTERVALS = {'evaluator': 60, 'exporter': 3600, 'trainer': 3600, 'controller': 300, 'portfolio': 60}
MODE = 'logical_same_user'


def initialize(root=ROOT):
    root = root.resolve()
    base = root/'.runtime/learning_roles_local'
    base.mkdir(parents=True, exist_ok=True)
    from logical_learning_authority import provision
    provision(base/'authority')
    path = base/'deployment.json'
    sid = current_sid()
    if path.exists():
        value = json.loads(path.read_bytes())
        if value.get('trainer_sid') != sid or value.get('isolation_mode') != MODE:
            raise ValueError('existing local deployment identity mismatch')
        return path
    registry = base/'evaluator'
    registry.mkdir(exist_ok=True)
    intake = base/'intake'
    trainer = base/'trainer'
    intake.mkdir(exist_ok=True)
    trainer.mkdir(exist_ok=True)
    value = {'isolation_mode': MODE, 'logical_isolation_accepted': True,
             'os_access_isolation': False, 'project_root': str(root),
             'trainer_sid': sid, 'evaluator_sid': sid,
             'registry': str(registry), 'dataset': str(root/'files/critic_dataset_v2.jsonl'),
             'training_input': str(intake/'training.jsonl'),
             'candidate_input': str(trainer/'candidate.json'),
             'candidate_output': str(trainer/'candidate.json'),
             'trainer_status': str(trainer/'status.json'),
             'status': str(registry/'status.json'),
             'export_status': str(registry/'training_export_latest.json'),
             'controller_status': str(registry/'controller_latest.json'),
             'portfolio_inputs': str(registry/'portfolio_inputs.json'),
             'portfolio_request': str(registry/'portfolio_request.json'),
             # Dedicated pointer: this runtime cannot race the protected service.
             'release_root': str(base/'release')}
    atomic(path, value)
    return path


def role_environment(role, source=None):
    # Allowlist, not a denylist: no inherited Telegram, exchange or evaluator keys.
    source = os.environ if source is None else source
    names = {'systemroot', 'windir', 'path', 'pathext', 'temp', 'tmp', 'userprofile',
             'localappdata', 'appdata', 'programdata', 'os', 'systemdrive'}
    env = {k: v for k, v in source.items() if k.lower() in names}
    if role == 'controller' and source.get('RANKER_EVALUATOR_KEY'):
        env['RANKER_EVALUATOR_KEY'] = source['RANKER_EVALUATOR_KEY']
    return env


def adopt_current_user(root=ROOT):
    """Explicit stopped-runtime migration, not automatic identity bypass."""
    base = root.resolve()/'.runtime/learning_roles_local'
    path = base/'deployment.json'
    value = json.loads(path.read_bytes())
    status = json.loads((base/'supervisor.json').read_bytes())
    if (not (base/'stop.request').exists() or status.get('state') != 'STOPPED'
            or value.get('isolation_mode') != MODE
            or value.get('logical_isolation_accepted') is not True):
        raise ValueError('identity migration requires stopped accepted logical runtime')
    from shutil import copyfile
    backup = base/('deployment.before_identity_'+str(time.time_ns())+'.json')
    copyfile(path, backup)
    sid = current_sid()
    value.update(trainer_sid=sid, evaluator_sid=sid, os_access_isolation=False)
    atomic(path, value)
    return path


def worker(path, role):
    base = path.parent
    with process_lock(base/(role+'.worker.lock')):
        while not (base/'stop.request').exists():
            started = time.monotonic()
            atomic(base/(role+'.lifecycle.json'), {'state': 'RUNNING', 'pid': os.getpid(),
                   'at': time.time(), 'isolation_mode': MODE})
            try:
                deployment = json.loads(path.read_bytes())
                if role == 'portfolio':
                    import asyncio
                    from prospective_policy_portfolios import tick
                    result = asyncio.run(tick(deployment))
                    from datetime import datetime, timezone
                    result['run_time'] = datetime.now(timezone.utc).isoformat()
                    atomic(base/'evaluator/portfolio_producer_latest.json', result)
                else:
                    if role == 'controller':
                        from learning_certificate_issuer import tick as certificate_tick
                        certificate_tick(deployment)
                        from portfolio_evidence_intake import tick as intake_tick
                        intake_tick(deployment)
                    result = run_tick(deployment, role)
                    if role == 'controller':
                        from learning_cohort_controller import tick as cohort_tick
                        cohort_tick(deployment)
                blocked = result.get('state') == 'BLOCKED'
                atomic(base/(role+'.lifecycle.json'), {'state': 'WAITING', 'pid': os.getpid(),
                       'at': time.time(), 'last_result': result.get('state', 'UNKNOWN'),
                       'isolation_mode': MODE})
            except Exception as exc:
                blocked = True
                error = {'state': 'BLOCKED', 'reason': str(exc),
                         'isolation_mode': MODE, 'runtime_eligible': False, 'at': time.time()}
                atomic(base/(role+'.error.json'), error)
                if role == 'portfolio':
                    atomic(base/'evaluator/portfolio_producer_latest.json', error)
            # Per-role cadence, no overlapping fit/export or catch-up burst.
            interval = min(60, INTERVALS[role]) if blocked else INTERVALS[role]
            deadline = time.monotonic()+max(1, interval-(time.monotonic()-started))
            while time.monotonic() < deadline and not (base/'stop.request').exists():
                time.sleep(min(1, max(0.01, deadline-time.monotonic())))


def supervise(path):
    base = path.parent
    children, handles = {}, []
    with process_lock(base/'supervisor.lock'):
        (base/'stop.request').unlink(missing_ok=True)
        try:
            while not (base/'stop.request').exists():
                for role in INTERVALS:
                    if role not in children or children[role].poll() is not None:
                        log = (base/(role+'.log')).open('ab')
                        handles.append(log)
                        children[role] = subprocess.Popen([sys.executable, str(Path(__file__).resolve()),
                            '--deployment', str(path), '--worker', role], cwd=ROOT,
                            env=role_environment(role), stdout=log, stderr=log,
                            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
                atomic(base/'supervisor.json', {'state': 'RUNNING', 'pid': os.getpid(),
                    'roles': {k: v.pid for k, v in children.items()}, 'at': time.time(),
                    'isolation_mode': MODE, 'os_access_isolation': False, 'closed_loop': False})
                time.sleep(2)
        finally:
            for process in children.values():
                if process.poll() is None:
                    process.terminate()
            for process in children.values():
                process.wait(timeout=30)
            for handle in handles:
                handle.close()
            atomic(base/'supervisor.json', {'state': 'STOPPED', 'pid': os.getpid(),
                    'at': time.time(), 'isolation_mode': MODE, 'closed_loop': False})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--deployment', type=Path)
    parser.add_argument('--worker', choices=tuple(INTERVALS))
    parser.add_argument('--stop', action='store_true')
    parser.add_argument('--adopt-current-user', action='store_true')
    args = parser.parse_args()
    path = adopt_current_user() if args.adopt_current_user else args.deployment or initialize()
    if args.adopt_current_user:
        print('Stopped logical deployment rebound to current user; backup preserved')
    elif args.stop:
        (path.parent/'stop.request').touch()
    elif args.worker:
        worker(path, args.worker)
    else:
        supervise(path)
