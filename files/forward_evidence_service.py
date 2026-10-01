"""Prospective evidence collection and scheduled fail-closed control (never a trainer)."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
import math
import os
from pathlib import Path
import subprocess
import sys

# The bundled interpreter's isolated ._pth points to the original checkout.
# Resolve role modules from the administrator-frozen source, not that checkout.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import independent_signal_evaluator as evaluator
import policy_provenance as provenance
from validated_ranker_rollout import canonical, sha, rollback, rollout


def atomic(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    with tmp.open('wb') as handle:
        handle.write(canonical(value))
        handle.flush()
        os.fsync(handle.fileno())
    tmp.replace(path)


def causal(row):
    return {k:v for k,v in row.items() if k not in
            ('labels', 'label_provenance', 'teacher', 'execution_reconstruction', 'reconstruction_integrity')}


def read_journal(path):
    observations, outcomes, previous = {}, {}, None
    if path.exists():
        for line in path.read_text(encoding='utf-8').splitlines():
            record = json.loads(line)
            body = record['body']
            if record['previous'] != previous or record['sha256'] != sha(canonical(body)) or body['previous'] != previous:
                raise ValueError('forward journal corrupt')
            previous = record['sha256']
            target = observations if body['kind'] == 'observation' else outcomes if body['kind'] == 'outcome' else None
            if target is None or body['id'] in target:
                raise ValueError('invalid/duplicate forward journal event')
            target[body['id']] = body['row']
    return observations, outcomes, previous


def collect(registry, dataset, now=None):
    now = now or datetime.now(timezone.utc)
    manifest = json.loads((registry/'manifest.json').read_bytes())
    if sha((registry/'candidate.json').read_bytes()) != manifest['candidate_sha256']:
        raise ValueError('frozen candidate changed')
    start = provenance.parse_utc(manifest['holdout_start'])
    journal = registry/'forward_journal.jsonl'
    counters = {'new_observations': 0, 'new_outcomes': 0, 'late_or_invalid': 0}
    lock = registry/'collector.lock'
    with lock.open('xb'):
        pass
    try:
        observations, outcomes, previous = read_journal(journal)
        def append(kind, identity, row):
            nonlocal previous
            body = {'previous': previous, 'kind': kind, 'id': identity,
                    'recorded_at': provenance.utc_iso(now), 'row': row}
            digest = sha(canonical(body))
            with journal.open('ab') as handle:
                handle.write(canonical({'previous': previous, 'sha256': digest, 'body': body})+b'\n')
                handle.flush()
                os.fsync(handle.fileno())
            previous = digest
        with dataset.open(encoding='utf-8') as handle:
            for line in handle:
                row = json.loads(line)
                identity = row.get('id')
                feature = provenance.parse_utc((row.get('provenance') or {}).get('feature_time'))
                if not identity or feature is None or feature < start:
                    continue
                clean = causal(row)
                if identity not in observations:
                    decision = provenance.parse_utc((row.get('decision_provenance') or {}).get('decision_time'))
                    if (not provenance.observation_provenance_valid(row) or not feature <= now <= feature+timedelta(seconds=120)
                            or decision is None or not feature <= decision <= min(now, feature+timedelta(minutes=1))
                            or row.get('labels', {}).get('ret_5') is not None):
                        counters['late_or_invalid'] += 1
                        continue
                    append('observation', identity, clean)
                    observations[identity] = clean
                    counters['new_observations'] += 1
                elif canonical(clean) != canonical(observations[identity]):
                    raise ValueError('forward observation identity conflict')
                if row.get('labels', {}).get('ret_5') is None:
                    continue
                due = provenance.forward_label_time(bar_ts=int(row['bar_ts']), tf=row['tf'], horizon=5)
                label = (row.get('label_provenance') or {}).get('ret_5') or {}
                recorded = provenance.parse_utc(label.get('recorded_at'))
                if (not provenance.label_provenance_valid(row, 'ret_5') or now < due
                        or provenance.parse_utc(label.get('label_time')) != due
                        or recorded is None or not due <= recorded <= now
                        or not math.isfinite(float(row['labels']['ret_5']))):
                    counters['late_or_invalid'] += 1
                    continue
                outcome = {'ret_5': row['labels']['ret_5'], 'provenance': label}
                if identity in outcomes:
                    if canonical(outcome) != canonical(outcomes[identity]):
                        raise ValueError('forward outcome identity conflict')
                else:
                    append('outcome', identity, outcome)
                    outcomes[identity] = outcome
                    counters['new_outcomes'] += 1
        rows = []
        for identity, clean in sorted(observations.items()):
            row = dict(clean)
            if identity in outcomes:
                row.update(labels={'ret_5': outcomes[identity]['ret_5']},
                           label_provenance={'ret_5': outcomes[identity]['provenance']})
            rows.append(row)
        path = registry/'forward_snapshot.jsonl'
        tmp = path.with_suffix('.tmp')
        with tmp.open('wb') as handle:
            for row in rows:
                handle.write(canonical(row)+b'\n')
        tmp.replace(path)
        return {**counters, 'observations': len(observations), 'outcomes': len(outcomes),
                'journal_sha256': sha(journal.read_bytes()) if journal.exists() else None,
                'runtime_eligible': False, 'scope': 'prospective_ret5_proxy_not_portfolio'}
    finally:
        lock.unlink()


def export_training(dataset, output, cutoff):
    """Evaluator writes only pre-holdout data into trainer-readable intake."""
    tmp = output.with_suffix('.tmp')
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with dataset.open(encoding='utf-8') as source, tmp.open('wb') as target:
        for line in source:
            row = json.loads(line)
            feature = provenance.parse_utc((row.get('provenance') or {}).get('feature_time'))
            if feature is None or feature >= cutoff:
                continue
            if not provenance.observation_provenance_valid(row) or not provenance.label_provenance_valid(row, 'ret_5'):
                continue
            labels = (row.get('label_provenance') or {}).values()
            if any(provenance.parse_utc(p.get('recorded_at')) is None or
                   provenance.parse_utc(p['recorded_at']) >= cutoff for p in labels):
                continue
            target.write(canonical(row)+b'\n')
            count += 1
    tmp.replace(output)
    return count


def controller_tick(request_path, release_root, now=None, harness_root=None):
    """No proxy approval; portfolio evidence failure clears a prior overlay."""
    from independent_portfolio_gate import authorize
    prior = sha((release_root/'active.json').read_bytes()) if (release_root/'active.json').exists() else None
    try:
        request = json.loads(request_path.read_bytes())
        evidence = {phase:(Path(v['bundle']).read_bytes(), json.loads(Path(v['certification']).read_bytes()))
                    for phase,v in request['evidence'].items()}
        model, champion = Path(request['candidate']).read_bytes(), Path(request['champion']).read_bytes()
        key = os.environ.get('RANKER_EVALUATOR_KEY', '').encode()
        ticket = authorize(evidence, os.environ.get('RANKER_COVERAGE_AUTHORITY_KEY', '').encode(),
                           key, model, champion, request.get('stage', 'CANARY'), now=now,
                           harness_root=harness_root)
        rollout(release_root, model, ticket, key, champion, expected=prior, now=now)
        return {'state': ticket['body']['stage'], 'runtime_eligible': True}
    except Exception as exc:
        result = {'state': 'BLOCKED', 'reason': str(exc), 'runtime_eligible': False}
        if prior is not None:
            try:
                rollback(release_root, prior)
                result['rollback'] = 'applied'
            except Exception as failure:
                result['rollback'] = 'blocked: '+str(failure)
        return result


def current_sid():
    import csv
    row = next(csv.reader(subprocess.check_output(['whoami', '/user', '/fo', 'csv', '/nh'], text=True).strip().splitlines()))
    return row[-1]


def verify_trainer_isolation(deployment):
    """Actual access probes, not an assertion based on the account name."""
    for name in ('registry', 'release_root'):
        try:
            list(Path(deployment[name]).iterdir())
        except PermissionError:
            continue
        raise PermissionError('trainer can inspect protected '+name)
    try:
        with Path(deployment['dataset']).open('rb') as handle:
            handle.read(1)
    except PermissionError:
        return
    raise PermissionError('trainer can read raw holdout dataset')


def run_tick(deployment, role):
    if current_sid() != deployment[role+'_sid']:
        raise PermissionError('service SID mismatch: '+role)
    registry = Path(deployment['registry'])
    if role == 'trainer':
        from ml_candidate_ranker import train_and_evaluate, build_live_model_payload
        try:
            verify_trainer_isolation(deployment)
            report = train_and_evaluate(Path(deployment['training_input']))
            atomic(Path(deployment['candidate_output']), build_live_model_payload(report))
            result = {'state': 'CANDIDATE_ONLY', 'runtime_eligible': False}
        except Exception as exc:
            result = {'state': 'BLOCKED', 'reason': str(exc), 'runtime_eligible': False}
        result['run_time'] = provenance.utc_iso(datetime.now(timezone.utc))
        atomic(Path(deployment['trainer_status']), result)
        return result
    result = {'state': 'BLOCKED', 'runtime_eligible': False,
              'run_time': provenance.utc_iso(datetime.now(timezone.utc))}
    try:
        if not (registry/'manifest.json').exists():
            result['bootstrap_training_rows'] = export_training(Path(deployment['dataset']),
                Path(deployment['training_input']), provenance.parse_utc(deployment['bootstrap_cutoff']))
        evaluator.register(Path(deployment['candidate_input']), registry)
        result['collection'] = collect(registry, Path(deployment['dataset']))
        manifest = json.loads((registry/'manifest.json').read_bytes())
        result['training_rows'] = export_training(Path(deployment['dataset']), Path(deployment['training_input']),
                                                 provenance.parse_utc(manifest['holdout_start']))
        report = evaluator.evaluate(registry, registry/'forward_snapshot.jsonl')
        atomic(registry/'evaluation_latest.json', report)
        result['evaluation'] = report['learning_quality']
    except Exception as exc:
        result['collection_blocker'] = str(exc)
    result['controller'] = controller_tick(Path(deployment['portfolio_request']), Path(deployment['release_root']),
                                           harness_root=deployment.get('project_root'))
    atomic(Path(deployment['status']), result)
    return result


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--deployment', type=Path, required=True)
    p.add_argument('--role', choices=('trainer', 'evaluator'), required=True)
    args = p.parse_args()
    result = run_tick(json.loads(args.deployment.read_text(encoding='utf-8-sig')), args.role)
    print(json.dumps(result))
