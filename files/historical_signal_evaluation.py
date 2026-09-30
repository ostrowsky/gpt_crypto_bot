"""Immutable maximum-available snapshot audit. Not a promotion portfolio replay."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

import independent_signal_evaluator as evaluator


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def freeze(path, raw):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as handle:
        handle.write(raw)


def audit(dataset, registry, run, now=None, predictor=None):
    """One frozen run. Restart verifies source/input bindings, never retunes."""
    now = now or datetime.now(timezone.utc)
    manifest_path = run / 'audit_manifest.json'
    sources = {name: sha(Path(__file__).with_name(name)) for name in
               ('independent_signal_evaluator.py', 'historical_signal_evaluation.py',
                'ml_candidate_ranker.py', 'ml_signal_model.py', 'policy_provenance.py', 'config.py')}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_bytes())
        if (manifest.get('sources') != sources or sha(run/'candidate.json') != manifest['candidate_sha256']
                or sha(run/'snapshot.jsonl') != manifest['dataset_sha256']
                or sha(run/'manifest.json') != manifest['registry_manifest_sha256']):
            raise ValueError('immutable historical input/source mismatch')
        result = json.loads((run/'result.json').read_bytes())
        if sha(run/'result.json') != json.loads((run/'receipt.json').read_bytes())['result_sha256']:
            raise ValueError('historical result changed')
        return result
    run.mkdir(parents=True, exist_ok=False)
    freeze(run/'candidate.json', (registry/'candidate.json').read_bytes())
    freeze(run/'manifest.json', (registry/'manifest.json').read_bytes())
    # The real dataset's cooperative lock guards against replacement/append races.
    import critic_dataset
    from contextlib import nullcontext
    lock = critic_dataset._dataset_io_lock() if dataset.resolve() == critic_dataset.CRITIC_FILE.resolve() else nullcontext()
    with lock:
        with (run/'snapshot.jsonl').open('xb') as target, dataset.open('rb') as source:
            shutil.copyfileobj(source, target)
    manifest = {'schema_version': 1, 'contract': 'maximum-post-exposure-audit-v1',
                'created_at': evaluator.provenance.utc_iso(now),
                'candidate_sha256': sha(run/'candidate.json'),
                'dataset_sha256': sha(run/'snapshot.jsonl'),
                'registry_manifest_sha256': sha(run/'manifest.json'),
                'sources': sources, 'scope': 'entire_available_file_no_date_or_symbol_filter',
                'sealed_historical_protocol': False, 'runtime_eligible': False}
    freeze(manifest_path, json.dumps(manifest, sort_keys=True, allow_nan=False).encode())
    result = evaluator.evaluate(run, run/'snapshot.jsonl', predictor, now, historical=True)
    result['audit_manifest_sha256'] = sha(manifest_path)
    result['coverage_verified'] = False
    result['limitations'].append('point-in-time universe and downtime coverage not independently certified')
    freeze(run/'result.json', json.dumps(result, sort_keys=True, allow_nan=False).encode())
    freeze(run/'receipt.json', json.dumps({'result_sha256': sha(run/'result.json')}).encode())
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--registry', type=Path, default=evaluator.REGISTRY)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--publish', action='store_true', help='Publish runtime audit pointer only after completion')
    args = parser.parse_args()
    report = audit(args.dataset, args.registry, args.run)
    if args.publish:
        evaluator.atomic_json(evaluator.ROOT/'.runtime/historical_signal_evaluation/current.json',
                              {'run': str(args.run.resolve()), 'candidate_sha256': report['candidate_sha256']})
    print(json.dumps({k: report[k] for k in ('evaluation_scope', 'available_history', 'evaluation_start', 'learning_quality', 'signal_quality')}))
