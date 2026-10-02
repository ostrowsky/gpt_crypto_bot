"""Independent phase-by-phase portfolio intake; no fabricated certificates."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import independent_portfolio_gate as gate
from coverage_public_verifier import authority_material
from validated_ranker_rollout import canonical, sha

CONTRACT = 'independent-portfolio-inputs-v1'


def atomic(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    with tmp.open('wb') as handle:
        handle.write(canonical(value))
        handle.flush()
        os.fsync(handle.fileno())
    tmp.replace(path)


def read_bound(root, descriptor):
    relative = Path(descriptor['path'])
    if relative.is_absolute() or relative.drive:
        raise ValueError('absolute evidence path forbidden')
    path = (root/relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError('evidence path escapes protected intake')
    raw = path.read_bytes()
    if sha(raw) != descriptor['sha256']:
        raise ValueError('evidence hash mismatch: '+relative.name)
    return path, raw


def confirm(manifest_path, request_path, authority_key, evaluator_key, now=None, harness_root=None):
    now = time.time() if now is None else now
    result = {'state': 'BLOCKED', 'runtime_eligible': False, 'checked_at': now,
              'contract': CONTRACT, 'phases': {}, 'blockers': [],
              'required_inputs': ['bound champion/candidate bytes', 'certified maximum historical replay',
                                  'separate sealed and prospective shadow portfolios',
                                  'closed valuation grid and BTC benchmark',
                                  'coverage authority and evaluator keys', 'full Truth Harness PASS']}
    try:
        raw_manifest = manifest_path.read_bytes()
        manifest = json.loads(raw_manifest)
        if manifest.get('contract') != CONTRACT:
            raise ValueError('unsupported portfolio input contract; proxies not accepted')
        stage = manifest['stage']
        if stage not in ('CANARY', 'PROMOTED'):
            raise ValueError('unsupported stage')
        phases = ['historical', 'sealed', 'shadow']+(['canary'] if stage == 'PROMOTED' else [])
        root = manifest_path.parent
        candidate_path, candidate = read_bound(root, manifest['candidate'])
        champion_path, champion = read_bound(root, manifest['champion'])
        result.update(stage=stage, manifest_sha256=sha(raw_manifest),
                      candidate_sha256=sha(candidate), champion_sha256=sha(champion))
        evidence, descriptors = {}, {}
        if set(manifest['evidence'])-set(phases):
            raise ValueError('unexpected evidence phases')
        for phase in phases:
            if phase not in manifest['evidence']:
                result['phases'][phase] = {'state': 'UNKNOWN', 'reason': 'missing phase'}
                result['blockers'].append(phase+': missing phase')
                continue
            try:
                entry = manifest['evidence'][phase]
                bundle_path, raw = read_bound(root, entry['bundle'])
                cert_path, cert_raw = read_bound(root, entry['certification'])
                cert = json.loads(cert_raw)
                checked = gate.evaluate_bundle(raw, cert, authority_key, sha(candidate), sha(champion), now)
                if checked['phase'] != phase:
                    raise ValueError('phase identity mismatch')
                result['phases'][phase] = dict(checked, state='PASS' if checked['passed'] else 'REJECTED')
                if not checked['passed']:
                    result['blockers'].append(phase+': after-cost/risk/uncertainty gate rejected')
                evidence[phase] = (raw, cert)
                descriptors[phase] = {'bundle': str(bundle_path), 'certification': str(cert_path)}
            except Exception as exc:
                result['phases'][phase] = {'state': 'UNKNOWN', 'reason': str(exc)}
                result['blockers'].append(phase+': '+str(exc))
        if result['blockers']:
            return result
        # Cross-phase chronology, freshness and Harness are checked independently
        # again; a phase PASS alone is never activation permission.
        gate.authorize(evidence, authority_key, evaluator_key, candidate, champion,
                       stage, now=now, harness_root=harness_root)
        # Detect mutable inputs during expensive evaluation before publishing.
        if manifest_path.read_bytes() != raw_manifest:
            raise ValueError('manifest changed during evaluation')
        for desc in (manifest['candidate'], manifest['champion']):
            read_bound(root, desc)
        for entry in manifest['evidence'].values():
            read_bound(root, entry['bundle'])
            read_bound(root, entry['certification'])
        atomic(request_path, {'stage': stage, 'candidate': str(candidate_path),
                              'champion': str(champion_path), 'evidence': descriptors,
                              'manifest_sha256': sha(raw_manifest)})
        result['state'] = 'READY'
        # READY is evidence eligibility only. Controller/config own activation.
        result['request'] = str(request_path)
    except Exception as exc:
        result['blockers'].append(str(exc))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    report = confirm(args.manifest, args.request,
                     authority_material(),
                     os.environ.get('RANKER_EVALUATOR_KEY', '').encode())
    atomic(args.report, report)
    print(json.dumps({'state': report['state'], 'blockers': report['blockers']}))
    raise SystemExit(0 if report['state'] == 'READY' else 1)
