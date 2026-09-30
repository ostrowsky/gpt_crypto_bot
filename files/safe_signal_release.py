"""Fail-closed release readiness and durable evidence ledger; no live write path."""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import policy_provenance as provenance
from historical_signal_evaluation import sha
from ml_candidate_ranker import predict_final_score_from_candidate_payload as predict_score

CONTRACT = 'safe-signal-release-readiness-v1'
ROOT = Path(__file__).resolve().parents[1]/'.runtime/safe_signal_release'
RELEASE_BLOCKERS = ('sealed_historical_protocol_missing',
                    'maximum_period_after_cost_ten_slot_replay_missing',
                    'point_in_time_coverage_not_certified',
                    'forward_shadow_and_canary_adapter_not_delivered')


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def reconstruct(report):
    """Independent numeric implementation: verdict strings are untrusted."""
    pairs = report.get('pairs')
    if not isinstance(pairs, list):
        raise ValueError('missing paired trace')
    daily, seen = defaultdict(list), set()
    base, candidate = 0, 0
    for pair in pairs:
        group = pair['group']
        if group in seen or not pair['baseline_id'] or not pair['candidate_id']:
            raise ValueError('duplicate or unidentified pair')
        seen.add(group)
        left, right = float(pair['baseline_ret5']), float(pair['candidate_ret5'])
        if not math.isfinite(left) or not math.isfinite(right):
            raise ValueError('nonfinite trace')
        delta = right-left
        if not math.isclose(delta, float(pair['delta']), abs_tol=1e-10):
            raise ValueError('paired trace arithmetic mismatch')
        daily[group[:10]].append(delta)
        base += left > 0
        candidate += right > 0
    values = [sum(v)/len(v) for _, v in sorted(daily.items())]
    ci = None
    if len(values) >= 30 and len(pairs) >= 100:
        rng = np.random.default_rng(20260930)
        draws = rng.choice(values, size=(2000, len(values)), replace=True).mean(axis=1)
        ci = [float(v) for v in np.quantile(draws, [.025, .975])]
    expected = {'paired_groups': len(pairs), 'valid_days': len(values),
                'baseline_positive': {'numerator': base, 'denominator': len(pairs)},
                'candidate_positive': {'numerator': candidate, 'denominator': len(pairs)},
                'mean_daily_ret5_delta_pp': sum(values)/len(values) if values else None,
                'positive_rate_lift': candidate/base if base else None,
                'paired_daily_95ci': ci}
    q = report['learning_quality']
    for key, value in expected.items():
        if canonical(q.get(key)) != canonical(value):
            raise ValueError('independent summary mismatch: '+key)
    if q.get('invalid_rows', -1) < 0 or q.get('overdue_rows', -1) < 0:
        raise ValueError('unknown data quality')
    verdict = ('BLOCKED' if q['invalid_rows'] or q['overdue_rows'] else 'UNKNOWN' if ci is None
               else 'PASS_PROXY' if ci[0] > 0 else 'REJECTED' if ci[1] < 0 else 'INCONCLUSIVE')
    if verdict != q.get('verdict'):
        raise ValueError('independent verdict mismatch')
    return verdict


def check_time(report, now):
    generated = provenance.parse_utc(report.get('generated_at'))
    if generated is None or generated > now or now-generated > timedelta(hours=24):
        raise ValueError('stale or future evaluation')


def verify_history(run):
    manifest = json.loads((run/'audit_manifest.json').read_bytes())
    result = json.loads((run/'result.json').read_bytes())
    receipt = json.loads((run/'receipt.json').read_bytes())
    if (manifest.get('contract') != 'maximum-post-exposure-audit-v1'
            or manifest.get('scope') != 'entire_available_file_no_date_or_symbol_filter'
            or result.get('evaluation_scope') != 'retrospective_post_exposure_audit'
            or result.get('sealed_holdout') is not False):
        raise ValueError('historical contract/scope mismatch')
    bindings = {'candidate.json': 'candidate_sha256', 'snapshot.jsonl': 'dataset_sha256',
                'manifest.json': 'registry_manifest_sha256'}
    for filename, key in bindings.items():
        if sha(run/filename) != manifest[key]:
            raise ValueError('historical artifact mismatch: '+filename)
    for name, expected in manifest['sources'].items():
        if Path(name).name != name or sha(Path(__file__).with_name(name)) != expected:
            raise ValueError('evaluator source changed')
    if (sha(run/'result.json') != receipt['result_sha256']
            or result['audit_manifest_sha256'] != sha(run/'audit_manifest.json')
            or result['candidate_sha256'] != manifest['candidate_sha256']
            or result['dataset_sha256'] != manifest['dataset_sha256']
            or result['manifest_sha256'] != manifest['registry_manifest_sha256']):
        raise ValueError('historical result binding mismatch')
    # Independently bind every selected return to the immutable raw observation.
    wanted = {p[k] for p in result['pairs'] for k in ('baseline_id', 'candidate_id')}
    selected_groups = {p['group'] for p in result['pairs']}
    raw_rows, choices, seen = {}, defaultdict(list), set()
    payload = json.loads((run/'candidate.json').read_bytes())
    epochs = {key for scope in payload['evaluation_provenance']['split_scopes'].values()
              for key in scope['policy_epoch_counts']}
    start = provenance.parse_utc(result['evaluation_start'])
    end = provenance.parse_utc(result['generated_at'])
    with (run/'snapshot.jsonl').open(encoding='utf-8') as handle:
        for line in handle:
            row = json.loads(line)
            if row.get('id') in wanted:
                if row['id'] in raw_rows:
                    raise ValueError('duplicate selected observation')
                raw_rows[row['id']] = row
            feature = provenance.parse_utc((row.get('provenance') or {}).get('feature_time'))
            decision = provenance.parse_utc((row.get('decision_provenance') or {}).get('decision_time'))
            if (feature is None or not start <= feature <= end or decision is None
                    or decision > feature+timedelta(minutes=1)
                    or not provenance.observation_provenance_valid(row)
                    or row.get('tf') not in {'15m', '1h', '4h'}
                    or provenance.canonical_policy_epoch(row['provenance'].get('policy_epoch')) not in epochs
                    or row['provenance'].get('dataset_contract') != 'candidate-outcome-v2'
                    or not row.get('id') or row['id'] in seen):
                continue
            seen.add(row['id'])
            closed = datetime.fromtimestamp(int(row['bar_ts'])/1000, timezone.utc)+provenance.timeframe_delta(row['tf'])
            group = provenance.utc_iso(closed)+'|'+row['tf']
            if group not in selected_groups:
                continue
            due = provenance.forward_label_time(bar_ts=int(row['bar_ts']), tf=row['tf'], horizon=5)
            target = (row.get('label_provenance') or {}).get('ret_5') or {}
            recorded = provenance.parse_utc(target.get('recorded_at'))
            if (row.get('labels', {}).get('ret_5') is None
                    or not provenance.label_provenance_valid(row, 'ret_5')
                    or provenance.parse_utc(target.get('label_time')) != due
                    or recorded is None or recorded > end or decision >= due):
                continue
            causal = {k: v for k, v in row.items() if k not in {'labels', 'teacher', 'label_provenance'}}
            baseline = float(row['decision']['candidate_score'])
            predicted = float(predict_score(payload, causal))
            if not math.isfinite(baseline) or not math.isfinite(predicted):
                continue
            choices[group].append((row['id'], baseline, predicted))
    for pair in result['pairs']:
        group_choices = choices[pair['group']]
        if (len(group_choices) < 2
                or sorted(group_choices, key=lambda r: (-r[1], r[0]))[0][0] != pair['baseline_id']
                or sorted(group_choices, key=lambda r: (-r[2], r[0]))[0][0] != pair['candidate_id']):
            raise ValueError('independent decision reconstruction mismatch')
        for role in ('baseline', 'candidate'):
            row = raw_rows[pair[role+'_id']]
            close = datetime.fromtimestamp(int(row['bar_ts'])/1000, timezone.utc)+provenance.timeframe_delta(row['tf'])
            if (pair['group'] != provenance.utc_iso(close)+'|'+row['tf']
                    or float(row['labels']['ret_5']) != pair[role+'_ret5']
                    or not provenance.label_provenance_valid(row, 'ret_5')):
                raise ValueError('selected trace does not match raw evidence')
    return result


def decide(history_run, forward_path, registry, now=None):
    now = now or datetime.now(timezone.utc)
    reasons = list(RELEASE_BLOCKERS)
    candidate_sha = sha(registry/'candidate.json')
    inputs = {'candidate_sha256': candidate_sha, 'history_run': str(history_run),
              'forward_path': str(forward_path)}
    state = 'BLOCKED'
    try:
        history = verify_history(history_run)
        forward = json.loads(forward_path.read_bytes())
        inputs.update(history_sha256=sha(history_run/'result.json'), forward_sha256=sha(forward_path))
        for report in (history, forward):
            check_time(report, now)
            if report.get('candidate_sha256') != candidate_sha or report.get('contract') != 'independent-forward-top1-v1':
                raise ValueError('candidate/metric contract mismatch')
        if (forward.get('evaluation_scope') != 'prospective_frozen_cohort'
                or forward.get('manifest_sha256') != sha(registry/'manifest.json')):
            raise ValueError('forward manifest/scope mismatch')
        hist_verdict, forward_verdict = reconstruct(history), reconstruct(forward)
        reasons.extend(['historical_proxy='+hist_verdict, 'forward_proxy='+forward_verdict])
        if 'REJECTED' in (hist_verdict, forward_verdict):
            state = 'REJECTED'
        elif hist_verdict == 'PASS_PROXY' and forward_verdict not in ('BLOCKED', 'REJECTED'):
            state = 'SHADOW'
    except (ValueError, KeyError, TypeError, OSError) as exc:
        reasons.append('evidence_verification_failed: '+str(exc))
    return {'contract': CONTRACT, 'generated_at': provenance.utc_iso(now),
            'controller_source_sha256': sha(Path(__file__)),
            'state': state, 'candidate_sha256': candidate_sha,
            'reasons': reasons, 'inputs': inputs,
            'owner': 'repository maintainer', 'repair_action': 'complete independent portfolio and forward release gates',
            'runtime_eligible': False, 'achievement_claimed': False}


def persist(root, decision):
    if (decision.get('state') not in {'BLOCKED', 'REJECTED', 'SHADOW'}
            or decision.get('runtime_eligible') is not False
            or decision.get('achievement_claimed') is not False):
        raise ValueError('unsupported release state or production eligibility')
    root.mkdir(parents=True, exist_ok=True)
    lock = root/'controller.lock'
    lock.open('xb').close()
    try:
        ledger = root/'transitions.jsonl'
        records = []
        previous = None
        if ledger.exists():
            for line in ledger.read_text(encoding='utf-8').splitlines():
                entry = json.loads(line)
                if (entry.get('previous') != previous
                        or entry['body'].get('previous_transition') != previous
                        or entry.get('sha256') != digest(entry['body'])):
                    raise ValueError('release ledger corrupt')
                previous = entry['sha256']
                records.append(entry)
        stable = {k: v for k, v in decision.items() if k != 'generated_at'}
        if records:
            latest = records[-1]['body']
            if latest['candidate_sha256'] != decision['candidate_sha256']:
                raise ValueError('candidate rotation requires a new release registry')
            if latest['state'] == 'REJECTED' and decision['state'] != 'REJECTED':
                raise ValueError('rejected candidate cannot silently re-enter')
            if canonical({k: v for k, v in latest.items() if k not in ('generated_at', 'previous_transition')}) == canonical(stable):
                return latest
        entry = {'previous': previous, 'body': dict(decision, previous_transition=previous)}
        entry['sha256'] = digest(entry['body'])
        with ledger.open('a', encoding='utf-8') as handle:
            handle.write(canonical(entry).decode()+'\n')
            handle.flush()
            import os
            os.fsync(handle.fileno())
        return decision
    finally:
        lock.unlink(missing_ok=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--history-run', type=Path)
    parser.add_argument('--registry', type=Path, required=True)
    parser.add_argument('--forward', type=Path, required=True)
    parser.add_argument('--state-dir', type=Path, default=ROOT)
    args = parser.parse_args()
    if args.history_run is None:
        pointer = json.loads((ROOT.parent/'historical_signal_evaluation/current.json').read_bytes())
        args.history_run = Path(pointer['run'])
        if not args.history_run.resolve().is_relative_to((ROOT.parent/'historical_signal_evaluation').resolve()):
            raise ValueError('audit pointer outside evaluator registry')
    decision = decide(args.history_run, args.forward, args.registry)
    print(json.dumps(persist(args.state_dir, decision)))
