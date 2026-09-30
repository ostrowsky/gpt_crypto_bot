"""Evaluator-owned frozen prospective cohort. Never trains or enables a policy."""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
from pathlib import Path

import policy_provenance as provenance

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / '.runtime/independent_evaluation'
CONTRACT = 'independent-forward-top1-v1'
MIN_DAYS = 30
MIN_GROUPS = 100
EMBARGO_HOURS = 48


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    tmp.replace(path)


def register(model_path, registry=REGISTRY, now=None):
    """Freeze once. Trainer replacement cannot reset this experiment's holdout."""
    manifest_path = registry / 'manifest.json'
    if manifest_path.exists():
        return json.loads(manifest_path.read_bytes())
    now = now or datetime.now(timezone.utc)
    raw = model_path.read_bytes()
    payload = json.loads(raw)
    epochs = set()
    for scope in (payload.get('evaluation_provenance') or {}).get('split_scopes', {}).values():
        epochs.update((scope.get('policy_epoch_counts') or {}).keys())
    if len(epochs) != 1:
        raise ValueError('unknown or mixed frozen candidate policy population')
    scopes = (payload.get('evaluation_provenance') or {}).get('split_scopes') or {}
    if set(scopes) != {'train', 'validation', 'test'}:
        raise ValueError('candidate lacks training/validation/test exposure boundaries')
    for scope in scopes.values():
        for key in ('last_feature_time', 'last_label_time', 'last_label_recorded_at'):
            when = provenance.parse_utc(scope.get(key))
            if when is None or when >= now:
                raise ValueError('unknown or future candidate exposure boundary')
    manifest = {
        'schema_version': 1, 'contract': CONTRACT, 'candidate_sha256': digest(raw),
        'registered_at': provenance.utc_iso(now),
        'holdout_start': provenance.utc_iso(now + timedelta(hours=EMBARGO_HOURS)),
        'baseline': 'candidate_score_top1_id_tiebreak',
        'target': 'ret_5_percent_proxy_not_portfolio_return',
        'min_days': MIN_DAYS, 'min_groups': MIN_GROUPS,
        'acceptance': 'paired_daily_delta_95ci_lower_bound_gt_zero',
        'runtime_eligible': False,
    }
    registry.mkdir(parents=True, exist_ok=True)
    # Exclusive creation is a fail-closed race check, never overwrite a frozen model.
    with (registry / 'candidate.json').open('xb') as handle:
        handle.write(raw)
    atomic_json(manifest_path, manifest)
    return manifest


def evaluate(registry, dataset, predictor=None, now=None, *, historical=False):
    now = now or datetime.now(timezone.utc)
    manifest_raw = (registry / 'manifest.json').read_bytes()
    manifest = json.loads(manifest_raw)
    raw = (registry / 'candidate.json').read_bytes()
    if digest(raw) != manifest['candidate_sha256'] or manifest.get('contract') != CONTRACT:
        raise ValueError('frozen candidate/contract mismatch')
    if (manifest.get('min_days') != MIN_DAYS or manifest.get('min_groups') != MIN_GROUPS
            or manifest.get('baseline') != 'candidate_score_top1_id_tiebreak'
            or manifest.get('acceptance') != 'paired_daily_delta_95ci_lower_bound_gt_zero'):
        raise ValueError('preregistered criteria changed')
    payload = json.loads(raw)
    epochs = set()
    for scope in (payload.get('evaluation_provenance') or {}).get('split_scopes', {}).values():
        epochs.update((scope.get('policy_epoch_counts') or {}).keys())
    if len(epochs) != 1:
        raise ValueError('unknown or mixed frozen candidate policy population')
    from ml_candidate_ranker import safe_feature_names, predict_final_score_from_candidate_payload
    names = payload.get('feature_names') or []
    if not names or not set(names).issubset(set(safe_feature_names())):
        raise ValueError('unknown or label-encoding feature')
    predictor = predictor or predict_final_score_from_candidate_payload
    start = provenance.parse_utc(manifest['holdout_start'])
    registered = provenance.parse_utc(manifest['registered_at'])
    if start is None or registered is None or start < registered + timedelta(hours=EMBARGO_HOURS):
        raise ValueError('holdout/embargo changed')
    if historical:
        scopes = (payload.get('evaluation_provenance') or {}).get('split_scopes') or {}
        if set(scopes) != {'train', 'validation', 'test'}:
            raise ValueError('unknown historical exposure')
        exposure = [provenance.parse_utc(s.get(k)) for s in scopes.values()
                    for k in ('last_feature_time', 'last_label_time', 'last_label_recorded_at')]
        if any(t is None or t >= registered for t in exposure):
            raise ValueError('unknown historical exposure')
        start = max(exposure) + timedelta(hours=EMBARGO_HOURS)
    groups, ids, incomplete_groups = defaultdict(list), set(), set()
    pending, invalid, eligible = 0, 0, 0
    invalid_reasons = defaultdict(int)
    total, unknown_time, excluded = 0, 0, 0
    first_available, last_available = None, None
    dataset_hash = hashlib.sha256()
    with dataset.open('rb') as handle:
        for line in handle:
            total += 1
            dataset_hash.update(line)
            try:
                row = json.loads(line)
            except (ValueError, TypeError):
                invalid += 1
                invalid_reasons['malformed_json'] += 1
                continue
            if not isinstance(row, dict):
                invalid += 1
                invalid_reasons['non_object'] += 1
                continue
            feature = provenance.parse_utc((row.get('provenance') or {}).get('feature_time'))
            if feature is None:
                unknown_time += 1
            else:
                first_available = min(first_available, feature) if first_available else feature
                last_available = max(last_available, feature) if last_available else feature
                if feature < start:
                    excluded += 1
            if feature is None or not start <= feature <= now:
                continue
            eligible += 1
            decision = provenance.parse_utc((row.get('decision_provenance') or {}).get('decision_time'))
            violations = []
            if not provenance.observation_provenance_valid(row):
                violations.append('invalid_observation_provenance')
            if provenance.canonical_policy_epoch((row.get('provenance') or {}).get('policy_epoch')) not in epochs:
                violations.append('other_policy_epoch')
            if (row.get('provenance') or {}).get('dataset_contract') != 'candidate-outcome-v2':
                violations.append('other_dataset_contract')
            if row.get('tf') not in {'15m', '1h', '4h'}:
                violations.append('unsupported_timeframe')
            if not row.get('id') or row['id'] in ids:
                violations.append('unknown_or_duplicate_id')
            if decision is None or decision > feature + timedelta(minutes=1):
                violations.append('unknown_or_late_decision')
            if violations:
                invalid += 1
                for reason in violations:
                    invalid_reasons[reason] += 1
                continue
            ids.add(row['id'])
            group = provenance.utc_iso(datetime.fromtimestamp(int(row['bar_ts'])/1000,
                                                              timezone.utc)
                                       + provenance.timeframe_delta(row['tf'])) + '|' + row['tf']
            labels = row.get('labels') or {}
            due = provenance.forward_label_time(bar_ts=int(row['bar_ts']), tf=row['tf'], horizon=5)
            label = (row.get('label_provenance') or {}).get('ret_5') or {}
            label_time = provenance.parse_utc(label.get('label_time'))
            recorded = provenance.parse_utc(label.get('recorded_at'))
            if labels.get('ret_5') is None:
                incomplete_groups.add(group)
                if now >= due + provenance.timeframe_delta(row['tf']) * 2:
                    pending += 1
                continue
            try:
                ret = float(labels['ret_5'])
                score = float((row.get('decision') or {}).get('candidate_score'))
                if (not math.isfinite(ret) or not math.isfinite(score)
                        or not provenance.label_provenance_valid(row, 'ret_5')
                        or label_time != due or recorded is None or recorded > now
                        or decision >= label_time):
                    raise ValueError('invalid causal target')
                # Prediction never receives outcomes, teachers or label provenance.
                causal = {k: v for k, v in row.items()
                          if k not in {'labels', 'teacher', 'label_provenance'}}
                prediction = float(predictor(payload, causal))
                if not math.isfinite(prediction):
                    raise ValueError('nonfinite prediction')
            except (ValueError, TypeError, KeyError):
                invalid += 1
                invalid_reasons['invalid_causal_target_or_prediction'] += 1
                continue
            groups[group].append((row['id'], score, prediction, ret))
    daily, pairs = defaultdict(list), []
    for key, rows in sorted(groups.items()):
        if len(rows) < 2 or key in incomplete_groups:
            continue
        baseline = sorted(rows, key=lambda r: (-r[1], r[0]))[0]
        challenger = sorted(rows, key=lambda r: (-r[2], r[0]))[0]
        delta = challenger[3] - baseline[3]
        daily[key[:10]].append(delta)
        pairs.append({'group': key, 'baseline_id': baseline[0], 'candidate_id': challenger[0],
                      'baseline_ret5': baseline[3], 'candidate_ret5': challenger[3], 'delta': delta})
    values = [sum(v) / len(v) for _, v in sorted(daily.items())]
    ci = None
    if len(values) >= MIN_DAYS and len(pairs) >= MIN_GROUPS:
        import numpy as np
        rng = np.random.default_rng(20260930)
        draws = rng.choice(values, size=(2000, len(values)), replace=True).mean(axis=1)
        ci = [float(x) for x in np.quantile(draws, [.025, .975])]
    verdict = ('BLOCKED' if invalid or pending else 'UNKNOWN' if ci is None
               else 'PASS_PROXY' if ci[0] > 0 else 'REJECTED' if ci[1] < 0 else 'INCONCLUSIVE')
    baseline_wins = sum(p['baseline_ret5'] > 0 for p in pairs)
    candidate_wins = sum(p['candidate_ret5'] > 0 for p in pairs)
    return {
        'schema_version': 1, 'contract': CONTRACT, 'generated_at': provenance.utc_iso(now),
        'evaluation_scope': 'retrospective_post_exposure_audit' if historical else 'prospective_frozen_cohort',
        'sealed_holdout': not historical,
        'evaluation_start': provenance.utc_iso(start),
        'available_history': {'rows': total, 'unknown_feature_time_rows': unknown_time,
                              'excluded_before_cutoff_rows': excluded,
                              'first_feature_time': provenance.utc_iso(first_available) if first_available else None,
                              'last_feature_time': provenance.utc_iso(last_available) if last_available else None},
        'candidate_sha256': digest(raw), 'manifest_sha256': digest(manifest_raw),
        'dataset_sha256': dataset_hash.hexdigest(), 'holdout_start': manifest['holdout_start'],
        'learning_quality': {'verdict': verdict, 'eligible_rows': eligible,
                             'invalid_rows': invalid, 'overdue_rows': pending,
                             'invalid_reason_counts': dict(sorted(invalid_reasons.items())),
                             'paired_groups': len(pairs), 'valid_days': len(values),
                             'excluded_incomplete_groups': len(incomplete_groups),
                             'baseline_positive': {'numerator': baseline_wins, 'denominator': len(pairs)},
                             'candidate_positive': {'numerator': candidate_wins, 'denominator': len(pairs)},
                             'positive_rate_lift': candidate_wins / baseline_wins if baseline_wins else None,
                             'mean_daily_ret5_delta_pp': sum(values)/len(values) if values else None,
                             'paired_daily_95ci': ci},
        'signal_quality': {'verdict': 'UNKNOWN', 'reason': 'portfolio replay and live outcomes required'},
        'pairs': pairs, 'runtime_eligible': False, 'achievement_claimed': False,
        'daily_deltas': {k: sum(v)/len(v) for k, v in sorted(daily.items())},
        'limitations': ['forward proxy only; no fees, exits or portfolio alpha',
                        'no automatic candidate replacement or production promotion',
                        'retrospective audit is not a sealed promotion holdout' if historical
                        else 'sealed future cohort; trainer historical test not reused'],
    }


def run_once(model, dataset, registry=REGISTRY):
    register(model, registry)
    report = evaluate(registry, dataset)
    atomic_json(registry / 'evaluation_latest.json', report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--registry', type=Path, default=REGISTRY)
    args = parser.parse_args()
    result = run_once(args.model, args.dataset, args.registry)
    print(json.dumps(result['learning_quality']))
