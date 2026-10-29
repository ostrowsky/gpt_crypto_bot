"""Immutable RDE-1 research registration and evaluation contracts.

No exchange adapter, secrets, live order authority or optional import at load.
"""
from pathlib import Path
import hashlib
import json
import math


SEEDS = (11, 23, 37, 53, 71)
THRESHOLDS = (0., .0005, .001, .002)


def _bytes(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def _immutable(path, value, error):
    path = Path(path)
    raw = _bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open('xb') as handle:
            handle.write(raw)
    except FileExistsError:
        if path.read_bytes() != raw:
            raise ValueError(error)
    return hashlib.sha256(raw).hexdigest()


def register_run(folder, value):
    if (value.get('evidence_kind') != 'SIMULATION' or value.get('promotion_allowed') is not False
            or value.get('holdout_kind') != 'RETROSPECTIVE_REUSED_HOLDOUT'):
        raise ValueError('RESEARCH_SCOPE_INVALID')
    return _immutable(Path(folder) / 'registration.json', value, 'REGISTRATION_CHANGED')


def freeze_predictions(path, records):
    return _immutable(path, records, 'FROZEN_PREDICTIONS_CHANGED')


def direction_metrics(labels, probs, actual_returns, expected_returns):
    if len({len(x) for x in (labels, probs, actual_returns, expected_returns)}) != 1:
        raise ValueError('PREDICTION_INVALID')
    rows = list(zip(labels, probs, actual_returns, expected_returns))
    for y, p, actual, expected in rows:
        if (y not in (0, 1) or not all(math.isfinite(float(v)) for v in (p, actual, expected))
                or not 0 <= p <= 1):
            raise ValueError('PREDICTION_INVALID')
    count = len(rows)
    positive = sum(y for y, *_ in rows)
    correct = sum((p >= .5) == y for y, p, *_ in rows)
    balanced = None
    if 0 < positive < count:
        balanced = .5 * (sum(y == 1 and p >= .5 for y, p, *_ in rows) / positive
                        + sum(y == 0 and p < .5 for y, p, *_ in rows) / (count-positive))
    return dict(rows_N=count, positive_n=positive, negative_n=count-positive, correct_n=correct,
        train_or_test_prevalence=positive/count if count else None,
        accuracy=correct/count if count else None, balanced_accuracy=balanced,
        brier=sum((p-y)**2 for y, p, *_ in rows)/count if count else None,
        logloss=-sum(y*math.log(max(1e-15,p))+(1-y)*math.log(max(1e-15,1-p))
                     for y,p,*_ in rows)/count if count else None,
        return_mae=sum(abs(a-e) for _,_,a,e in rows)/count if count else None,
        return_rmse=(sum((a-e)**2 for _,_,a,e in rows)/count)**.5 if count else None)


def forecast_records(raw, index, *, origin_ms, feature_end_ms, fit_cutoff_ms):
    if feature_end_ms >= origin_ms or fit_cutoff_ms > origin_ms:
        raise ValueError('FORECAST_TIME_INVALID')
    result = {}
    for name, prefix in (('CatBoost', 'cat'), ('GRU', 'gru'), ('PPO', 'gru')):
        result[name] = dict(origin_ms=origin_ms, feature_end_ms=feature_end_ms,
            fit_cutoff_ms=fit_cutoff_ms, p_up=list(map(float,raw[prefix+'_proba'][index])),
            expected_logreturn=list(map(float,raw[prefix+'_return'][index])),
            forecast_source='GRU_FROZEN' if name == 'PPO' else name)
    return result


def seed_daily_mean(rows, seeds):
    if (len(rows) != len(seeds) or len({r['seed'] for r in rows}) != len(rows)
            or {r['seed'] for r in rows} != set(seeds)
            or any(set(r['daily_pnl']) != set(rows[0]['daily_pnl']) for r in rows)):
        raise ValueError('SEED_OR_DAY_COVERAGE')
    return {day: sum(r['daily_pnl'][day] for r in rows)/len(rows) for day in rows[0]['daily_pnl']}


def comparison_verdict(*, paired_n, delta_lower, absolute_lower, stress_net, max_dd, reference_dd):
    if paired_n < 30 or any(v is None for v in (delta_lower,absolute_lower,stress_net,max_dd,reference_dd)):
        return 'INCONCLUSIVE'
    if delta_lower > .01 and absolute_lower > 0 and stress_net > 0 and max_dd <= reference_dd+.5:
        return 'EXPLORATORY_PATTERN_POWER_UNCERTIFIED'
    return 'NOT_SUPPORTED'


def select_threshold(rows):
    if not rows or any(r.get('segment') != 'tune' for r in rows):
        raise ValueError('TUNE_ONLY')
    if any(r['mean_daily_net_pnl_usdt'] is None for r in rows):
        raise ValueError('INCOMPLETE_TUNE')
    return max(rows, key=lambda r: (r['mean_daily_net_pnl_usdt'],r['threshold']))['threshold']
