"""Pure causal capacity features and retrospective two-option ranking pre-gate."""
from datetime import datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np

BAR = 900_000
HORIZON = 5 * BAR
TZ = ZoneInfo('Europe/Budapest')
FEATURES = ('return_1_pct', 'return_4_pct', 'return_16_pct',
            'volatility_16_pct', 'volume_ratio_16', 'range_16_pct',
            'range_position_16', 'candidate_role')


def causal_features(data, clock, role):
    """Closed, exact 17-bar prefix only. Return None for unknown past evidence."""
    closes = data['t'] + BAR
    i = int(np.searchsorted(closes, clock, side='right')) - 1
    if i < 16 or closes[i] != clock:
        return None
    rows = data[i-16:i+1]
    if not np.array_equal(rows['t'] + BAR, clock - np.arange(16, -1, -1)*BAR):
        return None
    a = np.column_stack([rows[n] for n in ('o', 'h', 'l', 'c', 'v')])
    if (not np.isfinite(a).all() or (a[:, :4] <= 0).any()
            or (a[:, 4] < 0).any() or (a[:, 1] < a[:, 2]).any()
            or (a[:, 1] < np.maximum(a[:, 0], a[:, 3])).any()
            or (a[:, 2] > np.minimum(a[:, 0], a[:, 3])).any()):
        return None
    c = rows['c']; r = np.diff(np.log(c))
    low, high = float(rows['l'][1:].min()), float(rows['h'][1:].max())
    v = float(rows['v'][1:].mean())
    return np.array([(c[-1]/c[-1-k]-1)*100 for k in (1, 4, 16)] + [
        float(r.std())*100, float(rows['v'][-1]/v) if v > 0 else 0.,
        (high/low-1)*100, (float(c[-1])-low)/(high-low) if high > low else .5,
        float(role)], dtype=float)


def forward_return(data, clock):
    """Require exact origin plus all five future closes, never stale asof labels."""
    closes = data['t'] + BAR
    i = int(np.searchsorted(closes, clock))
    if i >= len(data) or i+5 >= len(data):
        return None
    rows = data[i:i+6]
    if not np.array_equal(rows['t']+BAR, clock+np.arange(6)*BAR):
        return None
    if not np.isfinite(rows['c']).all() or (rows['c'] <= 0).any():
        return None
    return float((rows['c'][-1]/rows['c'][0]-1)*100)


def daily_identity(clock):
    local = datetime.fromtimestamp(clock/1000, timezone.utc).astimezone(TZ)
    cutoff = datetime.combine(local.date(), time(22), tzinfo=TZ)
    return local.date().isoformat(), int(cutoff.timestamp()*1000)


def split_boundaries(start, end):
    first = datetime.fromtimestamp(start/1000, timezone.utc).astimezone(TZ).date()
    last = datetime.fromtimestamp(end/1000, timezone.utc).astimezone(TZ).date()
    days = (last-first).days
    if days < 5:
        raise ValueError('too few chronological days')
    return tuple(int(datetime.combine(first+timedelta(days=int(days*f)),
                      time(), tzinfo=TZ).timestamp()*1000) for f in (.6, .8))


def cohort(clock, available_at, cuts):
    a, b = cuts
    if available_at < clock:
        raise ValueError('invalid label availability')
    if clock < a:
        return 'train' if available_at < a else 'purged'
    if clock < b:
        return 'validation' if available_at < b else 'purged'
    return 'test'


def relevance(returns):
    returns = np.asarray(returns, dtype=float)
    if returns.ndim != 2 or returns.shape[1] != 2 or not np.isfinite(returns).all():
        raise ValueError('finite two-option target required')
    return (returns > returns[:, ::-1]).astype(float)


def score_actions(scores):
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 2 or scores.shape[1] != 2 or not np.isfinite(scores).all():
        raise ValueError('finite two-option prediction required')
    return scores[:, 0] > scores[:, 1]  # strict ties keep incumbent


def block_interval(days, deltas, *, draws=5000):
    unique = sorted(set(days))
    if len(unique) < 3:
        return None
    sums = np.array([sum(d for day, d in zip(days, deltas) if day == key) for key in unique])
    counts = np.array([sum(day == key for day in days) for key in unique])
    rng = np.random.default_rng(42)
    starts = rng.integers(0, len(unique)-2, size=(draws, (len(unique)+2)//3))
    idx = (starts[:, :, None]+np.arange(3)).reshape(draws, -1)[:, :len(unique)]
    means = sums[idx].sum(axis=1)/counts[idx].sum(axis=1)
    return [float(x) for x in np.quantile(means, [.025, .975])]


def mission_stats(selected, leader, capture, known):
    idx = np.arange(len(selected))
    chosen_leader = leader[idx, selected]
    chosen_capture = capture[idx, selected]
    qualified = known & chosen_leader & np.isfinite(chosen_capture)
    return {'known_groups': int(known.sum()), 'leader_count': int((known & chosen_leader).sum()),
            'early_count': int((qualified & (chosen_capture >= .35)).sum()),
            'leader_rate': float(chosen_leader[known].mean()) if known.any() else None,
            'capture_mean': float(chosen_capture[qualified].mean()) if qualified.any() else None}


def evaluate(scores, returns, days, leader, capture, known, *, population_complete):
    """Selection diagnostics, NOT portfolio policy outcomes or deployment approval."""
    swap = score_actions(scores)
    select = np.where(swap, 0, 1)
    idx = np.arange(len(select))
    realized = returns[idx, select]
    keep = returns[:, 1]
    delta = realized-keep
    ci = block_interval(days, delta)
    baseline = mission_stats(np.ones(len(select), dtype=int), leader, capture, known)
    model = mission_stats(select, leader, capture, known)
    checks = {
        'sample': len(select) >= 100 and len(set(days)) >= 20,
        'mean_uplift': float(delta.mean()) >= .05 if len(delta) else False,
        'paired_lower_ci_positive': ci is not None and ci[0] > 0,
        'leader_noninferiority': model['leader_rate'] is not None and
            model['leader_rate'] >= baseline['leader_rate'],
        'capture_noninferiority': model['capture_mean'] is not None and
            baseline['capture_mean'] is not None and model['capture_mean'] >= baseline['capture_mean'],
        'mission_coverage': bool(known.all()) and len(known) > 0,
        'population_complete': bool(population_complete),
    }
    numeric = ('mean_uplift', 'paired_lower_ci_positive', 'leader_noninferiority', 'capture_noninferiority')
    if checks['sample'] and any(not checks[k] for k in numeric):
        verdict = 'REJECTED'
    elif all(checks.values()):
        verdict = 'ADVANCE_TO_SEPARATE_PORTFOLIO_REPLAY'
    else:
        verdict = 'INCONCLUSIVE'
    return {
        'verdict': verdict, 'runtime_eligible': False, 'achievement_claimed': False,
        'groups': len(select), 'test_days': len(set(days)), 'replacements': int(swap.sum()),
        'candidate_better_count': int((returns[:, 0] > returns[:, 1]).sum()),
        'decision_correct_count': int((realized >= returns.max(axis=1)).sum()),
        'keep_mean_return_pct': float(keep.mean()) if len(keep) else None,
        'always_replace_mean_return_pct': float(returns[:, 0].mean()) if len(returns) else None,
        'model_mean_return_pct': float(realized.mean()) if len(realized) else None,
        'paired_mean_uplift_pp': float(delta.mean()) if len(delta) else None,
        'paired_median_uplift_pp': float(np.median(delta)) if len(delta) else None,
        'paired_3day_95ci_pp': ci, 'keep_mission': baseline, 'model_mission': model,
        'checks': checks,
        'scope': 'retrospective selection proxy, not executable portfolio replacement',
    }
