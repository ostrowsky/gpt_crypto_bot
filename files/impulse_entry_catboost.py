"""Causal fixed-horizon impulse-entry edge, independent of active bot policy."""
from datetime import datetime, time, timezone
import numpy as np
from capacity_catboost import BAR, HORIZON, TZ, causal_features, forward_return

DAY = 86400000
FEATURE_NAMES = ('return1', 'return4', 'return16', 'vol16', 'volume_ratio16',
                 'range16', 'range_position16', 'tf_1h', 'utc_sin', 'utc_cos')
PARAMETERS = dict(iterations=400, depth=4, learning_rate=.03, l2_leaf_reg=10,
                  loss_function='RMSE', random_seed=42, thread_count=2,
                  allow_writing_files=False, verbose=False)


def features(data, clock, tf):
    if tf not in ('15m', '1h'):
        raise ValueError('unsupported timeframe')
    x = causal_features(data, clock, 0)
    if x is None:
        return None
    angle = 2*np.pi*(clock % DAY)/DAY
    return np.r_[x[:-1], float(tf == '1h'), np.sin(angle), np.cos(angle)]


def net_target(data, clock, fee_bps, slip_bps):
    if not 0 <= fee_bps < 10000 or not 0 <= slip_bps < 10000:
        raise ValueError('invalid costs')
    gross = forward_return(data, clock)
    if gross is None:
        return None
    f, s = fee_bps/10000, slip_bps/10000
    return 100*((1+gross/100)*(1-f)*(1-s)/((1+f)*(1+s))-1)


def fit_cohorts(clocks, valid, fit_at, start):
    """Inner whole-local-day validation, purged labels, strict matured history."""
    day0 = datetime.fromtimestamp(start/1000, timezone.utc).astimezone(TZ).date()
    day1 = datetime.fromtimestamp(fit_at/1000, timezone.utc).astimezone(TZ).date()
    from datetime import timedelta
    cut = int(datetime.combine(day0+timedelta(days=int((day1-day0).days*.8)),
                               time(), tzinfo=TZ).timestamp()*1000)
    available = clocks+HORIZON
    train = valid & (clocks < cut) & (available < cut)
    validation = valid & (clocks >= cut) & (available < fit_at)
    return train, validation, cut


def admit(prediction):
    return bool(np.isfinite(prediction) and prediction > 0)


def screen(snapshot, decisions):
    """Only explicitly scored impulse candidates change; initial untrained passthrough."""
    raw, times, _ = snapshot
    out = {}
    for at, candidates in raw.items():
        kept = [c for j, c in enumerate(candidates)
                if c.mode != 'impulse_speed' or decisions.get((at, j), True)]
        if kept:
            out[at] = kept
    return out, set(times), sum(map(len, out.values()))


def gate(control, target, full_missions, test_missions, ci, days):
    from turnover_economics import acceptance
    verdict = acceptance(control, target, full_missions, *test_missions, ci, days)
    verdict['checks'].pop('turnover_reduction')
    verdict['numerical_gate'] = ('PASS' if all(verdict['checks'].values()) else
                                'REJECTED' if days >= 30 else 'INCONCLUSIVE')
    return verdict
