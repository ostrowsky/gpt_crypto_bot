"""Bounded trainer search. Prediction diagnostics are never release authority."""
import numpy as np

CONTRACT = 'prediction-error-search-v1'
# Reference first makes ties deterministic and avoids needless parameter churn.
GRID = (
    {'iterations':250, 'depth':6, 'learning_rate':0.05, 'l2_leaf_reg':6.0},
    {'iterations':150, 'depth':4, 'learning_rate':0.03, 'l2_leaf_reg':10.0},
    {'iterations':200, 'depth':5, 'learning_rate':0.05, 'l2_leaf_reg':10.0},
)


def fingerprint():
    """Changed trainer/search code must not reuse an old fit receipt."""
    import hashlib
    from pathlib import Path
    root=Path(__file__).resolve().parent
    return hashlib.sha256(b''.join((root/name).read_bytes() for name in
        ('prediction_error_optimizer.py','ml_candidate_ranker.py'))).hexdigest()


def limit_cpu():
    """One logical CPU, low priority; failure prevents a heavy Windows fit."""
    import ctypes
    kernel=ctypes.windll.kernel32
    kernel.GetCurrentProcess.restype=ctypes.c_void_p
    kernel.SetPriorityClass.argtypes=[ctypes.c_void_p,ctypes.c_uint32]
    kernel.SetProcessAffinityMask.argtypes=[ctypes.c_void_p,ctypes.c_size_t]
    process=kernel.GetCurrentProcess()
    if not kernel.SetPriorityClass(process,0x4000) or not kernel.SetProcessAffinityMask(process,2):
        raise RuntimeError('single CPU/BelowNormal restriction unavailable')


def errors(y, p):
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    if y.ndim != 1 or p.shape != y.shape or not y.size:
        raise ValueError('missing or inconsistent prediction denominator')
    if not np.all(np.isfinite(y)) or not np.all(np.isfinite(p)):
        raise ValueError('nonfinite prediction evidence')
    if not np.all((y == 0) | (y == 1)) or not np.all((p >= 0) & (p <= 1)):
        raise ValueError('invalid binary target/probability')
    return (p-y)**2


def search(factory, X_train, y_train, X_validation, y_validation):
    """No test argument: model selection cannot consult the reserved test data."""
    trials, models = [], []
    errors(y_train, np.full(len(y_train), .5))
    for params in GRID:
        model = factory(dict(params))
        # Empty eval set prevents early stopping on the selection partition.
        model.fit(X_train, y_train, X_validation[:0], y_validation[:0])
        loss = float(errors(y_validation, model.predict_proba(X_validation)).mean())
        trials.append({'parameters':dict(params), 'validation_brier':loss,
                       'validation_rows':len(y_validation)})
        models.append(model)
    selected = min(range(len(trials)), key=lambda i: trials[i]['validation_brier'])
    report = {'contract':CONTRACT, 'objective':'raw_quality_probability_brier',
              'trials':trials, 'selected_index':selected,
              'selected_parameters':dict(GRID[selected]), 'reference_parameters':dict(GRID[0]),
              'runtime_eligible':False, 'scope':'prediction_only_not_portfolio_or_release'}
    return models[selected], models[0], report


def holdout(y, selected, reference, days, train_positive_rate):
    """Separate fixed-candidate test; never returns model selection instructions."""
    a, b = errors(y, selected), errors(y, reference)
    if len(days) != len(a) or any(not day for day in days):
        raise ValueError('missing test day identity')
    baseline = errors(y, np.full(len(a), train_positive_rate))
    days = np.asarray(days)
    delta = np.array([float((b-a)[days == day].mean()) for day in sorted(set(days))])
    interval = None
    if len(delta) >= 10:
        rng = np.random.default_rng(42)
        samples = [float(rng.choice(delta, size=len(delta), replace=True).mean())
                   for _ in range(2000)]
        interval = np.quantile(samples, [.025,.975]).tolist()
    state = ('UNKNOWN' if interval is None else
             'SUPPORTED' if interval[0] > 0 and a.mean() < baseline.mean() else 'NOT_PROVEN')
    y, selected = np.asarray(y), np.asarray(selected)
    return {'state':state, 'rows':len(a), 'days':len(delta), 'positive_labels':int(y.sum()),
            'selected_brier':float(a.mean()), 'reference_brier':float(b.mean()),
            'constant_train_prevalence_brier':float(baseline.mean()),
            'paired_daily_brier_reduction':float(delta.mean()), 'daily_95ci':interval,
            'false_positive_at_05':int(((selected >= .5) & (y == 0)).sum()),
            'false_negative_at_05':int(((selected < .5) & (y == 1)).sum()),
            'runtime_eligible':False, 'independent_forward_confirmation':'REQUIRED'}
