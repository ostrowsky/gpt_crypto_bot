"""Registered direct five-minute price-path regressors on frozen causal L2 inputs."""
from __future__ import annotations

import argparse
import copy
import json
import random
import shutil
import time
from pathlib import Path

import numpy as np

import evaluate_minute_direction as e
import minute_direction_models as m
from minute_direction_data import sha

HORIZONS = (1, 2, 3, 4, 5)
METHODS = ('Ridge', 'CatBoost', 'DeepLOB')


def targets(mid, segment):
    out = np.full((len(mid), 5), np.nan)
    changes = np.r_[0, np.cumsum(segment[1:] != segment[:-1])]
    for j, h in enumerate(HORIZONS):
        row = np.arange(max(0, len(mid) - h * 6)); future = row + h * 6
        good = (segment[row] >= 0) & (changes[row] == changes[future])
        good &= np.isfinite(mid[row]) & np.isfinite(mid[future]) & (mid[row] > 0) & (mid[future] > 0)
        out[row[good], j] = np.log(mid[future[good]] / mid[row[good]])
    return out


def train_scale(y):
    if y.ndim != 2 or y.shape[1] != 5 or not np.isfinite(y).all():
        raise ValueError('Invalid TRAIN targets')
    return y.mean(axis=0), np.maximum(y.std(axis=0), 1e-8)


def residual_quantiles(actual, predicted):
    residual = np.abs(actual - predicted)
    if not len(residual) or residual.shape[1] != 5 or not np.isfinite(residual).all():
        raise ValueError('Invalid independent CALIBRATION residuals')
    rank = min(len(residual), int(np.ceil((len(residual) + 1) * .90))) - 1
    return np.partition(residual, rank, axis=0)[rank]


def score(actual, predicted, widths, origin_price=None):
    actual = np.asarray(actual); predicted = np.asarray(predicted)
    if actual.shape != predicted.shape or not len(actual) or not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError('Invalid common score cohort')
    output = []
    for j, h in enumerate(HORIZONS):
        r = actual[:, j]; p = predicted[:, j]; error = p - r
        nz = r != 0
        covered = np.abs(error) <= widths[j]
        classes = lambda v: np.where(v < -.0002, 0, np.where(v > .0002, 2, 1))
        row = dict(horizon_min=h, n=len(r), mae_bp=float(np.mean(np.abs(error)) * 10000),
            rmse_bp=float(np.sqrt(np.mean(error ** 2)) * 10000),
            zero_return_mae_bp=float(np.mean(np.abs(r)) * 10000),
            zero_return_rmse_bp=float(np.sqrt(np.mean(r ** 2)) * 10000),
            direction_n=int(nz.sum()), direction_correct=int(((p > 0) == (r > 0))[nz].sum()),
            observed_majority_correct=int(max((r[nz] > 0).sum(), (r[nz] < 0).sum())),
            three_class_correct=int((classes(p) == classes(r)).sum()),
            interval_n=len(r), interval_covered=int(covered.sum()),
            interval_width_bp=float(2 * widths[j] * 10000))
        if origin_price is not None:
            delta = origin_price * (np.exp(p) - np.exp(r))
            row.update(mae_USDT=float(np.mean(np.abs(delta))), rmse_USDT=float(np.sqrt(np.mean(delta ** 2))))
        output.append(row)
    return output


def build_deeplob_regression():
    from torch import nn
    class Regression(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = m.build_deeplob()
            self.encoder.heads = nn.ModuleList([nn.Linear(32, 1) for _ in HORIZONS])
        def forward(self, x):
            return self.encoder(x).squeeze(-1)
    return Regression()


def deep_predictions(model, frames, rows, mean, scale):
    import torch
    model.eval(); output = []
    with torch.no_grad():
        for start in range(0, len(rows), 128):
            block = rows[start:start + 128]
            x = np.stack([m.normalize_sequence(frames[s]['book'][i - m.SEQUENCE + 1:i + 1]) for s, i in block])
            output.append(model(torch.from_numpy(x)).numpy())
    return np.concatenate(output) * scale + mean


def fit_deep(frames, cohorts, mean, scale, folder):
    import torch
    from torch.utils.data import DataLoader, Dataset
    from safetensors.torch import save_file
    torch.set_num_threads(2); torch.set_num_interop_threads(1)
    torch.manual_seed(42); np.random.seed(42); random.seed(42)
    torch.use_deterministic_algorithms(True)
    class Samples(Dataset):
        def __len__(self): return len(cohorts['train'])
        def __getitem__(self, k):
            s, i = cohorts['train'][k]
            x = m.normalize_sequence(frames[s]['book'][i - m.SEQUENCE + 1:i + 1])
            y = ((frames[s]['price_returns'][i] - mean) / scale).astype(np.float32)
            return torch.from_numpy(x), torch.from_numpy(y)
    loader = DataLoader(Samples(), batch_size=128, shuffle=True, num_workers=0,
        generator=torch.Generator().manual_seed(42))
    model = build_deeplob_regression()
    optimizer = torch.optim.Adam(model.parameters(), lr=.001, weight_decay=.0001)
    actual = e.matrix(frames, cohorts['validation'], 'price_returns')
    history = []; best = float('inf'); state = None; best_epoch = None; patience = 0
    for epoch in range(1, 9):
        model.train(); total = 0.; count = 0; started = time.monotonic()
        for x, y in loader:
            optimizer.zero_grad(); loss = torch.nn.functional.mse_loss(model(x), y)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1); optimizer.step()
            total += float(loss.detach()) * len(x); count += len(x)
        predicted = deep_predictions(model, frames, cohorts['validation'], mean, scale)
        val = float(np.mean(((actual - predicted) / scale) ** 2))
        if not np.isfinite(val): raise ValueError('Deep regression diverged')
        history.append(dict(epoch=epoch, train_mse=total / count, validation_mse=val,
            seconds=time.monotonic() - started))
        print('DeepLOB regression epoch', history[-1], flush=True)
        if val < best:
            best = val; state = copy.deepcopy(model.state_dict()); best_epoch = epoch; patience = 0
        else: patience += 1
        if patience >= 2: break
    model.load_state_dict(state)
    save_file({k: v.contiguous() for k, v in state.items()}, str(folder / 'deeplob_regression.safetensors'))
    return model, dict(best_epoch=best_epoch, history=history)


def run(books, classifier, output):
    old_result_path = classifier / 'result.json'; old_pred_path = classifier / 'test_predictions.npz'
    old_hashes = dict(result=sha(old_result_path), predictions=sha(old_pred_path))
    old = json.loads(old_result_path.read_text(encoding='utf-8'))
    receipt = json.loads((classifier / 'verification.json').read_text(encoding='utf-8'))
    if receipt.get('status') != 'PASS' or receipt['result_sha256'] != old_hashes['result'] or receipt['predictions_sha256'] != old_hashes['predictions']:
        raise ValueError('Classification evidence changed')
    for name, checksum in old['metadata']['registration']['source_hashes'].items():
        if sha(Path(__file__).with_name(name)) != checksum:
            raise ValueError('Causal feature source changed: ' + name)
    output.mkdir(parents=True, exist_ok=False)
    sources = dict(old['metadata']['registration']['source_hashes'])
    sources[Path(__file__).name] = sha(__file__)
    snapshot = output / 'source_snapshot'; snapshot.mkdir()
    for name in sources: shutil.copy2(Path(__file__).with_name(name), snapshot / name)
    frames, cuts, coverage = e.load_frames(books)
    if cuts != old['cuts']: raise ValueError('Clock boundaries changed')
    for a in frames.values():
        a['price_returns'] = targets(a['mid'], a['segment'])
        eligible = np.isfinite(a['price_returns']).all(axis=1)
        for split in ('train', 'validation', 'calibration', 'test'): a['masks'][split] &= eligible
    cohorts = {k: e.cohort(frames, k) for k in ('train', 'validation', 'calibration', 'test', 'inference')}
    registration = dict(registered_at=time.time(), source_hashes=sources, coverage_sha256=sha(books / 'coverage.json'),
        classifier_hashes=old_hashes, cuts=cuts, counts={k: len(v) for k, v in cohorts.items()},
        target_minutes=list(HORIZONS), disclosed_test=True, seed=42,
        parameters=dict(catboost=dict(loss='MultiRMSE', iterations=600, depth=6, lr=.05, l2=10, patience=60),
            ridge=dict(alpha=10), deeplob=dict(channels=16, lstm=32, sequence=100, lr=.001,
                weight_decay=.0001, batch=128, epochs=8, patience=2, loss='MSE'),
            intervals=dict(nominal=.90, source='separate calibration absolute residuals per horizon')))
    (output / 'registration.json').write_text(json.dumps(registration, indent=2), encoding='utf-8')
    print('REGISTERED price paths', registration, flush=True)
    x = {k: e.matrix(frames, r, 'x') for k, r in cohorts.items()}
    y = {k: e.matrix(frames, r, 'price_returns') for k, r in cohorts.items()}
    mean, scale = train_scale(y['train'])
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import Ridge
    from catboost import CatBoostRegressor
    from joblib import dump, load
    scaler = StandardScaler().fit(x['train'])
    ridge = Ridge(alpha=10).fit(scaler.transform(x['train']), (y['train'] - mean) / scale)
    dump(dict(model=ridge, scaler=scaler), output / 'ridge.joblib')
    cat = CatBoostRegressor(loss_function='MultiRMSE', iterations=600, depth=6, learning_rate=.05,
        l2_leaf_reg=10, random_seed=42, thread_count=2, allow_writing_files=False, verbose=False)
    cat.fit(x['train'], (y['train'] - mean) / scale,
        eval_set=(x['validation'], (y['validation'] - mean) / scale), early_stopping_rounds=60, use_best_model=True)
    cat.save_model(str(output / 'catboost_regression.cbm'))
    print('CatBoost regression frozen', cat.tree_count_, 'trees', flush=True)
    point = {}; cal = {}
    for split, sink in [('inference', point), ('calibration', cal)]:
        sink['Ridge'] = ridge.predict(scaler.transform(x[split])) * scale + mean
        sink['CatBoost'] = cat.predict(x[split]) * scale + mean
    np.savez_compressed(output / 'catboost_phase.npz', **point)
    deep, dlmeta = fit_deep(frames, cohorts, mean, scale, output)
    point['DeepLOB'] = deep_predictions(deep, frames, cohorts['inference'], mean, scale)
    cal['DeepLOB'] = deep_predictions(deep, frames, cohorts['calibration'], mean, scale)
    widths = {name: residual_quantiles(y['calibration'], cal[name]) for name in METHODS}
    rows = cohorts['inference']; test = set(cohorts['test'])
    scored = np.array([r in test for r in rows]); symbols = np.array([s for s, i in rows])
    clocks = np.array([frames[s]['time'][i] for s, i in rows]); p0 = e.matrix(frames, rows, 'mid')
    actual = y['inference']
    for p in point.values():
        if p.shape != actual.shape or not np.isfinite(p).all(): raise ValueError('Invalid frozen price forecast')
    metrics = {name: dict(pooled=score(actual[scored], p[scored], widths[name]),
        assets={s: score(actual[scored & (symbols == s)], p[scored & (symbols == s)], widths[name],
            p0[scored & (symbols == s)]) for s in frames}) for name, p in point.items()}
    np.savez_compressed(output / 'predictions.npz', time=clocks, symbol=symbols, origin_price=p0,
        actual_returns=actual, scored=scored, **point)
    # Native reload on preselected evenly spaced clocks, without outcome selection.
    indices = np.unique(np.linspace(0, len(rows) - 1, 24, dtype=int))
    native_cat = CatBoostRegressor(); native_cat.load_model(str(output / 'catboost_regression.cbm'))
    np.testing.assert_allclose(native_cat.predict(x['inference'][indices]) * scale + mean,
        point['CatBoost'][indices], rtol=1e-7, atol=1e-10)
    native_ridge = load(output / 'ridge.joblib')
    np.testing.assert_allclose(native_ridge['model'].predict(native_ridge['scaler'].transform(x['inference'][indices])) * scale + mean,
        point['Ridge'][indices], rtol=1e-7, atol=1e-10)
    from safetensors.torch import load_file
    native_deep = build_deeplob_regression(); native_deep.load_state_dict(load_file(str(output / 'deeplob_regression.safetensors')))
    np.testing.assert_allclose(deep_predictions(native_deep, frames, [rows[i] for i in indices], mean, scale),
        point['DeepLOB'][indices], rtol=1e-5, atol=1e-8)
    for name, checksum in sources.items():
        if sha(Path(__file__).with_name(name)) != checksum: raise ValueError('Source changed during run')
    for s, meta in coverage['assets'].items():
        if sha(books / (s + '.npz')) != meta['sha256']: raise ValueError('Book changed during run')
    if sha(old_result_path) != old_hashes['result'] or sha(old_pred_path) != old_hashes['predictions']:
        raise ValueError('Original classification evidence modified')
    result = dict(status='COMPLETED_RETROSPECTIVE_PRICE_PATHS', runtime_eligible=False, cuts=cuts,
        coverage=coverage, registration=registration, counts=registration['counts'], metrics=metrics,
        train_target_mean=mean.tolist(), train_target_scale=scale.tolist(),
        calibration_widths={k: v.tolist() for k, v in widths.items()},
        models=dict(catboost_trees=cat.tree_count_, deeplob=dlmeta),
        limitations=['Previously disclosed historical TEST; not fresh holdout or forward proof',
            'Perpetual mid-price forecasts, not execution/trading profits',
            'Marginal temporal residual intervals, no IID or joint-path coverage guarantee',
            'Full Truth Harness FAIL TH-11 remains separate'])
    (output / 'result.json').write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    verification = dict(status='PASS', native_samples=len(indices), classifier_unchanged=True,
        result_sha256=sha(output / 'result.json'), predictions_sha256=sha(output / 'predictions.npz'))
    (output / 'verification.json').write_text(json.dumps(verification, indent=2), encoding='utf-8')
    for name in METHODS: print(name, metrics[name]['pooled'], flush=True)
    print('COMPLETE price paths', output, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--books', type=Path, required=True)
    parser.add_argument('--classifier', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); run(args.books, args.classifier, args.output)
