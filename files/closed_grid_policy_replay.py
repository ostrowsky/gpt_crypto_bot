"""Immutable maximum recovered-archive current-rule baseline, not model approval."""
import argparse
import asyncio
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import re

import config
import replay_backtest as rb
from historical_signal_evaluation import freeze, sha
from portfolio_alpha import closed_price_series, evaluate_portfolio_alpha
from recover_rocket_history import valid
from research_rocket_capture import policy

DAY = 86_400_000
SOURCES = ('closed_grid_policy_replay.py', 'portfolio_alpha.py', 'replay_backtest.py',
           'strategy.py', 'indicators.py', 'monitor.py', 'config.py',
           'policy_provenance.py', 'research_rocket_capture.py', 'recover_rocket_history.py')


def validate_series(rows, step, start, end):
    if (not isinstance(rows, list) or end <= start or start % step or end % step
            or len(rows) != (end-start)//step):
        raise ValueError('unknown/missing closed-candle grid')
    for i, row in enumerate(rows):
        if not valid(row, step) or int(row['t']) != start+i*step:
            raise ValueError('invalid, unordered or duplicate closed candle')


def snapshot_archive(archive, output):
    summary_raw = (archive/'recovery_summary.json').read_bytes()
    summary = json.loads(summary_raw)
    start, end = int(summary['start']), int(summary['end'])
    if start >= end or end-start <= 10*DAY:
        raise ValueError('archive has insufficient warmup/window')
    output.mkdir(parents=True, exist_ok=False)
    inputs = output/'market'
    inputs.mkdir()
    freeze(output/'recovery_summary.json', summary_raw)
    bindings, complete, missing, seen = {}, set(), [], set()
    for item in summary['series']:
        sym, tf = item['symbol'], item['tf']
        if not re.fullmatch('[A-Z0-9]+', sym) or tf not in ('15m', '1h') or (sym, tf) in seen:
            raise ValueError('unknown/duplicate archive identity')
        seen.add((sym, tf))
        if item.get('start') != start or item.get('end') != end:
            raise ValueError('archive window identity changed')
        if item['status'] != 'complete':
            missing.append({'symbol': sym, 'tf': tf, 'missing_count': item.get('missing_count')})
            continue
        name = f'{sym}_{tf}_{start}_{end}.json'
        raw = (archive/name).read_bytes()
        meta_raw = (archive/Path(name).with_suffix('.manifest.json')).read_bytes()
        meta = json.loads(meta_raw)
        if (meta.get('status') != 'complete' or meta.get('sha256') != hashlib.sha256(raw).hexdigest()
                or any(meta.get(k) != item.get(k) for k in ('symbol', 'tf', 'start', 'end', 'sha256'))):
            raise ValueError('archive receipt identity/hash mismatch')
        validate_series(json.loads(raw), rb.BAR_MS[tf], start, end)
        freeze(inputs/name, raw)
        freeze(inputs/Path(name).with_suffix('.manifest.json'), meta_raw)
        bindings[name] = meta['sha256']
        bindings[Path(name).with_suffix('.manifest.json').name] = hashlib.sha256(meta_raw).hexdigest()
        complete.add((sym, tf))
    symbols = sorted({s for s, _ in seen})
    eligible = [s for s in symbols if (s, '15m') in complete and (s, '1h') in complete]
    if 'BTCUSDT' not in eligible:
        raise ValueError('complete BTC benchmark/context required')
    manifest = {'archive_start_ms': start, 'archive_end_ms': end, 'warmup_days': 10,
                'start_ms': start+10*DAY, 'end_ms': end,
                'requested_symbols': symbols, 'eligible_symbols': eligible,
                'missing_series': missing, 'input_hashes': bindings,
                'summary_sha256': hashlib.sha256(summary_raw).hexdigest(),
                'source_hashes': {name: sha(Path(__file__).with_name(name)) for name in SOURCES}}
    freeze(output/'manifest.json', json.dumps(manifest, sort_keys=True).encode())
    return manifest


def event_clock(raw, times, start, end):
    return ({t: list(v) for t, v in raw.items() if start <= t < end},
            {t for t in times if start <= t <= end} | {start, end},
            sum(len(v) for t, v in raw.items() if start <= t < end))


async def run(output, manifest):
    start, end = manifest['start_ms'], manifest['end_ms']
    symbols = manifest['eligible_symbols']
    cache = {}
    index = rb._build_market_cache_index(output/'market')
    for sym in symbols:
        for tf in ('15m', '1h'):
            data = rb._load_cached_klines(output/'market', sym, tf,
                                        manifest['archive_start_ms'], end, cache_index=index)
            if data is None:
                raise ValueError('frozen complete input disappeared')
            cache[sym, tf] = (data, rb.compute_features(data['o'], data['h'], data['l'], data['c'], data['v']))
        four = rb._aggregate_1h_to_4h(cache[sym, '1h'][0])
        cache[sym, '4h'] = (four, rb.compute_features(four['o'], four['h'], four['l'], four['c'], four['v']))
    c15 = {s: cache[s, '15m'] for s in symbols}
    c4 = {s: cache[s, '4h'] for s in symbols}
    ctx = rb._build_bull_day_context(cache['BTCUSDT', '1h'][0])
    print('closed archives/features ready; building current-rule candidates', flush=True)
    # Never import mutable learned model files as historical champion evidence.
    with policy('baseline'):
        raw, times, _ = await rb.build_replay_candidate_snapshot(
            symbols, ['15m', '1h'], cache, c15, c4, ctx, variant='score_replace_cluster')
        snapshot = event_clock(raw, times, start, end)
        print(f'candidate snapshot ready: {snapshot[2]}', flush=True)
        trades, stats = await rb.simulate_portfolio(
            symbols, ['15m', '1h'], cache, c15, c4, ctx, max_open_positions=10,
            enable_replacement=bool(getattr(config, 'PORTFOLIO_REPLACE_ENABLED', True)),
            replace_min_delta=float(getattr(config, 'PORTFOLIO_REPLACE_MIN_DELTA', 8)),
            variant='score_replace_cluster',
            top_gainer_score_min=float(config.TOP_GAINER_SCORE_GATE_MIN_SCORE),
            candidate_snapshot=snapshot)
    series = {s: closed_price_series(cache[s, '15m'][0], bar_ms=900000,
                                     start_ms=start, end_ms=end) for s in symbols}
    alpha = evaluate_portfolio_alpha(
        trades, price_series_by_symbol=series, benchmark_series=series['BTCUSDT'],
        window_start_ms=start, window_end_ms=end, requested_days=(end-start)//DAY,
        universe=symbols, variant='current_rule_only_baseline', capacity=10,
        fee_bps=float(config.PAPER_FEE_BPS), slippage_bps=5,
        source_hashes=manifest['source_hashes'])
    # Cash-account validity does not certify the historical population/policy.
    alpha['decision_grade'] = False
    alpha['evidence_grade'] = 'diagnostic_current_rule_partial_population'
    for name, expected in manifest['source_hashes'].items():
        if sha(Path(__file__).with_name(name)) != expected:
            raise ValueError('source drift during replay')
    for name, expected in manifest['input_hashes'].items():
        if sha(output/'market'/name) != expected:
            raise ValueError('input drift during replay')
    report = {'contract': 'maximum-archive-closed-grid-rule-baseline-v1',
              'state': 'BLOCKED', 'runtime_eligible': False, 'achievement_claimed': False,
              'manifest_sha256': sha(output/'manifest.json'), 'portfolio': alpha,
              'population': {'complete_symbols': len(symbols),
                             'requested_symbols': len(manifest['requested_symbols']),
                             'missing_series': manifest['missing_series']},
              'candidates': snapshot[2], 'trades': [asdict(t) for t in trades],
              'stats': asdict(stats), 'limitations': [
                  'current-rule baseline, not historical champion or sealed challenger comparison',
                  'learned scores disabled only within offline process',
                  'incomplete symbols excluded; not failures or zero capture',
                  'local-interior candles are not independently raw-exchange certified',
                  'point-in-time universe and agent-only admission policy unavailable',
                  'closed-bar simulated fills, not recorded decision-time executions',
                  '4h context derived from 1h; primary candidate timeframes 15m and 1h']}
    freeze(output/'result.json', json.dumps(report, sort_keys=True, allow_nan=False).encode())
    freeze(output/'receipt.json', json.dumps({
        'result.json': sha(output/'result.json'), 'manifest.json': sha(output/'manifest.json')}).encode())
    print(json.dumps({'state': report['state'], 'trades': len(trades),
                      'portfolio': alpha['portfolio'], 'btc': alpha['benchmark'],
                      'coverage': alpha['coverage']}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    frozen = snapshot_archive(args.archive, args.output)
    print(f"archive frozen: {len(frozen['eligible_symbols'])}/{len(frozen['requested_symbols'])} symbols", flush=True)
    asyncio.run(run(args.output, frozen))
