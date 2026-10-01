"""Maximum closed-archive diagnostic ablation; never approves production."""
import argparse
import asyncio
from dataclasses import asdict, replace
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np

import config
import replay_backtest as rb
from closed_grid_policy_replay import DAY, SOURCES, event_clock
from historical_signal_evaluation import freeze, sha
from portfolio_alpha import closed_price_series, evaluate_portfolio_alpha
from research_rocket_capture import policy

ARMS = ('baseline', 'extra_penalty_off', 'negative_terms_off')


def adjusted(candidate, arm):
    if arm not in ARMS:
        raise ValueError('unknown arm')
    delta = 0.0
    change = candidate.intraday_change_pct
    if arm != 'baseline' and candidate.tf == '15m' and candidate.mode in ('breakout', 'retest'):
        if change < -0.25:
            delta += 8.0
        if arm == 'negative_terms_off' and change < 0:
            delta -= max(-8.0, min(45.0, change * 6.0))
    return replace(candidate, top_gainer_score=round(candidate.top_gainer_score + delta, 4))


def windows(start, end):
    days = (end - start) // DAY
    if days < 5 or (end - start) % DAY:
        raise ValueError('whole-day chronological window required')
    a, b = start + int(days * .6) * DAY, start + int(days * .8) * DAY
    return [('full', start, end), ('discovery', start, a),
            ('validation', a, b), ('historical_test', b, end)]


def bounded_cache(cache, end):
    """Retain original indices/warmup, exclude every candle closing after end."""
    out = {}
    for key, (data, feat) in cache.items():
        n = int(np.searchsorted(data['t'], end - rb.BAR_MS[key[1]], side='right'))
        out[key] = (data[:n], {k: v[:n] for k, v in feat.items()})
    return out


def finalize_at_boundary(trades, cache, end):
    # Engine assumes an unclosed trailing candle and uses len-2 at forced close.
    # Our cache contains closed candles only; mark remaining positions at boundary.
    for trade in trades:
        if trade.exit_reason == 'open_at_end':
            data, _ = cache[trade.sym, trade.tf]
            trade.exit_price = float(data['c'][-1])
            trade.exit_ts = end
            rb._finalize_trade_metrics(trade)
        if trade.exit_ts > end:
            raise ValueError('future exit leaked across historical partition')


def trade_summary(trades, baseline):
    identity = lambda t: (t.sym, t.tf, t.mode, t.entry_ts)
    known = {identity(t) for t in baseline}
    added = [t for t in trades if identity(t) not in known]
    cost = 2 * (float(config.PAPER_FEE_BPS) + 5) / 100
    return {'trades': len(trades), 'added_entries': len(added),
            'net_losing_trades': sum(t.pnl_pct - cost < 0 for t in trades),
            'added_net_losing_trades': sum(t.pnl_pct - cost < 0 for t in added),
            'added_net_losing_fraction': (sum(t.pnl_pct - cost < 0 for t in added) / len(added)
                                          if added else None),
            'trx_trades': sum(t.sym == 'TRXUSDT' for t in trades),
            'trx_added_entries': sum(t.sym == 'TRXUSDT' for t in added)}


async def run(archive, output):
    manifest_raw = (archive / 'manifest.json').read_bytes()
    manifest = json.loads(manifest_raw)
    sources = {name: sha(Path(__file__).with_name(name))
               for name in (*SOURCES, 'audit_negative_day_rebound.py')}

    def verify():
        if (archive / 'manifest.json').read_bytes() != manifest_raw:
            raise ValueError('manifest drift')
        for name, expected in manifest['input_hashes'].items():
            if sha(archive / 'market' / name) != expected:
                raise ValueError('market input drift: ' + name)
        for name, expected in sources.items():
            if sha(Path(__file__).with_name(name)) != expected:
                raise ValueError('source drift: ' + name)

    verify()
    output.mkdir(parents=True, exist_ok=False)
    freeze(output / 'registration.json', json.dumps({
        'manifest_sha256': sha(archive / 'manifest.json'), 'source_hashes': sources,
        'arms': ARMS, 'windows': windows(manifest['start_ms'], manifest['end_ms']),
        'eligible_symbols': manifest['eligible_symbols'],
        'requested_symbols': manifest['requested_symbols'],
        'missing_series': manifest['missing_series']}, sort_keys=True).encode())
    cache = {}
    symbols = manifest['eligible_symbols']
    index = rb._build_market_cache_index(archive / 'market')
    for sym in symbols:
        for tf in ('15m', '1h'):
            data = rb._load_cached_klines(archive / 'market', sym, tf,
                        manifest['archive_start_ms'], manifest['end_ms'], cache_index=index)
            if data is None:
                raise ValueError('complete input missing')
            cache[sym, tf] = (data, rb.compute_features(data['o'], data['h'], data['l'], data['c'], data['v']))
        data = rb._aggregate_1h_to_4h(cache[sym, '1h'][0])
        cache[sym, '4h'] = (data, rb.compute_features(data['o'], data['h'], data['l'], data['c'], data['v']))
    c15, c4 = ({s: cache[s, tf] for s in symbols} for tf in ('15m', '4h'))
    ctx = rb._build_bull_day_context(cache['BTCUSDT', '1h'][0])
    print('features ready; building shared causal candidate population', flush=True)
    results = []
    with policy('baseline'):
        raw, times, _ = await rb.build_replay_candidate_snapshot(
            symbols, ['15m', '1h'], cache, c15, c4, ctx, variant='score_replace_cluster')
        print('candidate population ready', flush=True)
        for label, start, end in windows(manifest['start_ms'], manifest['end_ms']):
            baseline = []
            local = bounded_cache(cache, end)
            local15 = {s: local[s, '15m'] for s in symbols}
            local4 = {s: local[s, '4h'] for s in symbols}
            for arm in ARMS:
                snap = event_clock({t: [adjusted(c, arm) for c in cs] for t, cs in raw.items()},
                                   times, start, end)
                with patch.object(rb, '_load_temporal_scout_events', return_value=({}, {})):
                    trades, stats = await rb.simulate_portfolio(
                        symbols, ['15m', '1h'], local, local15, local4, ctx, max_open_positions=10,
                        enable_replacement=bool(getattr(config, 'PORTFOLIO_REPLACE_ENABLED', True)),
                        replace_min_delta=float(getattr(config, 'PORTFOLIO_REPLACE_MIN_DELTA', 8)),
                        variant='score_replace_cluster',
                        top_gainer_score_min=float(config.TOP_GAINER_SCORE_GATE_MIN_SCORE),
                        candidate_snapshot=snap)
                finalize_at_boundary(trades, local, end)
                if arm == 'baseline':
                    baseline = trades
                series = {s: closed_price_series(cache[s, '15m'][0], bar_ms=900000,
                                   start_ms=start, end_ms=end) for s in symbols}
                alpha = evaluate_portfolio_alpha(trades, price_series_by_symbol=series,
                    benchmark_series=series['BTCUSDT'], window_start_ms=start, window_end_ms=end,
                    requested_days=(end-start)//DAY, universe=symbols, variant=arm, capacity=10,
                    fee_bps=float(config.PAPER_FEE_BPS), slippage_bps=5, source_hashes=sources)
                alpha['decision_grade'] = False
                item = {'window': label, 'arm': arm, 'start_ms': start, 'end_ms': end,
                        'summary': trade_summary(trades, baseline), 'portfolio': alpha,
                        'candidates': snap[2], 'stats': asdict(stats)}
                freeze(output / f'{label}_{arm}.json', json.dumps(item, allow_nan=False).encode())
                results.append(item)
                print(json.dumps({'window': label, 'arm': arm, 'summary': item['summary'],
                                  'portfolio': alpha['portfolio']}), flush=True)
    verify()
    report = {'verdict': 'UNKNOWN', 'runtime_eligible': False, 'results': results,
              'limitations': ['reconstructed current-rule population, not certified live parity',
                'historical partitions are not sealed independent holdouts',
                'learned scores disabled offline; incomplete symbols excluded',
                'raw exchange provenance and point-in-time universe not certified',
                'mutable temporal scout logs excluded; open positions marked at partition end',
                'full truth harness FAIL TH-11 at registration',
                'net-losing trades are not top-mover false-positive labels']}
    freeze(output / 'result.json', json.dumps(report, allow_nan=False).encode())
    freeze(output / 'receipt.json', json.dumps({p.name: sha(p) for p in output.glob('*.json')}).encode())


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    asyncio.run(run(args.archive, args.output))
