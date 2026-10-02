"""Prospective paired full-policy paper state, never retrospective forward fills."""
import asyncio
import copy
from dataclasses import asdict
import json
from pathlib import Path
import time
from unittest.mock import patch

import aiohttp
import numpy as np
import config
import replay_backtest as rb
import certified_rule_score_policy as policy
from forward_evidence_service import atomic
from paired_full_policy_replay import frozen_arm
from process_lock import process_lock
from recover_rocket_history import exchange_rows, valid
from validated_ranker_rollout import canonical, sha

STEP = 900000
DAY = 86400000
SOURCES = (*policy.SOURCES, 'prospective_policy_portfolios.py', 'paired_full_policy_replay.py')


async def register(deployment, session, now):
    root = Path(deployment['registry'])/'portfolios'
    root.mkdir(parents=True, exist_ok=True)
    path = root/'registration.json'
    if path.exists():
        return root, json.loads(path.read_bytes())
    project = Path(deployment['project_root'])
    candidate = Path(deployment['candidate_input']).read_bytes()
    # A source cohort already registered by the evaluator cannot be substituted
    # with the next hourly candidate.
    signal_registry = Path(deployment['registry'])
    frozen_candidate = signal_registry/'candidate.json'
    if not frozen_candidate.exists():
        raise ValueError('independent frozen candidate not ready')
    candidate = frozen_candidate.read_bytes()
    async with session.get('https://api.binance.com/api/v3/exchangeInfo',
                           timeout=aiohttp.ClientTimeout(total=30)) as response:
        response.raise_for_status()
        exchange_raw = await response.read()
    exchange = json.loads(exchange_raw)
    active = {v['symbol'] for v in exchange['symbols'] if v['status'] == 'TRADING'}
    with patch.object(config, 'WATCHLIST_FILE', project/'files/watchlist.json'):
        watchlist = config.load_watchlist()
    symbols = sorted(set(watchlist) & active)
    if not symbols or 'BTCUSDT' not in symbols:
        raise ValueError('active watchlist lacks benchmark')
    source = Path(__file__).resolve().parent
    files = {'champion.json': policy.champion_bytes(), 'candidate.json': candidate,
             'general.json': (source/'ml_signal_model.json').read_bytes(),
             'base_ranker.json': (source/'ml_candidate_ranker.json').read_bytes(),
             'exchange_info.json': exchange_raw}
    for name, raw in files.items():
        target = root/name
        if target.exists() and target.read_bytes() != raw:
            raise ValueError('partial registration changed; operator review required')
        if not target.exists():
            with target.open('xb') as handle:
                handle.write(raw)
    manifest = {'contract': 'prospective-paired-policy-v1', 'registered_ms': now,
                'observation_start_ms': ((now+2*DAY+STEP-1)//STEP)*STEP,
                'symbols': symbols, 'excluded_inactive': sorted(set(watchlist)-active),
                'watchlist': watchlist, 'source_hashes': {n: sha((source/n).read_bytes()) for n in SOURCES},
                'files': {n: sha(raw) for n, raw in files.items()},
                'isolation_mode': 'logical_same_user', 'runtime_eligible': False}
    atomic(path, manifest)
    return root, manifest


def verify_registration(root, manifest):
    source = Path(__file__).resolve().parent
    for name, digest in manifest['source_hashes'].items():
        if sha((source/name).read_bytes()) != digest:
            raise ValueError('frozen portfolio source changed: '+name)
    for name, digest in manifest['files'].items():
        if sha((root/name).read_bytes()) != digest:
            raise ValueError('frozen portfolio input changed: '+name)


def merge_closed(old, incoming, step, frame):
    if not incoming:
        raise ValueError('missing market input')
    if any(not valid(r, step) for r in (*old, *incoming)):
        raise ValueError('invalid closed OHLCV')
    rows = {int(r['t']): r for r in old}
    for row in incoming:
        t = int(row['t'])
        if t+step > frame:
            raise ValueError('unclosed market candle')
        if t in rows and rows[t] != row:
            raise ValueError('closed market history revised')
        rows[t] = row
    result = [rows[t] for t in sorted(rows)]
    if any(b['t']-a['t'] != step for a, b in zip(result, result[1:])):
        raise ValueError('market history gap')
    if result[-1]['t'] != (frame//step)*step-step:
        raise ValueError('latest closed market candle missing')
    return result


def validate_clock(state, frame, now):
    if frame % STEP or not frame <= now <= frame+120000:
        raise ValueError('late/future prospective frame; no backfill as forward evidence')
    previous = state.get('last_frame')
    if previous is not None and frame != previous+STEP:
        raise ValueError('duplicate/gapped prospective frame; new cohort requires operator review')


def verify_chain(frames):
    previous = None
    for frame in frames:
        if frame['previous'] != previous or frame['sha256'] != sha(canonical(frame['body'])):
            raise ValueError('portfolio frame journal corrupt')
        previous = frame['sha256']
    return previous


def export_unsigned(root, manifest, state):
    """Full marked portfolios; open positions are not fake SELLs or closed N."""
    frames = state.get('frames', [])
    if not frames:
        return
    verify_chain(frames)
    start, end = frames[0]['body']['frame'], frames[-1]['body']['frame']
    if [f['body']['frame'] for f in frames] != list(range(start, end+1, STEP)):
        raise ValueError('incomplete portfolio valuation clock')
    bundle = {'phase': 'sealed', 'start_ms': start, 'end_ms': end,
              'maximum_available_bounds': [start, end],
              'last_model_exposure_ms': manifest['registered_ms'],
              'universe': manifest['symbols'], 'fee_bps': max(7.5, float(config.PAPER_FEE_BPS)),
              'slippage_bps': 5,
              'prices': {s: [[f['body']['frame'], f['body']['prices'][s]] for f in frames]
                         for s in manifest['symbols']},
              'scope': 'unsigned_prospective_full_policy_paper_not_real_fills'}
    for arm, owned in state['arms'].items():
        bundle[arm+'_trades'] = owned['closed_trades']+[
            dict(t, position_open=True) for t in owned['state']['open_positions']]
    atomic(root/'sealed_unsigned.json', bundle)


async def advance(root, manifest, state, incoming, frame, now):
    validate_clock(state, frame, now)
    expected = {s+'/'+tf for s in manifest['symbols'] for tf in ('15m','1h','4h')}
    if set(incoming) != expected:
        raise ValueError('partial or unexpected portfolio population')
    previous = verify_chain(state.get('frames', []))
    result = copy.deepcopy(state)
    market = result.setdefault('market', {})
    cache = {}
    for symbol in manifest['symbols']:
        for tf in ('15m', '1h', '4h'):
            key = symbol+'/'+tf
            rows = merge_closed(market.get(key, []), incoming[key], rb.BAR_MS[tf], frame)
            market[key] = rows
            # The shared candidate builder deliberately ignores its last row.
            # Repeat last *closed* OHLC as a non-executable tail. No future price
            # enters any executable index; tail is excluded from entry/exit clock.
            data = np.array([tuple(r[k] for k in ('t','o','h','l','c','v')) for r in rows],
                            dtype=rb._KLINE_DTYPE)
            tail = data[-1:].copy()
            tail['t'] += rb.BAR_MS[tf]
            data = np.concatenate((data, tail))
            cache[symbol, tf] = (data, rb.compute_features(data['o'], data['h'], data['l'], data['c'], data['v']))
    c15 = {s: cache[s, '15m'] for s in manifest['symbols']}
    c4 = {s: cache[s, '4h'] for s in manifest['symbols']}
    ctx = rb._build_bull_day_context(cache['BTCUSDT', '1h'][0])
    arms = result.setdefault('arms', {})
    diagnostics = {}
    for arm in ('champion', 'candidate'):
        owned = arms.setdefault(arm, {'state': {}, 'closed_trades': [], 'candidate_state': {}})
        with frozen_arm(root, arm):
            raw, _, _ = await rb.build_replay_candidate_snapshot(manifest['symbols'],
                ['15m','1h'], cache, c15, c4, ctx, variant='score_replace_cluster',
                candidate_stream_state=owned['candidate_state'], frame_ms=frame)
            candidates = raw.get(frame, [])
            trades, stats = await rb.simulate_portfolio(manifest['symbols'], ['15m','1h'],
                cache, c15, c4, ctx, max_open_positions=10,
                enable_replacement=bool(getattr(config, "PORTFOLIO_REPLACE_ENABLED", True)),
                replace_min_delta=float(getattr(config, "PORTFOLIO_REPLACE_MIN_DELTA", 8)),
                variant='score_replace_cluster', top_gainer_score_min=float(config.TOP_GAINER_SCORE_GATE_MIN_SCORE),
                candidate_snapshot=({frame: candidates}, {frame}, len(candidates)), stream_state=owned['state'])
            owned['closed_trades'].extend(asdict(t) for t in trades)
            # The objective day is not finished. Do not publish partial-day
            # ranges as final early-capture labels.
            for row in (*owned['closed_trades'], *owned['state']['open_positions']):
                row.update(capture_ratio_at_entry=None, lead_time_to_final_top_min=None,
                           day_final_price=0.0)
            diagnostics[arm] = {'candidates': len(candidates), 'closed_this_frame': len(trades),
                                'open_positions': len(owned['state']['open_positions'])}
    if time.time()*1000 > frame+120000:
        raise ValueError('policy computation exceeded causal arrival budget')
    prices = {s: market[s+'/15m'][-1]['c'] for s in manifest['symbols']}
    body = {'frame': frame, 'received_ms': now, 'prices': prices, 'arms': diagnostics,
            'input_sha256': sha(canonical(incoming)), 'previous': previous}
    result.setdefault('frames', []).append({'previous': previous, 'body': body,
                                          'sha256': sha(canonical(body))})
    result['last_frame'] = frame
    verify_registration(root, manifest)
    atomic(root/'checkpoint.json', {'body': result, 'sha256': sha(canonical(result))})
    return result


async def tick(deployment):
    now = int(time.time()*1000)
    async with aiohttp.ClientSession() as session:
        root, manifest = await register(deployment, session, now)
        verify_registration(root, manifest)
        if now < manifest['observation_start_ms']:
            return {'state': 'EMBARGO', 'observation_start_ms': manifest['observation_start_ms'],
                    'symbols': len(manifest['symbols']), 'runtime_eligible': False}
        frame = now//STEP*STEP
        with process_lock(root/'portfolio.lock'):
            state = {}
            if (root/'checkpoint.json').exists():
                envelope = json.loads((root/'checkpoint.json').read_bytes())
                if sha(canonical(envelope['body'])) != envelope['sha256']:
                    raise ValueError('portfolio checkpoint corrupt')
                state = envelope['body']
            if state.get('last_frame') == frame:
                return {'state': 'WAITING', 'last_frame': frame, 'runtime_eligible': False}
            validate_clock(state, frame, now)
            semaphore = asyncio.Semaphore(8)
            async def fetch(symbol, tf):
                step = rb.BAR_MS[tf]
                key = symbol+'/'+tf
                old = state.get('market', {}).get(key, [])
                start = old[-1]['t'] if old else (frame//step)*step-1000*step
                async with semaphore:
                    async with session.get(rb.BINANCE_URL, params={'symbol': symbol, 'interval': tf,
                        'startTime': start, 'endTime': frame-1, 'limit': 1000},
                        timeout=aiohttp.ClientTimeout(total=30)) as response:
                        response.raise_for_status()
                        payload = await response.json()
                return key, exchange_rows(payload, step, frame)
            incoming = dict(await asyncio.gather(*(fetch(s, tf) for s in manifest['symbols']
                                                   for tf in ('15m','1h','4h'))))
            result = await advance(root, manifest, state, incoming, frame, now)
            export_unsigned(root, manifest, result)
            return {'state': 'COLLECTED_FULL_POLICY_PAPER', 'last_frame': frame,
                    'frames': len(result['frames']), 'scope': 'prospective_paper_not_real_fills',
                    'runtime_eligible': False}
