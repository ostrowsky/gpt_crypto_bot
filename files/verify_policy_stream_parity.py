"""Maximum archive execution-state parity; not candidate/live-policy certification."""
import argparse
import asyncio
from dataclasses import asdict
import json
from pathlib import Path
import time

import config
import replay_backtest as rb
from closed_grid_policy_replay import event_clock
from forward_evidence_service import atomic
from paired_full_policy_replay import frozen_arm
from validated_ranker_rollout import canonical, sha
from replay_closed_trade_accounting import extend_index, reconcile


def differences(left, right, path=()):
    """Exact structural evidence: missing is not null and order is significant."""
    if isinstance(left, dict) and isinstance(right, dict):
        keys = sorted(set(left) | set(right))
    elif isinstance(left, list) and isinstance(right, list):
        keys = range(max(len(left), len(right)))
    else:
        if canonical(left) != canonical(right):
            yield {'path': list(path), 'bulk_present': True, 'stream_present': True,
                   'bulk': left, 'stream': right}
        return
    for key in keys:
        present_left = key in left if isinstance(left, dict) else key < len(left)
        present_right = key in right if isinstance(right, dict) else key < len(right)
        if present_left and present_right:
            yield from differences(left[key], right[key], (*path, key))
        else:
            yield {'path': [*path, key], 'bulk_present': present_left,
                   'stream_present': present_right,
                   'bulk': left[key] if present_left else None,
                   'stream': right[key] if present_right else None}


def normalized(trades, state):
    # Final-day objective labels are not decision inputs/portfolio execution.
    def row(value):
        value = dict(value)
        for key in ('capture_ratio_at_entry', 'lead_time_to_final_top_min', 'day_final_price'):
            value.pop(key, None)
        return value
    return {'closed': [row(asdict(t)) for t in trades],
            'state': dict(state, open_positions=[row(t) for t in state['open_positions']],
                last_closed_by_symbol={s: row(t) for s,t in state['last_closed_by_symbol'].items()})}


async def compare(symbols, cache, ctx, snapshot, progress=None, difference_output=None):
    c15, c4 = ({s: cache[s, tf] for s in symbols} for tf in ('15m', '4h'))
    kwargs = dict(max_open_positions=10,
        enable_replacement=bool(getattr(config, 'PORTFOLIO_REPLACE_ENABLED', True)),
        replace_min_delta=float(getattr(config, 'PORTFOLIO_REPLACE_MIN_DELTA', 8)),
        variant='score_replace_cluster', top_gainer_score_min=float(config.TOP_GAINER_SCORE_GATE_MIN_SCORE))
    bulk = {}
    if progress: progress({'phase': 'BATCH_SIMULATION', 'state': 'RUNNING'})
    left, _ = await rb.simulate_portfolio(symbols,['15m','1h'],cache,c15,c4,ctx,
        candidate_snapshot=snapshot,stream_state=bulk,**kwargs)
    streaming, right, history_index = {}, [], {}
    frames = sorted(snapshot[1])
    if progress: progress({'phase': 'STREAMING', 'state': 'RUNNING',
                           'frames_done': 0, 'frames_total': len(frames)})
    for n, frame in enumerate(frames):
        candidates = snapshot[0].get(frame, [])
        closed, _ = await rb.simulate_portfolio(symbols,['15m','1h'],cache,c15,c4,ctx,
            candidate_snapshot=({frame:candidates},{frame},len(candidates)),stream_state=streaming,**kwargs)
        right.extend(closed)
        extend_index(history_index, closed)
        reconcile(history_index, streaming)
        streaming = json.loads(canonical(streaming))  # real process/checkpoint boundary
        if progress and n % 96 == 0:
            progress({'phase': 'STREAMING', 'state': 'RUNNING',
                      'frames_done': n+1, 'frames_total': len(frames)})
    if progress: progress({'phase': 'COMPARISON', 'state': 'RUNNING',
                           'frames_done': len(frames), 'frames_total': len(frames)})
    a, b = normalized(left, bulk), normalized(right, streaming)
    detail = list(differences(a, b))
    for row in detail:
        if len(row['path']) > 1 and row['path'][0] == 'closed':
            i = row['path'][1]
            for name, source in (('bulk', a), ('stream', b)):
                if i < len(source['closed']):
                    row[name+'_trade_identity'] = {k: source['closed'][i][k]
                        for k in ('sym', 'tf', 'entry_ts', 'exit_ts')}
    evidence = {'difference_count': len(detail), 'differences': detail}
    if difference_output is not None: atomic(difference_output, evidence)
    return {'state': 'FAIL' if detail else 'PASS',
            'frames': len(frames), 'closed_trades': len(left),
            'stream_closed_trades': len(right), 'difference_count': len(detail),
            'difference_examples': detail[:20],
            'difference_artifact': None if difference_output is None else difference_output.name,
            'difference_sha256': None if difference_output is None else sha(difference_output.read_bytes()),
            'bulk_sha256': sha(canonical(a)), 'stream_sha256': sha(canonical(b)),
            'scope': 'execution_state_only_not_candidate_generation_or_live_parity'}


async def _run(archive, models, output):
    if output.exists(): raise ValueError('refusing to overwrite final parity evidence')
    verification_sources = {name: sha(Path(__file__).with_name(name).read_bytes()) for name in
                            ('verify_policy_stream_parity.py', 'replay_closed_trade_accounting.py')}
    manifest_raw = (archive/'manifest.json').read_bytes()
    manifest = json.loads(manifest_raw)
    registration = json.loads((models/'registration.json').read_bytes())
    registration_raw = (models/'registration.json').read_bytes()
    def progress(value):
        atomic(output.with_suffix('.progress.json'), dict(value, updated_at=time.time()))
    def verify():
        if (models/'registration.json').read_bytes() != registration_raw:
            raise ValueError('registration drift')
        if registration['maximum_available_bounds'] != [manifest['start_ms'], manifest['end_ms']]:
            raise ValueError('maximum interval drift')
        for name, digest in verification_sources.items():
            if sha(Path(__file__).with_name(name).read_bytes()) != digest:
                raise ValueError('verification source drift: '+name)
        if sha((archive/'manifest.json').read_bytes()) != registration['archive_sha256']:
            raise ValueError('archive registration mismatch')
        for name, digest in manifest['input_hashes'].items():
            if Path(name).name != name or sha((archive/'market'/name).read_bytes()) != digest:
                raise ValueError('archive market drift')
        for name, digest in registration['sources'].items():
            if sha(Path(__file__).with_name(name).read_bytes()) != digest:
                raise ValueError('registered source drift: '+name)
        for name, digest in registration['models'].items():
            if sha((models/(name+'.json')).read_bytes()) != digest:
                raise ValueError('frozen model drift')
    verify()
    progress({'phase': 'LOADING_MARKET', 'state': 'RUNNING'})
    index = rb._build_market_cache_index(archive/'market')
    cache, symbols = {}, manifest['eligible_symbols']
    for number, s in enumerate(symbols):
        for tf in ('15m','1h'):
            data = rb._load_cached_klines(archive/'market',s,tf,manifest['archive_start_ms'],
                                          manifest['end_ms'],cache_index=index)
            if data is None: raise ValueError('missing frozen market')
            cache[s,tf] = (data, rb.compute_features(data['o'],data['h'],data['l'],data['c'],data['v']))
        data = rb._aggregate_1h_to_4h(cache[s,'1h'][0])
        cache[s,'4h'] = (data,rb.compute_features(data['o'],data['h'],data['l'],data['c'],data['v']))
        progress({'phase': 'LOADING_MARKET', 'state': 'RUNNING',
                  'symbols_done': number+1, 'symbols_total': len(symbols)})
    ctx = rb._build_bull_day_context(cache['BTCUSDT','1h'][0])
    c15, c4 = ({s:cache[s,tf] for s in symbols} for tf in ('15m','4h'))
    result = {'contract':'maximum-stream-execution-parity-v2','runtime_eligible':False,
              'bounds':[manifest['start_ms'],manifest['end_ms']],
              'archive_sha256':sha(manifest_raw), 'registration_sha256':sha((models/'registration.json').read_bytes()),
              'verification_sources': verification_sources, 'arms':{}}
    for arm in ('champion','candidate'):
        with frozen_arm(models,arm):
            progress({'phase': 'BUILDING_CANDIDATES', 'state': 'RUNNING', 'arm': arm})
            raw,times,_ = await rb.build_replay_candidate_snapshot(symbols,['15m','1h'],cache,c15,c4,ctx,
                                                                  variant='score_replace_cluster')
            snapshot = event_clock(raw,times,manifest['start_ms'],manifest['end_ms'])
            result['arms'][arm] = await compare(symbols,cache,ctx,snapshot,
                lambda value: progress(dict(value, arm=arm)),
                output.with_name(output.stem+'.'+arm+'.differences.json'))
    progress({'phase': 'VERIFYING_RECEIPTS', 'state': 'RUNNING'})
    verify()
    result['state'] = 'PASS' if all(a['state']=='PASS' for a in result['arms'].values()) else 'FAIL'
    atomic(output,result)
    progress({'phase': 'FINISHED', 'state': result['state']})
    return result


async def run(archive, models, output):
    if output.exists(): raise ValueError('refusing to overwrite final parity evidence')
    try:
        return await _run(archive, models, output)
    except Exception as exc:
        result = {'contract': 'maximum-stream-execution-parity-v2', 'state': 'UNKNOWN',
                  'runtime_eligible': False, 'reason': str(exc),
                  'scope': 'execution_state_only_not_candidate_generation_or_live_parity'}
        atomic(output, result)
        atomic(output.with_suffix('.progress.json'),
               {'phase': 'FINISHED', 'state': 'UNKNOWN', 'updated_at': time.time()})
        return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('archive','models','output'): p.add_argument('--'+name,type=Path,required=True)
    args = p.parse_args()
    verdict = asyncio.run(run(args.archive,args.models,args.output))['state']
    print(verdict)
    raise SystemExit(0 if verdict == 'PASS' else 1)
