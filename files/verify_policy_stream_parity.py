"""Maximum archive execution-state parity; not candidate/live-policy certification."""
import argparse
import asyncio
from dataclasses import asdict
import json
from pathlib import Path

import config
import replay_backtest as rb
from closed_grid_policy_replay import event_clock
from forward_evidence_service import atomic
from paired_full_policy_replay import frozen_arm
from validated_ranker_rollout import canonical, sha


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


async def compare(symbols, cache, ctx, snapshot, progress=None):
    c15, c4 = ({s: cache[s, tf] for s in symbols} for tf in ('15m', '4h'))
    kwargs = dict(max_open_positions=10,
        enable_replacement=bool(getattr(config, 'PORTFOLIO_REPLACE_ENABLED', True)),
        replace_min_delta=float(getattr(config, 'PORTFOLIO_REPLACE_MIN_DELTA', 8)),
        variant='score_replace_cluster', top_gainer_score_min=float(config.TOP_GAINER_SCORE_GATE_MIN_SCORE))
    bulk = {}
    left, _ = await rb.simulate_portfolio(symbols,['15m','1h'],cache,c15,c4,ctx,
        candidate_snapshot=snapshot,stream_state=bulk,**kwargs)
    streaming, right = {}, []
    frames = sorted(snapshot[1])
    for n, frame in enumerate(frames):
        candidates = snapshot[0].get(frame, [])
        closed, _ = await rb.simulate_portfolio(symbols,['15m','1h'],cache,c15,c4,ctx,
            candidate_snapshot=({frame:candidates},{frame},len(candidates)),stream_state=streaming,**kwargs)
        right.extend(closed)
        streaming = json.loads(canonical(streaming))  # real process/checkpoint boundary
        if progress and n % 96 == 0:
            progress(n+1, len(frames))
    a, b = normalized(left, bulk), normalized(right, streaming)
    return {'state': 'PASS' if canonical(a) == canonical(b) else 'FAIL',
            'frames': len(frames), 'closed_trades': len(left),
            'bulk_sha256': sha(canonical(a)), 'stream_sha256': sha(canonical(b)),
            'scope': 'execution_state_only_not_candidate_generation_or_live_parity'}


async def run(archive, models, output):
    manifest_raw = (archive/'manifest.json').read_bytes()
    manifest = json.loads(manifest_raw)
    registration = json.loads((models/'registration.json').read_bytes())
    def verify():
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
    index = rb._build_market_cache_index(archive/'market')
    cache, symbols = {}, manifest['eligible_symbols']
    for s in symbols:
        for tf in ('15m','1h'):
            data = rb._load_cached_klines(archive/'market',s,tf,manifest['archive_start_ms'],
                                          manifest['end_ms'],cache_index=index)
            if data is None: raise ValueError('missing frozen market')
            cache[s,tf] = (data, rb.compute_features(data['o'],data['h'],data['l'],data['c'],data['v']))
        data = rb._aggregate_1h_to_4h(cache[s,'1h'][0])
        cache[s,'4h'] = (data,rb.compute_features(data['o'],data['h'],data['l'],data['c'],data['v']))
    ctx = rb._build_bull_day_context(cache['BTCUSDT','1h'][0])
    c15, c4 = ({s:cache[s,tf] for s in symbols} for tf in ('15m','4h'))
    result = {'contract':'maximum-stream-execution-parity-v1','runtime_eligible':False,
              'bounds':[manifest['start_ms'],manifest['end_ms']],
              'archive_sha256':sha(manifest_raw), 'registration_sha256':sha((models/'registration.json').read_bytes()),
              'arms':{}}
    for arm in ('champion','candidate'):
        with frozen_arm(models,arm):
            raw,times,_ = await rb.build_replay_candidate_snapshot(symbols,['15m','1h'],cache,c15,c4,ctx,
                                                                  variant='score_replace_cluster')
            snapshot = event_clock(raw,times,manifest['start_ms'],manifest['end_ms'])
            result['arms'][arm] = await compare(symbols,cache,ctx,snapshot,
                lambda done,total: atomic(output.with_suffix('.progress.json'),
                    {'state':'RUNNING','arm':arm,'frames_done':done,'frames_total':total}))
    verify()
    result['state'] = 'PASS' if all(a['state']=='PASS' for a in result['arms'].values()) else 'FAIL'
    atomic(output,result)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('archive','models','output'): p.add_argument('--'+name,type=Path,required=True)
    args = p.parse_args()
    print(asyncio.run(run(args.archive,args.models,args.output))['state'])
