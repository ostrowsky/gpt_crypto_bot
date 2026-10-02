"""Frozen complete BUY/SELL replay of the supported bounded ranker overlay.

Unsigned numerical evidence is never authority to change the running bot.
"""
import argparse
import asyncio
from contextlib import contextmanager, ExitStack
from dataclasses import asdict
import json
from pathlib import Path
import time
from unittest.mock import patch

import config
import monitor
import replay_backtest as rb
import validated_ranker_rollout as rollout
import certified_rule_score_policy as rule_score
from audit_negative_day_rebound import bounded_cache, finalize_at_boundary
from closed_grid_policy_replay import SOURCES, event_clock, validate_series
from historical_signal_evaluation import freeze, sha
from independent_portfolio_gate import compare_accounts
from portfolio_alpha import closed_price_series


def preflight(champion):
    if champion.get('contract') == rule_score.CONTRACT:
        blockers = []
        if rollout.canonical(champion) != rule_score.champion_bytes():
            blockers.append('rule champion descriptor differs from current sources/config/models')
        if getattr(config, 'VALIDATED_RANKER_ROLLOUT_ENABLED', False):
            blockers.append('active legacy overlay cannot be reconstructed as rule champion')
        return blockers
    blockers = []
    if not getattr(config, 'ML_CANDIDATE_RANKER_RUNTIME_ENABLED', False):
        blockers.append('live ranker runtime disabled')
    if float(getattr(config, 'ML_CANDIDATE_RANKER_SCORE_WEIGHT', 0)) == 0:
        blockers.append('live ranker weight zero')
    if (getattr(config, 'POLICY_PROVENANCE_REQUIRED_FOR_RANKER', True)
            and not champion.get('runtime_eligible')):
        blockers.append('champion rejected by live provenance loader')
    return blockers


@contextmanager
def frozen_arm(directory, arm):
    if arm not in ('champion', 'candidate'):
        raise ValueError('unsupported full policy arm')
    champion = json.loads((directory/'champion.json').read_bytes())
    candidate = json.loads((directory/'candidate.json').read_bytes())
    general = json.loads((directory/'general.json').read_bytes())
    with ExitStack() as stack:
        is_rule = champion.get('contract') == rule_score.CONTRACT
        stack.enter_context(patch.object(monitor, '_RANKER_MODEL_FILE',
            directory/('base_ranker.json' if is_rule else 'champion.json')))
        stack.enter_context(patch.object(monitor, '_RANKER_MODEL_CACHE', None))
        stack.enter_context(patch.object(rb, '_load_ml_model_payload', return_value=general))
        # Same champion components/exit rules in both arms. Only bounded overlay
        # selection is substituted; this does not issue a production ticket.
        stack.enter_context(patch.object(config, 'VALIDATED_RANKER_ROLLOUT_ENABLED',
                                         arm == 'candidate' and not is_rule))
        stack.enter_context(patch.object(config, 'CERTIFIED_RULE_SCORE_POLICY_ENABLED',
                                         arm == 'candidate' and is_rule))
        stack.enter_context(patch.object(rule_score, 'champion_bytes',
                                         return_value=(directory/'champion.json').read_bytes()))
        stack.enter_context(patch.object(rollout, 'select', return_value=(candidate,
            {'candidate_sha256': sha(directory/'candidate.json'), 'stage':'OFFLINE_REPLAY'})
            if arm == 'candidate' else None))
        stack.enter_context(patch.object(rb, '_load_temporal_scout_events', return_value=({}, {})))
        if json.loads((directory/'champion.json').read_bytes()) != champion:
            raise ValueError('frozen champion drift')
        yield


async def run_arms(cache, symbols, start, end, directory):
    local = bounded_cache(cache, end)
    c15 = {s: local[s, '15m'] for s in symbols}
    c4 = {s: local[s, '4h'] for s in symbols}
    ctx = rb._build_bull_day_context(local['BTCUSDT', '1h'][0])
    result = {}
    for arm in ('champion', 'candidate'):
        with frozen_arm(directory, arm):
            raw, times, _ = await rb.build_replay_candidate_snapshot(
                symbols, ['15m','1h'], local, c15, c4, ctx, variant='score_replace_cluster')
            snapshot = event_clock(raw, times, start, end)
            trades, stats = await rb.simulate_portfolio(
                symbols, ['15m','1h'], local, c15, c4, ctx, max_open_positions=10,
                enable_replacement=bool(getattr(config, 'PORTFOLIO_REPLACE_ENABLED', True)),
                replace_min_delta=float(getattr(config, 'PORTFOLIO_REPLACE_MIN_DELTA', 8)),
                variant='score_replace_cluster',
                top_gainer_score_min=float(config.TOP_GAINER_SCORE_GATE_MIN_SCORE),
                candidate_snapshot=snapshot)
            finalize_at_boundary(trades, local, end)
            result[arm] = {'trades': [asdict(t) for t in trades],
                           'stats': asdict(stats), 'candidates': snapshot[2]}
    return result


async def run(archive, champion_path, candidate_path, output, *, policy_family='legacy-ranker'):
    manifest_raw = (archive/'manifest.json').read_bytes()
    manifest = json.loads(manifest_raw)
    sources = {n: sha(Path(__file__).with_name(n)) for n in
        (*SOURCES, 'paired_full_policy_replay.py', 'audit_negative_day_rebound.py',
         'independent_portfolio_gate.py', 'validated_ranker_rollout.py',
         'process_lock.py', 'ml_candidate_ranker.py', 'ml_signal_model.py',
         'certified_rule_score_policy.py')}
    output.mkdir(parents=True, exist_ok=False)
    for name, path in [('champion', champion_path), ('candidate', candidate_path),
                       ('general', Path(rb.__file__).with_name('ml_signal_model.json'))]:
        raw = (rule_score.champion_bytes() if name == 'champion' and policy_family == 'rule-score'
               else path.read_bytes())
        if not isinstance(json.loads(raw), dict):
            raise ValueError('model must be a mapping')
        freeze(output/(name+'.json'), raw)
    if policy_family == 'rule-score':
        freeze(output/'base_ranker.json', champion_path.read_bytes())
    elif policy_family != 'legacy-ranker':
        raise ValueError('unknown policy family')
    models = {p.stem:sha(p) for p in output.glob('*.json')}
    def verify():
        if (archive/'manifest.json').read_bytes() != manifest_raw:
            raise ValueError('archive manifest drift')
        for name, checksum in manifest['input_hashes'].items():
            relative = Path(name)
            if relative.is_absolute() or relative.name != name:
                raise ValueError('unsafe frozen archive path')
            if sha(archive/'market'/name) != checksum:
                raise ValueError('market input drift')
        for name, checksum in sources.items():
            if sha(Path(__file__).with_name(name)) != checksum:
                raise ValueError('source drift')
        for name, checksum in models.items():
            if sha(output/(name+'.json')) != checksum:
                raise ValueError('model drift')
    verify()
    freeze(output/'registration.json', rollout.canonical({
        'contract':'paired-full-policy-replay-v1', 'maximum_available_bounds':
        [manifest['start_ms'],manifest['end_ms']], 'sources':sources, 'models':models,
        'archive_sha256':sha(archive/'manifest.json'), 'policy_family':policy_family,
        'runtime_eligible':False}))
    blockers = preflight(json.loads((output/'champion.json').read_bytes()))
    if blockers:
        report = {'state':'BLOCKED', 'runtime_eligible':False, 'blockers':blockers,
                  'comparison':None, 'reason':'no effective supported live challenger policy'}
    else:
        start, end = manifest['start_ms'], manifest['end_ms']
        cache, symbols = {}, manifest['eligible_symbols']
        index = rb._build_market_cache_index(archive/'market')
        for sym in symbols:
            for tf in ('15m','1h'):
                data = rb._load_cached_klines(archive/'market', sym, tf,
                    manifest['archive_start_ms'], end, cache_index=index)
                if data is None:
                    raise ValueError('frozen market unavailable')
                rows = [dict(t=int(r['t']), o=float(r['o']), h=float(r['h']),
                    l=float(r['l']), c=float(r['c']), v=float(r['v'])) for r in data]
                validate_series(rows, rb.BAR_MS[tf], manifest['archive_start_ms'], end)
                cache[sym,tf] = (data, rb.compute_features(data['o'],data['h'],data['l'],data['c'],data['v']))
            four = rb._aggregate_1h_to_4h(cache[sym,'1h'][0])
            cache[sym,'4h'] = (four, rb.compute_features(four['o'],four['h'],four['l'],four['c'],four['v']))
        arms = await run_arms(cache, symbols, start, end, output)
        bundle = {'phase':'historical', 'start_ms':start, 'end_ms':end,
            'maximum_available_bounds':[start,end], 'last_model_exposure_ms':end,
            'universe':symbols, 'fee_bps':max(7.5,float(config.PAPER_FEE_BPS)),
            'slippage_bps':5, 'prices':{s:closed_price_series(cache[s,'15m'][0],
                bar_ms=900000,start_ms=start,end_ms=end) for s in symbols},
            **{a+'_trades':arms[a]['trades'] for a in arms}}
        freeze(output/'historical_unsigned.json', rollout.canonical(bundle))
        comparison = compare_accounts(bundle, time.time())
        report = {'state':'UNKNOWN', 'runtime_eligible':False, 'comparison':comparison,
            'arms':{a:{k:v for k,v in data.items() if k != 'trades'} for a,data in arms.items()},
            'blockers':['unsigned reconstructed population, no raw/PIT certification',
                        'historical model exposure; not sealed holdout',
                        'separate prospective full-policy shadow and canary evidence required']}
    verify()
    freeze(output/'result.json', rollout.canonical(report))
    freeze(output/'receipt.json', rollout.canonical({p.name:sha(p) for p in output.glob('*.json')}))
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for arg in ('archive','champion','candidate','output'):
        p.add_argument('--'+arg, type=Path, required=True)
    p.add_argument('--policy-family', choices=('legacy-ranker','rule-score'), default='legacy-ranker')
    args = p.parse_args()
    result = asyncio.run(run(args.archive,args.champion,args.candidate,args.output,
                            policy_family=args.policy_family))
    print(json.dumps({'state':result['state'], 'blockers':result['blockers']}))
    raise SystemExit(0 if result['comparison'] is not None else 1)
