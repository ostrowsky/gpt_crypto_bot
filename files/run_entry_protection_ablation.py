"""Maximum-history research only. Never toggles the production bot's guards."""
import argparse
import asyncio
from dataclasses import asdict
import json
from pathlib import Path
from unittest.mock import patch

import config
import replay_backtest as rb
from audit_negative_day_rebound import bounded_cache, finalize_at_boundary
from closed_grid_policy_replay import SOURCES, event_clock, validate_series
from historical_signal_evaluation import freeze, sha
from paired_full_policy_replay import frozen_arm
from portfolio_alpha import closed_price_series, evaluate_portfolio_alpha
from validated_ranker_rollout import canonical
import certified_rule_score_policy as rule

ARMS = ('control', 'no_capacity', 'no_chase', 'both')


def options(arm):
    if arm not in ARMS:
        raise ValueError('unknown arm')
    return arm not in ('no_capacity', 'both'), arm not in ('no_chase', 'both')


def financial_state(report):
    if not report.get('decision_grade'):
        return 'INVALID_ACCOUNT_EVIDENCE'
    return 'DIAGNOSTIC_ONLY'


def publish(path, value):
    freeze(path, canonical(value))


async def run(archive, output):
    manifest_raw = (archive/'manifest.json').read_bytes()
    m = json.loads(manifest_raw)
    start, end = m['start_ms'], m['end_ms']
    if end <= start or not m['eligible_symbols']:
        raise ValueError('empty maximum archive')
    output.mkdir(parents=True, exist_ok=False)
    names = set((*SOURCES, *rule.SOURCES, 'run_entry_protection_ablation.py',
        'paired_full_policy_replay.py', 'audit_negative_day_rebound.py',
        'validated_ranker_rollout.py', 'ml_signal_model.py'))
    sources = {n: sha(Path(__file__).with_name(n)) for n in sorted(names)}
    for name, raw in (
        ('champion', rule.champion_bytes()),
        ('base_ranker', Path(__file__).with_name('ml_candidate_ranker.json').read_bytes()),
        ('general', Path(__file__).with_name('ml_signal_model.json').read_bytes())):
        freeze(output/(name+'.json'), raw)
    # The common frozen-arm interface reads candidate.json; no candidate model
    # overlay is selected in this experiment. Both populations use the champion.
    freeze(output/'candidate.json', (output/'base_ranker.json').read_bytes())
    descriptor=json.loads((output/'champion.json').read_bytes())
    for source_name, frozen_name in (
        ('ml_candidate_ranker.json','base_ranker.json'),('ml_signal_model.json','general.json')):
        if descriptor['models'][source_name] != sha(output/frozen_name):
            raise ValueError('model changed during freeze')
    models = {p.stem: sha(p) for p in output.glob('*.json')}
    def verify():
        if (archive/'manifest.json').read_bytes() != manifest_raw:
            raise ValueError('maximum archive drift')
        for name, digest in m['input_hashes'].items():
            if Path(name).name != name or sha(archive/'market'/name) != digest:
                raise ValueError('market input drift')
        for name, digest in sources.items():
            if sha(Path(__file__).with_name(name)) != digest:
                raise ValueError('source drift: '+name)
        for name, digest in models.items():
            if sha(output/(name+'.json')) != digest:
                raise ValueError('frozen model drift')
    verify()
    publish(output/'registration.json', {'contract':'entry-protection-ablation-v1',
        'maximum_available_bounds':[start,end], 'archive_sha256':sha(archive/'manifest.json'),
        'sources':sources, 'models':models, 'arms':list(ARMS),
        'capital_contract':'unchanged_ten_slot_no_leverage', 'runtime_eligible':False})
    if any(getattr(config, name, False) for name in (
        'VALIDATED_RANKER_ROLLOUT_ENABLED', 'CERTIFIED_RULE_SCORE_POLICY_ENABLED',
        'LOCAL_LOGICAL_POLICY_ROLLOUT_ENABLED')):
        raise ValueError('active overlay requires a separate frozen reconstruction')
    symbols = m['eligible_symbols']
    index = rb._build_market_cache_index(archive/'market')
    cache = {}
    for sym in symbols:
        print(json.dumps({'phase':'LOAD', 'symbol':sym}), flush=True)
        for tf in ('15m','1h'):
            data = rb._load_cached_klines(archive/'market',sym,tf,
                m['archive_start_ms'],end,cache_index=index)
            if data is None:
                raise ValueError('missing closed series')
            rows = [dict(t=int(r['t']), **{k:float(r[k]) for k in ('o','h','l','c','v')})
                    for r in data]
            validate_series(rows, rb.BAR_MS[tf],m['archive_start_ms'],end)
            cache[sym,tf] = (data, rb.compute_features(*[data[k] for k in ('o','h','l','c','v')]))
        four = rb._aggregate_1h_to_4h(cache[sym,'1h'][0])
        cache[sym,'4h'] = (four,rb.compute_features(*[four[k] for k in ('o','h','l','c','v')]))
    cache = bounded_cache(cache,end)
    c15={s:cache[s,'15m'] for s in symbols}
    c4={s:cache[s,'4h'] for s in symbols}
    context=rb._build_bull_day_context(cache['BTCUSDT','1h'][0])
    populations = {}
    original_builder = rb._build_candidates_for_symbol
    with frozen_arm(output,'champion'):
        for chase in (True,False):
            completed=0
            async def logged_builder(sym,tf,*args,**kwargs):
                nonlocal completed
                print(json.dumps({'phase':'BUILD','chase':chase,'symbol':sym,'tf':tf,
                    'completed_series':completed,'total_series':2*len(symbols)}),flush=True)
                built=await original_builder(sym,tf,*args,**kwargs)
                completed+=1
                return built
            enabled = getattr(config,'TOP_GAINER_CHASE_GUARD_ENABLED',True) if chase else False
            with patch.object(config,'TOP_GAINER_CHASE_GUARD_ENABLED',enabled), \
                 patch.object(rb,'_build_candidates_for_symbol',logged_builder):
                raw,times,_=await rb.build_replay_candidate_snapshot(symbols,['15m','1h'],
                    cache,c15,c4,context,variant='score_replace_cluster')
            populations[chase]=event_clock(raw,times,start,end)
        reports={}
        prices={s:closed_price_series(c15[s][0],bar_ms=900000,start_ms=start,end_ms=end)
                for s in symbols}
        for arm in ARMS:
            capacity,chase=options(arm)
            print(json.dumps({'phase':'SIMULATE','arm':arm}),flush=True)
            trades,stats=await rb.simulate_portfolio(symbols,['15m','1h'],cache,c15,c4,context,
                max_open_positions=10,enforce_position_capacity=capacity,
                enable_replacement=bool(getattr(config,'PORTFOLIO_REPLACE_ENABLED',True)),
                replace_min_delta=float(getattr(config,'PORTFOLIO_REPLACE_MIN_DELTA',8)),
                variant='score_replace_cluster',top_gainer_score_min=float(config.TOP_GAINER_SCORE_GATE_MIN_SCORE),
                candidate_snapshot=populations[chase])
            finalize_at_boundary(trades,cache,end)
            publish(output/('trades_'+arm+'.json'),[asdict(t) for t in trades])
            account=evaluate_portfolio_alpha(trades,price_series_by_symbol=prices,
                benchmark_series=prices['BTCUSDT'],window_start_ms=start,window_end_ms=end,
                requested_days=(end-start)//86400000,universe=symbols,variant=arm,
                fee_bps=max(7.5,float(config.PAPER_FEE_BPS)),slippage_bps=5,capacity=10)
            publish(output/('account_'+arm+'.json'),account)
            reports[arm]={'candidates':populations[chase][2], 'trades':len(trades),
                'capacity_blocked':stats.skipped_portfolio_full,
                'financial_state':financial_state(account),
                'portfolio':account['portfolio'], 'net_alpha':account['net_alpha_after_costs'],
                'violations':account['coverage']['contract_violations']}
    verify()
    result={'state':'UNKNOWN','runtime_eligible':False,'arms':reports,
        'blockers':['reconstructed historical population; raw/PIT and full live BUY parity unverified',
                    'model exposure overlaps history; untouched forward validation required',
                    'cash/capacity violations invalidate unrestricted-signal portfolio profitability'],
        'production_guards_changed':False}
    publish(output/'result.json',result)
    publish(output/'receipt.json',{p.name:sha(p) for p in output.glob('*.json')})
    print(json.dumps({'phase':'COMPLETE','state':'UNKNOWN','runtime_eligible':False}),flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    from prediction_error_optimizer import limit_cpu
    limit_cpu()
    asyncio.run(run(a.archive,a.output))
