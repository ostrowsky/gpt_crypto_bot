"""Maximum recovered archive, observe-only conflict capture and frozen ranker study."""
import argparse
import asyncio
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np

import config
import replay_backtest as rb
from capacity_catboost import (FEATURES, HORIZON, causal_features, cohort,
    daily_identity, evaluate, forward_return, relevance, split_boundaries)
from closed_grid_policy_replay import SOURCES, event_clock, validate_series
from historical_signal_evaluation import freeze, sha
from portfolio_alpha import closed_price_series, evaluate_portfolio_alpha
from research_rocket_capture import policy

LOCAL_SOURCES = tuple(dict.fromkeys((*SOURCES, 'capacity_catboost.py',
    'run_capacity_catboost.py', 'ml_candidate_ranker.py', 'ml_signal_model.py')))


def publish(path, value):
    freeze(path, json.dumps(value, sort_keys=True, allow_nan=False, indent=2).encode())


def validated_trace(archive, directory):
    """Reuse numerical traces only with exact receipt/source/maximum-window parity."""
    receipt = json.loads((directory/'receipt.json').read_bytes())
    for name, expected in receipt.items():
        if Path(name).name != name or sha(directory/name) != expected:
            raise ValueError('trace receipt mismatch: '+name)
    for required in ('registration.json','result.json','historical_unsigned.json','champion.json'):
        if required not in receipt: raise ValueError('missing trace receipt: '+required)
    reg = json.loads((directory/'registration.json').read_bytes())
    m = json.loads((archive/'manifest.json').read_bytes())
    if not set(SOURCES).issubset(reg['sources']):
        raise ValueError('incomplete trace source bindings')
    if (reg['maximum_available_bounds'] != [m['start_ms'],m['end_ms']]
            or reg['archive_sha256'] != sha(archive/'manifest.json')):
        raise ValueError('trace does not cover identical maximum archive')
    for name, expected in reg['sources'].items():
        if Path(name).name != name or sha(Path(__file__).with_name(name)) != expected:
            raise ValueError('trace source drift: '+name)
    result = json.loads((directory/'result.json').read_bytes())
    bundle = json.loads((directory/'historical_unsigned.json').read_bytes())
    if (bundle['start_ms'] != m['start_ms'] or bundle['end_ms'] != m['end_ms']
            or bundle['universe'] != m['eligible_symbols']):
        raise ValueError('trace trade population/window mismatch')
    return result,bundle,receipt


def mission(data, clock, sym, objective):
    day, cutoff = daily_identity(clock)
    if day not in objective['eligible_days'] or clock > cutoff:
        return False, None, False
    is_leader = (day, sym) in objective['label_pairs']
    # The complete day grid is required; lookup the exact canonical open/close.
    tz = rb._objective_tz()
    from datetime import time
    start = int(datetime.combine(datetime.fromisoformat(day).date(), time(), tzinfo=tz).timestamp()*1000)
    opens = data['t']; closes = opens+rb.BAR_MS['15m']
    i = int(np.searchsorted(opens, start)); j = int(np.searchsorted(closes, cutoff))-1
    if j+1 < len(closes) and closes[j+1] == cutoff: j += 1
    p = int(np.searchsorted(closes, clock))
    if i >= len(data) or j < i or p >= len(data) or opens[i] != start or closes[j] != cutoff or closes[p] != clock:
        return False, None, False
    denominator = float(data['c'][j]-data['o'][i])
    capture = float(np.clip((data['c'][j]-data['c'][p])/denominator, 0., 1.5)) if is_leader and denominator > 0 else None
    return bool(is_leader), capture, True


async def run(archive, output, trace_directory=None):
    from catboost import CatBoostRanker, Pool
    manifest_raw = (archive/'manifest.json').read_bytes()
    m = json.loads(manifest_raw)
    output.mkdir(parents=True, exist_ok=False)
    source_dir = output/'source_snapshot'; source_dir.mkdir()
    sources = {}
    for name in LOCAL_SOURCES:
        path = Path(__file__).with_name(name)
        freeze(source_dir/name, path.read_bytes()); sources[name] = sha(path)
    start, end = m['start_ms'], m['end_ms']
    cuts = split_boundaries(start, end)
    trace = validated_trace(archive,trace_directory) if trace_directory else None
    if trace:
        frozen_trace = output/'trace_snapshot'; frozen_trace.mkdir()
        for name in trace[2]: freeze(frozen_trace/name,(trace_directory/name).read_bytes())
        freeze(frozen_trace/'receipt.json',(trace_directory/'receipt.json').read_bytes())
    publish(output/'registration.json', {
        'contract': 'capacity-catboost-pre-gate-v1', 'registered_at': datetime.now(timezone.utc).isoformat(),
        'archive_manifest_sha256': sha(archive/'manifest.json'), 'sources': sources,
        'start_ms': start, 'end_ms': end, 'cuts_ms': cuts, 'features': FEATURES,
        'horizon_ms': HORIZON, 'replacement_incremental_cost_pp': .25,
        'parameters': {'loss_function':'YetiRank','iterations':400,'depth':4,
            'learning_rate':.03,'l2_leaf_reg':10,'random_seed':42,'thread_count':2,'early_stopping_rounds':40},
        'test_exposure':'retrospective previously exposed archive',
        'requested_symbols':m['requested_symbols'], 'complete_symbols':m['eligible_symbols'],
        'missing_series':m['missing_series'], 'runtime_eligible':False})
    publish(output/'trace_mode.json',{'mode':'verified_existing_champion' if trace else 'recompute_rule_only',
        'receipt_sha256':sha(trace_directory/'receipt.json') if trace else None})

    def verify():
        if (archive/'manifest.json').read_bytes() != manifest_raw:
            raise ValueError('archive manifest drift')
        for name, expected in sources.items():
            if sha(Path(__file__).with_name(name)) != expected:
                raise ValueError('source drift: '+name)
        for name, expected in m['input_hashes'].items():
            if Path(name).name != name or sha(archive/'market'/name) != expected:
                raise ValueError('archive input identity/hash mismatch: '+name)
        if trace:
            validated_trace(archive,trace_directory)
            for name,expected in trace[2].items():
                if sha(output/'trace_snapshot'/name) != expected:
                    raise ValueError('frozen trace drift')
    verify()
    cache = {}; symbols = m['eligible_symbols']
    index = rb._build_market_cache_index(archive/'market')
    for n, sym in enumerate(symbols):
        print(json.dumps({'phase':'LOAD','completed':n,'total':len(symbols),'symbol':sym}),flush=True)
        for tf in (('15m',) if trace else ('15m','1h')):
            data = rb._load_cached_klines(archive/'market',sym,tf,m['archive_start_ms'],end,cache_index=index)
            if data is None: raise ValueError('missing frozen input')
            rows = [dict(t=int(r['t']),**{k:float(r[k]) for k in ('o','h','l','c','v')}) for r in data]
            validate_series(rows,rb.BAR_MS[tf],m['archive_start_ms'],end)
            cache[sym,tf] = (data,{} if trace else rb.compute_features(data['o'],data['h'],data['l'],data['c'],data['v']))
        if not trace:
            four = rb._aggregate_1h_to_4h(cache[sym,'1h'][0])
            cache[sym,'4h'] = (four,rb.compute_features(four['o'],four['h'],four['l'],four['c'],four['v']))
    c15 = {s:cache[s,'15m'] for s in symbols}
    c4 = {s:cache[s,'4h'] for s in symbols} if not trace else {}
    ctx = rb._build_bull_day_context(cache['BTCUSDT','1h'][0]) if not trace else None
    conflicts = []; seen = set(); counters = {'recorded_calls':0,'duplicate_pairs':0,'invalid_past':0}
    recorder = rb._record_capacity_competition
    def observed(stats, *, ts_ms, candidate, open_positions, variant):
        # Delegate exactly once; instrumentation cannot alter the replay decisions.
        recorder(stats,ts_ms=ts_ms,candidate=candidate,open_positions=open_positions,variant=variant)
        counters['recorded_calls'] += 1
        if not open_positions: return
        incumbent = min(open_positions.values(),key=rb._trade_allocation_score)
        identity = (ts_ms,candidate.sym,incumbent.sym)
        if identity in seen:
            counters['duplicate_pairs'] += 1; return
        seen.add(identity)
        x = [causal_features(c15[s][0],ts_ms,role) for s,role in ((candidate.sym,1),(incumbent.sym,0))]
        if any(v is None for v in x):
            counters['invalid_past'] += 1; return
        conflicts.append({'clock':int(ts_ms),'symbols':[candidate.sym,incumbent.sym],
                          'features':[v.tolist() for v in x]})
    original = rb._build_candidates_for_symbol
    async def logged(sym, tf, *args, **kwargs):
        print(json.dumps({'phase':'BUILD','symbol':sym,'tf':tf}),flush=True)
        result = await original(sym,tf,*args,**kwargs)
        print(json.dumps({'phase':'BUILT','symbol':sym,'tf':tf,'candidates':len(result)}),flush=True)
        return result
    if trace:
        result,bundle,_ = trace
        stats = result['arms']['champion']['stats']
        trades = bundle['champion_trades']; candidate_count = result['arms']['champion']['candidates']
        for event in stats['capacity_competition_events']:
            counters['recorded_calls'] += 1
            ts=event['ts_ms']; syms=[event['candidate_sym'],event['incumbent_sym']]
            identity=(ts,*syms)
            if identity in seen: counters['duplicate_pairs'] += 1; continue
            seen.add(identity)
            if not start <= ts < end: raise ValueError('capacity event outside maximum window')
            x=[causal_features(c15[s][0],ts,role) for s,role in zip(syms,(1,0))]
            if any(v is None for v in x): counters['invalid_past'] += 1; continue
            conflicts.append({'clock':int(ts),'symbols':syms,'features':[v.tolist() for v in x]})
    else:
        with policy('baseline'), patch.object(rb,'_load_temporal_scout_events',return_value=({},{})):
            with patch.object(rb,'_build_candidates_for_symbol',logged):
                raw,times,_ = await rb.build_replay_candidate_snapshot(symbols,['15m','1h'],cache,c15,c4,ctx,variant='score_replace_cluster')
            snapshot = event_clock(raw,times,start,end)
            print(json.dumps({'phase':'SIMULATE','candidates':snapshot[2]}),flush=True)
            with patch.object(rb,'_record_capacity_competition',observed):
                trades,stats = await rb.simulate_portfolio(symbols,['15m','1h'],cache,c15,c4,ctx,
                    max_open_positions=10,enable_replacement=bool(config.PORTFOLIO_REPLACE_ENABLED),
                    replace_min_delta=float(config.PORTFOLIO_REPLACE_MIN_DELTA),variant='score_replace_cluster',
                    top_gainer_score_min=float(config.TOP_GAINER_SCORE_GATE_MIN_SCORE),candidate_snapshot=snapshot)
            trades=[asdict(t) for t in trades]; candidate_count=snapshot[2]
    objective = rb._daily_top_objective(symbols,cache,start_ms=start,end_ms=end,top_n=15)
    publish(output/'conflicts.json',conflicts)
    data_rows = []
    counters.update({'immature_forward':0,'purged':0,'mission_unknown':0})
    for row in conflicts:
        ts = row['clock']; syms = row['symbols']; day, cutoff = daily_identity(ts)
        y = [forward_return(c15[s][0],ts) for s in syms]
        if any(v is None for v in y): counters['immature_forward'] += 1; continue
        y[0] -= .25
        available = max(ts+HORIZON,cutoff)
        part = cohort(ts,available,cuts)
        if part == 'purged': counters['purged'] += 1; continue
        ms = [mission(c15[s][0],ts,s,objective) for s in syms]
        known = all(v[2] for v in ms)
        if not known: counters['mission_unknown'] += 1
        data_rows.append(dict(row,returns=y,day=day,label_available_ms=available,cohort=part,
            leader=[v[0] for v in ms],capture=[v[1] for v in ms],mission_known=known))
    publish(output/'dataset.json',data_rows)
    cohorts = {k:[r for r in data_rows if r['cohort']==k] for k in ('train','validation','test')}
    print(json.dumps({'phase':'TRAIN','counts':{k:len(v) for k,v in cohorts.items()},'audit':counters}),flush=True)
    if min(len(v) for v in cohorts.values()) < 20:
        raise ValueError('insufficient dataset; immutable capture retained')
    def arrays(rows):
        return (np.asarray([r['features'] for r in rows]),np.asarray([r['returns'] for r in rows]))
    def pool(rows):
        x,y = arrays(rows)
        return Pool(x.reshape(-1,len(FEATURES)),relevance(y).ravel(),
                    group_id=np.repeat(np.arange(len(rows)),2),feature_names=list(FEATURES))
    model = CatBoostRanker(loss_function='YetiRank',iterations=400,depth=4,learning_rate=.03,
        l2_leaf_reg=10,random_seed=42,thread_count=2,allow_writing_files=False)
    model.fit(pool(cohorts['train']),eval_set=pool(cohorts['validation']),early_stopping_rounds=40,verbose=False)
    model.save_model(str(output/'ranker.cbm'))
    rows = cohorts['test']; x,y = arrays(rows)
    scores = model.predict(x.reshape(-1,len(FEATURES))).reshape(-1,2)
    reloaded = CatBoostRanker(); reloaded.load_model(str(output/'ranker.cbm'))
    check = np.linspace(0,len(x)*2-1,min(64,len(x)*2),dtype=int)
    np.testing.assert_array_equal(model.predict(x.reshape(-1,len(FEATURES))[check]),
                                  reloaded.predict(x.reshape(-1,len(FEATURES))[check]))
    leader = np.array([r['leader'] for r in rows],dtype=bool)
    capture = np.array([[np.nan if v is None else v for v in r['capture']] for r in rows])
    known = np.array([r['mission_known'] for r in rows],dtype=bool)
    days = [r['day'] for r in rows]
    result = evaluate(scores,y,days,leader,capture,known,
        population_complete=len(symbols)==len(m['requested_symbols']))
    prices = {s:closed_price_series(c15[s][0],bar_ms=900000,start_ms=start,end_ms=end) for s in symbols}
    alpha = evaluate_portfolio_alpha(trades,price_series_by_symbol=prices,benchmark_series=prices['BTCUSDT'],
        window_start_ms=start,window_end_ms=end,requested_days=(end-start)//86400000,
        universe=symbols,variant='current_rule_only_baseline',capacity=10,fee_bps=7.5,slippage_bps=5,
        source_hashes=sources)
    alpha['decision_grade']=False; alpha['evidence_grade']='diagnostic_current_rule_partial_population'
    result.update({'cohort_counts':{k:len(v) for k,v in cohorts.items()},'audit':counters,
        'tree_count':model.tree_count_,'feature_importance':dict(zip(FEATURES,map(float,
            model.get_feature_importance(type='PredictionValuesChange')))),
        'baseline':alpha,'baseline_candidates':candidate_count,'baseline_trade_count':len(trades),
        'period_ms':[start,end],'split_boundaries_ms':cuts,'source_hashes':sources,
        'limitations':['retrospective exposed archive, not fresh forward confirmation',
            '93 complete symbols of 105 requested; historical watchlist membership unavailable',
            'verified frozen historical champion' if trace else 'rule-only baseline; learned scores disabled',
            'historical policy trace is not complete current live agent parity',
            'blocked pair opportunity, not proof replacement is admissible under all existing guards',
            'selection return and leader proxies are not cash-account policy uplift',
            'event-time closed-candle assumption; receive-time and exchange fill records unavailable'],
        'native_reload_rows':len(check)})
    np.savez_compressed(output/'test_predictions.npz',scores=scores,returns=y,leader=leader,
        capture=capture,known=known,days=np.asarray(days),clocks=np.array([r['clock'] for r in rows]),features=x)
    verify()
    publish(output/'result.json',result)
    publish(output/'baseline_trades.json',trades)
    publish(output/'receipt.json',{p.name:sha(p) for p in output.iterdir() if p.is_file()})
    print(json.dumps({k:v for k,v in result.items() if k not in ('baseline','source_hashes')},indent=2),flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive',type=Path,required=True); p.add_argument('--output',type=Path,required=True)
    p.add_argument('--trace-bundle',type=Path)
    args=p.parse_args(); asyncio.run(run(args.archive,args.output,args.trace_bundle))
