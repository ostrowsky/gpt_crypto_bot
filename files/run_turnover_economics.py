"""Registered offline full-policy entry/replacement economics experiment."""
import argparse
import asyncio
from dataclasses import asdict
from datetime import datetime,time,timedelta,timezone
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np

import config
import replay_backtest as rb
from audit_negative_day_rebound import finalize_at_boundary
from capacity_catboost import split_boundaries,TZ
from compare_price_volatility_bot import read_feature_pack,reuse_features,reuse_candidate_snapshot,replacement_options
from historical_signal_evaluation import freeze,sha
from portfolio_alpha import _simulate_account,_benchmark_result,closed_price_series
from research_rocket_capture import policy
from turnover_economics import (BAR,ARMS,screened_snapshot,replacement_enabled,cash_ledger,
    grouped_attribution,acceptance)

ROOT=Path(__file__).resolve().parent.parent
SOURCES=('run_turnover_economics.py','turnover_economics.py','compare_price_volatility_bot.py',
    'replay_backtest.py','portfolio_alpha.py','config.py','indicators.py','strategy.py','monitor.py',
    'research_rocket_capture.py','audit_negative_day_rebound.py','capacity_catboost.py')


def publish(path,obj):freeze(path,json.dumps(obj,indent=2,allow_nan=False).encode())


def objective(trades,cache,symbols,start,end):
    daily=rb._daily_top_objective(symbols,cache,start_ms=start,end_ms=end,top_n=15)
    result=rb._make_report(label='economics',start=datetime.fromtimestamp(start/1000,timezone.utc),
        end=datetime.fromtimestamp(end/1000,timezone.utc),symbols=symbols,timeframes=['15m','1h'],
        trades=trades,run_stats=rb.ReplayRunStats(),daily_objective=daily)
    return result['objective']


def paired_test_interval(control,candidate,boundary,end,draws=5000):
    base=dict(control);target=dict(candidate)
    first=datetime.fromtimestamp(boundary/1000,timezone.utc).astimezone(TZ).date()
    last=datetime.fromtimestamp(end/1000,timezone.utc).astimezone(TZ).date()
    days=[];deltas=[];day=first
    while day<last:
        a=int(datetime.combine(day,time(),tzinfo=TZ).timestamp()*1000)
        b=int(datetime.combine(day+timedelta(days=1),time(),tzinfo=TZ).timestamp()*1000)
        if boundary<=a and b<=end and a in base and b in base and a in target and b in target:
            days.append(day.isoformat());deltas.append(np.log(target[b]/target[a])-np.log(base[b]/base[a]))
        day+=timedelta(days=1)
    if len(days)<3:return {'days':len(days),'corrected_interval':None}
    rng=np.random.default_rng(42);starts=rng.integers(0,len(days)-2,size=(draws,(len(days)+2)//3))
    indices=(starts[:,:,None]+np.arange(3)).reshape(draws,-1)[:,:len(days)]
    values=np.asarray(deltas)[indices].mean(axis=1)*10000
    q=.05/(2*3)
    return {'days':len(days),'mean_daily_log_uplift_bp':float(np.mean(deltas)*10000),
            'corrected_interval':[float(v) for v in np.quantile(values,[q,1-q])],
            'confidence':1-.05/3,'block_calendar_days':3,'draws':draws}


def validate_data(cache,m):
    for sym in m['eligible_symbols']:
        for tf in ('15m','1h'):
            d=cache[sym,tf][0];step=rb.BAR_MS[tf]
            if not np.array_equal(d['t'],np.arange(m['archive_start_ms'],m['end_ms'],step)):
                raise ValueError('incomplete market grid')
            a=np.column_stack([d[k] for k in ('o','h','l','c','v')])
            if (not np.isfinite(a).all() or (a[:,:4]<=0).any() or (a[:,4]<0).any()
                or (a[:,1]<np.maximum(a[:,0],a[:,3])).any()
                or (a[:,2]>np.minimum(a[:,0],a[:,3])).any() or (a[:,1]<a[:,2]).any()):
                raise ValueError('invalid OHLCV')


async def run(market,features,candidates,output):
    for directory in (market,features,candidates):
        if not directory.resolve().is_relative_to(ROOT/'.runtime'):
            raise ValueError('reuse restricted to trusted local runtime checkpoints')
    manifest_raw=(market/'manifest.json').read_bytes();m=json.loads(manifest_raw)
    output.mkdir(parents=True,exist_ok=False);source_dir=output/'source_snapshot';source_dir.mkdir()
    sources={}
    for name in SOURCES:
        p=Path(__file__).with_name(name);freeze(source_dir/name,p.read_bytes());sources[name]=sha(p)
    start,end=m['start_ms'],m['end_ms'];cuts=split_boundaries(start,end)
    fee=max(7.5,float(config.PAPER_FEE_BPS));slip=5.
    publish(output/'registration.json',{'contract':'turnover-economics-replay-v1','arms':ARMS,
        'start_ms':start,'end_ms':end,'sources':sources,'market_manifest_sha256':sha(market/'manifest.json'),
        'feature_hashes_sha256':sha(features/'hashes.json'),
        'candidate_receipt_sha256':sha(candidates/'candidate_checkpoint_receipt.json'),
        'fee_bps':fee,'slippage_bps':slip,'split_boundaries_ms':cuts,'runtime_eligible':False,
        'population_complete':len(m['eligible_symbols']),'population_requested':len(m['requested_symbols'])})
    def verify():
        if (market/'manifest.json').read_bytes()!=manifest_raw:raise ValueError('market manifest drift')
        for name,expected in sources.items():
            if sha(Path(__file__).with_name(name))!=expected:raise ValueError('source drift: '+name)
        for name,expected in m['input_hashes'].items():
            if Path(name).name!=name or sha(market/'market'/name)!=expected:raise ValueError('market SHA drift')
    verify()
    print('Loading verified native feature checkpoints',flush=True)
    cache,unused_frames,hashes=reuse_features(features,m,output);del unused_frames
    validate_data(cache,m)
    symbols=m['eligible_symbols'];c15={s:cache[s,'15m'] for s in symbols};c4={s:cache[s,'4h'] for s in symbols}
    context=rb._build_bull_day_context(cache['BTCUSDT','1h'][0])
    with policy('baseline'),patch.object(rb,'_load_temporal_scout_events',return_value=({},{})):
        snapshot=reuse_candidate_snapshot(candidates,m,output)
        # Current serial kernel regeneration validates all fields for fixed assets.
        spot=[]
        for sym,tf in (('BTCUSDT','15m'),('ETHUSDT','1h')):
            data,feat=cache[sym,tf]
            rows=await rb._build_candidates_for_symbol(sym,tf,data,feat,c15,c4,context,variant='score_replace_cluster',include_trend_start=False)
            expected=[asdict(c) for at in sorted(snapshot[0]) for c in snapshot[0][at] if c.sym==sym and c.tf==tf]
            actual=[asdict(c) for c in rows if start<=c.ts_ms<end]
            if actual!=expected:raise ValueError('serial candidate kernel equivalence failed')
            spot.append({'symbol':sym,'tf':tf,'candidates':len(actual)})
        publish(output/'candidate_spot_verification.json',{'status':'PASS','pairs':spot})
        filtered,audit=screened_snapshot(snapshot,cache,fee,slip)
        publish(output/'filter_audit.json',audit)
        trades_by_arm={};stats={};enabled,delta=replacement_options(config)
        for name in ARMS:
            stream=filtered if name in ('amplitude_cost','combined') else snapshot
            print(json.dumps({'phase':'SIMULATE','arm':name,'candidates':stream[2]}),flush=True)
            trades,stat=await rb.simulate_portfolio(symbols,['15m','1h'],cache,c15,c4,context,
                max_open_positions=10,enable_replacement=replacement_enabled(name,enabled),
                replace_min_delta=delta,variant='score_replace_cluster',
                top_gainer_score_min=float(config.TOP_GAINER_SCORE_GATE_MIN_SCORE),candidate_snapshot=stream)
            finalize_at_boundary(trades,cache,end);trades_by_arm[name]=trades;stats[name]=asdict(stat)
            publish(output/f'trades_{name}.json',[asdict(t) for t in trades])
    series={s:closed_price_series(c15[s][0],bar_ms=BAR,start_ms=start,end_ms=end) for s in symbols}
    grid=list(range(start,end+1,BAR));accounts={};missions={};test_missions={};stress={};attribution={}
    benchmark=_benchmark_result(series['BTCUSDT'],initial_capital=10000,fee_bps=fee,slippage_bps=slip)
    for name,trades in trades_by_arm.items():
        print(json.dumps({'phase':'ACCOUNT','arm':name,'trades':len(trades)}),flush=True)
        account=cash_ledger(trades,series,grid,fee,slip)
        ref=_simulate_account(trades,price_series_by_symbol=series,valuation_timestamps=grid,
            capacity=10,initial_capital=10000,fee_bps=fee,slippage_bps=slip)
        if ref.violations or ref.fully_valued_points!=len(grid):raise ValueError('canonical cash/coverage violation')
        np.testing.assert_allclose(account['curve'],ref.equity_curve,rtol=1e-10,atol=1e-7)
        for k,v in (('fees',ref.fees_quote),('slippage',ref.slippage_quote)):
            np.testing.assert_allclose(account['totals'][k],v,rtol=1e-10,atol=1e-7)
        curves=dict(account['curve']);test_return=100*(curves[end]/curves[cuts[1]]-1)
        ledger=account.pop('ledger');publish(output/f'ledger_{name}.json',ledger)
        accounts[name]={**account,'test_return_pct':test_return,'trades':len(trades),
            'alpha_pp':account['net_return_pct']-benchmark['net_return_after_costs_pct'],
            'replacements':stats[name]['replacements_total']}
        missions[name]=objective(trades,cache,symbols,start,end)
        # Existing positions at TEST origin are not relabeled as new TEST buys.
        test_trades=[t for t in trades if t.entry_ts>=cuts[1]]
        test_missions[name]=objective(test_trades,cache,symbols,cuts[1],end)
        for row in ledger:
            row['exit_class']='REPLACEMENT' if row['exit_reason'].startswith('replaced_by_') else 'ATR_TRAIL' if 'ATR trail' in row['exit_reason'] else row['exit_reason'].split('(')[0].strip()
            row['entry_cohort']='discovery' if row['entry_ts']<cuts[0] else 'validation' if row['entry_ts']<cuts[1] else 'test'
        attribution[name]={k:grouped_attribution(ledger,k) for k in ('mode','tf','exit_class','entry_cohort')}
        stress_account=cash_ledger(trades,series,grid,2*fee,2*slip)
        stress[name]={k:v for k,v in stress_account.items() if k not in ('curve','ledger')}
    comparisons={}
    for name in ARMS[1:]:
        ci=paired_test_interval(accounts['control']['curve'],accounts[name]['curve'],cuts[1],end)
        comparisons[name]={**acceptance(accounts['control'],accounts[name],(missions['control'],missions[name]),
            test_missions['control'],test_missions[name],ci['corrected_interval'],ci['days']),**ci}
    verify()
    for s,h in hashes.items():
        if sha(features/f'{s}.npz')!=h:raise ValueError('feature drift')
    result={'status':'COMPLETED_RETROSPECTIVE_DIAGNOSTIC','runtime_eligible':False,'achievement_claimed':False,
        'sources':sources,'start_ms':start,'end_ms':end,'days':(end-start)/86400000,
        'population_complete':len(symbols),'population_requested':len(m['requested_symbols']),
        'costs':{'fee_bps':fee,'slippage_bps':slip},'benchmark':benchmark,'accounts':accounts,
        'missions':missions,'test_missions':test_missions,'comparisons':comparisons,'cost_stress':stress,
        'attribution':attribution,'filter_audit':audit,'split_boundaries_ms':cuts,
        'limitations':['fixed recovered 93/105 symbols; historical PIT universe not certified',
            'rule-only replay with mutable learned scores and temporal logs disabled',
            'closed-bar idealized fills and static costs; not recorded exchange executions',
            'previously exposed retrospective history; no fresh forward confirmation',
            'TEST account carries holdings and policy-dependent equity from earlier cohorts',
            'per-trade attribution groups by entry cohort; not period-realized cash PnL']}
    publish(output/'result.json',result)
    publish(output/'receipt.json',{p.name:sha(p) for p in output.iterdir() if p.is_file()})
    print(json.dumps({n:{k:a[k] for k in ('net_return_pct','test_return_pct','trades','max_drawdown_pct','alpha_pp')} for n,a in accounts.items()},indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('market','features','candidates','output'):p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();asyncio.run(run(a.market,a.features,a.candidates,a.output))
