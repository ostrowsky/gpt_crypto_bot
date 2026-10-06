"""Registered maximum-history causal leader accompaniment policy experiment."""
from __future__ import annotations
import argparse,asyncio,json,shutil,time
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch
import numpy as np
import config,replay_backtest as rb
import leader_mission_metrics as m
from leader_trend_continuation import TrendPolicy,diagnostics,paired_accompaniment,verdict
from run_leader_mission_reaudit import load_market
from run_impulse_entry_catboost import SOURCES as PARENT_SOURCES
from run_joint_direction_amplitude import reuse_parent
from run_turnover_economics import publish,validate_data
from compare_price_volatility_bot import reuse_features,reuse_candidate_snapshot,replacement_options
from historical_signal_evaluation import sha,freeze
from research_rocket_capture import policy
from audit_negative_day_rebound import finalize_at_boundary
from portfolio_alpha import closed_price_series,_simulate_account
from turnover_economics import cash_ledger

ROOT=Path(__file__).resolve().parent.parent
SOURCES=tuple(dict.fromkeys(('leader_trend_continuation.py','run_leader_trend_continuation.py',
    'leader_mission_metrics.py','run_leader_mission_reaudit.py','run_joint_direction_amplitude.py')+PARENT_SOURCES))


def skip_record(trade,candidate):
    return dict(symbol=candidate.sym,tf=candidate.tf,mode=candidate.mode,at=candidate.ts_ms,
        price=candidate.price,score=candidate.top_gainer_score,
        previous_entry=trade.entry_ts if trade else None,previous_exit=trade.exit_ts if trade else None,
        previous_reason=trade.exit_reason if trade else None,previous_mode=trade.mode if trade else None,
        previous_tf=trade.tf if trade else None,
        cooldown_until=trade.exit_ts+rb._cooldown_bars_after_exit(trade.mode,trade.exit_reason)*rb.BAR_MS[trade.tf] if trade else None)


async def run(parent,output):
    if not parent.resolve().is_relative_to(ROOT/'.runtime'):raise ValueError('trusted parent only')
    old=json.loads((parent/'registration.json').read_bytes());market_path=Path(old['market'])
    features_path=Path(old['features']);candidate_path=Path(old['candidates'])
    output.mkdir(parents=True,exist_ok=False);(output/'source_snapshot').mkdir()
    sources={n:sha(Path(__file__).with_name(n)) for n in SOURCES}
    for n in sources:shutil.copy2(Path(__file__).with_name(n),output/'source_snapshot'/n)
    spec=ROOT/'docs/specs/leader-trend-continuation.md';shutil.copy2(spec,output/'registered_spec.md')
    manifest=json.loads((market_path/'manifest.json').read_bytes());start,end=manifest['start_ms'],manifest['end_ms']
    prior=json.loads((parent/'result.json').read_bytes());boundary=prior['split_boundaries_ms'][1]
    fee=max(7.5,float(config.PAPER_FEE_BPS));slip=5.
    registration=dict(contract='leader-trend-continuation-v1',registered_at=time.time(),sources=sources,
        spec_sha256=sha(output/'registered_spec.md'),market=str(market_path.resolve()),
        market_manifest_sha256=sha(market_path/'manifest.json'),features=str(features_path.resolve()),
        candidates=str(candidate_path.resolve()),parent=str(parent.resolve()),parent_result_sha256=sha(parent/'result.json'),
        start_ms=start,end_ms=end,test_boundary_ms=boundary,fee_bps=fee,slippage_bps=slip,
        runtime_eligible=False,exposed_retrospective=True,rule=dict(ema=20,atr=14,entry_rise_atr=1,drawdown_atr=1,max_delay_minutes=60))
    publish(output/'registration.json',registration);print('REGISTERED one fixed trend rule',flush=True)
    reuse_parent(parent,market_path,output)
    cache,frames,hashes=reuse_features(features_path,manifest,output);del frames;validate_data(cache,manifest)
    market=load_market(market_path,manifest);features={s:m.indicators(d) for s,d in market.items()}
    labels,days=m.daily_labels(market,start,end);publish(output/'daily_labels.json',labels)
    inherited=json.loads((output/'trades_control.json').read_bytes())
    symbols=manifest['eligible_symbols'];c15={s:cache[s,'15m'] for s in symbols};c4={s:cache[s,'4h'] for s in symbols}
    snapshot=reuse_candidate_snapshot(candidate_path,manifest,output)
    context=rb._build_bull_day_context(cache['BTCUSDT','1h'][0]);progress=rb._update_trade_progress;record_skip=rb._record_cooldown_skip
    enabled,delta=replacement_options(config);all_trades={};all_skips={};stats={}
    with policy('baseline'),patch.object(rb,'_load_temporal_scout_events',return_value=({},{})):
        for name in ('control','trend'):
            skips=[]
            def recorder(trade,candidate,cache,stat):
                skips.append(skip_record(trade,candidate));return record_skip(trade,candidate,cache,stat)
            wrapper=TrendPolicy(progress,market,features) if name=='trend' else None
            print('SIMULATE '+name,flush=True)
            with patch.object(rb,'_record_cooldown_skip',side_effect=recorder),patch.object(rb,'_update_trade_progress',side_effect=wrapper or progress):
                trades,stat=await rb.simulate_portfolio(symbols,['15m','1h'],cache,c15,c4,context,
                    max_open_positions=10,enable_replacement=enabled,replace_min_delta=delta,
                    variant='score_replace_cluster',top_gainer_score_min=float(config.TOP_GAINER_SCORE_GATE_MIN_SCORE),candidate_snapshot=snapshot)
            finalize_at_boundary(trades,cache,end);rows=[asdict(t) for t in trades]
            if name=='control':
                if rows!=inherited:raise ValueError('control differs from pinned prior full trades')
                publish(output/'control_parity.json',dict(status='PASS_ALL_FIELDS',trades=len(rows),parent_sha256=sha(output/'trades_control.json')))
            else:
                publish(output/'trades_trend.json',rows)
                publish(output/'policy_trace.json',dict(decisions=wrapper.decisions,ticks=wrapper.ticks,remaining_active=len(wrapper.active),
                    scope='remaining state can correspond to replacement/boundary; live positions define effective exit'))
            publish(output/('cooldown_'+name+'.json'),skips);all_skips[name]=skips;all_trades[name]=trades;stats[name]=asdict(stat)
            print(json.dumps(dict(arm=name,trades=len(rows),cooldown_skips=len(skips),deferrals=sum(r['defer'] for r in wrapper.decisions) if wrapper else 0)),flush=True)
    arms={};daily={};episodes={};causes={};accounts={}
    series={s:closed_price_series(c15[s][0],bar_ms=m.BAR,start_ms=start,end_ms=end) for s in symbols};grid=list(range(start,end+1,m.BAR))
    for name,trades in all_trades.items():
        rows=[asdict(t) for t in trades];full,fd,entries=m.mission(rows,labels);test,td,_=m.mission(rows,labels,boundary)
        by_symbol={s:[t for t in rows if t['sym']==s] for s in symbols}
        ep=[m.episode(e,market[e['symbol']],features[e['symbol']],by_symbol[e['symbol']]) for e in entries]
        tep=[e for e in ep if m.day_bounds(e['day'])[0]>=boundary]
        arms[name]=dict(full=full,test=test,exits_full=m.exit_summary(ep),exits_test=m.exit_summary(tep))
        daily[name]=dict(full=fd,test=td);episodes[name]=ep
        causes[name]=dict(full=diagnostics(ep,all_skips[name]),test=diagnostics(ep,all_skips[name],boundary))
        publish(output/f'leader_entries_{name}.json',entries);publish(output/f'episodes_{name}.json',ep)
        a=cash_ledger(trades,series,grid,fee,slip);ledger=a.pop('ledger');publish(output/f'ledger_{name}.json',ledger)
        ref=_simulate_account(trades,price_series_by_symbol=series,valuation_timestamps=grid,capacity=10,initial_capital=10000,fee_bps=fee,slippage_bps=slip)
        if ref.violations or ref.fully_valued_points!=len(grid):raise ValueError('account violation')
        np.testing.assert_allclose(a['curve'],ref.equity_curve,rtol=1e-10,atol=1e-7)
        np.testing.assert_allclose([a['totals']['fees'],a['totals']['slippage']],[ref.fees_quote,ref.slippage_quote],rtol=1e-10,atol=1e-7)
        if name=='control':np.testing.assert_allclose(a['curve'],prior['accounts']['control']['curve'],rtol=0,atol=1e-9)
        curve=dict(a['curve']);accounts[name]={**a,'test_return_pct':100*(curve[end]/curve[boundary]-1),'trades':len(rows)}
        print('METRICS '+name+' '+json.dumps(arms[name]),flush=True)
    base,target=([e for e in episodes[k] if m.day_bounds(e['day'])[0]>=boundary] for k in ('control','trend'))
    comparison=dict(entry=m.paired_intervals(daily['control']['test'],daily['trend']['test']),
        exits=m.compare_exits(base,target),accompaniment=paired_accompaniment(base,target))
    decision=verdict(arms,comparison)
    result=dict(status='COMPLETED_RETROSPECTIVE_TREND_RULE',runtime_eligible=False,achievement_claimed=False,
        start_ms=start,end_ms=end,population_complete=len(symbols),population_requested=len(manifest['requested_symbols']),
        eligible_days=len(days),arms=arms,comparison=comparison,verdict=decision,causes=causes,accounts=accounts,stats=stats,
        policy_counts=dict(first_soft_decisions=len(wrapper.decisions),deferrals=sum(r['defer'] for r in wrapper.decisions),active_ticks=len(wrapper.ticks)),
        limitations=['TEST exposed; fixed rule motivated by previous retrospective diagnostics','availability-selected93/105 symbols',
            '15m closed-bar decisions, idealized fills, no PIT/receive/live parity','operational marker not ultimate future peak; early hard exits may be correct',
            'cooldown blocker events do not prove feasible successful counterfactual entry','old stored early annotations misaligned; corrected raw labels used'])
    for n,h in sources.items():
        if sha(Path(__file__).with_name(n))!=h:raise ValueError('source drift')
    if sha(market_path/'manifest.json')!=registration['market_manifest_sha256']:raise ValueError('manifest drift')
    for n,h in manifest['input_hashes'].items():
        if sha(market_path/'market'/n)!=h:raise ValueError('market drift')
    for s,h in hashes.items():
        if sha(features_path/(s+'.npz'))!=h:raise ValueError('feature drift')
    publish(output/'daily_metrics.json',daily);publish(output/'result.json',result)
    publish(output/'receipt.json',{p.name:sha(p) for p in output.iterdir() if p.is_file()})
    print('COMPLETE '+json.dumps(dict(verdict=decision,comparison=comparison)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--parent',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();asyncio.run(run(a.parent,a.output))
