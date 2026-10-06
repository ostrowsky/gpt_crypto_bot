"""Sequential mission re-evaluation of frozen hypotheses; no economic gate."""
from __future__ import annotations
import argparse,json,shutil,time
from pathlib import Path
import numpy as np
from historical_signal_evaluation import sha
import leader_mission_metrics as m

PARENTS={
    'turnover':'.runtime/turnover_economics_20261005_v1',
    'impulse':'.runtime/impulse_entry_catboost_20261006_v1',
    'joint':'.runtime/joint_direction_amplitude_20261006_v1',
    'exit':'.runtime/exit_action_advantage_20261006_v2',
    'capacity':'.runtime/capacity_catboost_20261005_v2',
    'execution':'.runtime/order_flow_execution_20261006_v2'}
ARMS=[('control','turnover','control'),('amplitude','turnover','amplitude_cost'),
    ('no_replacement','turnover','no_replacement'),('combined','turnover','combined'),
    ('impulse_catboost','impulse','catboost'),('joint','joint','joint'),('exit_model','exit','exit_model')]


def publish(path,value):
    with path.open('x',encoding='utf-8') as file:json.dump(value,file,indent=2,allow_nan=False)


def load_market(directory,manifest):
    market={}
    for symbol in manifest['eligible_symbols']:
        name=symbol+'_15m.json';p=directory/'market'/name
        if sha(p)!=manifest['input_hashes'][name]:raise ValueError('raw hash drift')
        rows=json.loads(p.read_bytes());a={k:np.array([r[k] for r in rows]) for k in ('t','o','h','l','c')}
        if not np.array_equal(a['t'],np.arange(manifest['archive_start_ms'],manifest['end_ms'],m.BAR)):raise ValueError('raw grid gap')
        if not all(np.isfinite(a[k]).all() and (a[k]>0).all() for k in ('o','h','l','c')):raise ValueError('invalid prices')
        market[symbol]=dict(time=a['t']+m.BAR,open=a['o'],high=a['h'],low=a['l'],close=a['c'])
    return market


def verify_parents():
    hashes={};results={}
    for name,value in PARENTS.items():
        path=Path(value);receipt=json.loads((path/'receipt.json').read_bytes())
        audit=json.loads((path/'independent_verification.json').read_bytes())
        if audit['status']!='PASS':raise ValueError('parent audit failed')
        for file,digest in receipt.items():
            if Path(file).name!=file or sha(path/file)!=digest:raise ValueError('parent artifact drift '+name+'/'+file)
        if audit.get('result_sha256') and audit['result_sha256']!=sha(path/'result.json'):raise ValueError('audit binding')
        reg=json.loads((path/'registration.json').read_bytes())
        sources=reg.get('sources',reg.get('source_hashes',{}))
        for file,digest in sources.items():
            if not (path/'source_snapshot'/file).exists() or sha(path/'source_snapshot'/file)!=digest:raise ValueError('source snapshot drift')
        hashes[name]=dict(path=str(path.resolve()),result=sha(path/'result.json'),receipt=sha(path/'receipt.json'),audit=sha(path/'independent_verification.json'))
        results[name]=json.loads((path/'result.json').read_bytes())
    control=sha(Path(PARENTS['turnover'])/'trades_control.json')
    for name in ('impulse','joint','exit'):
        if sha(Path(PARENTS[name])/'trades_control.json')!=control:raise ValueError('unmatched control policies')
    return hashes,results


def capacity_diagnostic(market,labels,output):
    path=Path(PARENTS['capacity']);rows=json.loads((path/'dataset.json').read_bytes());test=[r for r in rows if r['cohort']=='test']
    with np.load(path/'test_predictions.npz',allow_pickle=False) as z:scores=z['scores']
    if len(test)!=len(scores):raise ValueError('capacity count')
    choices=np.where(scores[:,0]>scores[:,1],0,1);known=[];records=[]
    for i,r in enumerate(test):
        day=m.day_key(r['clock']);at=r['clock'];options=[]
        for sym in r['symbols']:
            values=labels.get(day,{}).get('values',{});d=market.get(sym)
            if sym not in values or d is None or at>m.day_bounds(day)[1]:options.append(None);continue
            idx=np.searchsorted(d['time'],at)
            if idx>=len(d['time']) or d['time'][idx]!=at:options.append(None);continue
            dayrow=values[sym];rise=dayrow['close']-dayrow['open'];cap=(dayrow['close']-d['close'][idx])/rise if rise>0 else None
            options.append(dict(leader=sym in labels[day]['leaders'],capture=float(np.clip(cap,0,1.5)) if cap is not None else None))
        ready=all(o is not None for o in options);known.append(ready)
        records.append(dict(time=at,day=day,choice=int(choices[i]),known=ready,options=options))
    out=dict(scope='repeated capacity conflict options, not unique captured leaders or full portfolio replay',
        issued=len(test),known=sum(known),unknown=len(known)-sum(known),swaps=int((choices==0).sum()),runtime_eligible=False)
    for name in ('keep','model'):
        chosen=[r['options'][1 if name=='keep' else r['choice']] for r in records if r['known']]
        out[name]=dict(leader_n=sum(o['leader'] for o in chosen),leader_N=len(chosen),
            early_n=sum(o['leader'] and o['capture'] is not None and o['capture']>=.35 for o in chosen),early_N=len(chosen))
    out['mission_verdict']='NO_PROXY_LEADER_GAIN' if out['model']['leader_n']<=out['keep']['leader_n'] and out['model']['early_n']<=out['keep']['early_n'] else 'PROXY_CHANGE_REQUIRES_FULL_POLICY_REPLAY'
    publish(output/'capacity_records.json',records);return out


def run(market_dir,output):
    output.mkdir(parents=True,exist_ok=False);manifest=json.loads((market_dir/'manifest.json').read_bytes())
    sources={name:sha(Path(__file__).with_name(name)) for name in ('leader_mission_metrics.py','run_leader_mission_reaudit.py')}
    snapshot=output/'source_snapshot';snapshot.mkdir()
    for name in sources:shutil.copy2(Path(__file__).with_name(name),snapshot/name)
    hashes,parents=verify_parents();spec=Path('docs/specs/leader-mission-reaudit.md');shutil.copy2(spec,output/'registered_spec.md')
    start,end=manifest['start_ms'],manifest['end_ms'];boundary=parents['turnover']['split_boundaries_ms'][1]
    registration=dict(registered_at=time.time(),source_hashes=sources,spec_sha256=sha(spec),market=str(market_dir.resolve()),
        market_manifest_sha256=sha(market_dir/'manifest.json'),parents=hashes,start_ms=start,end_ms=end,test_boundary_ms=boundary,
        criterion='early detection, leader coverage, selection and causal weakening accompaniment',economic_gate=False,
        retrained=False,exposed_retrospective=True,runtime_eligible=False)
    publish(output/'registration.json',registration);print('REGISTERED mission-only re-audit',flush=True)
    market=load_market(market_dir,manifest);labels,days=m.daily_labels(market,start,end)
    publish(output/'daily_labels.json',labels);features={s:m.indicators(d) for s,d in market.items()}
    results={};records={};daily={};episodes={}
    for arm,parent,trade_name in ARMS:
        path=Path(PARENTS[parent]);trades=json.loads((path/f'trades_{trade_name}.json').read_bytes())
        full,full_days,leader_records=m.mission(trades,labels)
        test,test_days,test_records=m.mission(trades,labels,boundary)
        if full['captured']!=parents[parent]['missions'][trade_name]['captured_pair_count'] or full['precision_n']!=parents[parent]['missions'][trade_name]['objective_trade_count'] or full['precision_N']!=parents[parent]['missions'][trade_name]['eligible_trade_count']:
            raise ValueError('leader/precision discrepancy outside registered early correction')
        if full['stored_early']!=parents[parent]['missions'][trade_name]['early_pair_count']:raise ValueError('stored annotation reconstruction')
        by_symbol={s:[t for t in trades if t['sym']==s] for s in market}
        ep=[m.episode(r,market[r['symbol']],features[r['symbol']],by_symbol[r['symbol']]) for r in leader_records]
        test_ep=[r for r in ep if m.day_bounds(r['day'])[0]>=boundary]
        results[arm]=dict(full=full,test=test,exits_full=m.exit_summary(ep),exits_test=m.exit_summary(test_ep))
        records[arm]=leader_records;daily[arm]=dict(full=full_days,test=test_days);episodes[arm]=ep
        publish(output/f'episodes_{arm}.json',ep);publish(output/f'leader_entries_{arm}.json',leader_records)
        print('MISSION',arm,json.dumps(dict(full=full,test=test,exits=results[arm]['exits_test'])),flush=True)
    comparisons={}
    for arm in results:
        if arm=='control':continue
        ci=m.paired_intervals(daily['control']['test'],daily[arm]['test'])
        exits=m.compare_exits([r for r in episodes['control'] if m.day_bounds(r['day'])[0]>=boundary],
            [r for r in episodes[arm] if m.day_bounds(r['day'])[0]>=boundary])
        comparisons[arm]=dict(entry=ci,exits=exits,mission_verdict=m.verdict(results['control']['test'],results[arm]['test'],ci,exits))
    capacity=capacity_diagnostic(market,labels,output)
    execution=parents['execution']['actual_orders']
    execution=dict(scope='execution timing, not discovery/selection/leader exit policy',issued=execution['issued'],
        joined=execution['states']['JOINED_PERPETUAL_PROXY'],actual_deferrals={h:r['deferred'] for h,r in execution['scores'].items()},
        mission_verdict='NO_CHANGED_JOINED_LEADER_DECISIONS_NO_PORTFOLIO_PROOF',runtime_eligible=False)
    result=dict(status='COMPLETED_MISSION_REAUDIT',runtime_eligible=False,start_ms=start,end_ms=end,
        population_complete=len(market),population_requested=len(manifest['requested_symbols']),eligible_days=len(days),
        order=['capacity','amplitude/no_replacement/combined','impulse_catboost','joint','exit_model','execution'],
        arms=results,comparisons=comparisons,capacity=capacity,execution=execution,
        manual_truth_findings=[dict(id='TH-03/TH-05',status='FAIL_HISTORICAL_EARLY_ANNOTATION_ALIGNMENT',
            impact='old early counts use timeframe open clocks/final prices inconsistent with common22:00 close; corrected in this evaluator')],
        limitations=['retrospective mission re-score, no retraining or sealed confirmation','93/105 availability-selected population',
            'hypotheses retain original return-based training targets; this does not test newly trained leader-target models',
            'trend-end marker operational, not ultimate top; risk exits can correctly precede it',
            'capacity proxy and perpetual quote tasks are not full-policy leader outcomes','no certified live/PIT/receive-time parity'])
    publish(output/'daily_metrics.json',daily);publish(output/'result.json',result)
    for name,digest in sources.items():
        if sha(Path(__file__).with_name(name))!=digest:raise ValueError('source changed')
    check,_=verify_parents()
    if check!=hashes or sha(market_dir/'manifest.json')!=registration['market_manifest_sha256']:raise ValueError('input changed')
    publish(output/'receipt.json',{p.name:sha(p) for p in output.iterdir() if p.is_file()});print('COMPLETE mission re-audit',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--market',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.market,a.output)
