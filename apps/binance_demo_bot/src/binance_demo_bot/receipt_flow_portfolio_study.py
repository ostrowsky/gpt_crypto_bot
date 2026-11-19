"""RFP-1 fixed, source-bound execution comparison. Offline conditional portfolios."""
import argparse
from dataclasses import asdict
from decimal import Decimal as D
import gzip
import html
import json
from pathlib import Path
import platform
import time
from . import receipt_flow_direction as io
from .receipt_flow_portfolio import Policy,simulate,fit_microprice,digest
from .receipt_flow_portfolio_audit import audit_events
from .receipt_flow_quotes import ReceiptQuotes

PROFILES={'base':dict(fee_bps='10',slippage_bps='5',latency_ms=1000),
          'stress':dict(fee_bps='20',slippage_bps='10',latency_ms=2000)}


def case_definitions():
    result=[dict(case_id='CASH',arm='CASH',horizon_seconds=0,admission='cash',profile='neutral')]
    for arm in ('M0','M1','P','B','F'):
        for h in (5,10,30):
            for admission in ('direction','cost-aware'):
                for profile in ('base','stress'):
                    result.append(dict(case_id=f'{arm}-h{h}-{admission}-{profile}',arm=arm,
                        horizon_seconds=h,admission=admission,profile=profile))
    return result


def signal_rows(rows,*,arm,horizon,coefficients):
    result=[]
    for r in rows:
        if arm=='M0':score=r['micro_bps']
        elif arm=='M1':
            c=coefficients[str(horizon)];score=c['intercept']+c['slope']*r['micro_bps']
        else:score=r['predictions'][f'{arm}_{horizon}']
        row=dict(at_ms=r['at_ms'],feature_end_ms=r.get('feature_end_ms',r['at_ms']-1),
            symbol=r['symbol'],score_bps=float(score),spread_bps=r['spread_bps'],
            source_row=r['source_row'],feature_sha256=r['features_sha256'])
        if 'origin_mid' in r:row['origin_mid']=r['origin_mid']
        result.append(row)
    return result


def _events(path):
    with gzip.open(path,'rt',encoding='utf-8') as stream:
        for line in stream:yield json.loads(line)


def paired_daily(left,right,*,test_start_ms):
    a={r['day']:r for r in left if r['start_ms']>=test_start_ms and r['complete']}
    b={r['day']:r for r in right if r['start_ms']>=test_start_ms and r['complete']}
    dates=sorted(set(a)&set(b));delta=[D(a[d]['pnl_usdt'])-D(b[d]['pnl_usdt']) for d in dates]
    return dict(common_days=dates,paired_days_n=len(dates),
                mean_delta_usdt=str(sum(delta,D(0))/len(delta)) if delta else None,
                interpretation='CONDITIONAL_POINT_ONLY_NO_SIGNIFICANCE')


def _summary(result,case,test_start):
    full=[r for r in result['daily'] if r['full_calendar_day']]
    test=[r for r in full if r['start_ms']>=test_start]
    good=[D(r['pnl_usdt']) for r in test if r['complete']]
    delta=[D(r['boundary_delta_usdt']) for r in test if r['boundary_delta_usdt'] is not None]
    return dict(case,net_interval_usdt=result['net_interval_usdt'],
        test_boundary_net_diagnostic_usdt=str(sum(delta,D(0))) if len(delta)==len(test) else None,
        test_mean_daily_net_pnl_usdt=str(sum(good,D(0))/len(good)) if good else None,
        test_complete_days_n=len(good),test_requested_days_N=len(test),test_profitable_days_n=sum(v>0 for v in good),
        oos_complete_days_n=result['complete_days_n'],oos_requested_days_N=result['requested_days_N'],
        entries_n=result['entries_n'],closed_cycles_n=result['closed_cycles_n'],fees_usdt=result['fees_usdt'],
        turnover_usdt=result['turnover_usdt'],drawdown_observed_usdt=result['drawdown_observed_usdt'],
        unmarked_held_ticks_n=result['unmarked_held_ticks_n'],open_exposure=result['position'],
        event_counts=result['event_counts'])


def _render(folder,report,results,first_trades,quotes):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import math
    from datetime import datetime,timezone
    plots=folder/'plots';plots.mkdir()
    colors=dict(M0='#888888',M1='#844aa2',P='#3975b9',B='#e68a20',F='#239461')
    fig,axes=plt.subplots(2,3,figsize=(15,8),sharex=True)
    for i,profile in enumerate(('base','stress')):
        for j,h in enumerate((5,10,30)):
            ax=axes[i,j]
            for arm,color in colors.items():
                key=f'{arm}-h{h}-cost-aware-{profile}';daily=results[key]['daily']
                ys=[float(r['pnl_usdt']) if r['complete'] else math.nan for r in daily]
                ax.plot([r['day'][5:] for r in daily],ys,'o-',label=arm,color=color,alpha=.8)
            ax.axhline(0,color='black',linestyle=':',label='CASH');ax.grid(alpha=.2)
            ax.set_title(f'{profile} | hold {h}s | cost-aware');ax.set_ylabel('Complete-day net USDT')
            if i==0 and j==0:ax.legend(fontsize=8)
    fig.suptitle('RFP-1 | One shared100USDT portfolio per case | Nov9-12, 2026 Budapest\nConditional simulation, exposed TEST Nov11-12; gaps stay gaps, no actual PnL certification')
    fig.tight_layout();fig.savefig(plots/'daily-pnl.png',dpi=110);plt.close(fig)
    trade_images={}
    for key,trade in first_trades.items():
        buy,sell=trade['buy'],trade.get('sell');origin=buy['origin_ms'];symbol=buy['symbol']
        h=next(c['horizon_seconds'] for c in report['cases'] if c['case_id']==key)
        at_end=min(report['end_ms'],max(origin+(h+15)*1000,(sell['at_ms']+5000) if sell else origin+(h+15)*1000))
        at_end=min(at_end,origin+300000)
        clocks=range(origin-60000,at_end+1,1000);values=[];offsets=[]
        for t in clocks:
            q=quotes.quote_at(symbol,t);offsets.append((t-origin)/1000)
            values.append(float((D(q['bid'])+D(q['ask']))/2) if q else math.nan)
        fig,ax=plt.subplots(figsize=(11,5));before=[i for i,x in enumerate(offsets) if x<=0];after=[i for i,x in enumerate(offsets) if x>=0]
        ax.plot([offsets[i] for i in before],[values[i] for i in before],color='black',label='History before issue')
        ax.plot([offsets[i] for i in after],[values[i] for i in after],color='black',linestyle=':',label='Later actual receipt mid')
        ax.scatter([(buy['at_ms']-origin)/1000],[float(buy['price'])],marker='^',color='#239461',s=75,label='Sim BUY ask+slip')
        if sell and sell['at_ms']<=at_end:ax.scatter([(sell['at_ms']-origin)/1000],[float(sell['price'])],marker='v',color='#c13e3e',s=75,label='Sim SELL bid-slip')
        selected=trade['selected']
        if 'origin_mid' in selected:
            target=selected['origin_mid']*math.exp(selected['score_bps']/10000)
            ax.plot([0,h],[selected['origin_mid'],target],'D--',color='#844aa2',label='Frozen h-return anchor (connector only)')
        ax.axvline(0,color='#777');ax.set_title(key+' | '+symbol+' | '+datetime.fromtimestamp(origin/1000,timezone.utc).isoformat()+'\nFirst chronological fill; conditional execution, no best-trade selection')
        ax.set_xlabel('Seconds from frozen decision');ax.set_ylabel('USDT');ax.grid(alpha=.2);ax.legend(fontsize=8)
        fig.tight_layout();name=key+'.png';fig.savefig(plots/name,dpi=100);plt.close(fig);trade_images[key]=name
    table=[]
    for c in report['cases']:
        def fmt(v):return 'unknown' if v is None else f'{D(v):.6f}'
        table.append('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in
            (c['case_id'],fmt(c['net_interval_usdt']),fmt(c['test_mean_daily_net_pnl_usdt']),
             f"{c['test_complete_days_n']}/{c['test_requested_days_N']}",c['closed_cycles_n'],fmt(c['fees_usdt']),
             c['unmarked_held_ticks_n'],bool(c['open_exposure'])))+'</tr>')
    document='''<!doctype html><meta charset="utf-8"><title>RFP-1 financial replay</title>
<style>body{font:16px system-ui;margin:30px auto;max-width:1450px;padding:15px}table{border-collapse:collapse;font-size:13px}td,th{padding:6px;border:1px solid #ccc}img{max-width:100%}pre{white-space:pre-wrap}</style>
<h1>RFP-1: delayed shared-cash microprice/CatBoost portfolios</h1>
<p>OOS Nov9-12, TEST Nov11-12, Europe/Budapest. One100USDT portfolio per case, max30USDT per entry. MID accounting, bid/ask fills plus fees/slip. Four-day continuous cash and inventory.</p>
<p>SIMULATION ONLY. Exposed two-day TEST, three Demo assets, receipt-age/partial-book execution, fractional quantity and historical filters not certified. Full Truth Harness FAIL TH-11 remains. No actual orders, income, learning handoff or promotion.</p>
<p>M0 raw microprice; M1 TRAIN-only linear microprice; P/B/F fixed CatBoost price/book/trades. All60 policy cases and CASH preregistered; no threshold search.</p>
<img src="plots/daily-pnl.png"><table><tr><th>Case</th><th>OOS boundary net diagnostic USDT</th><th>TEST complete-day mean</th><th>TEST complete/requested</th><th>Closed cycles</th><th>Fees</th><th>Unmarked held clocks</th><th>Open exposure</th></tr>'''+''.join(table)+'</table><h2>First chronological trade per case</h2>'
    document+=''.join('<h3>'+key+'</h3><img src="plots/'+name+'">' for key,name in trade_images.items())
    document+='<h2>Limitations</h2><pre>'+html.escape(json.dumps(report['limitations'],indent=2))+'</pre>'
    (folder/'comparison.html').write_text(document,encoding='utf-8')


def run(rfd,capture,folder,truth_report):
    import numpy as np
    rfd=Path(rfd).resolve();capture=Path(capture).resolve();folder=Path(folder).resolve();truth_report=Path(truth_report).resolve()
    if folder.exists():raise ValueError('OUTPUT_EXISTS')
    folder.mkdir(parents=True)
    original=io.read_frozen(rfd/'registration.json');prepared=io.read_frozen(rfd/'prepared.json')
    predicted=io.read_frozen(rfd/'predictions.json');old_audit=io.read_frozen(rfd/'independent-audit.json')
    if old_audit['state']!='PASS_ARTIFACT_AUDIT_ONLY' or prepared['state']!='COMPLETE_RAW_REVALIDATION':raise ValueError('RFD_NOT_VERIFIED')
    if predicted['registration_sha256']!=io.file_sha(rfd/'registration.json') or prepared['registration_sha256']!=io.file_sha(rfd/'registration.json'):
        raise ValueError('RFD_REGISTRATION_BINDING')
    if predicted['file']!='frozen-predictions.npz':raise ValueError('PREDICTION_PATH')
    npz=rfd/predicted['file']
    if io.file_sha(npz)!=predicted['sha256']:raise ValueError('FROZEN_PREDICTIONS_HASH')
    manifest=json.loads((capture/'manifest.json').read_text());registration=json.loads((capture/'registration.json').read_text())
    full=prepared['parallel_validation']
    if (full['original_validator_all_records'] is not True or full['verified_records_n']!=manifest['records_n'] or
            full['verified_chunks_n']!=len(manifest['chunks'])):raise ValueError('RFD_RAW_VALIDATION_BINDING')
    inputs={str(npz):predicted['sha256']}
    for name in ('registration.json','prepared.json','predictions.json','independent-audit.json'):
        inputs[str(rfd/name)]=io.file_sha(rfd/name)
    for name in ('registration.json','manifest.json','status.json'):
        path=capture/name;sha=io.file_sha(path)
        if original['inputs'].get(str(path))!=sha:raise ValueError('CAPTURE_METADATA_CHANGED')
        inputs[str(path)]=sha
    expected={item['file']:original['inputs'].get(str(capture/item['file'])) for item in manifest['chunks']}
    quotes=ReceiptQuotes(capture,manifest,expected)
    source=Path(__file__).parent;spec=source.parents[3]/'docs/specs/binance-demo-receipt-flow-portfolio.md'
    for name in ('receipt_flow_portfolio.py','receipt_flow_portfolio_study.py','receipt_flow_portfolio_audit.py',
                 'receipt_flow_quotes.py','financial.py','receipt_flow_direction.py','flow_capture_audit.py','public_flow_dataset.py'):
        inputs[str(source/name)]=io.file_sha(source/name)
    inputs[str(spec)]=io.file_sha(spec);inputs[str(truth_report)]=io.file_sha(truth_report)
    truth=json.loads(truth_report.read_text())
    if truth['profile']!='full' or truth['status'] not in ('pass','fail','unknown'):raise ValueError('TRUTH_REPORT_REQUIRED')
    environment=dict(python=platform.python_version(),numpy=np.__version__)
    if environment!=dict(python='3.11.9',numpy='2.4.4'):raise ValueError('UNPINNED_ENVIRONMENT')
    start=original['splits']['tune'][0];end=original['splits']['test'][1];test_start=original['splits']['test'][0]
    plan=dict(protocol='RFP-1',cases=case_definitions(),profiles=PROFILES,source_symbols=original['source_symbols'],
        start_ms=start,end_ms=end,test_start_ms=test_start,objective='daily_net_equity_pnl_v1',
        inputs=inputs,environment=environment,truth_harness=dict(status=truth['status'],profile=truth['profile']),
        source_semantics='RFD_PRIOR_FULL_RAW_VALIDATION_PLUS_HASHED_ASOF_TOP_QUOTES',
        simulation_only=True,financial_certification=False,promotion_allowed=False,
        risk=dict(initial_cash='100',max_debit='30',min_debit='5',day_loss_and_peak_drawdown='1',global_drawdown='5',cooldown_ms=5000),
        calibration='M1_OLS_TRAIN_ONLY',test_exposed=True)
    plan['fee_source']='REGISTERED_SIMULATION_ASSUMPTION_NOT_ACTUAL_FEES'
    io.freeze(folder/'registration.json',plan)
    snapshot=folder/'source-snapshot';snapshot.mkdir()
    for path in inputs:
        p=Path(path)
        if p.suffix in ('.py','.md'):(snapshot/p.name).write_bytes(p.read_bytes())
    io.write_json(folder/'status.json',dict(phase='EXTRACTING_FROZEN_SIGNALS',promotion_allowed=False))
    arrays=np.load(npz,allow_pickle=False);times=arrays['times'];symbols=arrays['symbols'];x=arrays['features_F'];y=arrays['targets'];mids=arrays['mids']
    names=original['feature_names']['F'];micro=names.index('micro_bps');spread=names.index('spread_bps')
    train_start,train_end=original['splits']['train'];idx=np.flatnonzero((times>=train_start)&(times<train_end)&np.isfinite(y).all(axis=1))
    coefficients={}
    for j,h in enumerate((5,10,30)):
        rows=[dict(at_ms=int(times[i]),feature_end_ms=int(times[i])-1,label_available_ms=int(times[i])+30000,
                   micro_bps=float(x[i,micro]),target_bps=float(y[i,j])) for i in idx]
        coefficients[str(h)]=fit_microprice(rows,train_start_ms=train_start,train_end_ms=train_end)
        if coefficients[str(h)]['train_rows_n']!=predicted['models'][f'F_{h}']['training_rows_n']:raise ValueError('BASELINE_TRAIN_COHORT')
    io.freeze(folder/'microprice-calibration.json',coefficients)
    common=[];source_rows=np.flatnonzero((times>=start)&(times<end))
    predictions={f'{a}_{h}':arrays[f'{a}_{h}'] for a in ('P','B','F') for h in (5,10,30)}
    for i in source_rows:
        i=int(i);t=int(times[i]);symbol=original['source_symbols'][int(symbols[i])]
        common.append(dict(at_ms=t,feature_end_ms=t-1,symbol=symbol,source_row=i,
            features_sha256=digest(dict(source_row=i,at_ms=t,symbol=symbol,features=x[i].tolist())),
            micro_bps=float(x[i,micro]),spread_bps=float(x[i,spread]),origin_mid=float(mids[i]),
            predictions={key:float(values[i]) for key,values in predictions.items()}))
    io.freeze(folder/'signals-manifest.json',dict(issued_rows_n=len(common),all_future_label_states_retained=True,
        provenance_sha256=digest([(r['at_ms'],r['symbol'],r['source_row'],r['features_sha256']) for r in common]),
        prediction_file_sha256=predicted['sha256'],train_rows_n=len(idx),intervals_used_as_trading_inputs=False,
        microprice_calibration_sha256=io.file_sha(folder/'microprice-calibration.json')))
    summaries=[];results={};audits={};first_trades={};last_pair=None;grouped={}
    for case in plan['cases']:
        key=case['case_id'];out=folder/'cases'/key;out.mkdir(parents=True)
        if case['arm']=='CASH':cfg=Policy();signals={}
        else:
            pair=(case['arm'],case['horizon_seconds'])
            if pair!=last_pair:
                grouped={}
                for row in signal_rows(common,arm=pair[0],horizon=pair[1],coefficients=coefficients):
                    grouped.setdefault(row['at_ms'],[]).append(row)
                last_pair=pair
            signals=grouped;cfg=Policy(horizon_seconds=case['horizon_seconds'],admission=case['admission'],**PROFILES[case['profile']])
        io.write_json(folder/'status.json',dict(phase='REPLAYING_PORTFOLIO',case_id=key,complete_n=len(summaries),cases_N=61,promotion_allowed=False))
        pending_file=out/'events.jsonl.gz.partial';first=None;selected={}
        with gzip.open(pending_file,'wt',encoding='utf-8',compresslevel=3) as stream:
            def emit(e):
                nonlocal first
                stream.write(json.dumps(e,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n')
                if e['kind']=='DECISION' and e['selected'] is not None:selected[e['decision_id']]=e['selected']
                elif e['kind']=='BUY_FILL' and first is None:first=dict(buy=e,selected=selected[e['decision_id']])
                elif e['kind']=='SELL_FILL' and first is not None and 'sell' not in first and e['decision_id']==first['buy']['decision_id']:first['sell']=e
            result=simulate(signals,start_ms=start,end_ms=end,quote_at=quotes.quote_at,policy=cfg,emit=emit)
        journal=out/'events.jsonl.gz';pending_file.replace(journal)
        io.freeze(out/'result.json',result)
        checked=audit_events(_events(journal),result,cfg,quote_at=quotes.quote_at)
        io.freeze(out/'audit.json',checked);audits[key]=checked
        io.freeze(out/'manifest.json',dict(case=case,policy=asdict(cfg),events_sha256=io.file_sha(journal),
            result_sha256=io.file_sha(out/'result.json'),audit_sha256=io.file_sha(out/'audit.json'),
            registration_sha256=io.file_sha(folder/'registration.json'),
            microprice_calibration_sha256=io.file_sha(folder/'microprice-calibration.json')))
        summaries.append(_summary(result,case,test_start));results[key]=result
        if first is not None:first_trades[key]=first
    limitations=['CONDITIONAL_SIMULATION_NOT_ACTUAL_ACCOUNT_PNL','EXPOSED_TEST_TWO_DAYS_NO_INDEPENDENT_SIGNIFICANCE',
        'THREE_LIQUID_DEMO_ASSETS_NOT_ALL_USDT_POPULATION','EXCHANGE_AGE_UNKNOWN_RECEIPT_CLOCK',
        'TOP_OF_BOOK_AND_SLIPPAGE_FILL_PROXY_NOT_QUEUE_CERTIFICATION','FRACTIONAL_QUANTITY_HISTORICAL_FILTERS_NOT_CERTIFIED',
        'DAYS_WITH_UNMARKED_HELD_EXPOSURE_EXCLUDED_NOT_ZERO','NO_MODEL_OR_POLICY_PROMOTION']
    comparisons=[]
    for h in (5,10,30):
        for admission in ('direction','cost-aware'):
            for profile in ('base','stress'):
                left=f'F-h{h}-{admission}-{profile}'
                for reference in ('M1','B'):
                    right=f'{reference}-h{h}-{admission}-{profile}'
                    comparisons.append(dict(left=left,right=right,
                        **paired_daily(results[left]['daily'],results[right]['daily'],test_start_ms=test_start)))
    report=dict(state='COMPLETE_CONDITIONAL_SIMULATION',cases=summaries,start_ms=start,end_ms=end,test_start_ms=test_start,
        prior_issued_rows_n=len(common),raw_validation_proof_reused=True,limitations=limitations,
        paired_test_comparisons=comparisons,
        truth_harness=plan['truth_harness'],financial_certification=False,promotion_allowed=False)
    _render(folder,report,results,first_trades,quotes)
    quotes.verify_used()
    for path,expected_sha in inputs.items():
        if io.file_sha(path)!=expected_sha:raise ValueError('INPUT_CHANGED:'+path)
    io.freeze(folder/'quote-source-evidence.json',quotes.evidence());io.freeze(folder/'report.json',report)
    io.freeze(folder/'ledger-audits.json',dict(state='PASS_LEDGER_ARTIFACT_AUDIT_ONLY',cases_n=len(audits),cases=audits,
        financial_certification=False,promotion_allowed=False))
    io.write_json(folder/'status.json',dict(phase='COMPLETE_CONDITIONAL_SIMULATION',cases_n=61,promotion_allowed=False))
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('rfd','capture','output','truth-report'):parser.add_argument('--'+key,required=True)
    args=parser.parse_args();app=Path(__file__).resolve().parents[2]
    from .public_flow_dataset import own_runtime_path
    rfd=own_runtime_path(app,args.rfd);capture=own_runtime_path(app,args.capture)
    folder=own_runtime_path(app,args.output);truth=own_runtime_path(app,args.truth_report)
    if folder.exists():raise ValueError('OUTPUT_EXISTS')
    if any(folder==p or folder.is_relative_to(p) or p.is_relative_to(folder) for p in (rfd,capture,truth)):
        raise ValueError('OVERLAPPING_PATHS')
    try:
        report=run(rfd,capture,folder,truth)
        print(json.dumps(dict(state=report['state'],cases_n=len(report['cases']),promotion_allowed=False)),flush=True)
    except BaseException as error:
        if folder.exists():
            io.write_json(folder/'incident.json',dict(exception=type(error).__name__,detail=str(error),promotion_allowed=False))
            io.write_json(folder/'status.json',dict(phase='INCOMPLETE',exception=type(error).__name__,promotion_allowed=False))
        raise


if __name__=='__main__':main()
