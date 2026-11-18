"""RFD-1 fixed CatBoost ablation on independently revalidated Demo receipts."""
import argparse
from collections import Counter
import gzip
import hashlib
import html
import json
import math
from pathlib import Path
import platform
import time

from . import receipt_flow_direction as core

MODEL_PARAMS=dict(iterations=300,depth=5,learning_rate=.05,l2_leaf_reg=10,
    loss_function='RMSE',random_seed=42,thread_count=2,bootstrap_type='No',
    has_time=True,allow_writing_files=False,verbose=False)
P_NAMES=sorted(['return_1','return_5','return_10','rv_10'])
B_NAMES=sorted(P_NAMES+['spread_bps','micro_bps','depth_imbalance_1',
    'depth_imbalance_5','depth_imbalance_20','log_depth_20','quote_age_ms'])
F_NAMES=sorted(B_NAMES+[f'{name}_{w}' for name in ('trade_imbalance','log_trade_qty',
    'trade_count','book_ofi') for w in (1,5,10)])
NAMES={'P':P_NAMES,'B':B_NAMES,'F':F_NAMES}


def masks(grid,start,end):
    import numpy as np
    issued=np.arange(start,end,dtype='int64')
    issued=issued[(issued%5==0)&(grid[issued,3]==1)]
    candidates=issued[issued+30<end]
    bad=np.concatenate(([0],np.cumsum(grid[:,2]!=1)))
    good=(bad[candidates+31]-bad[candidates]==0)&(grid[candidates,1]==grid[candidates+30,1])
    return issued,candidates[good]


def make_plan(capture,inputs):
    return dict(protocol='RFD-1',evidence_kind='PUBLIC_DEMO_FORWARD_RETURN_DIAGNOSTIC',
        source_symbols=capture['symbols'],splits=core.calendar_splits(capture),
        horizon_clock='RECEIPT_STRICT_BEFORE',quote_max_age_ms=1000,
        trade_max_age_ms=5000,feature_context_seconds=10,origin_step_seconds=5,
        horizons_seconds=list(core.HORIZONS),arms=['P','B','F'],feature_names=NAMES,
        model_params=MODEL_PARAMS,tune_use='DIAGNOSTIC_ONLY_NO_SELECTION',
        calibration='FINITE_SAMPLE_90PCT_ABSOLUTE_RESIDUAL_CAL_ONLY',inputs=inputs,
        validation_workers=4,validation='ORIGINAL_VALIDATOR_ALL_FRAMES_WITH_EXACT_BOUNDARY_CHAIN',
        test_days=2,verdict='DIAGNOSTIC_ONLY_INSUFFICIENT_INDEPENDENT_DAYS',
        net_daily_pnl=None,promotion_allowed=False)


def fit_model(x,y,names,path):
    import numpy as np
    from catboost import CatBoostRegressor,Pool
    if len(y)<100:raise ValueError('INSUFFICIENT_TRAIN_ROWS')
    model=CatBoostRegressor(**MODEL_PARAMS)
    model.fit(Pool(x,label=y,feature_names=names));model.save_model(str(path))
    reload=CatBoostRegressor();reload.load_model(str(path))
    n=min(32,len(y));a=model.predict(x[:n]);b=reload.predict(x[:n])
    delta=float(np.max(np.abs(a-b)))
    if delta>1e-12:raise ValueError('NATIVE_RELOAD_MISMATCH')
    return reload,dict(reload_rows_n=n,max_reload_delta=delta,model_sha256=core.file_sha(path))


def _environment():
    import catboost,numpy,scipy
    env=dict(python=platform.python_version(),catboost=catboost.__version__,
             numpy=numpy.__version__,scipy=scipy.__version__)
    if env!=dict(python='3.11.9',catboost='1.2.10',numpy='2.4.4',scipy='1.17.1'):
        raise ValueError('UNPINNED_ENVIRONMENT')
    return env


def prepare(capture,folder):
    import numpy as np
    from . import public_flow_dataset as flow
    from .flow_capture_audit import _verified_chunk_lines
    from .receipt_flow_parallel import BoundaryState,BoundedValidation
    capture=Path(capture).resolve();folder=Path(folder).resolve()
    if folder.exists():raise ValueError('OUTPUT_EXISTS')
    folder.mkdir(parents=True)
    registration=json.loads((capture/'registration.json').read_bytes());flow._validate_registration(registration)
    manifest=json.loads((capture/'manifest.json').read_bytes())
    status=json.loads((capture/'status.json').read_bytes())
    if not status.get('calendar_schedule_finished'):raise ValueError('CAPTURE_NOT_FINISHED')
    if registration['source_sha256']!=core.file_sha(flow.__file__):raise ValueError('PRODUCER_CHANGED')
    if manifest['registration_sha256']!=flow._sha(registration) or status['registration_sha256']!=flow._sha(registration):
        raise ValueError('CAPTURE_REGISTRATION_BINDING')
    source=Path(__file__).parent
    spec=source.parents[3]/'docs/specs/binance-demo-receipt-flow-direction.md'
    inputs={str(capture/name):core.file_sha(capture/name) for name in ('registration.json','manifest.json','status.json')}
    for name in ('receipt_flow_direction.py','receipt_flow_study.py','receipt_flow_parallel.py','public_flow_dataset.py','flow_capture_audit.py'):
        inputs[str(source/name)]=core.file_sha(source/name)
    inputs[str(spec)]=core.file_sha(spec)
    for i,item in enumerate(manifest['chunks']):
        if item['file']!='chunk-%08d.jsonl.gz'%i:raise ValueError('CHUNK_ORDER')
        inputs[str(capture/item['file'])]=item['sha256']
    plan=make_plan(registration,inputs);plan['environment']=_environment()
    core.freeze(folder/'registration.json',plan)
    snap=folder/'source-snapshot';snap.mkdir()
    for path in inputs:
        p=Path(path)
        if p.suffix in ('.py','.md'):(snap/p.name).write_bytes(p.read_bytes())
    start,end=registration['full_days_start_ms'],registration['planned_end_ms']
    n=(end-start)//1000
    arrays={s:np.lib.format.open_memmap(folder/(s+'.npy'),mode='w+',dtype='float64',shape=(n,4+len(F_NAMES))) for s in registration['symbols']}
    for grid in arrays.values():grid[:]=np.nan
    engine=core.ReceiptGrid(registration['symbols'],start_ms=start,end_ms=end)
    shadow=BoundaryState();checks=BoundedValidation()
    counts=Counter();last_mono=None;active=None;session_ids=set()
    def save(rows):
        for row in rows:
            i=(row['at_ms']-start)//1000
            values=[np.nan if row['mid'] is None else row['mid'],row['epoch'],int(row['valid']),int(row['features'] is not None)]
            values+=([row['features']['F'][k] for k in F_NAMES] if row['features'] else [np.nan]*len(F_NAMES))
            arrays[row['symbol']][i]=values
    compressed=0
    for i,item in enumerate(manifest['chunks']):
        raw=(capture/item['file']).read_bytes()
        if len(raw)!=item['compressed_bytes'] or hashlib.sha256(raw).hexdigest()!=item['sha256']:
            raise ValueError('CHUNK_HASH')
        compressed+=len(raw);times=[];sids=set();records=[];before=shadow.snapshot()
        for line in _verified_chunk_lines(raw,item):
            record=json.loads(line);sid=record['session_id'];kind=record['kind']
            times.append(record['wall_ms']);sids.add(sid);counts[kind]+=1
            if last_mono is not None and record['mono_ns']<last_mono:raise ValueError('MONOTONIC_REGRESSION')
            last_mono=record['mono_ns']
            if kind=='SESSION_START':
                if active is not None or sid in session_ids:raise ValueError('SESSION_BOUNDARY')
                active=sid;session_ids.add(sid)
            elif active!=sid:raise ValueError('SESSION_OWNERSHIP')
            parsed=json.loads(record['raw'])['data'] if kind=='MARKET' else None
            shadow.advance(record,parsed);records.append(record)
            save(engine.feed(record,parsed=parsed))
            if kind=='SESSION_END':active=None
        if (min(times)!=item['first_receipt_ms'] or max(times)!=item['last_receipt_ms'] or
                sorted(sids)!=item['session_ids']):raise ValueError('CHUNK_METADATA')
        checks.submit(records,before,shadow.snapshot())
        if i%64==0 or i+1==len(manifest['chunks']):
            core.write_json(folder/'status.json',dict(phase='RAW_REVALIDATION',chunks_n=i+1,
                chunks_N=len(manifest['chunks']),records_n=sum(counts.values()),checked_at_ms=time.time_ns()//1000000,promotion_allowed=False))
    parallel_proof=checks.finish()
    if parallel_proof['verified_records_n']!=sum(counts.values()):raise ValueError('UNVERIFIED_RAW_RECORDS')
    save(engine.finish())
    if active is not None or sum(counts.values())!=manifest['records_n'] or compressed!=manifest['compressed_bytes']:
        raise ValueError('CAPTURE_TOTALS')
    if len(list(capture.glob('chunk-*.jsonl.gz')))!=len(manifest['chunks']):raise ValueError('UNLISTED_CHUNK')
    summary=shadow.validator.summary()
    for key in ('sessions_n','gaps_n','invalid_n','duplicates_n','conflicts_n','first_receipt_ms','last_receipt_ms','messages_per_topic'):
        if summary[key]!=status[key]:raise ValueError('REVALIDATED_COUNTERS:'+key)
    for path,digest in inputs.items():
        if core.file_sha(path)!=digest:raise ValueError('INPUT_CHANGED:'+path)
    files={}
    for symbol,grid in arrays.items():
        grid.flush();files[symbol]=core.file_sha(folder/(symbol+'.npy'))
    core.freeze(folder/'prepared.json',dict(state='COMPLETE_RAW_REVALIDATION',counts=dict(counts),
        summary=summary,parallel_validation=parallel_proof,files=files,columns=['mid','epoch','valid','feature_valid']+F_NAMES,
        registration_sha256=core.file_sha(folder/'registration.json'),exchange_age='UNKNOWN',promotion_allowed=False))
    core.write_json(folder/'status.json',dict(phase='PREPARED_NOT_TRAINED',promotion_allowed=False))
    return plan


def _load_samples(folder,plan):
    import numpy as np
    start=plan['splits']['train'][0];parts=[];coverage=[]
    for si,symbol in enumerate(plan['source_symbols']):
        grid=np.load(folder/(symbol+'.npy'),mmap_mode='r')
        for segment,(lo,hi) in plan['splits'].items():
            issued,scored=masks(grid,(lo-start)//1000,(hi-start)//1000)
            mature=np.isin(issued,scored)
            targets=np.full((len(issued),3),np.nan)
            for j,h in enumerate(core.HORIZONS):
                targets[mature,j]=np.log(grid[scored+h,0]/grid[scored,0])*10000
            parts.append((issued*1000+start,np.full(len(issued),si),grid[issued,4:].copy(),targets,grid[issued,0].copy()))
            coverage.append(dict(symbol=symbol,segment=segment,requested_origins_N=((hi-lo)//1000)//5,
                issued_n=len(issued),scored_n=len(scored),excluded_feature_n=((hi-lo)//1000)//5-len(issued),
                excluded_label_n=len(issued)-len(scored)))
    data=[np.concatenate([p[i] for p in parts]) for i in range(5)]
    order=np.lexsort((data[1],data[0]));data=[v[order] for v in data]
    return data,coverage


def evaluate(folder,plan):
    import numpy as np
    from zoneinfo import ZoneInfo
    from datetime import datetime
    folder=Path(folder);prepared=core.read_frozen(folder/'prepared.json')
    if (folder/'fit-started.json').exists():raise ValueError('FIT_ALREADY_STARTED')
    if _environment()!=plan['environment']:raise ValueError('ENVIRONMENT_CHANGED')
    for path,digest in plan['inputs'].items():
        if core.file_sha(path)!=digest:raise ValueError('INPUT_CHANGED:'+path)
    for symbol,digest in prepared['files'].items():
        if core.file_sha(folder/(symbol+'.npy'))!=digest:raise ValueError('PREPARED_DATA_CHANGED')
    core.freeze(folder/'fit-started.json',dict(at_ms=time.time_ns()//1000000,registration_sha256=core.file_sha(folder/'registration.json')))
    (times,symbols,x,y,mids),coverage=_load_samples(folder,plan)
    good=np.isfinite(y).all(axis=1)
    segment_masks={name:(times>=lo)&(times<hi) for name,(lo,hi) in plan['splits'].items()}
    train=segment_masks['train']&good;cal=segment_masks['cal']&good
    if train.sum()<100 or cal.sum()<100:raise ValueError('INSUFFICIENT_TRAIN_OR_CAL')
    preds={};radii={};proofs={};rows=[];majorities={}
    days=np.array([datetime.fromtimestamp(int(t)/1000,ZoneInfo('Europe/Budapest')).date().isoformat() for t in times])
    for j,h in enumerate(core.HORIZONS):
        majority=core.majority_direction(y[train,j].tolist());majorities[str(h)]=majority
        for arm in ('P','B','F'):
            key=f'{arm}_{h}';columns=[F_NAMES.index(k) for k in NAMES[arm]]
            core.write_json(folder/'status.json',dict(phase='FITTING_FIXED_MODEL',arm=arm,horizon_seconds=h,train_n=int(train.sum()),promotion_allowed=False))
            model,proof=fit_model(x[train][:,columns],y[train,j],NAMES[arm],folder/(key+'.cbm'))
            proof['training_rows_n']=int(train.sum());proofs[key]=proof
            prediction=np.asarray(model.predict(x[:,columns]));preds[key]=prediction
            radius=core.conformal_radius(np.abs(y[cal,j]-prediction[cal]).tolist());radii[key]=radius
            for segment,mask in segment_masks.items():
                for label,asset_mask in [('ALL',np.ones(len(times),dtype=bool))]+[(s,symbols==i) for i,s in enumerate(plan['source_symbols'])]:
                    selected=mask&asset_mask&good
                    metric=core.metrics(y[selected,j].tolist(),prediction[selected].tolist(),majority_sign=majority)
                    micro=core.metrics(y[selected,j].tolist(),x[selected,F_NAMES.index('micro_bps')].tolist(),majority_sign=majority)
                    metric.update(segment=segment,symbol=label,arm=arm,horizon_seconds=h,
                        interval_radius_bps=radius,interval_covered_n=int((np.abs(y[selected,j]-prediction[selected])<=radius).sum()),
                        interval_N=int(selected.sum()),micro_direction_correct_n=micro['direction_correct_n'],
                        issued_N=int((mask&asset_mask).sum()),model_sha256=proof['model_sha256'])
                    rows.append(metric)
    contrasts=[]
    for j,h in enumerate(core.HORIZONS):
        for day in sorted(set(days[segment_masks['test']])):
            for label,asset_mask in [('ALL',np.ones(len(times),dtype=bool))]+[(s,symbols==i) for i,s in enumerate(plan['source_symbols'])]:
                mask=segment_masks['test']&(days==day)&asset_mask&good;n=int(mask.sum())
                contrasts.append(dict(day=day,symbol=label,horizon_seconds=h,N=n,
                    mae_delta_F_minus_B_bps=float((np.abs(y[mask,j]-preds[f'F_{h}'][mask])-np.abs(y[mask,j]-preds[f'B_{h}'][mask])).mean()) if n else None))
    prediction_file=folder/'frozen-predictions.npz'
    np.savez_compressed(prediction_file,times=times,symbols=symbols,features_F=x,targets=y,mids=mids,**preds)
    core.freeze(folder/'predictions.json',dict(file=prediction_file.name,sha256=core.file_sha(prediction_file),
        models=proofs,radii_bps=radii,rows_n=len(times),feature_end_rule='origin_ms-1',
        label_validity='COMMON_30_SECOND_CONTINUOUS_RECEIPT_MASK',registration_sha256=core.file_sha(folder/'registration.json')))
    report=dict(state='COMPLETE_DIAGNOSTIC_NOT_FINANCIAL',metrics=rows,coverage=coverage,
        train_majority_sign=majorities,paired_test_day_deltas=contrasts,native_reload=proofs,
        verdict='INCONCLUSIVE_INDEPENDENT_TEST_DAYS_2_LT_7',test_days_N=2,
        source_semantics='DEMO_STANDING_TOP20_AND_AGGRESSOR_AGGTRADES',exchange_age='UNKNOWN',
        net_daily_pnl=None,financial_certification=False,promotion_allowed=False,
        registration_sha256=core.file_sha(folder/'registration.json'))
    core.freeze(folder/'report.json',report)
    render(folder,plan,report,(times,symbols,x,y,mids),preds,radii)
    core.write_json(folder/'status.json',dict(phase='COMPLETE_DIAGNOSTIC_NOT_FINANCIAL',models_n=9,promotion_allowed=False))
    return report


def render(folder,plan,report,data,preds,radii):
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from datetime import datetime,timezone
    folder=Path(folder);plots=folder/'plots';plots.mkdir()
    times,symbols,x,y,mids=data;test_start,test_end=plan['splits']['test'];base=plan['splits']['train'][0]
    issued=[dict(symbol=plan['source_symbols'][int(s)],at_ms=int(t),features=True) for s,t in zip(symbols,times) if test_start<=t<test_end]
    origins=core.chart_origins(issued,plan['source_symbols'],(test_start+12*3600000,test_start+36*3600000))
    core.freeze(folder/'chart-origins.json',dict(selection='FIRST_ISSUED_AT_FIXED_TEST_CLOCK_NO_LABEL_FILTER',origins=origins))
    images=[]
    for symbol,at in origins:
        si=plan['source_symbols'].index(symbol);idx=int(np.flatnonzero((times==at)&(symbols==si))[0]);p0=mids[idx]
        grid=np.load(folder/(symbol+'.npy'),mmap_mode='r');origin=(at-base)//1000
        offsets=np.arange(-60,31);actual=grid[origin+offsets,0]
        fig,axes=plt.subplots(3,1,figsize=(12,10),sharex=True,sharey=True)
        for ax,arm,color in zip(axes,('P','B','F'),('#3975b9','#e68a20','#239461')):
            ax.plot(offsets[offsets<=0],actual[offsets<=0],color='black',label='Observed history')
            ax.plot(offsets[offsets>=0],actual[offsets>=0],color='black',linestyle=':',label='Actual after issue')
            forecast=[p0]+[p0*math.exp(float(preds[f'{arm}_{h}'][idx])/10000) for h in core.HORIZONS]
            lower=[p0]+[p0*math.exp((float(preds[f'{arm}_{h}'][idx])-radii[f'{arm}_{h}'])/10000) for h in core.HORIZONS]
            upper=[p0]+[p0*math.exp((float(preds[f'{arm}_{h}'][idx])+radii[f'{arm}_{h}'])/10000) for h in core.HORIZONS]
            ax.plot([0,5,10,30],forecast,'o--',color=color,label='Independent forecast anchors (connector only)')
            ax.fill_between([0,5,10,30],lower,upper,color=color,alpha=.15,label='CAL residual 90% interval')
            ax.axvline(0,color='#777');ax.set_title(arm+': '+{'P':'price','B':'price + standing book','F':'price + book + trades'}[arm])
            ax.set_ylabel('USDT');ax.grid(alpha=.25);ax.legend(fontsize=8,loc='upper left')
        axes[-1].set_xlabel('Seconds from frozen issue (receipt clock)')
        fig.suptitle(symbol+' | '+datetime.fromtimestamp(at/1000,timezone.utc).isoformat()+'\nDemo diagnostic, future fact never used to choose origin')
        fig.tight_layout();name=f'{symbol}-{at}.png';fig.savefig(plots/name,dpi=110);plt.close(fig);images.append(name)
    table=[]
    for r in report['metrics']:
        if r['segment']!='test' or r['symbol']!='ALL':continue
        def fmt(v):return 'unknown' if v is None else f'{v:.4f}'
        table.append('<tr>'+''.join('<td>'+str(v)+'</td>' for v in
            (r['arm'],r['horizon_seconds'],r['N'],fmt(r['mae_bps']),fmt(r['zero_mae_bps']),
             fmt(r['rmse_bps']),f"{r['direction_correct_n']}/{r['direction_N']}",
             f"{r['majority_correct_n']}/{r['direction_N']}",f"{r['micro_direction_correct_n']}/{r['direction_N']}",
             f"{r['interval_covered_n']}/{r['interval_N']}"))+'</tr>')
    document='''<!doctype html><meta charset="utf-8"><title>RFD-1 Demo flow CatBoost</title>
<style>body{font:16px system-ui;max-width:1300px;margin:30px auto;padding:15px}td,th{padding:8px;border:1px solid #ccc}table{border-collapse:collapse}img{width:100%}pre{white-space:pre-wrap}</style>
<h1>RFD-1: CatBoost 5/10/30 seconds</h1><p>P price; B + standing book; F + aggressor trades and snapshot OFI.</p>
<p>TEST: Nov 11–12, 2026 Europe/Budapest. Same issued origins and common30-second label mask. Strict receipt timing, gaps excluded.</p>
<p>DIAGNOSTIC ONLY. Two TEST days; exchange-age UNKNOWN; no portfolio PnL or promotion. Full Truth Harness FAIL TH-11 retained.</p>
<table><tr><th>Arm</th><th>h sec</th><th>N</th><th>MAE bp</th><th>Zero MAE</th><th>RMSE</th><th>Direction correct/N</th><th>TRAIN majority correct/N</th><th>Microprice correct/N</th><th>90% covered/N</th></tr>'''+''.join(table)+'</table>'
    document+='<h2>Coverage by split and asset</h2><pre>'+html.escape(json.dumps(report['coverage'],indent=2))+'</pre>'
    document+='<h2>Frozen forecasts and later observed fact</h2><p>Lines between5/10/30-second anchors are connectors, not learned intra-horizon paths. No fabricated wiggles or actual-driven forecast updates.</p>'
    document+=''.join('<img src="plots/'+name+'">' for name in images)
    (folder/'comparison.html').write_text(document,encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args()
    from .public_flow_dataset import own_runtime_path
    app=Path(__file__).resolve().parents[2]
    capture=own_runtime_path(app,args.capture);folder=own_runtime_path(app,args.output)
    if capture==folder or capture.is_relative_to(folder) or folder.is_relative_to(capture):raise ValueError('OVERLAPPING_PATHS')
    if folder.exists():raise ValueError('OUTPUT_EXISTS')
    try:
        plan=prepare(capture,folder);report=evaluate(folder,plan)
        print(json.dumps(dict(state=report['state'],models_n=9,net_daily_pnl=None,promotion_allowed=False)),flush=True)
    except BaseException as error:
        if folder.exists():
            core.write_json(folder/'incident.json',dict(exception=type(error).__name__,detail=str(error),promotion_allowed=False))
            core.write_json(folder/'status.json',dict(phase='INCOMPLETE',exception=type(error).__name__,promotion_allowed=False))
        raise


if __name__=='__main__':main()
