"""FSA-1 isolated retrospective feature ablation; no account/order dependency."""
import argparse
from collections import deque
import hashlib
import html
import json
import math
import os
from pathlib import Path
import platform
import sqlite3
import time
import zlib

MODEL_PARAMS=dict(iterations=300,depth=5,learning_rate=.05,l2_leaf_reg=10,
    loss_function='RMSE',random_seed=42,thread_count=2,bootstrap_type='No',
    has_time=True,allow_writing_files=False,verbose=False)
SYMBOLS=['BTCUSDT','ETHUSDT']
CUTOFF=1793197800000

def sha(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix('.next')
    with temp.open('w',encoding='utf-8') as stream:
        json.dump(value,stream,sort_keys=True,separators=(',',':'),allow_nan=False)
        stream.flush();os.fsync(stream.fileno())
    os.replace(temp,path)

def freeze_registration(folder,registration):
    path=Path(folder)/'registration.json'
    if path.exists():
        if json.loads(path.read_text(encoding='utf-8'))!=registration:raise ValueError('REGISTRATION_CHANGED')
    else:write(path,registration)
    return sha(registration)

def read_archive(path,symbols,cutoff):
    from .trend_research import STEP
    series={};manifest={}
    con=sqlite3.connect(Path(path).resolve().as_uri()+'?mode=ro',uri=True,timeout=10)
    try:
        for symbol in symbols:
            rows=[];vectors=[];receipts=[];cursor=-1;raw_n=short_n=0;previous=None
            while True:
                page=con.execute('SELECT start_ms,source_sha,blob_sha,body,received_ms FROM pages WHERE symbol=? AND start_ms>? ORDER BY start_ms LIMIT 1',(symbol,cursor)).fetchone()
                if page is None:break
                cursor,digest,blob_sha,blob,received=page
                if hashlib.sha256(blob).hexdigest()!=blob_sha:raise ValueError('ARCHIVE_HASH')
                try:body=json.loads(zlib.decompress(blob))
                except (ValueError,zlib.error):raise ValueError('ARCHIVE_HASH') from None
                if sha(body)!=digest:raise ValueError('ARCHIVE_HASH')
                if not body:raise ValueError('ARCHIVE_EMPTY_PAGE')
                vectors.append([cursor,digest,blob_sha]);receipts.append(received)
                for row in body:
                    if previous is not None and row[0]<=previous:raise ValueError('ARCHIVE_NON_MONOTONE')
                    previous=row[0]
                    if row[6]>=cutoff:continue
                    if received<row[6]:raise ValueError('RECEIPT_BEFORE_CLOSE')
                    raw_n+=1
                    if row[0]%STEP or row[6]!=row[0]+STEP-1:short_n+=1;continue
                    rows.append(row)
            if not rows:raise ValueError('MISSING_SYMBOL:'+symbol)
            series[symbol]=rows
            expected=(cutoff-rows[0][0])//STEP
            manifest[symbol]=dict(raw_rows_n=raw_n,regular_rows_n=len(rows),shortened_rows_n=short_n,
                expected_grid_N=expected,missing_regular_n=expected-len(rows),pages_n=len(vectors),
                first_open_ms=rows[0][0],last_close_ms=rows[-1][6],page_vector_sha256=sha(vectors),
                received_min_ms=min(receipts),received_max_ms=max(receipts),historical_receipts_certified=False)
    finally:con.close()
    return series,manifest

def normalize_interval(raw):
    return dict(raw,lower=raw.get('ci_low_usdt'),upper=raw.get('ci_high_usdt'),mean=raw.get('mean_delta_usdt'))

def verdict(delta,*,absolute_lower,drawdown,reference_drawdown,stress_pnl):
    if delta.get('paired_days_N',0)<30 or delta.get('lower') is None:return 'INCONCLUSIVE'
    return ('POSITIVE_PATTERN_POWER_UNCERTIFIED' if delta['lower']>.01 and absolute_lower is not None
        and absolute_lower>0 and drawdown<=reference_drawdown+.5 and stress_pnl is not None and stress_pnl>0
        else 'NOT_SUPPORTED')

def training_examples(series,start,end):
    from .trend_research import STEP,POLICY,_events,trend_signal
    from .feature_ablation import build_features,training_label
    histories={s:deque(maxlen=32) for s in series};vectors={a:[] for a in ('A0','A1','A2')}
    labels=[];provenance=[];counts=dict(candidate_rows_N=0,invalid_context_n=0,immature_or_gapped_label_n=0)
    for at,group in _events(series):
        if at>=end:break
        if at>=start:
            for symbol in sorted(histories):
                if trend_signal(histories[symbol],now_ms=at)['action']!='BUY':continue
                counts['candidate_rows_N']+=1
                features=build_features(histories,symbol,at)
                if features is None:counts['invalid_context_n']+=1;continue
                target=training_label(series[symbol],at,end)
                if target is None:counts['immature_or_gapped_label_n']+=1;continue
                for arm in vectors:vectors[arm].append(features[arm])
                labels.append(target);provenance.append(dict(symbol=symbol,origin_ms=at,
                    feature_end_ms=at-1,label_available_ms=at+16*STEP))
        for symbol,row in group.items():histories[symbol].append(row)
    counts['fitted_rows_n']=len(labels)
    return vectors,labels,provenance,counts

def fit_model(vectors,labels,feature_names,path):
    from catboost import CatBoostRegressor,Pool
    data=[[v[k] for k in feature_names] for v in vectors]
    model=CatBoostRegressor(**MODEL_PARAMS)
    model.fit(Pool(data,label=labels,feature_names=feature_names))
    model.save_model(str(path))
    return model

def render_report(folder,report,segments):
    names=['C0','A0','A1','A2'];colors=['#555','#2676b8','#e58b20','#1b976a']
    rows=[]
    for arm in names:
        r=segments['test_base_'+arm]
        def fmt(key):
            v=r.get(key);return 'null' if v is None else f'{v:.4f}'
        rows.append('<tr><td>'+arm+'</td><td>'+fmt('net_pnl_usdt')+'</td><td>'+fmt('mean_daily_net_pnl_usdt')+
            '</td><td>'+fmt('max_drawdown_usdt')+'</td><td>'+str(r['trades_n'])+'</td><td>'+fmt('fees_usdt')+'</td></tr>')
    paths=[];all_values=[0.];curves={}
    for arm in names:
        total=0.;values=[]
        for day,pnl in sorted(segments['test_base_'+arm]['daily_pnl'].items()):total+=pnl;values.append(total)
        curves[arm]=values;all_values.extend(values)
    lo,hi=min(all_values),max(all_values);span=max(hi-lo,.01)
    for arm,color in zip(names,colors):
        values=curves[arm];n=max(1,len(values)-1)
        points=' '.join(f'{50+900*i/n:.2f},{30+300*(hi-v)/span:.2f}' for i,v in enumerate(values))
        paths.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2"/>')
    document='''<!doctype html><meta charset="utf-8"><title>FSA-1 comparison</title>
<style>body{font:16px system-ui;max-width:1100px;margin:32px auto;padding:16px}td,th{padding:10px;border-bottom:1px solid #ddd}pre{white-space:pre-wrap}svg{width:100%}</style>
<h1>Feature-source ablation: BTC/ETH, SIMULATION</h1>
<p>Test: 2025-06-11 16:30 — 2026-10-28 14:30 UTC. Capital 100 USDT each arm. Net fees/slippage included.</p>
<p>C0 frozen rule; A0 OHLCV; A1 + closed BTC/ETH context; A2 + candle taker-flow proxy.</p>
<table><tr><th>Arm</th><th>Total net USDT</th><th>Mean daily net</th><th>Max DD USDT</th><th>Trades</th><th>Fees USDT</th></tr>'''+''.join(rows)+'''</table>
<h2>Cumulative complete-day net PnL (USDT)</h2>'''+f'<svg viewBox="0 0 1000 370"><text x="5" y="24">{hi:.2f}</text><text x="5" y="340">{lo:.2f}</text>'+''.join(paths)+'</svg><p>'+ ' | '.join(f'<span style="color:{c}">{a}</span>' for a,c in zip(names,colors))+'</p><h2>Preregistered comparisons</h2><pre>'+html.escape(json.dumps(report['comparisons'],indent=2))+'</pre><h2>Limits</h2><p>'+html.escape('; '.join(report['limitations']))+'</p>'
    (Path(folder)/'comparison.html').write_text(document,encoding='utf-8')

def run(archive,folder):
    from .trend_research import POLICY,STEP,time_splits
    from .feature_ablation import build_features,replay,paired_bootstrap
    import catboost,numpy,scipy
    from catboost import CatBoostRegressor
    folder=Path(folder).resolve();folder.mkdir(parents=True,exist_ok=True)
    package=Path(__file__).parent;spec=package.parents[3]/'docs/specs/binance-demo-feature-source-ablation.md'
    source={n:hashlib.sha256((package/n).read_bytes()).hexdigest() for n in ('feature_ablation.py','feature_ablation_runner.py','trend_research.py','financial.py')}
    environment=dict(python=platform.python_version(),catboost=catboost.__version__,numpy=numpy.__version__,scipy=scipy.__version__)
    if environment!=dict(python='3.11.9',catboost='1.2.10',numpy='2.4.4',scipy='1.17.1'):raise ValueError('UNPINNED_ENVIRONMENT')
    registration=dict(attempt='FSA-1-BTCETH-20261029-v2',symbols=SYMBOLS,cutoff_ms=CUTOFF,model_params=MODEL_PARAMS,
        policy=POLICY,source_sha256=source,spec_sha256=hashlib.sha256(spec.read_bytes()).hexdigest(),environment=environment,
        objective='daily_net_equity_pnl_v1',evidence_kind='SIMULATION',promotion_allowed=False,
        costs={'base':[10,5],'stress':[20,10]},split='calendar70/15/15',feature_arms=['C0','A0','A1','A2'])
    registration_sha=freeze_registration(folder,registration)
    write(folder/'status.json',dict(state='VERIFYING_ARCHIVE',registration_sha256=registration_sha))
    series,manifest=read_archive(archive,SYMBOLS,CUTOFF)
    first=max(v[0][0] for v in series.values());splits=time_splits(first,CUTOFF)
    # Reject incomplete tune/test before training; no future-valid symbol substitution.
    for name,start,end in splits[1:]:
        for symbol,rows in series.items():
            times=[r[0] for r in rows if start<=r[0]<end]
            if times!=list(range(start,end,STEP)):raise ValueError('INCOMPLETE_EVALUATION_GRID:'+name+':'+symbol)
    freeze_registration(folder/'data',dict(manifest=manifest,splits=splits,data_sha256=sha(manifest)))
    write(folder/'status.json',dict(state='EXTRACTING_TRAIN',registration_sha256=registration_sha))
    vectors,labels,provenance,counts=training_examples(series,splits[0][1],splits[0][2])
    write(folder/'training-provenance.json',dict(counts=counts,rows=provenance,labels_sha256=sha(labels),features_sha256=sha(vectors)))
    if len(labels)<100:raise ValueError('INSUFFICIENT_TRAIN_CANDIDATES')
    models={};names={}
    for arm in ('A0','A1','A2'):
        write(folder/'status.json',dict(state='FITTING',arm=arm,counts=counts))
        names[arm]=sorted(vectors[arm][0])
        models[arm]=fit_model(vectors[arm],labels,names[arm],folder/(arm+'.cbm'))
    predictions={}
    def score(arm):
        def predict(symbol,histories,at):
            key=(symbol,at)
            if key not in predictions:
                features=build_features(histories,symbol,at)
                predictions[key]={a:None for a in models} if features is None else {
                    a:float(models[a].predict([[features[a][k] for k in names[a]]])[0]) for a in models}
            return predictions[key][arm]
        return predict
    segments={}
    for segment,start,end in splits[1:]:
        for cost,fee,slip in (('base',10,5),('stress',20,10)):
            for arm in ('C0','A0','A1','A2'):
                key=segment+'_'+cost+'_'+arm;write(folder/'status.json',dict(state='REPLAYING',segment=key))
                r=replay(series,start_ms=start,end_ms=end,scorer=None if arm=='C0' else score(arm),fee_bps=fee,slippage_bps=slip)
                r.update(registration_sha256=registration_sha,data_sha256=sha(manifest));write(folder/(key+'.json'),r);segments[key]=r
    comparisons={}
    for arm,reference in (('A2','A1'),('A1','A0')):
        candidate=segments['test_base_'+arm];baseline=segments['test_base_'+reference]
        delta=normalize_interval(paired_bootstrap(candidate['daily_pnl'],baseline['daily_pnl'],confidence=.9875))
        absolute=normalize_interval(paired_bootstrap(candidate['daily_pnl'],{k:0. for k in candidate['daily_pnl']},confidence=.9875))
        # Normalise core interval names to stable runner financial verdict schema.
        result=verdict(delta,absolute_lower=absolute.get('lower'),drawdown=candidate['max_drawdown_usdt'],
            reference_drawdown=baseline['max_drawdown_usdt'],stress_pnl=segments['test_stress_'+arm]['net_pnl_usdt'])
        comparisons[arm+'-'+reference]=dict(paired_daily_delta=delta,absolute_daily=absolute,verdict=result,promotion_allowed=False)
    report=dict(state='RESEARCH_COMPLETE',evidence_kind='SIMULATION',training_state='DIAGNOSTIC_MODELS_FIT',
        registration_sha256=registration_sha,data_sha256=sha(manifest),splits=splits,training=counts,comparisons=comparisons,confirmatory_power_assessed=False,
        results={k:{a:b for a,b in v.items() if a not in ('trades','days','daily_pnl')} for k,v in segments.items()},
        models_sha256={a:hashlib.sha256((folder/(a+'.cbm')).read_bytes()).hexdigest() for a in models},
        promotion_allowed=False,full_truth_harness='FAIL_TH11_LEGACY_SOURCE_HASH',limitations=[
        'Two liquid current-survivor pairs; not all519 candidate-population evidence',
        'Candle-aggregated taker volume is a proxy, not raw trades or L2',
        'Uncertified historical receipt latency, membership, filters and OHLC matching',
        'Historical test exposed to prior research; no untouched prospective financial proof',
        'Full Truth Harness FAIL TH-11 remains unresolved; no live adoption',
        'Simulated fees/slippage, not actual Demo commission receipts; no actual learning handoff'])
    report['report_sha256']=sha(report);write(folder/'report.json',report);render_report(folder,report,segments)
    write(folder/'status.json',dict(state='RESEARCH_COMPLETE',report_sha256=report['report_sha256'],promotion_allowed=False))
    return report

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--archive',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    try:report=run(args.archive,args.output)
    except Exception as error:
        write(args.output/'incident.json',dict(state='FAILED',reason=type(error).__name__,detail=str(error),promotion_allowed=False));raise
    print(json.dumps(dict(state=report['state'],comparisons=report['comparisons'],promotion_allowed=False)))

if __name__=='__main__':main()
