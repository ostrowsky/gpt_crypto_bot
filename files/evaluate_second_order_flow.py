"""Frozen CatBoost OFI/book ablation on all verified second-scale source books."""
from __future__ import annotations
import argparse,json,shutil,time,platform
from importlib.metadata import version
from pathlib import Path
import numpy as np
from minute_direction_data import sha,SYMBOLS
import second_order_flow_models as m

def load_asset(root,symbol,coverage):
    chunks=[]
    for part in coverage['assets'][symbol]['parts']:
        path=root/symbol/part['file']
        if sha(path)!=part['sha256']:raise ValueError('Prepared OFI source drift')
        with np.load(path,allow_pickle=False) as f:chunks.append({k:f[k] for k in f.files})
    data={k:np.concatenate([c[k] for c in chunks]) for k in chunks[0]}
    if np.any(np.diff(data['time'])!=1000):raise ValueError('One-second source clock broken')
    return data

def validate_source(raw,books,coverage):
    manifest=json.loads((raw/'manifest.json').read_text(encoding='utf-8'))
    if manifest['dataset']!='predict-quant/binance-future-orderbook' or manifest['revision']!='b8590b83452d7a32fbb274ff7741b6db000b3984' or len(manifest['files'])!=90:raise ValueError('Wrong maximum archive')
    if sha(raw/'manifest.json')!=coverage['source_manifest_sha256'] or coverage['source_hash']!=sha(Path(__file__).with_name('second_order_flow_data.py')):raise ValueError('Preparation provenance drift')
    expected={r['path'] for r in manifest['files']}
    actual={p.relative_to(raw).as_posix() for s in SYMBOLS for p in (raw/s).glob('*.parquet')}
    if expected!=actual:raise ValueError('Original archive files do not match registration')
    for row in manifest['files']:
        if sha(raw/row['path'])!=row['lfs']['oid']:raise ValueError('Raw source hash drift')
    for s in SYMBOLS:
        if len(coverage['assets'][s]['parts'])!=sum(r['path'].startswith(s+'/') for r in manifest['files']):raise ValueError('Prepared archive coverage incomplete')


def run(books,anchor,output,raw):
    old_paths=[anchor/'result.json',anchor/'test_predictions.npz',anchor.parent/'minute_price_paths_20261005'/'result.json',anchor.parent/'minute_price_paths_20261005'/'predictions.npz']
    old_hashes={str(p):sha(p) for p in old_paths}
    cuts=json.loads((anchor/'result.json').read_text(encoding='utf-8'))['cuts']
    coverage=json.loads((books/'coverage.json').read_text(encoding='utf-8'))
    if coverage['status']!='PREPARED_CAUSAL_BOOK_OFI':raise ValueError('Unprepared source')
    validate_source(raw,books,coverage)
    output.mkdir(parents=True,exist_ok=False);snapshot=output/'source_snapshot';snapshot.mkdir()
    sources={}
    for name in ('minute_direction_data.py','second_order_flow_data.py','prepare_second_order_flow_parallel.py','second_order_flow_models.py','evaluate_second_order_flow.py'):
        path=Path(__file__).with_name(name);sources[name]=sha(path);shutil.copy2(path,snapshot/name)
    frames={};asset_coverage={}
    for si,symbol in enumerate(SYMBOLS):
        data=load_asset(books,symbol,coverage);frame=m.features(data,si)
        mid=(data['book'][:,0]+data['book'][:,2])/2
        frame['returns']=m.targets(mid,data['segment'])[frame.pop('indices')]
        frame['masks']=m.masks(frame['time'],frame['good'],frame['returns'],cuts)
        frames[symbol]=frame
        asset_coverage[symbol]=dict(source_seconds=len(mid),source_valid=int((data['segment']>=0).sum()),
            eligible_origins=int(frame['good'].sum()),counts={k:int(v.sum()) for k,v in frame['masks'].items()})
        del data,mid
        print('Features ready',symbol,asset_coverage[symbol],flush=True)
    cohorts={}
    for split in ('train','validation','calibration','test','inference'):
        rows=[(s,i) for s,a in frames.items() for i in np.flatnonzero(a['masks'][split])]
        rows.sort(key=lambda r:(frames[r[0]]['time'][r[1]],r[0]));cohorts[split]=rows
        if not rows:raise ValueError('No '+split+' cohort')
    matrix=lambda rows,key:np.array([frames[s][key][i] for s,i in rows])
    registration=dict(registered_at=time.time(),source_hashes=sources,coverage_sha256=sha(books/'coverage.json'),cuts=cuts,
        environment=dict(python=platform.python_version(),packages={name:version(name) for name in ('numpy','scipy','scikit-learn','catboost','pyarrow','joblib')}),
        old_evidence_hashes=old_hashes,counts={k:len(v) for k,v in cohorts.items()},horizon_seconds=list(m.HORIZONS),
        disclosed_test=True,seed=42,parameters=dict(catboost=dict(depth=6,learning_rate=.05,l2=10,iterations=600,patience=60,loss='MultiRMSE'),
        ridge=dict(alpha=10),calibration=dict(nominal=.9,separate=True),direction_epsilon=m.DIRECTION_EPS))
    (output/'registration.json').write_text(json.dumps(registration,indent=2),encoding='utf-8');print('REGISTERED',registration,flush=True)
    x={k:matrix(r,'x') for k,r in cohorts.items()};y={k:matrix(r,'returns') for k,r in cohorts.items()}
    mean=y['train'].mean(axis=0);scale=np.maximum(y['train'].std(axis=0),1e-8)
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import Ridge
    from joblib import dump,load
    from catboost import CatBoostRegressor
    scaler=StandardScaler().fit(x['train']);ridge=Ridge(alpha=10).fit(scaler.transform(x['train']),(y['train']-mean)/scale)
    dump(dict(model=ridge,scaler=scaler),output/'ridge_ofi.joblib')
    predictions={};cal={};modelmeta={};state_width=frames[SYMBOLS[0]]['state_width']
    for split,sink in [('inference',predictions),('calibration',cal)]:
        sink['Ridge_OFI']=ridge.predict(scaler.transform(x[split]))*scale+mean
        sink['Zero']=np.zeros_like(y[split]);sink['Microprice']=np.repeat(matrix(cohorts[split],'micro_return')[:,None],len(m.HORIZONS),axis=1)
    for name,width in [('CatBoost_OFI',x['train'].shape[1]),('CatBoost_Book',state_width)]:
        model=CatBoostRegressor(loss_function='MultiRMSE',iterations=600,depth=6,learning_rate=.05,l2_leaf_reg=10,
            random_seed=42,thread_count=2,verbose=False,allow_writing_files=False)
        model.fit(x['train'][:,:width],(y['train']-mean)/scale,
            eval_set=(x['validation'][:,:width],(y['validation']-mean)/scale),early_stopping_rounds=60,use_best_model=True)
        model.save_model(str(output/(name+'.cbm')))
        predictions[name]=model.predict(x['inference'][:,:width])*scale+mean
        cal[name]=model.predict(x['calibration'][:,:width])*scale+mean
        modelmeta[name]=dict(trees=model.tree_count_,input_width=width,validation_loss=float(model.best_score_['validation']['MultiRMSE']),parameters=model.get_all_params())
        print('Frozen',name,modelmeta[name],flush=True)
    widths={k:m.quantiles(y['calibration'],v) for k,v in cal.items()}
    rows=cohorts['inference'];test=set(cohorts['test']);scored=np.array([r in test for r in rows]);clock=matrix(rows,'time');symbol=np.array([s for s,i in rows]);p0=matrix(rows,'mid')
    scores={name:dict(pooled=m.metrics(y['inference'][scored],p[scored],widths[name]),
        assets={s:m.metrics(y['inference'][scored&(symbol==s)],p[scored&(symbol==s)],widths[name],p0[scored&(symbol==s)]) for s in SYMBOLS}) for name,p in predictions.items()}
    paired=m.paired_days(clock[scored],y['inference'][scored],{k:v[scored] for k,v in predictions.items()})
    np.savez_compressed(output/'predictions.npz',time=clock,symbol=symbol,origin_price=p0,actual_returns=y['inference'],scored=scored,**predictions)
    pick=np.unique(np.linspace(0,len(rows)-1,32,dtype=int))
    for name,meta in modelmeta.items():
        native=CatBoostRegressor();native.load_model(str(output/(name+'.cbm')))
        np.testing.assert_allclose(native.predict(x['inference'][pick,:meta['input_width']])*scale+mean,predictions[name][pick],rtol=1e-7,atol=1e-10)
    native=load(output/'ridge_ofi.joblib')
    np.testing.assert_allclose(native['model'].predict(native['scaler'].transform(x['inference'][pick]))*scale+mean,predictions['Ridge_OFI'][pick],rtol=1e-7,atol=1e-10)
    for path,checksum in old_hashes.items():
        if sha(path)!=checksum:raise ValueError('Old evidence modified')
    for name,checksum in sources.items():
        if sha(Path(__file__).with_name(name))!=checksum:raise ValueError('Experiment code changed during run')
    if sha(books/'coverage.json')!=registration['coverage_sha256']:raise ValueError('Prepared coverage changed during run')
    for s in SYMBOLS:
        for part in coverage['assets'][s]['parts']:
            if sha(books/s/part['file'])!=part['sha256']:raise ValueError('Prepared source changed during run')
    validate_source(raw,books,coverage)
    result=dict(status='COMPLETED_RETROSPECTIVE_SECOND_OFI',runtime_eligible=False,registration=registration,coverage=coverage,
        asset_coverage=asset_coverage,models=modelmeta,train_target_mean=mean.tolist(),train_target_scale=scale.tolist(),
        calibration_widths={k:v.tolist() for k,v in widths.items()},metrics=scores,paired=paired,
        limitations=['Book-state OFI, not executed trade tape or identified cancellations','E availability assumption; receipt timestamps absent',
            'Disclosed short historical TEST; no independent forward/portfolio claim','Full Truth Harness FAIL TH-11'])
    (output/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    verification=dict(status='PASS',native_samples=len(pick),old_evidence_unchanged=True,result_sha256=sha(output/'result.json'),predictions_sha256=sha(output/'predictions.npz'),
        native_sha256={p.name:sha(p) for p in [output/'ridge_ofi.joblib',output/'CatBoost_OFI.cbm',output/'CatBoost_Book.cbm']})
    (output/'verification.json').write_text(json.dumps(verification,indent=2),encoding='utf-8')
    for name in predictions:print(name,scores[name]['pooled'],flush=True)
    print('COMPLETE second OFI',output,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--books',type=Path,required=True);p.add_argument('--anchor',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--data',type=Path,required=True)
    a=p.parse_args();run(a.books,a.anchor,a.output,a.data)
