"""Reproduce all nine research models, portable release and causal assertions.

Writes requested model/data artifacts only to the chosen ignored runtime folder.
August results for newly added methods are retrospective, never a new sealed test.
"""
import argparse
import json
import sys
from pathlib import Path
from dataclasses import replace
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'files'))
import research_forecast as rf
from research_forecast_release import pack_release,unpack_release


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',default='forecast_demo_artifacts')
    parser.add_argument('--historical-end',default=rf.ForecastConfig.end_utc)
    parser.add_argument('--release-end',default=pd.Timestamp.now(tz='UTC').floor('D').isoformat())
    args=parser.parse_args()
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    cfg=rf.ForecastConfig(end_utc=args.historical_end)
    market,manifest={},{}
    for symbol in cfg.symbols:
        market[symbol],manifest[symbol]=rf.fetch_history(symbol,cfg,out)
        print('Historical input',symbol,len(market[symbol]),flush=True)
    experiment=rf.run_experiment(market,cfg)
    rf.save_evidence(experiment,manifest,out/'benchmark_v32.json')
    assert all(n in experiment['predictions'][s] for s in cfg.symbols for n in cfg.models),experiment['status']
    print('Historical benchmark COMPLETE',flush=True)
    cv_results,cv_traces=rf.expanding_window_check(experiment,cfg,model_names=cfg.models,return_traces=True)
    (out/'cv_v32.json').write_text(json.dumps(cv_results,ensure_ascii=False,indent=2),encoding='utf-8')
    # Store historical SINGLE-ORIGIN evidence rather than misleading connected h15s.
    examples={}
    for s in cfg.symbols:
        examples[s]={}
        for name in cfg.models:
            examples[s][name]={}
            trace=next(t for t in cv_traces if t['symbol']==s and t['model']==name and t['fold']==3)
            for stage,rows,pred in [('train_oof',trace['rows'],trace['predictions']),('test',experiment['grids'][s]['test'],experiment['predictions'][s][name])]:
                row=rows.iloc[-1];origin=row.open_time+pd.Timedelta(minutes=1)
                full=experiment['prepared'][s]
                real=full.loc[(full.open_time+pd.Timedelta(minutes=1)>=origin-pd.Timedelta(minutes=15)) &
                              (full.open_time+pd.Timedelta(minutes=1)<=origin+pd.Timedelta(minutes=15))]
                examples[s][name][stage]=dict(origin=origin.isoformat(),price=float(row.close),prediction=(row.close*np.exp(pred[-1])).tolist(),
                    actual_times=[t.isoformat() for t in real.open_time+pd.Timedelta(minutes=1)],actual=real.close.tolist())
    # New release has new training weights: retrospective benchmark never certifies it.
    end=rf.utc(args.release_end)
    release_cfg=replace(cfg,end_utc=end.isoformat(),history_days=42)
    release=dict(policies={},widths={},predictions={},choices={},metadata=experiment['metadata'])
    release.update({k:experiment[k] for k in ('results','status','direction_baselines')})
    provenance=dict(status='UNPROVEN_FORWARD_CANDIDATE',end_utc=end.isoformat(),
        train_start=(end-pd.Timedelta(days=42)).isoformat(),train_cutoff=(end-pd.Timedelta(days=7)).isoformat(),
        tune_cutoff=(end-pd.Timedelta(days=3)).isoformat(),calibration_cutoff=end.isoformat(),
        train_days=35,tune_days=4,calibration_days=3,historical_benchmark_config=rf.asdict(cfg),
        historical_benchmark_scope='Retrospective August comparison; new methods are not independently sealed after this period was viewed',
        historical_weights_differ_from_release=True,cv_results=cv_results,examples=examples,inputs={},causality=[],roundtrip=[],versions=experiment['metadata']['versions'])
    for symbol in cfg.symbols:
        raw,receipt=rf.fetch_archive_history(symbol,release_cfg,out/'archives')
        print('Release input',symbol,len(raw),flush=True)
        provenance['inputs'][symbol]=receipt
        full=rf.prepare(raw,release_cfg)
        finite=np.isfinite(full[rf.FEATURES+rf.target_columns(cfg)]).all(axis=1)
        train=full.loc[finite & (full.label_available_at<end-pd.Timedelta(days=7))]
        tune=full.loc[finite & (full.available_at>=end-pd.Timedelta(days=7)) & (full.label_available_at<end-pd.Timedelta(days=3))]
        cal=full.loc[finite & (full.available_at>=end-pd.Timedelta(days=3)) & (full.label_available_at<end)]
        assert train.label_available_at.max()<tune.available_at.min() and tune.label_available_at.max()<cal.available_at.min()
        cal=rf.evaluation_grid(cal,cfg.calibration_stride)
        release['policies'][symbol]={};release['widths'][symbol]={};release['predictions'][symbol]={};release['choices'][symbol]='Persistence'
        release['policies'][symbol]['Persistence']=None
        release['widths'][symbol]['Persistence']=rf.finite_sample_widths(cal[rf.target_columns(cfg)].to_numpy(),np.zeros((len(cal),cfg.horizon)))
        release['predictions'][symbol]['Persistence']=np.zeros((len(cal),cfg.horizon))
        for name in cfg.models:
            print('Release fit',symbol,name,flush=True)
            model=rf.build_policy(name,train,tune,full,release_cfg)
            pred=model.predict(full,cal)
            q=rf.finite_sample_widths(cal[rf.target_columns(cfg)].to_numpy(),pred)
            release['policies'][symbol][name]=model;release['widths'][symbol][name]=q;release['predictions'][symbol][name]=pred
            # Change every future OHLCV value and prepared feature; use identical frozen weights.
            origin=cal.iloc[[0]];idx=int(origin.time_idx.iloc[0])
            changed=raw.copy();future=changed.index>idx
            for column in ('open','high','low','close'):changed.loc[future,column]*=1.3
            changed.loc[future,'volume']*=3
            changed_full=rf.prepare(changed,release_cfg)
            before=model.predict(full,origin);after=model.predict(changed_full,changed_full.iloc[[idx]])
            prefix=model.predict(full.iloc[:idx+1],origin)
            np.testing.assert_allclose(before,after,rtol=1e-6,atol=1e-8)
            np.testing.assert_allclose(before,prefix,rtol=1e-6,atol=1e-8)
            provenance['causality'].append(dict(symbol=symbol,model=name,future_mutation='PASS',closed_prefix='PASS'))
        envelope=pack_release(release,replace(release_cfg,symbols=tuple(release['policies'])),provenance)
        restored,restore_cfg,_=unpack_release(envelope)
        for name in cfg.models:
            before=release['policies'][symbol][name].predict(full,cal.tail(1))
            after=restored['policies'][symbol][name].predict(full,cal.tail(1))
            np.testing.assert_allclose(before,after,rtol=1e-6,atol=1e-8)
            provenance['roundtrip'].append(dict(symbol=symbol,model=name,status='PASS',max_error=float(np.max(np.abs(before-after)))))
        # Partial checkpoint outside Git for interrupted preparation.
        envelope=pack_release(release,replace(release_cfg,symbols=tuple(release['policies'])),provenance)
        (out/'release_v32.json').write_text(json.dumps(envelope),encoding='utf-8')
    print('RELEASE COMPLETE; all nine methods/all three assets',flush=True)


if __name__=="__main__":
    main()
