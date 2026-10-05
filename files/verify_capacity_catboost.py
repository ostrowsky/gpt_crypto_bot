"""Independent receipts, native inference, raw candle and selection-metric audit."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def checksum(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(directory, archive):
    from catboost import CatBoostRanker
    receipt = json.loads((directory/'receipt.json').read_bytes())
    for name, expected in receipt.items():
        if Path(name).name != name or checksum(directory/name) != expected:
            raise ValueError('changed result artifact: '+name)
    rows = json.loads((directory/'dataset.json').read_bytes())
    reg = json.loads((directory/'registration.json').read_bytes())
    report = json.loads((directory/'result.json').read_bytes())
    for name,expected in reg['sources'].items():
        if Path(name).name != name or checksum(directory/'source_snapshot'/name) != expected:
            raise ValueError('changed source snapshot: '+name)
    a,b = reg['cuts_ms']
    if any(r['label_available_ms'] >= (a if r['cohort']=='train' else b)
           for r in rows if r['cohort'] != 'test'):
        raise ValueError('split boundary leakage')
    test = [r for r in rows if r['cohort']=='test']
    with np.load(directory/'test_predictions.npz',allow_pickle=False) as p:
        scores,returns,x = p['scores'],p['returns'],p['features']
        if not np.isfinite(scores).all() or not np.isfinite(returns).all():
            raise ValueError('nonfinite scored evidence')
        np.testing.assert_array_equal(returns,np.array([r['returns'] for r in test]))
        np.testing.assert_array_equal(x,np.array([r['features'] for r in test]))
        np.testing.assert_array_equal(p['clocks'],[r['clock'] for r in test])
        model=CatBoostRanker();model.load_model(str(directory/'ranker.cbm'))
        native=model.predict(x.reshape(-1,x.shape[-1])).reshape(-1,2)
        np.testing.assert_array_equal(scores,native)
        swap=scores[:,0]>scores[:,1]
        selected=np.where(swap,returns[:,0],returns[:,1])
        delta=selected-returns[:,1]
        numeric={
            'groups':len(test),'replacements':int(np.count_nonzero(swap)),
            'candidate_better_count':int(np.count_nonzero(returns[:,0]>returns[:,1])),
            'decision_correct_count':int(np.count_nonzero(selected>=returns.max(axis=1))),
            'paired_mean_uplift_pp':float(delta.mean()),'paired_median_uplift_pp':float(np.median(delta)),
            'keep_mean_return_pct':float(returns[:,1].mean()),
            'always_replace_mean_return_pct':float(returns[:,0].mean()),
            'model_mean_return_pct':float(selected.mean())}
        for key, value in numeric.items():
            if not math.isclose(value,report[key],rel_tol=1e-12,abs_tol=1e-12):
                raise ValueError('metric mismatch: '+key)
        for name,choice in [('keep_mission',np.ones(len(test),dtype=int)),
                            ('model_mission',np.where(swap,0,1))]:
            known=p['known']; leader=p['leader'][np.arange(len(test)),choice]
            capture=p['capture'][np.arange(len(test)),choice]
            good=known & leader & np.isfinite(capture)
            expected={'known_groups':int(known.sum()),'leader_count':int((known & leader).sum()),
                      'early_count':int((good & (capture>=.35)).sum())}
            if any(report[name][k]!=v for k,v in expected.items()):
                raise ValueError('mission count mismatch')
    # Raw JSON prefix audit uses no experiment feature/label helper.
    m=json.loads((archive/'manifest.json').read_bytes()); raw_by_symbol={}
    indices=np.linspace(0,len(test)-1,min(32,len(test)),dtype=int)
    audited=0
    for i in indices:
        row=test[int(i)]; clock=row['clock']
        for option,symbol in enumerate(row['symbols']):
            if symbol not in raw_by_symbol:
                name=f"{symbol}_15m_{m['archive_start_ms']}_{m['archive_end_ms']}.json"
                if checksum(archive/'market'/name)!=m['input_hashes'][name]:
                    raise ValueError('raw input mismatch')
                raw_by_symbol[symbol]={int(r['t'])+900000:r for r in json.loads((archive/'market'/name).read_bytes())}
            lookup=raw_by_symbol[symbol]
            origin=lookup[clock]
            terminal=lookup[clock+4500000]
            for t in range(clock,clock+4500001,900000):
                if t not in lookup: raise ValueError('missing forward grid')
            expected=(terminal['c']/origin['c']-1)*100-(.25 if option==0 else 0)
            if not math.isclose(expected,row['returns'][option],abs_tol=1e-12):
                raise ValueError('raw return/cost mismatch')
            prefix=[lookup[clock-k*900000] for k in range(16,-1,-1)]
            closes=[v['c'] for v in prefix]
            logs=[math.log(closes[k]/closes[k-1]) for k in range(1,17)]
            mean=sum(logs)/16; volume=sum(v['v'] for v in prefix[1:])/16
            high=max(v['h'] for v in prefix[1:]);low=min(v['l'] for v in prefix[1:])
            features=[(closes[-1]/closes[-1-k]-1)*100 for k in (1,4,16)]+[
                math.sqrt(sum((v-mean)**2 for v in logs)/16)*100,
                prefix[-1]['v']/volume if volume else 0,(high/low-1)*100,
                (closes[-1]-low)/(high-low) if high>low else .5,1. if option==0 else 0.]
            np.testing.assert_allclose(features,row['features'][option],rtol=1e-10,atol=1e-12)
            audited+=1
    return {'status':'PASS','native_prediction_rows':len(test)*2,
            'raw_option_audits':audited,'raw_origin_audits':len(indices),
            'chronological_rows_checked':len(rows),'numeric_metrics':numeric,
            'scope':'numerical/provenance checks, not live parity or promotion authority'}


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--archive',type=Path,required=True)
    args=p.parse_args();r=verify(args.run,args.archive)
    with (args.run/'independent_verification.json').open('x',encoding='utf-8') as f:
        json.dump(r,f,indent=2,allow_nan=False)
    print(json.dumps(r,indent=2))
