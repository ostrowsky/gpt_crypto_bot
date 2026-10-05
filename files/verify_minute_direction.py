"""Reload frozen models and verify issued probabilities against native state."""
import argparse,json
from pathlib import Path
import numpy as np
import minute_direction_models as m
import evaluate_minute_direction as e
from minute_direction_data import sha

def verify(folder,books):
    result=json.loads((folder/'result.json').read_text(encoding='utf-8'))
    registration=result['metadata']['registration']
    for name,expected in registration['source_hashes'].items():
        if sha(folder/'source_snapshot'/name)!=expected or sha(Path(__file__).with_name(name))!=expected:
            raise ValueError('Frozen source no longer matches: '+name)
    if sha(books/'coverage.json')!=registration['coverage_sha256']:raise ValueError('Coverage changed')
    frames,cuts,_=e.load_frames(books)
    if cuts!=registration['cuts']:raise ValueError('Boundaries changed')
    rows=e.cohort(frames,'inference');scored=set(e.cohort(frames,'test'))
    with np.load(folder/'test_predictions.npz',allow_pickle=False) as f:pred={k:f[k] for k in f.files}
    np.testing.assert_array_equal(pred['time'],[frames[s]['time'][i] for s,i in rows])
    np.testing.assert_array_equal(pred['symbol'],[s for s,i in rows])
    np.testing.assert_array_equal(pred['labels'],e.matrix(frames,rows,'labels'))
    np.testing.assert_array_equal(pred['returns'],e.matrix(frames,rows,'returns'))
    np.testing.assert_array_equal(pred['scored'],[r in scored for r in rows])
    choose=np.unique(np.linspace(0,len(rows)-1,24,dtype=int));sample=[rows[i] for i in choose]
    x=e.matrix(frames,sample,'x')
    from catboost import CatBoostClassifier
    for j,h in enumerate(m.HORIZONS):
        model=CatBoostClassifier();model.load_model(str(folder/f'catboost_h{h}.cbm'))
        t=result['metadata']['catboost'][j]['temperature']
        np.testing.assert_allclose(m.temperature(model.predict_proba(x),t),pred['CatBoost'][choose,j],rtol=1e-7,atol=1e-8)
    import torch
    from safetensors.torch import load_file
    torch.set_num_threads(2);model=m.build_deeplob();model.load_state_dict(load_file(str(folder/'deeplob.safetensors')));model.eval()
    p=e.deep_probabilities(model,frames,sample)
    for j,t in enumerate(result['metadata']['deeplob']['temperature']):
        np.testing.assert_allclose(m.temperature(p[:,j],t),pred['DeepLOB'][choose,j],rtol=1e-5,atol=1e-6)
    for j,h in enumerate(m.HORIZONS):
        mask=pred['scored']
        for name in ('CatBoost','DeepLOB','Prior','Momentum','Logistic'):
            score=m.metrics(pred['labels'][mask,j],pred[name][mask,j],pred['returns'][mask,j])
            if score!=result['metrics'][str(h)][name]['pooled']:raise ValueError('Stored metrics disagree with immutable predictions')
    report=dict(status='PASS',native_state_samples=len(choose),inference_rows=len(rows),scored_rows=int(pred['scored'].sum()),
        result_sha256=sha(folder/'result.json'),predictions_sha256=sha(folder/'test_predictions.npz'))
    (folder/'verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(report)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('folder',type=Path);p.add_argument('--books',type=Path,required=True)
    a=p.parse_args();verify(a.folder,a.books)
