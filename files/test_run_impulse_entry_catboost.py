import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from impulse_entry_catboost import BAR,DAY,HORIZON
from run_impulse_entry_catboost import fit_predict,single_interval


class RunnerTests(unittest.TestCase):
    def test_fits_use_only_mature_labels_and_inference_has_no_label_gate(self):
        rows=[(t,0,'A','15m') for t in range(0,62*DAY,BAR)]
        x=np.ones((len(rows),10));y=np.ones(len(rows))
        y[-8:]=np.nan
        seen=[]
        class Fake:
            tree_count_=3
            def __init__(self,**kwargs):pass
            def fit(self,a,b,eval_set,**kwargs):seen.append((a.copy(),b.copy(),eval_set))
            def predict(self,a):return np.ones(len(a))
            def save_model(self,p):Path(p).write_bytes(b'frozen')
        with tempfile.TemporaryDirectory() as d,patch('run_impulse_entry_catboost.CatBoostRegressor',Fake):
            decisions,folds,pred,ids=fit_predict(rows,x,y,0,62*DAY,Path(d))
        self.assertEqual(len(folds),2)
        for f in folds:
            self.assertLess(f['train_max_label_at'],f['validation_boundary'])
            self.assertLess(f['validation_max_label_at'],f['fit_at'])
        self.assertTrue(np.isfinite(pred[-8:]).all())
        self.assertTrue(all(decisions[tuple(r[:2])] for r in rows[-8:]))
        self.assertTrue(np.all(ids[:30*96]==-1))

    def test_future_labels_do_not_change_fit_or_predictions(self):
        # Actual native fit: mutating unavailable future labels must leave forecasts identical.
        rng=np.random.default_rng(42);c=np.arange(45*DAY,step=BAR)
        rows=[(int(t),0,'A','15m') for t in c]
        x=rng.normal(size=(len(c),10));y=.2*x[:,0]+rng.normal(size=len(c))
        changed=y.copy();changed[c>=30*DAY]=1e8
        from impulse_entry_catboost import PARAMETERS
        small=dict(PARAMETERS,iterations=10,depth=2,thread_count=1)
        with tempfile.TemporaryDirectory() as a,tempfile.TemporaryDirectory() as b,patch('run_impulse_entry_catboost.PARAMETERS',small):
            first=fit_predict(rows,x,y,0,45*DAY,Path(a))
            second=fit_predict(rows,x,changed,0,45*DAY,Path(b))
        np.testing.assert_array_equal(first[2],second[2]);self.assertEqual(first[0],second[0])

    def test_single_comparison_bootstrap(self):
        c=[(t,100.) for t in range(0,40*DAY+1,BAR)]
        r=single_interval(c,c,0,40*DAY)
        self.assertEqual(r['interval'],[0.,0.]);self.assertEqual(r['confidence'],.95)
        self.assertGreaterEqual(r['days'],38)


if __name__=='__main__':unittest.main()
