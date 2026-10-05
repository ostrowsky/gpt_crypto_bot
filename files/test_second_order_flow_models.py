import unittest
import numpy as np
import second_order_flow_models as m


def source(n=1000):
    t=1_770_000_000_000+np.arange(n)*1000
    p=100+np.arange(n)*.0001
    q=np.zeros((n,20))
    for k in range(5):
        q[:,4*k]=p-.01-k*.01;q[:,4*k+1]=10+k
        q[:,4*k+2]=p+.01+k*.01;q[:,4*k+3]=12+k
    return dict(time=t,book=q,segment=np.ones(n,dtype=int),flow=np.tile(np.arange(5),(n,1)).astype(float),events=np.full(n,10),age=np.zeros(n))


class OFIModelTests(unittest.TestCase):
    def test_features_are_invariant_to_future_mutation(self):
        d=source();a=m.features(d,0)
        d['book'][401:]*=4;d['flow'][401:]*=100;d['events'][401:]=10000
        b=m.features(d,0);past=a['time']<=d['time'][400]
        np.testing.assert_allclose(a['x'][past],b['x'][past],equal_nan=True)
        np.testing.assert_array_equal(a['good'][past],b['good'][past])
        self.assertEqual(a['state_width'],41);self.assertEqual(a['x'].shape[1],81)

    def test_gaps_invalidate_past_context_and_entire_future(self):
        d=source();d['segment'][400]=-1;d['book'][400]=np.nan
        f=m.features(d,0);self.assertFalse(f['good'][(f['indices']>=400)&(f['indices']<=460)].any())
        p=(d['book'][:,0]+d['book'][:,2])/2;y=m.targets(p,d['segment'])
        self.assertTrue(np.isnan(y[390,-1]));self.assertTrue(np.isfinite(y[390,0]))
        for j,h in enumerate(m.HORIZONS):self.assertAlmostEqual(y[500,j],np.log(p[500+h]/p[500]))

    def test_strict_purge_and_inference_does_not_read_labels(self):
        t=np.arange(0,200000,5000);g=np.ones(len(t),dtype=bool);y=np.ones((len(t),6))
        cuts=dict(start=0,validation=50000,calibration=100000,test=150000,end=200000)
        a=m.masks(t,g,y,cuts)
        for key,hi in [('train',50000),('validation',100000),('calibration',150000),('test',200000)]:
            self.assertTrue((t[a[key]]+31000<hi).all())
        self.assertTrue((t[a['train']]%30000==0).all())
        y[:]=np.nan;b=m.masks(t,g,y,cuts)
        np.testing.assert_array_equal(a['inference'],b['inference']);self.assertFalse(b['test'].any())

    def test_calibration_rank_and_metrics_count_actual_zeros(self):
        actual=np.repeat(np.arange(10)[:,None],6,axis=1)*1e-4;pred=np.zeros_like(actual)
        q=m.quantiles(actual,pred);np.testing.assert_allclose(q,.0009)
        row=m.metrics(actual,pred,q)[0]
        self.assertEqual(row['direction_n'],9);self.assertEqual(row['direction_correct'],0)
        self.assertEqual(row['sign_correct'],1);self.assertEqual(row['sign_n'],10)
        self.assertEqual(row['actual_sign_counts'],[0,1,9]);self.assertEqual(row['predicted_sign_counts'],[0,10,0])
        self.assertEqual(row['interval_covered'],10);self.assertAlmostEqual(row['mae_bp'],4.5)
        balanced=actual-.00045
        self.assertEqual(m.metrics(balanced,np.zeros_like(balanced),q)[0]['direction_correct'],0)
        tiny=np.full_like(actual,1e-16)
        self.assertEqual(m.metrics(actual,tiny,q)[0]['predicted_sign_counts'],[0,10,0])
        with self.assertRaises(ValueError):m.quantiles(actual,np.full_like(actual,np.nan))

    def test_short_daily_bootstrap_cannot_claim_robust_edge(self):
        t=np.arange(8*86400000,17*86400000,3600000);a=np.ones((len(t),6))*.0001
        pred=dict(CatBoost_OFI=a.copy(),CatBoost_Book=np.zeros_like(a),Ridge_OFI=np.zeros_like(a),Zero=np.zeros_like(a))
        report=m.paired_days(t,a,pred)
        self.assertEqual(len(report),9)
        for v in report.values():
            self.assertEqual(v['n_days'],7);self.assertFalse(v['robust_claim_allowed']);self.assertAlmostEqual(v['mean_mae_gain_bp'],1)

    def test_catboost_multioutput_and_ridge_native_roundtrip(self):
        import tempfile
        from pathlib import Path
        from catboost import CatBoostRegressor
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import Ridge
        from joblib import dump,load
        rng=np.random.default_rng(42);x=rng.normal(size=(200,81));y=rng.normal(size=(200,6))
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);scaler=StandardScaler().fit(x[:120]);ridge=Ridge(alpha=10).fit(scaler.transform(x[:120]),y[:120])
            dump(dict(model=ridge,scaler=scaler),p/'ridge');native=load(p/'ridge')
            np.testing.assert_allclose(ridge.predict(scaler.transform(x[160:])),native['model'].predict(native['scaler'].transform(x[160:])))
            model=CatBoostRegressor(iterations=4,depth=2,loss_function='MultiRMSE',thread_count=1,verbose=False,allow_writing_files=False)
            model.fit(x[:120],y[:120],eval_set=(x[120:160],y[120:160]),early_stopping_rounds=2,use_best_model=True)
            model.save_model(str(p/'cat'));native=CatBoostRegressor();native.load_model(str(p/'cat'))
            np.testing.assert_allclose(model.predict(x[160:]),native.predict(x[160:]))


if __name__=='__main__':unittest.main()
