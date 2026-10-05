"""Honest horizons/splits and unchanged future-blind inference features."""
import unittest
import warnings
from unittest.mock import patch
import numpy as np
import minute_direction_models as m
import evaluate_minute_direction as e
from minute_direction_partial import validate_state

def frames(n=1200,start=1772668800000):
    time=start+np.arange(n)*10000;mid=100*np.exp(.0002*np.sin(np.arange(n)/9))
    book=np.empty((n,40))
    for k in range(10):
        book[:,4*k]=mid+.001*(k+1);book[:,4*k+2]=mid-.001*(k+1)
        book[:,4*k+1]=1+k/10;book[:,4*k+3]=2+k/10
    return dict(time=time,book=book,flow=np.ones((n,5)),segment=np.ones(n,dtype=np.int64),trade_gap=np.zeros(n,dtype=bool))

class ModelTests(unittest.TestCase):
    def test_unconverged_control_is_failure_not_a_valid_baseline(self):
        from sklearn.exceptions import ConvergenceWarning
        def bad_fit(*args):warnings.warn('not converged',ConvergenceWarning)
        with patch('sklearn.linear_model.LogisticRegression') as ctor:
            ctor.return_value.fit.side_effect=bad_fit
            with self.assertRaises(ConvergenceWarning):e.fit_logistic(np.zeros((3,2)),np.array([0,1,2]))
    def test_future_mutation_and_prefix_equivalence_for_features(self):
        a=frames();cut=600
        original=m.causal_features(**a,symbol_index=0)
        changed={k:v.copy() for k,v in a.items()};changed['book'][cut+1:]*=3;changed['flow'][cut+1:]*=9
        other=m.causal_features(**changed,symbol_index=0)
        prefix=m.causal_features(**{k:v[:cut+1] for k,v in a.items()},symbol_index=0)
        np.testing.assert_array_equal(original[0][:cut+1],other[0][:cut+1]);np.testing.assert_array_equal(original[0][:cut+1],prefix[0])
        np.testing.assert_array_equal(original[1][:cut+1],prefix[1])
    def test_targets_exact_wall_clock_and_neutral_band(self):
        mid=np.exp(np.arange(80)*.0001);segment=np.ones(80,dtype=int)
        r,y=m.targets(mid,segment)
        for j,h in enumerate(m.HORIZONS):self.assertAlmostEqual(r[0,j],h*6*.0001);self.assertEqual(y[0,j],2)
        self.assertTrue((y[-30:,2]==-1).all())
        segment[10:]=2;_,y=m.targets(mid,segment);self.assertEqual(y[0,1],-1)
    def test_purging_all_labels_close_before_next_split(self):
        a=frames(n=3000);cuts=m.boundaries(int(a['time'][0]),int(a['time'][-1]));good=np.ones(len(a['time']),dtype=bool);labels=np.ones((len(good),3),dtype=int)
        masks=m.split_masks(a['time'],good,labels,cuts)
        for name,bound in [('train','validation'),('validation','calibration'),('calibration','test')]:
            self.assertTrue((a['time'][masks[name]]+5*m.MINUTE+10000<cuts[bound]).all())
        self.assertTrue((a['time'][masks['train']]%(5*m.MINUTE)==0).all())
    def test_gap_never_bridges_sequence_or_target(self):
        a=frames();a['segment'][400]=-1
        _,good,mid=m.causal_features(**a,symbol_index=0)
        self.assertFalse(good[402]);self.assertFalse(good[498])
        a=frames();a['trade_gap'][400]=True
        _,good,_=m.causal_features(**a,symbol_index=0);self.assertFalse(good[402])
    def test_normalization_causal_price_ratios_not_future_statistics(self):
        a=frames();book=a['book'][100:200];x=m.normalize_sequence(book)
        self.assertEqual(x.shape,(100,40));self.assertTrue(np.isfinite(x).all())
        scaled=book.copy();scaled[:,0::4]*=2;scaled[:,2::4]*=2;scaled[:,1::4]*=3;scaled[:,3::4]*=3
        np.testing.assert_allclose(x,m.normalize_sequence(scaled),rtol=1e-5,atol=1e-5)
    def test_calibration_and_counts_preserve_base_rate(self):
        y=np.tile([0,1,2],30);p=np.tile([.1,.8,.1],(90,1));t=m.calibrate(p,y)
        self.assertGreater(t,1)
        result=m.metrics(y,m.temperature(p,t),np.where(y==2,.001,np.where(y==0,-.001,0)))
        self.assertEqual(result['class_counts'],[30,30,30]);self.assertEqual(result['n'],90)
        self.assertEqual(sum(map(sum,result['confusion'])),90);self.assertEqual(result['nonneutral_n'],60)
    def test_actual_cnn_inception_lstm_three_horizon_topology(self):
        import torch
        torch.set_num_threads(2)
        from torch import nn
        model=m.build_deeplob().eval()
        self.assertGreater(sum(isinstance(k,nn.Conv2d) for k in model.modules()),8)
        self.assertTrue(hasattr(model,'branch3') and hasattr(model,'branch5') and hasattr(model,'branchpool'))
        self.assertIsInstance(model.lstm,nn.LSTM)
        with torch.no_grad():self.assertEqual(tuple(model(torch.zeros(2,100,40)).shape),(2,3,3))
    def test_standing_top20_validation(self):
        bids=[[100-i*.01,1] for i in range(20)];asks=[[100.01+i*.01,2] for i in range(20)]
        self.assertEqual(validate_state(bids,asks).shape,(40,))
        bids[0][1]=0
        with self.assertRaises(ValueError):validate_state(bids,asks)
        with self.assertRaises(ValueError):validate_state(bids[:10],asks)
    def test_deeplob_native_checkpoint_roundtrip(self):
        import tempfile
        from pathlib import Path
        import torch
        from safetensors.torch import save_file,load_file
        torch.set_num_threads(2);torch.manual_seed(42)
        model=m.build_deeplob().eval();x=torch.randn(2,100,40)
        with torch.no_grad():expected=model(x).numpy()
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'model.safetensors';save_file(model.state_dict(),str(path))
            restored=m.build_deeplob().eval();restored.load_state_dict(load_file(str(path)))
            with torch.no_grad():np.testing.assert_array_equal(expected,restored(x).numpy())

class AccountTests(unittest.TestCase):
    def setup_account(self):
        a=frames(n=150);a['book'][:,0::4]=100.01+np.arange(10)*.01;a['book'][:,2::4]=100-np.arange(10)*.01
        cuts=dict(test=int(a['time'][0]),end=int(a['time'][-1]))
        return {'BTCUSDT':a},[('BTCUSDT',0)],np.array([[.1,.1,.8]]),cuts
    def test_costs_delay_and_no_future_gap_filtering(self):
        f,r,p,c=self.setup_account();base=e.account(f,r,p,1,c)
        self.assertLess(base['return_pct'],0);self.assertEqual(base['trades_detail'][0]['entry'],c['test']+10000)
        f['BTCUSDT']['segment'][7:12]=-1
        gap=e.account(f,r,p,1,c)
        self.assertEqual(gap['trades'],1);self.assertEqual(gap['trades_detail'][0]['exit'],c['test']+120000)
        self.assertGreater(gap['missing_holding_marks'],0);self.assertFalse(gap['drawdown_fully_observed'])
    def test_funding_and_double_fees_have_real_cost(self):
        f,r,p,c=self.setup_account();base=e.account(f,r,p,1,c)
        stress=e.account(f,r,p,1,c,15,10);self.assertLess(stress['return_pct'],base['return_pct'])
        f['BTCUSDT']['funding']=np.array([[c['test']+60000,.01,100]])
        funded=e.account(f,r,p,1,c);self.assertGreater(funded['funding_paid_initial_capital'],0)
        self.assertLess(funded['return_pct'],base['return_pct'])
    def test_zero_trades_cash_control_and_slots(self):
        f,r,p,c=self.setup_account();p[:]=[.3,.4,.3]
        empty=e.account(f,r,p,1,c);self.assertEqual(empty['trades'],0);self.assertEqual(empty['return_pct'],0)
        self.assertEqual(empty['max_positions'],0)

if __name__=='__main__':unittest.main()
