import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from joint_direction_amplitude import DAY,BAR,PARAMETERS
from run_joint_direction_amplitude import fit_blocks


class FitTests(unittest.TestCase):
    def test_native_future_mutation_and_tail_issuance(self):
        rng=np.random.default_rng(42);clock=np.arange(45*DAY,step=BAR)
        x=rng.normal(size=(len(clock),10));gross=.3*x[:,0]+rng.normal(size=len(clock))
        def payload(g):
            return dict(clock=clock,index=np.zeros(len(clock),dtype=int),x=x.copy(),gross=g.copy())
        changed=gross.copy();changed[clock>=30*DAY]=np.nan
        import joint_direction_amplitude as module
        with tempfile.TemporaryDirectory() as a,tempfile.TemporaryDirectory() as b,patch.object(module,'PARAMETERS',dict(PARAMETERS,iterations=8,depth=2,thread_count=1)):
            za=payload(gross);zb=payload(changed)
            da,fa=fit_blocks(za,0,45*DAY,7.5,5,Path(a));db,fb=fit_blocks(zb,0,45*DAY,7.5,5,Path(b))
        np.testing.assert_array_equal(za['prediction'],zb['prediction']);self.assertEqual(da,db)
        self.assertTrue(np.isfinite(zb['prediction'][clock>=30*DAY]).all())
        f=fa[0];self.assertLess(f['train_max_label_at'],f['boundaries'][0])
        self.assertLess(f['validation_max_label_at'],f['boundaries'][1]);self.assertLess(f['calibration_max_label_at'],f['fit_at'])
        self.assertEqual(len(f['models']),3)

    def test_small_sample_explicitly_passthrough(self):
        z=dict(clock=np.array([30*DAY]),index=np.array([0]),x=np.ones((1,10)),gross=np.array([1.]))
        with tempfile.TemporaryDirectory() as d:
            decisions,folds=fit_blocks(z,0,40*DAY,7.5,5,Path(d))
        self.assertEqual(decisions,{});self.assertEqual(folds[0]['state'],'UNTRAINED_PASSTHROUGH')
        self.assertEqual(z['fold'][0],-1)


if __name__=='__main__':unittest.main()
